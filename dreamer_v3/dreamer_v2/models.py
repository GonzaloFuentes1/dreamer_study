import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import OneHotCategorical, OneHotCategoricalStraightThrough, Distribution

class RSSM_V2(nn.Module):
    """
    RSSM with Discrete Latents (Categorical) and KL Balancing.
    """
    def __init__(self, action_dim, stoch_dim=32, stoch_classes=32, deter_dim=200, hidden_dim=200, embed_dim=1024):
        super().__init__()
        self.stoch_dim = stoch_dim         # e.g., 32 categorical variables
        self.stoch_classes = stoch_classes # e.g., 32 classes each
        self.deter_dim = deter_dim
        
        # The total flat size of the stochastic state is (stoch_dim * stoch_classes)
        # because we use one-hot encoding.
        stoch_flat_dim = stoch_dim * stoch_classes

        # RNN Cell (LayerNorm GRU is often used in V2, using standard GRU for simplicity here)
        self.cell = nn.GRUCell(hidden_dim, deter_dim)

        # Pre-GRU processing: Combine previous stochastic state + previous action
        self.pre_gru_net = nn.Sequential(
            nn.Linear(stoch_flat_dim + action_dim, hidden_dim),
            nn.ELU(),
            nn.LayerNorm(hidden_dim) # LayerNorm added for V2 stability
        )

        # Prior (Transition Model): Predicts next s_t from h_t
        # Output is logits for each of the 'stoch_dim' categorical variables
        self.prior_net = nn.Sequential(
            nn.Linear(deter_dim, hidden_dim),
            nn.ELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, stoch_flat_dim)
        )

        # Posterior (Representation Model): Predicts s_t from h_t and e_t (image embedding)
        self.posterior_net = nn.Sequential(
            nn.Linear(deter_dim + embed_dim, hidden_dim),
            nn.ELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, stoch_flat_dim)
        )
    
    def get_stoch_state(self, logits):
        """
        Takes flat logits (B, 32*32), reshapes to (B, 32, 32), 
        and returns the distribution + samples (straight through).
        """
        shape = logits.shape
        # Reshape to (Batch, stoch_dim, stoch_classes)
        logits = logits.view(shape[:-1] + (self.stoch_dim, self.stoch_classes))
        
        # In DreamerV2 original: dist = tfd.Independent(tfd.OneHotCategorical(logits=logits), 1)
        # OneHotCategoricalStraightThrough allows gradient flow via hard samples
        # but torch.distributions doesn't support "StraightThrough" natively in all versions cleanly like TF.
        # usually we sample and use gumbel_softmax manually or use the RelaxedOneHotCategorical.
        # Here we use a safe manual implementation.
        
        dist = OneHotCategorical(logits=logits)
        log_prob = dist.logits
        # Sample with straight-through gradients
        # Use gumbel-softmax: (B, 32, 32)
        stoch = F.gumbel_softmax(logits, tau=1.0, hard=True, dim=-1)
        
        # Flatten back for input to next step: (B, 32*32)
        stoch_flat = stoch.view(shape[:-1] + (self.stoch_dim * self.stoch_classes,))
        
        return dist, stoch, stoch_flat

    def step(self, prev_stoch_flat, prev_action, prev_deter):
        x = torch.cat([prev_stoch_flat, prev_action], dim=-1)
        x = self.pre_gru_net(x)
        deter = self.cell(x, prev_deter)
        return deter

    def observe(self, embed, action, state=None):
        batch_size, seq_len, _ = embed.shape
        if state is None:
            deter = torch.zeros(batch_size, self.deter_dim, device=embed.device)
            # Initial stochastic state is zeros? Or learned? Usually zeros works for start.
            stoch_flat = torch.zeros(batch_size, self.stoch_dim * self.stoch_classes, device=embed.device)
        else:
            stoch_flat, deter = state

        prior_logits_list = []
        post_logits_list = []
        deters_list = []
        stochs_flat_list = []

        for t in range(seq_len):
            deter = self.step(stoch_flat, action[:, t], deter)
            
            # Prior
            prior_logits = self.prior_net(deter)
            
            # Posterior
            post_logits = self.posterior_net(torch.cat([deter, embed[:, t]], dim=-1))
            
            # Sample from Posterior for training
            dist_post, stoch, stoch_flat = self.get_stoch_state(post_logits)

            prior_logits_list.append(prior_logits)
            post_logits_list.append(post_logits)
            deters_list.append(deter)
            stochs_flat_list.append(stoch_flat)

        prior_logits = torch.stack(prior_logits_list, dim=1)
        post_logits = torch.stack(post_logits_list, dim=1)
        deters = torch.stack(deters_list, dim=1)
        stochs_flat = torch.stack(stochs_flat_list, dim=1)

        return prior_logits, post_logits, stochs_flat, deters

    def imagine(self, actor, start_state, horizon, is_discrete=False):
        """
        Imagine ahead using the PRIOR only (no images).
        """
        stoch_flat, deter = start_state
        
        stochs_list = []
        deters_list = []
        actions_list = []
        
        curr_stoch_flat = stoch_flat
        curr_deter = deter
        
        for t in range(horizon):
            # Actor decides action based on internal state s_t, h_t
            feat = torch.cat([curr_deter, curr_stoch_flat], dim=-1)
            action = actor(feat.detach()) # Stop grad for actor
            
            # For continuous actions, add exploration noise during imagination
            if not is_discrete:
                action = torch.tanh(action)
            
            curr_deter = self.step(curr_stoch_flat, action, curr_deter)
            prior_logits = self.prior_net(curr_deter)
            dist, stoch, curr_stoch_flat = self.get_stoch_state(prior_logits)
            
            stochs_list.append(curr_stoch_flat)
            deters_list.append(curr_deter)
            actions_list.append(action)

        return torch.stack(deters_list, dim=1), torch.stack(stochs_list, dim=1), torch.stack(actions_list, dim=1)

    def kl_loss(self, post_logits, prior_logits, alpha=0.8):
        """
        KL Balancing:
        loss = alpha * KL(stop_grad(post), prior) + (1-alpha) * KL(post, stop_grad(prior))
        """
        # Shapes are (B, T, 32*32). Reshape to (B, T, 32, 32)
        shape = post_logits.shape
        post_logits = post_logits.view(shape[:-1] + (self.stoch_dim, self.stoch_classes))
        prior_logits = prior_logits.view(shape[:-1] + (self.stoch_dim, self.stoch_classes))
        
        post_dist = OneHotCategorical(logits=post_logits)
        prior_dist = OneHotCategorical(logits=prior_logits)
        
        # Direct KL Divergence for Categorical distributions
        # KL(p, q) = sum p * (log p - log q)
        
        # Stop gradient versions
        post_dist_sg = OneHotCategorical(logits=post_logits.detach())
        prior_dist_sg = OneHotCategorical(logits=prior_logits.detach())

        # Calculate both directions
        kl_lhs = torch.distributions.kl_divergence(post_dist_sg, prior_dist)
        kl_rhs = torch.distributions.kl_divergence(post_dist, prior_dist_sg)
        
        # V2 paper: sum over the 32 discrete variables.
        # kl_lhs shape: (B, T, 32). Sum over dim -1 -> (B, T)
        loss = alpha * kl_lhs.sum(dim=-1) + (1.0 - alpha) * kl_rhs.sum(dim=-1)
        
        return loss.mean()


class ConvEncoder(nn.Module):
    def __init__(self, input_channels=3, depth=32, stride=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, depth, 4, stride),
            nn.ReLU(),
            nn.Conv2d(depth, depth * 2, 4, stride),
            nn.ReLU(),
            nn.Conv2d(depth * 2, depth * 4, 4, stride),
            nn.ReLU(),
            nn.Conv2d(depth * 4, depth * 8, 4, stride),
            nn.ReLU()
        )
        
    def forward(self, obs):
        x = self.net(obs)
        return x.reshape(x.shape[0], -1)


class ConvDecoder(nn.Module):
    def __init__(self, input_dim, depth=32, output_channels=3):
        super().__init__()
        self.linear = nn.Linear(input_dim, 32 * 1 * 1)
        self.convs = nn.Sequential(
            nn.ConvTranspose2d(32, depth * 4, 5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(depth * 4, depth * 2, 5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(depth * 2, depth, 6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(depth, output_channels, 6, stride=2),
        )

    def forward(self, features):
        x = self.linear(features)
        x = x.view(x.shape[0], 32, 1, 1)
        x = self.convs(x)
        return x


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden=400, layers=2, act=nn.ELU, dist=None):
        super().__init__()
        model = []
        for _ in range(layers):
            model.append(nn.Linear(input_dim, hidden))
            model.append(act())
            input_dim = hidden
        model.append(nn.Linear(hidden, output_dim))
        self.net = nn.Sequential(*model)
        self.dist = dist

    def forward(self, x):
        x = self.net(x)
        if self.dist == 'normal':
            from torch.distributions import Normal
            return Normal(x, 1.0)
        if self.dist == 'bernoulli':
            from torch.distributions import Bernoulli
            return Bernoulli(logits=x)
        return x
