import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import OneHotCategorical, OneHotCategoricalStraightThrough, Distribution

class RSSM_V2(nn.Module):
    def __init__(self, action_dim, stoch_dim=32, stoch_classes=32, deter_dim=200, hidden_dim=200, embed_dim=1024):
        super().__init__()
        self.stoch_dim = stoch_dim
        self.stoch_classes = stoch_classes
        self.deter_dim = deter_dim
        
        stoch_flat_dim = stoch_dim * stoch_classes

        self.cell = nn.GRUCell(hidden_dim, deter_dim)

        self.pre_gru_net = nn.Sequential(
            nn.Linear(stoch_flat_dim + action_dim, hidden_dim),
            nn.ELU(),
            nn.LayerNorm(hidden_dim)
        )

        self.prior_net = nn.Sequential(
            nn.Linear(deter_dim, hidden_dim),
            nn.ELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, stoch_flat_dim)
        )

        self.posterior_net = nn.Sequential(
            nn.Linear(deter_dim + embed_dim, hidden_dim),
            nn.ELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, stoch_flat_dim)
        )
    
    def get_stoch_state(self, logits):
        shape = logits.shape
        logits = logits.view(shape[:-1] + (self.stoch_dim, self.stoch_classes))
        logits = torch.clamp(logits, -10.0, 10.0) # Clamp for stability
        
        dist = OneHotCategorical(logits=logits)
        log_prob = dist.logits
        stoch = F.gumbel_softmax(logits, tau=1.0, hard=True, dim=-1)
        
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
            stoch_flat = torch.zeros(batch_size, self.stoch_dim * self.stoch_classes, device=embed.device)
        else:
            stoch_flat, deter = state

        prior_logits_list = []
        post_logits_list = []
        deters_list = []
        stochs_flat_list = []

        for t in range(seq_len):
            deter = self.step(stoch_flat, action[:, t], deter)
            
            prior_logits = self.prior_net(deter)
            prior_logits = torch.clamp(prior_logits, -10.0, 10.0) # Clamp for stability
            
            post_logits = self.posterior_net(torch.cat([deter, embed[:, t]], dim=-1))
            post_logits = torch.clamp(post_logits, -10.0, 10.0) # Clamp for stability
            
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
        stoch_flat, deter = start_state
        
        stochs_list = []
        deters_list = []
        actions_list = []
        
        curr_stoch_flat = stoch_flat
        curr_deter = deter
        
        for t in range(horizon):
            feat = torch.cat([curr_deter, curr_stoch_flat], dim=-1)
            
            action = actor(feat)
            
            if not is_discrete:
                action = torch.tanh(action)
            
            curr_deter = self.step(curr_stoch_flat, action, curr_deter)
            prior_logits = self.prior_net(curr_deter)
            prior_logits = torch.clamp(prior_logits, -10.0, 10.0) # Clamp for stability
            dist, stoch, curr_stoch_flat = self.get_stoch_state(prior_logits)
            
            stochs_list.append(curr_stoch_flat)
            deters_list.append(curr_deter)
            actions_list.append(action)

        return torch.stack(deters_list, dim=1), torch.stack(stochs_list, dim=1), torch.stack(actions_list, dim=1)

    def kl_loss(self, post_logits, prior_logits, alpha=0.8, free_nats=1.0):
        shape = post_logits.shape
        post_logits = post_logits.view(shape[:-1] + (self.stoch_dim, self.stoch_classes))
        prior_logits = prior_logits.view(shape[:-1] + (self.stoch_dim, self.stoch_classes))
        
        post_dist = OneHotCategorical(logits=post_logits)
        prior_dist = OneHotCategorical(logits=prior_logits)
        
        post_dist_sg = OneHotCategorical(logits=post_logits.detach())
        prior_dist_sg = OneHotCategorical(logits=prior_logits.detach())

        kl_lhs = torch.distributions.kl_divergence(post_dist_sg, prior_dist)
        kl_rhs = torch.distributions.kl_divergence(post_dist, prior_dist_sg)
        
        kl_lhs_sum = kl_lhs.sum(dim=-1)
        kl_rhs_sum = kl_rhs.sum(dim=-1)
        
        loss = alpha * kl_lhs_sum + (1.0 - alpha) * kl_rhs_sum
        
        loss = torch.maximum(loss, torch.ones_like(loss) * free_nats)
        
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
