import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import OneHotCategorical, OneHotCategoricalStraightThrough
from dreamer_v2.networks import ConvEncoder, ConvDecoder

class RSSM_V2(nn.Module):
    def __init__(self, action_dim, deter_dim=600, hidden_dim=600):
        super().__init__()
        self.stoch_dim = 32
        self.stoch_classes = 32
        self.deter_dim = deter_dim
        self.hidden_dim = hidden_dim
        
        stoch_flat_dim = self.stoch_dim * self.stoch_classes

        self.cell = nn.GRUCell(hidden_dim, deter_dim)

        self.pre_gru_net = nn.Sequential(
            nn.Linear(stoch_flat_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU()
        )
        
        output_size = self.stoch_dim * self.stoch_classes

        self.prior_net = nn.Sequential(
            nn.Linear(deter_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, output_size)
        )

        self.posterior_net = nn.Sequential(
            nn.Linear(deter_dim + 1024, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, output_size)
        )
    
    def sample_stoch(self, params):
        shape = params.shape
        logits = params.view(*shape[:-1], self.stoch_dim, self.stoch_classes)
        logits = torch.clamp(logits, -10.0, 10.0)
        
        dist = OneHotCategoricalStraightThrough(logits=logits)
        stoch = dist.rsample()
        
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
            
            prior_params = self.prior_net(deter)
            
            post_params = self.posterior_net(torch.cat([deter, embed[:, t]], dim=-1))
            
            dist_post, stoch, stoch_flat = self.sample_stoch(post_params)

            prior_logits_list.append(prior_params)
            post_logits_list.append(post_params)
            deters_list.append(deter)
            stochs_flat_list.append(stoch_flat)

        # Stack results: [batch, seq_len, ...]
        prior_logits = torch.stack(prior_logits_list, dim=1)  # [batch, seq, 1024]
        post_logits = torch.stack(post_logits_list, dim=1)    # [batch, seq, 1024]
        deters = torch.stack(deters_list, dim=1)              # [batch, seq, 600]
        stochs_flat = torch.stack(stochs_flat_list, dim=1)    # [batch, seq, 1024] flat
        
        # Note: logits can be reshaped to [batch, seq, 32, 32] for KL calculation
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
            dist, stoch, curr_stoch_flat = self.sample_stoch(prior_logits)
            
            stochs_list.append(curr_stoch_flat)
            deters_list.append(curr_deter)
            actions_list.append(action)

        return torch.stack(deters_list, dim=1), torch.stack(stochs_list, dim=1), torch.stack(actions_list, dim=1)

    def kl_loss(self, post_logits, prior_logits, alpha=0.8, free_nats=1.0):
        """KL Balancing with Automatic Differentiation (Section 2.2)
        
        KL[sg(post) || prior] trains the dynamics (prior)
        KL[post || sg(prior)] trains the representation (posterior)
        
        We balance these with alpha to prevent one from dominating.
        free_nats prevents over-optimization when KL is already small.
        """
        shape = post_logits.shape
        # Reshape to [batch, seq, 32, 32] - 32 categorical distributions
        post_logits = post_logits.view(shape[:-1] + (self.stoch_dim, self.stoch_classes))
        prior_logits = prior_logits.view(shape[:-1] + (self.stoch_dim, self.stoch_classes))
        
        # Create distributions
        post_dist = OneHotCategorical(logits=post_logits)
        prior_dist = OneHotCategorical(logits=prior_logits)
        
        # Stop-gradient versions for KL balancing
        post_dist_sg = OneHotCategorical(logits=post_logits.detach())
        prior_dist_sg = OneHotCategorical(logits=prior_logits.detach())

        # KL divergence for each of the 32 categorical distributions
        # kl_divergence returns [batch, seq, 32] - one value per categorical
        kl_lhs = torch.distributions.kl_divergence(post_dist_sg, prior_dist)
        kl_rhs = torch.distributions.kl_divergence(post_dist, prior_dist_sg)
        
        # Sum over the 32 categorical dimensions to get total KL per timestep
        # Result: [batch, seq]
        kl_lhs_sum = kl_lhs.sum(dim=-1)
        kl_rhs_sum = kl_rhs.sum(dim=-1)
        
        # KL balancing: alpha controls dynamics vs representation learning
        loss = alpha * kl_lhs_sum + (1.0 - alpha) * kl_rhs_sum
        
        # Free nats: don't penalize KL below this threshold (prevents over-compression)
        loss = torch.maximum(loss, torch.ones_like(loss) * free_nats)
        
        return loss.mean()


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden=400, layers=4, act=nn.ELU, dist=None):
        """Multi-Layer Perceptron
        
        Paper Table: MPL number of layers = 4, MPL number of units = 400
        This creates: input → 400 → 400 → 400 → 400 → output
        """
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


class WorldModel(nn.Module):
    """Dreamer V2 World Model.
    
    Learns latent dynamics and reconstructs observations.
    Composed of:
    - Encoder: obs → embedding
    - RSSM: latent dynamics with categorical stochastic states (32×32)
    - Decoder: latent → obs reconstruction
    - Reward predictor: latent → reward
    - Discount predictor: latent → continuation probability (NEW in V2)
    
    The world model is trained with:
    1. Observation reconstruction loss
    2. KL divergence loss (with balancing for V2)
    3. Reward prediction loss
    4. Discount prediction loss (predicts episode continuation)
    
    Note: Discount predictor is NEW in V2. It predicts whether the episode
    continues (used for discount γ_t in value estimation).
    """
    
    def __init__(self, obs_shape, action_dim, config):
        super().__init__()
        
        # Extract config params
        use_grayscale = config['model'].get('grayscale', False)
        input_channels = 1 if use_grayscale else obs_shape[0]
        output_channels = input_channels
        embed_dim = config['model'].get('embed_dim', 1024)
        deter_dim = config['model']['rssm']['deter_dim']
        hidden_dim = config['model']['rssm']['hidden_dim']
        
        # Create components
        self.encoder = ConvEncoder(input_channels=input_channels, embed_dim=embed_dim)
        
        self.rssm = RSSM_V2(
            action_dim=action_dim,
            deter_dim=deter_dim,
            hidden_dim=hidden_dim
        )
        
        stoch_size = self.rssm.stoch_dim * self.rssm.stoch_classes
        feature_dim = deter_dim + stoch_size
        
        self.decoder = ConvDecoder(feature_dim, output_channels=output_channels)
        
        mlp_hidden = config.get('network', {}).get('mlp_units', 400)
        mlp_layers = config.get('network', {}).get('mlp_layers', 4)
        
        self.reward_model = RewardModel(feature_dim, hidden=mlp_hidden, layers=mlp_layers)
        self.discount_model = DiscountModel(feature_dim, hidden=mlp_hidden, layers=mlp_layers)
    
    def parameters(self):
        """Return all trainable parameters of the world model."""
        return list(self.encoder.parameters()) + \
               list(self.rssm.parameters()) + \
               list(self.decoder.parameters()) + \
               list(self.reward_model.parameters()) + \
               list(self.discount_model.parameters())


class RewardModel(nn.Module):
    """Reward predictor for Dreamer V2.
    
    Predicts scalar reward from latent state (deter + stoch).
    Part of the world model, trained with actual rewards.
    
    Architecture: 4-layer MLP with 400 units (paper spec).
    """
    
    def __init__(self, feature_dim, hidden=400, layers=4):
        super().__init__()
        self.net = MLP(feature_dim, 1, hidden=hidden, layers=layers)
    
    def forward(self, features):
        """Predict reward from features.
        
        Args:
            features: [batch, feature_dim] concatenated [deter, stoch]
        
        Returns:
            [batch, 1] predicted reward
        """
        return self.net(features)


class DiscountModel(nn.Module):
    """Discount (continuation) predictor for Dreamer V2.
    
    Predicts whether episode continues (discount = γ_t).
    NEW in V2 compared to V1 (V1 uses fixed gamma).
    
    Outputs logits for binary classification:
    - High logit → episode continues (discount ≈ γ)
    - Low logit → episode ends (discount ≈ 0)
    
    Architecture: 4-layer MLP with 400 units (paper spec).
    Loss: Binary cross-entropy with (1 - terminal) as target.
    """
    
    def __init__(self, feature_dim, hidden=400, layers=4):
        super().__init__()
        self.net = MLP(feature_dim, 1, hidden=hidden, layers=layers)
    
    def forward(self, features):
        """Predict continuation probability from features.
        
        Args:
            features: [batch, feature_dim] concatenated [deter, stoch]
        
        Returns:
            [batch, 1] logits for continuation (use sigmoid or BCEWithLogitsLoss)
        """
        return self.net(features)


class ActorModel(nn.Module):
    """Actor (policy) network for Dreamer V2.
    
    Maps latent states to actions. NOT part of world model - used for
    policy learning via actor-critic.
    
    For continuous actions:
    - Outputs mean and log_std
    - Actions sampled from Normal(mean, std)
    - Applies tanh squashing to bound actions
    
    For discrete actions:
    - Outputs logits
    - Actions sampled from Categorical
    
    Architecture: 4-layer MLP with 400 units (paper spec).
    """
    
    def __init__(self, feature_dim, action_dim, hidden=400, layers=4, discrete=False):
        super().__init__()
        self.discrete = discrete
        self.action_dim = action_dim
        
        if discrete:
            out_dim = action_dim
        else:
            out_dim = action_dim  # Mean only, log_std is a parameter
        
        self.net = MLP(feature_dim, out_dim, hidden=hidden, layers=layers)
        
        if not discrete:
            # Separate log_std parameter (not state-dependent)
            self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, features):
        """Compute action distribution.
        
        Args:
            features: [batch, feature_dim] concatenated [deter, stoch]
        
        Returns:
            For continuous: mean [batch, action_dim] (use self.log_std for std)
            For discrete: logits [batch, action_dim]
        """
        return self.net(features)


class ValueModel(nn.Module):
    """Value function (critic) for Dreamer V2.
    
    Estimates expected cumulative return V(s_t) from latent state.
    NOT part of world model - used for actor-critic policy learning.
    
    Trained with TD(λ) targets from imagined trajectories.
    Used to compute actor loss (policy gradient with value baseline).
    
    Architecture: 4-layer MLP with 400 units (paper spec).
    """
    
    def __init__(self, feature_dim, hidden=400, layers=4):
        super().__init__()
        self.net = MLP(feature_dim, 1, hidden=hidden, layers=layers)
    
    def forward(self, features):
        """Predict value from features.
        
        Args:
            features: [batch, feature_dim] concatenated [deter, stoch]
        
        Returns:
            [batch, 1] predicted value (expected cumulative return)
        """
        return self.net(features)
