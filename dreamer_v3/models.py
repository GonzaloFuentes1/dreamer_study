"""
Dreamer V3 Models: High-level world model and RSSM components.

This module contains the recurrent state space model (RSSM_V3) and
the WorldModel wrapper that combines encoder, RSSM, decoder, and
prediction heads.

Basic network components (MLP, Conv, etc.) are in networks.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import OneHotCategoricalStraightThrough

from .networks import RMSNorm, BlockLinear, GRUCell, MLP, Encoder, Decoder


class RSSM_V3(nn.Module):
    """DreamerV3 RSSM with 32x32 categorical + straight-through gradients"""
    def __init__(self, action_dim, config=None):
        super().__init__()
        # V3 uses same discrete representation as V2: 32x32 categorical (FIXED)
        self.stoch = 32
        self.classes = 32
        
        # V3 default dims (can be overridden)
        if config:
            self.deter = config.get('deter', config.get('deter_dim', 512))
            self.hidden = config.get('hidden', config.get('hidden_dim', 512))
            self.embed = config.get('embed', config.get('embed_dim', 1024))
        else:
            self.deter = 512
            self.hidden = 512
            self.embed = 1024
        
        self.stoch_channels = self.stoch * self.classes  # 1024
        self.act = nn.SiLU()  # V3 uses SiLU instead of ELU
        
        # 1. Deterministic Path Input (from "Imagination")
        # Input: [stoch_flat, action]
        self.img_in = nn.Sequential(
            nn.Linear(self.stoch_channels + action_dim, self.hidden),
            RMSNorm(self.hidden),
            self.act
        )
        
        # 2. GRU Cell with BlockLinear (V3 optimization)
        blocks = config.get('blocks', 16) if config else 16
        self.cell_layer = GRUCell(self.hidden, self.deter, norm=True, blocks=blocks)
        
        # 3. Posterior (Observation step)
        # Input: [deter, embed]
        self.obs_out = nn.Sequential(
             nn.Linear(self.deter + self.embed, self.hidden),
             RMSNorm(self.hidden),
             self.act,
             nn.Linear(self.hidden, self.stoch_channels)
        )
        
        # 4. Prior (Imagination step)
        # Input: [deter]
        self.img_out = nn.Sequential(
             nn.Linear(self.deter, self.hidden),
             RMSNorm(self.hidden),
             self.act,
             nn.Linear(self.hidden, self.stoch_channels)
        )

    def initial(self, batch_size, device):
        return {
            'stoch': torch.zeros(batch_size, self.stoch, self.classes, device=device),
            'deter': torch.zeros(batch_size, self.deter, device=device)
        }

    def cell(self, x, h):
        return self.cell_layer(x, h)

    def get_dist(self, logits, unimix_ratio=0.01):
        """Get categorical distribution with uniform mixing for stability (V3 trick)"""
        shape = logits.shape
        logits = logits.view(*(shape[:-1] + (self.stoch, self.classes)))
        
        # V3: Mix with uniform for stability
        if unimix_ratio > 0:
            probs = F.softmax(logits, dim=-1)
            uniform = torch.ones_like(probs) / self.classes
            probs = (1.0 - unimix_ratio) * probs + unimix_ratio * uniform
            logits = torch.log(torch.clamp(probs, min=1e-8))
        
        # Use PyTorch's OneHotCategoricalStraightThrough for proper gradients
        return OneHotCategoricalStraightThrough(logits=logits)
    
    def sample_stoch(self, logits):
        """Sample stochastic state with straight-through gradients"""
        dist = self.get_dist(logits)
        stoch = dist.rsample()  # [batch, 32, 32] - uses rsample for straight-through
        stoch_flat = stoch.view(stoch.shape[0], -1)  # [batch, 1024] for linear layers
        return dist, stoch, stoch_flat
        
    def observe(self, embed, action, state=None):
        """Process observation sequence and return posterior states"""
        batch_size = embed.shape[0]
        if state is None:
            state = self.initial(batch_size, embed.device)
        
        stoch = state['stoch']  # [B, 32, 32]
        stoch_flat = stoch.view(batch_size, -1)  # [B, 1024]
        deter = state['deter']  # [B, deter_dim]
        
        post_logits_list = []
        prior_logits_list = []
        deters_list = []
        stochs_list = []
        
        # Loop over time
        for t in range(embed.shape[1]):
            # 1. Deterministic state update
            inp = torch.cat([stoch_flat, action[:, t]], dim=-1)
            x = self.img_in(inp)
            deter = self.cell(x, deter)
            deters_list.append(deter)
            
            # 2. Prior (before seeing observation)
            prior_logit = self.img_out(deter)
            prior_logits_list.append(prior_logit)
            
            # 3. Posterior (after seeing observation)
            obs_inp = torch.cat([deter, embed[:, t]], dim=-1)
            post_logit = self.obs_out(obs_inp)
            post_logits_list.append(post_logit)
            
            # 4. Sample from posterior for next step
            _, stoch, stoch_flat = self.sample_stoch(post_logit)
            stochs_list.append(stoch)
            
        return {
            'deter': torch.stack(deters_list, dim=1),
            'stoch': torch.stack(stochs_list, dim=1),  # [B, T, 32, 32]
            'logit': torch.stack(post_logits_list, dim=1),
            'prior_logit': torch.stack(prior_logits_list, dim=1)
        }

    def imagine(self, policy, start_state, horizon):
        """Imagine trajectory using learned dynamics"""
        deter = start_state['deter']
        stoch = start_state['stoch']  # [B, 32, 32]
        stoch_flat = stoch.reshape(stoch.shape[0], -1)  # [B, 1024]
        
        deters_list = []
        stochs_list = []
        actions_list = []
        
        for t in range(horizon):
            # Get action from policy
            feat = torch.cat([deter, stoch_flat], dim=-1)
            action = policy(feat)
            actions_list.append(action)
            
            # Update deterministic state
            inp = torch.cat([stoch_flat, action], dim=-1)
            x = self.img_in(inp)
            deter = self.cell(x, deter)
            deters_list.append(deter)
            
            # Sample from prior
            prior_logit = self.img_out(deter)
            _, stoch, stoch_flat = self.sample_stoch(prior_logit)
            stochs_list.append(stoch)
            
        return {
            'deter': torch.stack(deters_list, dim=1),
            'stoch': torch.stack(stochs_list, dim=1),
            'action': torch.stack(actions_list, dim=1)
        }
    
    def kl_loss(self, post_logits, prior_logits, free_nats=1.0):
        """V3 KL Loss with separate dynamics and representation losses."""
        post_dist = self.get_dist(post_logits, unimix_ratio=0)
        prior_dist = self.get_dist(prior_logits, unimix_ratio=0)
        
        # Stop-gradient versions for separate losses
        post_dist_sg = self.get_dist(post_logits.detach(), unimix_ratio=0)
        prior_dist_sg = self.get_dist(prior_logits.detach(), unimix_ratio=0)

        # L_dyn = KL(sg(posterior) || prior) -> Trains the prior
        loss_dyn = torch.distributions.kl_divergence(post_dist_sg, prior_dist)
        
        # L_rep = KL(posterior || sg(prior)) -> Regularizes the posterior
        loss_rep = torch.distributions.kl_divergence(post_dist, prior_dist_sg)
        
        # Apply free nats to both losses
        loss_dyn = torch.maximum(loss_dyn.sum(dim=-1), torch.tensor(free_nats)).mean()
        loss_rep = torch.maximum(loss_rep.sum(dim=-1), torch.tensor(free_nats)).mean()
        
        return loss_dyn, loss_rep


class RewardModel(nn.Module):
    """Reward predictor for Dreamer V3.
    
    Predicts reward using symexp twohot distribution (paper eq. 10-11).
    Part of the world model, trained with actual rewards.
    
    Architecture: 5-layer MLP with 640 units outputting 255 bins.
    Paper: Section "World model learning", uses categorical distribution
    with exponentially spaced bins.
    """
    
    def __init__(self, feature_dim, hidden=640, layers=5, num_bins=255, device='cpu'):
        super().__init__()
        from .utils import create_symlog_bins
        
        self.num_bins = num_bins
        self.net = MLP(feature_dim, num_bins, hidden=hidden, layers=layers)
        
        # Create symlog-spaced bins: symexp(-20 ... +20)
        self.register_buffer('bins', create_symlog_bins(num_bins, device=device))
        
        # Initialize output layer to zeros (paper recommendation)
        nn.init.zeros_(self.net.net[-1].weight)
        nn.init.zeros_(self.net.net[-1].bias)
    
    def forward(self, features):
        """Predict reward distribution from features.
        
        Args:
            features: [batch, feature_dim] concatenated [deter, stoch]
        
        Returns:
            logits: [batch, num_bins] logits for categorical distribution
        """
        return self.net(features)
    
    def predict(self, features):
        """Get expected reward value.
        
        Args:
            features: [batch, feature_dim]
            
        Returns:
            [batch] expected reward values
        """
        logits = self.forward(features)
        probs = torch.softmax(logits, dim=-1)
        return (probs * self.bins).sum(dim=-1)


class ContinueModel(nn.Module):
    """Continue (continuation) predictor for Dreamer V3.
    
    Predicts whether episode continues via logistic regression.
    Paper: "continue predictor via logistic regression" (eq. 1).
    
    Different from V2 DiscountModel: predicts binary continuation c_t ∈ {0,1}
    not discount factor. Used in loss_cont = -log p(c_t | h_t, z_t).
    
    Architecture: 5-layer MLP with 640 units outputting 1 logit.
    """
    
    def __init__(self, feature_dim, hidden=640, layers=5):
        super().__init__()
        self.net = MLP(feature_dim, 1, hidden=hidden, layers=layers)
    
    def forward(self, features):
        """Predict continuation logits from features.
        
        Args:
            features: [batch, feature_dim] concatenated [deter, stoch]
        
        Returns:
            [batch, 1] logits for continuation (use BCEWithLogitsLoss)
        """
        return self.net(features)


class CriticModel(nn.Module):
    """Critic (value function) for Dreamer V3.
    
    Predicts return distribution using symexp twohot (paper page 5-6).
    NOT part of world model - used for actor-critic learning.
    
    Paper: "we parameterize the critic as categorical distribution with
    exponentially spaced bins" - enables learning across different reward scales.
    
    Architecture: 5-layer MLP with 640 units outputting 255 bins.
    """
    
    def __init__(self, feature_dim, hidden=640, layers=5, num_bins=255, device='cpu'):
        super().__init__()
        from .utils import create_symlog_bins
        
        self.num_bins = num_bins
        self.net = MLP(feature_dim, num_bins, hidden=hidden, layers=layers)
        
        # Create symlog-spaced bins: symexp(-20 ... +20)
        self.register_buffer('bins', create_symlog_bins(num_bins, device=device))
        
        # Initialize output layer to zeros (paper recommendation)
        nn.init.zeros_(self.net.net[-1].weight)
        nn.init.zeros_(self.net.net[-1].bias)
    
    def forward(self, features):
        """Predict value distribution from features.
        
        Args:
            features: [batch, feature_dim] concatenated [deter, stoch]
        
        Returns:
            logits: [batch, num_bins] logits for categorical distribution
        """
        return self.net(features)
    
    def predict(self, features):
        """Get expected value (for reading out value estimates).
        
        Args:
            features: [batch, feature_dim]
            
        Returns:
            [batch] expected return values
        """
        logits = self.forward(features)
        probs = torch.softmax(logits, dim=-1)
        return (probs * self.bins).sum(dim=-1)


class ActorModel(nn.Module):
    """Actor (policy) network for Dreamer V3.
    
    Outputs action distribution with return normalization (paper eq. 6-7).
    
    For continuous actions: outputs mean and log_std
    For discrete actions: outputs logits for OneHotCategorical
    
    Architecture: 5-layer MLP with 640 units.
    """
    
    def __init__(self, feature_dim, action_dim, hidden=640, layers=5, is_discrete=True):
        super().__init__()
        self.is_discrete = is_discrete
        self.action_dim = action_dim
        
        if is_discrete:
            output_dim = action_dim
        else:
            # Continuous: mean and log_std
            output_dim = action_dim * 2
        
        self.net = MLP(feature_dim, output_dim, hidden=hidden, layers=layers)
    
    def forward(self, features):
        """Predict action distribution from features.
        
        Args:
            features: [batch, feature_dim] concatenated [deter, stoch]
        
        Returns:
            For discrete: [batch, action_dim] logits
            For continuous: [batch, action_dim * 2] concatenated [mean, log_std]
        """
        return self.net(features)
    
    def get_distribution(self, features):
        """Get action distribution.
        
        Args:
            features: [batch, feature_dim]
            
        Returns:
            torch.distributions.Distribution
        """
        output = self.forward(features)
        
        if self.is_discrete:
            return torch.distributions.OneHotCategorical(logits=output)
        else:
            mean, log_std = torch.chunk(output, 2, dim=-1)
            std = torch.exp(log_std)
            # Use Independent to sum log_prob over action dimensions
            return torch.distributions.Independent(torch.distributions.Normal(mean, std), 1)


class WorldModel(nn.Module):
    """Dreamer V3 World Model.
    
    Learns latent dynamics and reconstructs observations.
    Composed of:
    - Encoder: obs → embedding
    - RSSM_V3: latent dynamics with categorical stochastic states
    - Decoder: latent → obs reconstruction
    - RewardModel: latent → reward (twohot distribution)
    - ContinueModel: latent → continuation probability
    """
    
    def __init__(self, obs_shape, action_dim, config):
        super().__init__()
        
        model_cfg = config['model']
        rssm_cfg = model_cfg['rssm'].copy()
        device = config.get('device', 'cpu')
        
        # Encoder
        self.encoder = Encoder(obs_shape, config=model_cfg)
        
        # RSSM
        self.rssm = RSSM_V3(action_dim, config=rssm_cfg)
        
        # Feature dim: deter + stoch_flat
        feature_dim = rssm_cfg.get('deter_dim', 1024) + \
                     rssm_cfg.get('stoch_dim', 32) * rssm_cfg.get('stoch_classes', 32)
        
        # Decoder
        self.decoder = Decoder(feature_dim, shape=obs_shape, config=model_cfg)
        
        # Prediction heads
        mlp_hidden = model_cfg.get('mlp_hidden', 640)
        mlp_layers = model_cfg.get('mlp_layers', 5)
        
        self.reward_model = RewardModel(feature_dim, hidden=mlp_hidden, layers=mlp_layers, device=device)
        self.continue_model = ContinueModel(feature_dim, hidden=mlp_hidden, layers=mlp_layers)
