import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import OneHotCategorical, OneHotCategoricalStraightThrough, Distribution


def symlog(x):
    """Symlog: sign(x) * ln(|x| + 1)"""
    return torch.sign(x) * torch.log(torch.abs(x) + 1)


def symexp(x):
    """Inverse of symlog: sign(x) * (exp(|x|) - 1)"""
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


def create_symexp_bins(num_bins=255, device='cuda'):
    """Create exponentially spaced bins from -20 to +20 in symexp space.
    Returns bins in normal space for direct comparison with reward/return values.
    """
    bins = torch.linspace(-20, 20, num_bins, device=device)
    return symexp(bins)


def twohot_encode(x, bins):
    """Twohot encode scalars into vectors with weights on two closest bins.
    Args:
        x: tensor of any shape
        bins: (num_bins,) sorted bin positions
    Returns:
        target: (*x.shape, num_bins) with twohot encoding
    """
    x = x.unsqueeze(-1)  # Add bin dimension
    below = (bins <= x).sum(dim=-1, dtype=torch.int64) - 1
    below = torch.clamp(below, 0, len(bins) - 2)
    above = below + 1
    
    below_val = bins[below]
    above_val = bins[above]
    weight_above = torch.clamp((x.squeeze(-1) - below_val) / (above_val - below_val + 1e-8), 0, 1)
    weight_below = 1 - weight_above
    
    target = torch.zeros((*x.shape[:-1], len(bins)), device=x.device)
    target.scatter_(-1, below.unsqueeze(-1), weight_below.unsqueeze(-1))
    target.scatter_(-1, above.unsqueeze(-1), weight_above.unsqueeze(-1))
    return target


class BlockLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_blocks: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_blocks = num_blocks
        assert in_features % num_blocks == 0
        assert out_features % num_blocks == 0
        
        # GroupConv1d is mathematically equivalent to Block Diagonal Linear
        # Input shape: [B, C_in, 1] -> Output: [B, C_out, 1]
        self.conv = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            groups=num_blocks,
            bias=bias
        )
        
        # Initialize weights properly (Conv1d init is different from Linear)
        # We want to mimic Kaiming Uniform
        fan_in = (in_features // num_blocks)
        bound = (1 / fan_in) ** 0.5 if fan_in > 0 else 0
        nn.init.uniform_(self.conv.weight, -bound, bound)
        if bias and self.conv.bias is not None:
            nn.init.uniform_(self.conv.bias, -bound, bound)

    def forward(self, x):
        original_shape = x.shape
        x = x.view(-1, self.in_features, 1)
        
        y = self.conv(x)
        
        out_shape = original_shape[:-1] + (self.out_features,)
        return y.view(out_shape)


class RSSM_V3(nn.Module):
    """
    RSSM with Block-Diagonal GRU and Discrete Latents.
    """
    def __init__(self, action_dim, stoch_dim=32, stoch_classes=32, deter_dim=512, hidden_dim=512, embed_dim=1024, blocks=8):
        super().__init__()
        self.stoch_dim = stoch_dim
        self.stoch_classes = stoch_classes
        self.deter_dim = deter_dim
        self.hidden_dim = hidden_dim
        self.blocks = blocks
        
        stoch_flat_dim = stoch_dim * stoch_classes
        
        # 1. Input Embeddings (Dense)
        self.img_in = nn.Sequential(nn.Linear(embed_dim, hidden_dim, bias=False))
        self.action_in = nn.Sequential(nn.Linear(action_dim, hidden_dim, bias=False))
        self.stoch_in = nn.Sequential(nn.Linear(stoch_flat_dim, hidden_dim, bias=False))
        self.deter_in = nn.Sequential(nn.Linear(deter_dim, hidden_dim, bias=False))
        
        # 2. Recurrent Step (Block Diagonal Deep Cell)
        # Structure: Concat Inputs -> Layer -> Layer -> GRU Cell
        self.obs_out_norm = RMSNorm(hidden_dim)
        
        # We process the concatenated inputs (4 * hidden_dim) down to hidden_dim using blocks?
        # Actually JAX code repeats the input across blocks.
        # Simplification for PyTorch: Dense projection to deter_dim then BlockGRU
        
        # Implementation following JAX "core": 
        # Inputs are processed and concatenated.
        # Then BlockLinear layers.
        
        self.pre_gru_net = nn.Sequential(
            BlockLinear(hidden_dim, hidden_dim, blocks),
            RMSNorm(hidden_dim),
            nn.SiLU()
        )
        
        # Block GRU Gates
        # Input to GRU is hidden_dim, State is deter_dim
        # We need gates for (reset, cand, update)
        # Weights for Input: [hidden_dim -> 3*deter_dim] (Block)
        # Weights for Hidden: [deter_dim -> 3*deter_dim] (Block)
        self.gru_x = BlockLinear(hidden_dim, 3 * deter_dim, blocks)
        self.gru_h = BlockLinear(deter_dim, 3 * deter_dim, blocks)

        # 3. Posteriors / Priors
        self.prior_net = nn.Sequential(
            nn.Linear(deter_dim, hidden_dim),
            RMSNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, stoch_flat_dim)
        )

        self.posterior_net = nn.Sequential(
            nn.Linear(deter_dim + embed_dim, hidden_dim),
            RMSNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, stoch_flat_dim)
        )
    
    def get_stoch_state(self, logits, mix_ratio=0.01):
        """Mix with 1% uniform in probability space."""
        shape = logits.shape
        logits = logits.view(shape[:-1] + (self.stoch_dim, self.stoch_classes))
        probs = F.softmax(logits, dim=-1)
        probs = (1 - mix_ratio) * probs + mix_ratio / self.stoch_classes
        # dist = OneHotCategorical(probs=probs) # Removed for speed
        mixed_logits = torch.log(probs + 1e-8)
        stoch = F.gumbel_softmax(mixed_logits, tau=1.0, hard=True, dim=-1)
        stoch_flat = stoch.view(shape[:-1] + (self.stoch_dim * self.stoch_classes,))
        return None, stoch, stoch_flat # Return None as dist

    def step(self, prev_stoch_flat, x_action, prev_deter):
        # 1. Embed inputs (Dense mixing)
        x_stoch = self.stoch_in(prev_stoch_flat)
        # x_action is now precomputed and passed in
        x_deter = self.deter_in(prev_deter)
        
        # Sum embeddings (as they project to same hidden_dim) - equivalent to JAX concat then matmul if weights aligned
        x = x_stoch + x_action + x_deter
        x = F.silu(self.obs_out_norm(x))
        
        # 2. Deep Block Processing
        x = self.pre_gru_net(x)
        
        # 3. Block GRU Step
        # Calculate gates
        gates_x = self.gru_x(x)
        gates_h = self.gru_h(prev_deter)
        gates = gates_x + gates_h
        
        reset, cand, update = torch.chunk(gates, 3, dim=-1)
        
        reset = torch.sigmoid(reset)
        update = torch.sigmoid(update - 1) # Bias -1 for forget gate (Dreamer trick)
        cand = torch.tanh(reset * cand)
        
        deter = update * cand + (1 - update) * prev_deter
        return deter

    def observe(self, embed, action, state=None):
        batch_size, seq_len, _ = embed.shape
        if state is None:
            deter = torch.zeros(batch_size, self.deter_dim, device=embed.device)
            stoch_flat = torch.zeros(batch_size, self.stoch_dim * self.stoch_classes, device=embed.device)
        else:
            stoch_flat, deter = state


        # [B, T, A] -> [B, T, H]
        x_action_seq = self.action_in(action)

        prior_logits_list = []
        post_logits_list = []
        deters_list = []
        stochs_flat_list = []

        for t in range(seq_len):
            deter = self.step(stoch_flat, x_action_seq[:, t], deter)
            
            post_logits = self.posterior_net(torch.cat([deter, embed[:, t]], dim=-1))
            
            _, _, stoch_flat = self.get_stoch_state(post_logits)
            
            post_logits_list.append(post_logits)
            deters_list.append(deter)
            stochs_flat_list.append(stoch_flat)
        
        deters = torch.stack(deters_list, dim=1)
        
        prior_logits = self.prior_net(deters)
        
        post_logits = torch.stack(post_logits_list, dim=1)
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
            action = actor(feat)  # Keep gradients for dynamics backprop
            
            if not is_discrete:
                action = torch.tanh(action)
            
            # Embed action for step
            x_action = self.action_in(action)
            
            curr_deter = self.step(curr_stoch_flat, x_action, curr_deter)
            prior_logits = self.prior_net(curr_deter)
            _, stoch, curr_stoch_flat = self.get_stoch_state(prior_logits)
            
            stochs_list.append(curr_stoch_flat)
            deters_list.append(curr_deter)
            actions_list.append(action)
        
        img_stoch = torch.stack(stochs_list, dim=1)
        img_deter = torch.stack(deters_list, dim=1)
        img_actions = torch.stack(actions_list, dim=1)
        
        return img_deter, img_stoch, img_actions

    def kl_loss(self, post_logits, prior_logits, free_nats=1.0, beta_dyn=1.0, beta_rep=0.1):
        """V3 KL loss with mixture distribution"""
        # Helper to get mixed distribution
        def _get_dist(l):
            shape = l.shape
            l = l.view(shape[:-1] + (self.stoch_dim, self.stoch_classes))
            p = F.softmax(l, dim=-1)
            p = 0.99 * p + 0.01 / self.stoch_classes
            return OneHotCategorical(probs=p)

        # L_dyn: sg(posterior) || prior
        post_dist_detached = _get_dist(post_logits.detach())
        prior_dist = _get_dist(prior_logits)
        kl_dyn = torch.distributions.kl_divergence(post_dist_detached, prior_dist).sum(dim=-1)
        loss_dyn = torch.maximum(kl_dyn, torch.ones_like(kl_dyn) * free_nats).mean()
        
        # L_rep: posterior || sg(prior)
        post_dist = _get_dist(post_logits)
        prior_dist_detached = _get_dist(prior_logits.detach())
        kl_rep = torch.distributions.kl_divergence(post_dist, prior_dist_detached).sum(dim=-1)
        loss_rep = torch.maximum(kl_rep, torch.ones_like(kl_rep) * free_nats).mean()
        
        return beta_dyn * loss_dyn + beta_rep * loss_rep


class ConvEncoder(nn.Module):
    def __init__(self, input_channels=3, depth=48, stride=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, depth, 4, stride),
            RMSNorm([depth, 31, 31]),
            nn.SiLU(),
            nn.Conv2d(depth, depth * 2, 4, stride),
            RMSNorm([depth * 2, 14, 14]),
            nn.SiLU(),
            nn.Conv2d(depth * 2, depth * 4, 4, stride),
            RMSNorm([depth * 4, 6, 6]),
            nn.SiLU(),
            nn.Conv2d(depth * 4, depth * 8, 4, stride),
            RMSNorm([depth * 8, 2, 2]),
            nn.SiLU()
        )
        
    def forward(self, obs):
        x = self.net(obs)
        return x.reshape(x.shape[0], -1)


class ConvDecoder(nn.Module):
    def __init__(self, input_dim, depth=48, output_channels=3):
        super().__init__()
        self.linear = nn.Linear(input_dim, 32 * 1 * 1)
        self.convs = nn.Sequential(
            nn.ConvTranspose2d(32, depth * 4, 5, stride=2),
            RMSNorm([depth * 4, 5, 5]),
            nn.SiLU(),
            nn.ConvTranspose2d(depth * 4, depth * 2, 5, stride=2),
            RMSNorm([depth * 2, 13, 13]),
            nn.SiLU(),
            nn.ConvTranspose2d(depth * 2, depth, 6, stride=2),
            RMSNorm([depth, 30, 30]),
            nn.SiLU(),
            nn.ConvTranspose2d(depth, output_channels, 6, stride=2),
        )

    def forward(self, features):
        x = self.linear(features)
        x = x.view(x.shape[0], 32, 1, 1)
        x = self.convs(x)
        return torch.sigmoid(x)


class SymexpTwohotMLP(nn.Module):
    """MLP that outputs categorical distribution over symexp bins (paper Eq 10-11)."""
    def __init__(self, input_dim, num_bins=255, hidden=512, layers=3):
        super().__init__()
        self.num_bins = num_bins
        
        model = []
        for _ in range(layers):
            model.append(nn.Linear(input_dim, hidden))
            model.append(RMSNorm(hidden))
            model.append(nn.SiLU())
            input_dim = hidden
        
        # Output layer with zero initialization (paper p.6)
        output_layer = nn.Linear(hidden, num_bins)
        nn.init.zeros_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)
        model.append(output_layer)
        
        self.net = nn.Sequential(*model)
        self.register_buffer('bins', create_symexp_bins(num_bins=num_bins, device='cuda'))
    
    def forward(self, x):
        """Returns logits for categorical distribution."""
        return self.net(x)
    
    def predict(self, x):
        """Returns predicted value as weighted average of bins."""
        logits = self(x)
        probs = F.softmax(logits, dim=-1)
        return (probs * self.bins).sum(dim=-1)
    
    def loss(self, x, target):
        """Compute twohot categorical cross entropy loss (paper Eq 11).
        
        Bins are in normal space (created as symexp(linspace(-20, 20))).
        Target is in normal space (reward/return values).
        No need to apply symlog to target since bins are already in symexp space.
        """
        logits = self(x)
        # Bins are already in symexp space, target should be compared directly
        target_encoded = twohot_encode(target, self.bins)
        return -(target_encoded * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden=512, layers=3, act=nn.SiLU, dist=None):
        super().__init__()
        model = []
        for _ in range(layers):
            model.append(nn.Linear(input_dim, hidden))
            model.append(RMSNorm(hidden))
            model.append(act())
            input_dim = hidden
        model.append(nn.Linear(hidden, output_dim))
        self.net = nn.Sequential(*model)
        self.dist = dist
    
    def forward(self, x):
        return self.net(x)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        # If dim is int, convert to tensor; if list, convert to tensor
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = torch.mean(x ** 2, dim=-1, keepdim=True)
        return x * torch.rsqrt(norm + self.eps) * self.scale
