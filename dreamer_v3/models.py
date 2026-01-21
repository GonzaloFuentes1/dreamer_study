import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from torch.distributions import OneHotCategorical, Distribution
import math

def symlog(x):
    """Symlog: sign(x) * ln(|x| + 1)"""
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


def symexp(x):
    """Inverse of symlog: sign(x) * (exp(|x|) - 1)"""
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


def create_symexp_bins(num_bins=255, device='cuda'):
    bins = torch.linspace(-20, 20, num_bins, device=device)
    return symexp(bins)


def twohot_encode(x, bins):
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


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        # Ensure dim is int (channels/features)
        if isinstance(dim, (list, tuple)): 
            dim = dim[0]
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        original_dtype = x.dtype
        x_f32 = x.float()
        
        if x.ndim == 4: # Image [B, C, H, W]
            norm = torch.mean(x_f32 ** 2, dim=1, keepdim=True)
            scale = self.scale.view(1, -1, 1, 1)
            out = x_f32 * torch.rsqrt(norm + self.eps) * scale
        else: # Vector [B, ..., D]
            norm = torch.mean(x_f32 ** 2, dim=-1, keepdim=True)
            out = x_f32 * torch.rsqrt(norm + self.eps) * self.scale
            
        return out.to(original_dtype)


class BlockLinear(nn.Module):
    def __init__(self, in_features, out_features, num_blocks, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_blocks = num_blocks
        
        # Ensure divisibility
        if in_features % num_blocks != 0:
            raise ValueError(f"in_features {in_features} not divisible by blocks {num_blocks}")
        if out_features % num_blocks != 0:
            raise ValueError(f"out_features {out_features} not divisible by blocks {num_blocks}")
            
        self.chunk_in = in_features // num_blocks
        self.chunk_out = out_features // num_blocks
        
        # Initialize weights with standard initialization (truncated normal approximation)
        self.weight = nn.Parameter(torch.empty(num_blocks, self.chunk_out, self.chunk_in))
        nn.init.trunc_normal_(self.weight, std=1.0 / math.sqrt(self.chunk_in), a=-2.0, b=2.0)
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(num_blocks, self.chunk_out))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        # x: [B, C] or [B, T, C]
        shape = x.shape
        x_flat = x.view(-1, self.in_features) # Collapse batch/time
        
        B = x_flat.shape[0]
        x_blocked = x_flat.view(B, self.num_blocks, self.chunk_in)
        
        # Block-wise matrix mult: [B, K, I] @ [K, I, O] -> [B, K, O]
        # Weight is [K, O, I]. Transpose to [K, I, O] for matmul
        w = self.weight.transpose(1, 2)
        
        # Parallel mm over blocks
        # We can use einsum: b k i, k i o -> b k o
        y = torch.einsum('bki,kio->bko', x_blocked, w)
        
        if self.bias is not None:
            y = y + self.bias
            
        y = y.reshape(B, self.out_features)
        return y


class GRUCell(nn.Module):
    """GRU Cell with optional Layer Norm and Block Matrices."""
    def __init__(self, input_size, hidden_size, norm=True, blocks=None):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.norm = norm
        # Standard Linear for Input->Hidden (not part of recurrent state per se)
        self.fc_x = nn.Linear(input_size, 3 * hidden_size)
        
        # Block Linear for Hidden->Hidden (The recurrent part)
        if blocks:
             self.fc_h = BlockLinear(hidden_size, 3 * hidden_size, num_blocks=blocks)
        else:
             self.fc_h = nn.Linear(hidden_size, 3 * hidden_size)

        if norm:
            self.norm_x = RMSNorm(3 * hidden_size)
            self.norm_h = RMSNorm(3 * hidden_size)

    def forward(self, x, h):
        # x: Input
        # h: Hidden state from previous step
        
        # Force h to match x's batch size to prevent dimension mismatches
        if x.shape[0] != h.shape[0]:
            h = h[:x.shape[0]].contiguous()
        
        x_out = self.fc_x(x)
        h_out = self.fc_h(h)
        
        if self.norm:
            x_out = self.norm_x(x_out)
            h_out = self.norm_h(h_out)
        
        x_r, x_z, x_n = torch.chunk(x_out, 3, dim=-1)
        h_r, h_z, h_n = torch.chunk(h_out, 3, dim=-1)
        
        r = torch.sigmoid(x_r + h_r)
        z = torch.sigmoid(x_z + h_z)
        
        # New h candidate
        n = torch.tanh(x_n + r * h_n)
        
        next_h = (1 - z) * n + z * h
        return next_h


class OneHotDist(Distribution):
    def __init__(self, logits=None, probs=None, unimix_ratio=0.01):
        if logits is not None and probs is None:
            # Handle numerical stability for softmax
            # mix 1% uniform (paper default)
            probs = F.softmax(logits, dim=-1)
            if unimix_ratio > 0:
                uniform = torch.ones_like(probs) / probs.shape[-1]
                probs = (1.0 - unimix_ratio) * probs + unimix_ratio * uniform
                logits = torch.log(torch.clamp(probs, min=1e-8))
                
        self.logits = logits
        self.probs = probs
        # Ensure safe probs for Validation with stronger clamping
        self.probs = torch.clamp(self.probs, min=1e-6, max=1.0)
        # Normalize with numerical stability
        probs_sum = self.probs.sum(dim=-1, keepdim=True)
        probs_sum = torch.clamp(probs_sum, min=1e-6)
        self.probs = self.probs / probs_sum
        
        self.cat = OneHotCategorical(probs=self.probs)

    def sample(self, sample_shape=torch.Size()):
        return self.cat.sample(sample_shape)
        
    def log_prob(self, value):
        return self.cat.log_prob(value)
        
    @property
    def mode(self):
        return self.cat.mode


class RSSM(nn.Module):
    def __init__(self, action_dim, config=None):
        super().__init__()
        # Defaults + accept both *_dim and legacy keys
        if config:
            self.stoch = config.get('stoch', config.get('stoch_dim', 32))
            self.classes = config.get('classes', config.get('stoch_classes', 32))
            self.deter = config.get('deter', config.get('deter_dim', 512))
            self.hidden = config.get('hidden', config.get('hidden_dim', 512))
            self.embed = config.get('embed', config.get('embed_dim', 1024))
        else:
            self.stoch = 32
            self.classes = 32
            self.deter = 512
            self.hidden = 512
            self.embed = 1024
        
        self.stoch_channels = self.stoch * self.classes
        self.act = nn.SiLU()
        
        # 1. Deterministic Path Input (from "Imagination")
        # Input: [stoch, action]
        # Renamed to img_in (Imagination Input) because agent.py expects it
        self.img_in = nn.Sequential(
            nn.Linear(self.stoch_channels + action_dim, self.hidden),
            RMSNorm(self.hidden),
            self.act
        )
        
        # Use BlockLinear in GRU if blocks specified
        blocks = config.get('blocks', 16) if config else 16
        self.cell_layer = GRUCell(self.hidden, self.deter, norm=True, blocks=blocks)
        
        # 3. Output from Deter to Posterior/Prior 
        # (This is implicitly used by prior_net and posterior_net)
        
        # 4. Posterior (Observe)
        # Input: [deter, embed]
        self.obs_out = nn.Sequential(
             nn.Linear(self.deter + self.embed, self.hidden),
             RMSNorm(self.hidden),
             self.act,
             nn.Linear(self.hidden, self.stoch_channels)
        )
        
        # 5. Prior (Imagine)
        # Input: [deter]
        self.img_out = nn.Sequential(
             nn.Linear(self.deter, self.hidden),
             RMSNorm(self.hidden),
             self.act,
             nn.Linear(self.hidden, self.stoch_channels)
        )

    def initial(self, batch_size, device):
        return (
            torch.zeros(batch_size, self.stoch_channels, device=device),
            torch.zeros(batch_size, self.deter, device=device)
        )

    def cell(self, x, h):
        return self.cell_layer(x, h)

    def get_dist(self, logits):
        shape = logits.shape
        logits = logits.view(*(shape[:-1] + (self.stoch, self.classes)))
        
        # Using Mixed Distribution (Uniform Mix) for stability
        dist = OneHotDist(logits, unimix_ratio=0.01)
        return dist
        
    def observe(self, embed, action, state=None):
        if state is None:
            state = self.initial(embed.shape[0], embed.device)
        
        stoch_flat, deter = state
        
        post_logits = []
        deters = []
        stochs_flat = []
        prior_logits = []
        
        # Loop over time
        for t in range(embed.shape[1]):
            # 1. Deterministic Path
            # Embed stoch+action
            inp = torch.cat([stoch_flat, action[:, t]], dim=-1)
            x = self.img_in(inp)
            deter = self.cell(x, deter)
            deters.append(deter)
            
            # 2. Stochastic Path
            # Posterior
            obs_inp = torch.cat([deter, embed[:, t]], dim=-1)
            logit = self.obs_out(obs_inp)
            post_logits.append(logit)
            
            # Prior (for KL later)
            prior_logit = self.img_out(deter)
            prior_logits.append(prior_logit)
            
            # Sample Posterior for next step
            dist = self.get_dist(logit)
            stoch = dist.sample()
            stoch_flat = stoch.view(stoch.shape[0], -1)
            stochs_flat.append(stoch_flat)
            
        return {
            'deter': torch.stack(deters, dim=1),
            'stoch': torch.stack(stochs_flat, dim=1).view(embed.shape[0], embed.shape[1], self.stoch, self.classes),
            'logit': torch.stack(post_logits, dim=1),
            'prior_logit': torch.stack(prior_logits, dim=1)
        }

    def imagine(self, policy, start_state, horizon):
        deter = start_state['deter']
        stoch = start_state['stoch']
        stoch_flat = stoch.reshape(stoch.shape[0], -1)
        
        deters = []
        stochs = []
        actions = []
        
        for t in range(horizon):
            feat = torch.cat([deter, stoch_flat], dim=-1)
            action = policy(feat)
            actions.append(action)
            
            inp = torch.cat([stoch_flat, action], dim=-1)
            x = self.img_in(inp)
            deter = self.cell(x, deter)
            deters.append(deter)
            
            prior_logit = self.img_out(deter)
            dist = self.get_dist(prior_logit)
            stoch = dist.sample()
            stoch_flat = stoch.view(stoch.shape[0], -1)
            stochs.append(stoch)
            
        return {
            'deter': torch.stack(deters, dim=1),
            'stoch': torch.stack(stochs, dim=1),
            'action': torch.stack(actions, dim=1)
        }


class ConvEncoder(nn.Module):
    def __init__(self, input_shape, config):
        super().__init__()
        depth = config.get('depth', 48)
        self.net = nn.Sequential(
            nn.Conv2d(input_shape[0], depth, 4, 2),
            RMSNorm(depth),
            nn.SiLU(),
            nn.Conv2d(depth, depth * 2, 4, 2),
            RMSNorm(depth * 2),
            nn.SiLU(),
            nn.Conv2d(depth * 2, depth * 4, 4, 2),
            RMSNorm(depth * 4),
            nn.SiLU(),
            nn.Conv2d(depth * 4, depth * 8, 4, 2),
            RMSNorm(depth * 8),
            nn.SiLU()
        )
        
    def forward(self, obs):
        # inputs: dict with image or tensor
        if isinstance(obs, dict):
            x = obs['image']
        else:
            x = obs
            
        # Ensure float and normalized to [-0.5, 0.5] if input is uint8
        if x.dtype == torch.uint8:
            x = x.float() / 255.0 - 0.5
            
        # x is [B, T, C, H, W]
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        y = self.net(x)
        return y.reshape(B, T, -1)


class Decoder(nn.Module):
    def __init__(self, input_dim, shape, config):
        super().__init__()
        depth = config.get('depth', 48)
        output_channels = shape[0]
        
        self.linear = nn.Linear(input_dim, 32 * depth)
        self.convs = nn.Sequential(
            nn.ConvTranspose2d(32 * depth, depth * 4, 5, 2),
            RMSNorm(depth * 4),
            nn.SiLU(),
            nn.ConvTranspose2d(depth * 4, depth * 2, 5, 2),
            RMSNorm(depth * 2),
            nn.SiLU(),
            nn.ConvTranspose2d(depth * 2, depth, 6, 2),
            RMSNorm(depth),
            nn.SiLU(),
            nn.ConvTranspose2d(depth, output_channels, 6, 2),
        )

    def forward(self, features):
        B, T, _ = features.shape
        x = self.linear(features.view(B*T, -1))
        x = x.view(x.shape[0], -1, 1, 1) 
        x = self.convs(x)
        return {'image': x.view(B, T, *x.shape[1:])}


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
    
    def forward(self, x):
        return self.net(x)

# Encoder with projection to specified embed_dim
class Encoder(nn.Module):
    def __init__(self, input_shape, config):
        super().__init__()
        self.cnn = ConvEncoder(input_shape, config)
        
        # Calculate CNN output size dynamically with a dummy forward pass
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, *input_shape)  # [B=1, T=1, C, H, W]
            cnn_out = self.cnn(dummy_input)
            cnn_out_size = cnn_out.shape[-1]
        
        # Project to desired embed_dim
        self.embed_dim = config.get('embed_dim', 1024)
        self.proj = nn.Linear(cnn_out_size, self.embed_dim)
    
    def forward(self, obs):
        features = self.cnn(obs)  # [B, T, cnn_out_size]
        return self.proj(features)  # [B, T, embed_dim]

