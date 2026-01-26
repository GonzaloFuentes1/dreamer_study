"""
Dreamer V3 Networks: Basic building blocks and encoder/decoder architectures.

Normalization: [0, 255] uint8 → [0, 1] float32
Decoder output: Sigmoid [0, 1] (NOT Normal distribution like V1/V2)

Architecture differences from V1/V2:
- RMSNorm instead of no normalization
- SiLU activation instead of ReLU
- Sigmoid output instead of Gaussian
- Bottleneck design in decoder
- BlockLinear for efficient GRU computation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.
    
    More stable than LayerNorm for deep networks.
    Used throughout V3 architecture.
    """
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        if isinstance(dim, (list, tuple)): 
            dim = dim[0]
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        original_dtype = x.dtype
        x_f32 = x.float()
        
        if x.ndim == 4:  # Image [B, C, H, W]
            norm = torch.mean(x_f32 ** 2, dim=1, keepdim=True)
            scale = self.scale.view(1, -1, 1, 1)
            out = x_f32 * torch.rsqrt(norm + self.eps) * scale
        else:  # Vector [B, ..., D]
            norm = torch.mean(x_f32 ** 2, dim=-1, keepdim=True)
            out = x_f32 * torch.rsqrt(norm + self.eps) * self.scale
            
        return out.to(original_dtype)


class BlockLinear(nn.Module):
    """Block-diagonal linear layer for efficient computation.
    
    Instead of full matrix [in_features, out_features], uses block-diagonal
    structure with num_blocks independent matrices. Reduces computation and
    memory while maintaining expressiveness for recurrent operations.
    
    Used in GRUCell for V3's optimized RSSM implementation.
    """
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
        
        # Initialize weights with truncated normal (V3 paper recommendation)
        self.weight = nn.Parameter(torch.empty(num_blocks, self.chunk_out, self.chunk_in))
        nn.init.trunc_normal_(self.weight, std=1.0 / math.sqrt(self.chunk_in), a=-2.0, b=2.0)
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(num_blocks, self.chunk_out))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        """Apply block-diagonal linear transformation.
        
        Args:
            x: [batch, in_features] or [batch, time, in_features]
        
        Returns:
            [batch, out_features] or [batch, time, out_features]
        """
        shape = x.shape
        x_flat = x.view(-1, self.in_features)
        
        B = x_flat.shape[0]
        x_blocked = x_flat.view(B, self.num_blocks, self.chunk_in)
        
        # Block-wise matrix multiplication using einsum
        # [B, K, I] @ [K, I, O] -> [B, K, O]
        w = self.weight.transpose(1, 2)  # [K, O, I] -> [K, I, O]
        y = torch.einsum('bki,kio->bko', x_blocked, w)
        
        if self.bias is not None:
            y = y + self.bias
            
        y = y.reshape(B, self.out_features)
        
        # Restore original shape
        if len(shape) > 2:
            y = y.view(*shape[:-1], self.out_features)
            
        return y


class GRUCell(nn.Module):
    """GRU Cell with optional RMSNorm and BlockLinear.
    
    V3 uses RMSNorm for stability and BlockLinear for efficiency
    in the recurrent hidden-to-hidden connections.
    """
    def __init__(self, input_size, hidden_size, norm=True, blocks=None):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.norm = norm
        
        # Input-to-hidden transformation (non-recurrent)
        self.fc_x = nn.Linear(input_size, 3 * hidden_size)
        
        # Hidden-to-hidden transformation (recurrent, can use BlockLinear)
        if blocks:
            self.fc_h = BlockLinear(hidden_size, 3 * hidden_size, num_blocks=blocks)
        else:
            self.fc_h = nn.Linear(hidden_size, 3 * hidden_size)

        if norm:
            self.norm_x = RMSNorm(3 * hidden_size)
            self.norm_h = RMSNorm(3 * hidden_size)

    def forward(self, x, h):
        """GRU forward pass.
        
        Args:
            x: [batch, input_size] current input
            h: [batch, hidden_size] previous hidden state
        
        Returns:
            [batch, hidden_size] next hidden state
        """
        # Force h to match x's batch size to prevent dimension mismatches
        if x.shape[0] != h.shape[0]:
            h = h[:x.shape[0]].contiguous()
        
        x_out = self.fc_x(x)
        h_out = self.fc_h(h)
        
        if self.norm:
            x_out = self.norm_x(x_out)
            h_out = self.norm_h(h_out)
        
        # Split into reset, update, and new gates
        x_r, x_z, x_n = torch.chunk(x_out, 3, dim=-1)
        h_r, h_z, h_n = torch.chunk(h_out, 3, dim=-1)
        
        r = torch.sigmoid(x_r + h_r)  # Reset gate
        z = torch.sigmoid(x_z + h_z)  # Update gate
        n = torch.tanh(x_n + r * h_n)  # New candidate
        
        next_h = (1 - z) * n + z * h  # Interpolate
        return next_h


class MLP(nn.Module):
    """Multi-layer perceptron for V3.
    
    V3 MLP: 5 layers with 640 units (vs V2: 4 layers with 400 units)
    Uses RMSNorm and SiLU for stability and performance.
    
    Used for actor, critic, reward, and discount prediction heads.
    """
    def __init__(self, input_dim, output_dim, hidden=640, layers=5, act=nn.SiLU, dist=None):
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
        """Forward pass through MLP.
        
        Args:
            x: [batch, input_dim] or [batch, time, input_dim]
        
        Returns:
            [batch, output_dim] or [batch, time, output_dim]
        """
        x = self.net(x)
        if self.dist == 'symlog_disc':
            # V3: Symlog discretization for value/reward
            return x  # Return logits for TwoHot distribution
        return x


class Encoder(nn.Module):
    """Encoder for Dreamer V3.
    
    Architecture: 64x64 RGB → 4 Conv2d with RMSNorm + SiLU → projection to embed_dim
    Normalization: [0, 255] uint8 → [0, 1] float32
    
    Input: [batch, time, 3, 64, 64]
    Output: [batch, time, embed_dim]
    
    Key differences from V1/V2:
    - Normalizes to [0, 1] not [-0.5, 0.5]
    - Uses RMSNorm after each conv
    - Uses SiLU instead of ReLU
    - Linear projection to configurable embed_dim
    """
    
    def __init__(self, input_shape, config):
        super().__init__()
        depth = config.get('depth', 48)
        input_channels = input_shape[0] if isinstance(input_shape, (list, tuple)) else input_shape
        
        # 4 convolutional layers with RMSNorm and SiLU
        # depth progression: C → 48 → 96 → 192 → 384
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, depth, kernel_size=4, stride=2),
            RMSNorm(depth),
            nn.SiLU(),
            nn.Conv2d(depth, depth * 2, kernel_size=4, stride=2),
            RMSNorm(depth * 2),
            nn.SiLU(),
            nn.Conv2d(depth * 2, depth * 4, kernel_size=4, stride=2),
            RMSNorm(depth * 4),
            nn.SiLU(),
            nn.Conv2d(depth * 4, depth * 8, kernel_size=4, stride=2),
            RMSNorm(depth * 8),
            nn.SiLU()
        )
        
        # Calculate CNN output size dynamically and add projection
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, *input_shape)
            if dummy_input.dtype == torch.uint8:
                dummy_input = dummy_input.float() / 255.0
            B, T, C, H, W = dummy_input.shape
            dummy_flat = dummy_input.view(B*T, C, H, W)
            cnn_out = self.cnn(dummy_flat)
            cnn_out_size = cnn_out.view(B, T, -1).shape[-1]
        
        # Project to desired embed_dim
        self.embed_dim = config.get('embed_dim', 1024)
        self.proj = nn.Linear(cnn_out_size, self.embed_dim)
        
    def forward(self, obs):
        """Encode observations to latent embeddings.
        
        Args:
            obs: [batch, time, C, H, W] uint8 [0, 255] or float32 [0, 1]
                 Can also be dict with 'image' key
        
        Returns:
            [batch, time, embed_dim] embeddings
        """
        # Handle dict input
        if isinstance(obs, dict):
            x = obs['image']
        else:
            x = obs
            
        # Normalize to [0, 1] if uint8 (V3 convention)
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
            
        # x is [B, T, C, H, W]
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        y = self.cnn(x)
        y = y.reshape(B, T, -1)
        return self.proj(y)


class Decoder(nn.Module):
    """VAE Decoder for Dreamer V3.
    
    Architecture: latent → bottleneck (compress/expand) → ConvTranspose → sigmoid
    Output: Sigmoid [0, 1] (NOT Gaussian like V1/V2)
    
    Input: [batch, time, feature_dim]
    Output: {'image': [batch, time, 3, 64, 64]} with values in [0, 1]
    
    Key differences from V1/V2:
    - Bottleneck architecture (funnel down then up)
    - RMSNorm instead of no normalization
    - SiLU instead of ReLU
    - Sigmoid output instead of Normal distribution
    
    Loss computation:
        obs_normalized = obs.float() / 255.0  # [0, 1]
        loss = F.mse_loss(decoder_output, obs_normalized)
        # or: loss = F.binary_cross_entropy(decoder_output, obs_normalized)
    """
    
    def __init__(self, input_dim, shape, config):
        super().__init__()
        depth = config.get('depth', 48)
        output_channels = shape[0] if isinstance(shape, (list, tuple)) else 3
        
        # Bottleneck architecture
        bottleneck_size = 320
        
        # Compression to bottleneck (funnel down)
        self.fc_compress = nn.Linear(input_dim, bottleneck_size)
        self.norm_compress = RMSNorm(bottleneck_size)
        
        # Expansion from bottleneck (funnel up)
        self.fc_expand = nn.Linear(bottleneck_size, 12 * depth)
        self.norm_expand = RMSNorm(12 * depth)
        
        # ConvTranspose layers with RMSNorm and SiLU
        self.convs = nn.Sequential(
            nn.ConvTranspose2d(12 * depth, depth * 3, kernel_size=5, stride=2),
            RMSNorm(depth * 3),
            nn.SiLU(),
            nn.ConvTranspose2d(depth * 3, depth * 2, kernel_size=5, stride=2),
            RMSNorm(depth * 2),
            nn.SiLU(),
            nn.ConvTranspose2d(depth * 2, depth, kernel_size=6, stride=2),
            RMSNorm(depth),
            nn.SiLU(),
            nn.ConvTranspose2d(depth, output_channels, kernel_size=6, stride=2),
        )

    def forward(self, features):
        """Decode latent features to images.
        
        Args:
            features: [batch, time, feature_dim]
        
        Returns:
            dict with 'image' key: [batch, time, C, H, W] with values in [0, 1]
        """
        B, T, _ = features.shape
        
        # VAE funnel: compress → bottleneck → expand → spatial
        x = self.norm_compress(F.silu(self.fc_compress(features.view(B*T, -1))))
        x = self.norm_expand(F.silu(self.fc_expand(x)))
        x = x.view(x.shape[0], -1, 1, 1)
        x = self.convs(x)
        x = torch.sigmoid(x)  # V3 output in [0, 1] range
        
        return {'image': x.view(B, T, *x.shape[1:])}
