"""
Dreamer V1 Networks: Encoder and Decoder for Gaussian latent world models.

Normalization: [0, 255] uint8 → [-0.5, 0.5] float32
Decoder output: Normal distribution with mean ∈ ℝ and fixed std=1.0
"""

import torch
import torch.nn as nn
from torch.distributions import Normal, Independent


class ConvEncoder(nn.Module):
    """VAE Encoder for Dreamer V1.
    
    Architecture: 64x64 RGB → 4 Conv2d layers → 1024 embedding
    Normalization: [0, 255] uint8 → [0, 1] float32
    
    Note: Papers don't specify [-0.5, 0.5] explicitly. Using [0, 1] is simpler
    and mathematically equivalent for Gaussian decoder output. The key is
    consistency between encoder normalization and decoder training targets.
    
    Input: [batch, 3, 64, 64]
    Output: [batch, 1024]
    """
    
    def __init__(self, input_channels=3, embed_dim=1024):
        super().__init__()
        self.embed_dim = embed_dim
        
        # 4 convolutional layers with stride 2
        # 64x64 → 31x31 → 14x14 → 6x6 → 2x2
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.ReLU(),
        )
        
        self.flatten = nn.Flatten()
        # 2x2x256 = 1024 features
        self.fc = nn.Linear(1024, embed_dim) if embed_dim != 1024 else nn.Identity()
    
    def forward(self, obs):
        """Encode observations to latent embeddings.
        
        Args:
            obs: [batch, 3, 64, 64] uint8 [0, 255] or float32 [0, 1]
        
        Returns:
            [batch, 1024] embeddings
        """
        # Normalize to [0, 1] if uint8
        if obs.dtype == torch.uint8:
            obs = obs.float() / 255.0
        
        x = self.net(obs)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class ConvDecoder(nn.Module):
    """VAE Decoder for Dreamer V1.
    
    Architecture: latent → FC → 5 ConvTranspose2d layers → 64x64 RGB
    Output: Gaussian distribution Normal(mean, std=1)
    
    Input: [batch, feature_dim]
    Output: Independent(Normal(mean, std=1), reinterpreted_batch_ndims=3)
            where mean.shape = [batch, 3, 64, 64]
    
    Loss computation:
        obs_normalized = obs.float() / 255.0  # [0, 1]
        loss = -decoder_dist.log_prob(obs_normalized).mean()
        
    Note: [-0.5, 0.5] normalization is NOT required. Papers don't specify it.
    Using [0, 1] is simpler and gives identical gradients for Normal(μ, σ=1).
    """
    
    def __init__(self, feature_dim, output_channels=3, std=1.0):
        super().__init__()
        self.output_channels = output_channels
        self.std = std
        
        # Project to spatial: 256 channels, 2x2
        self.fc = nn.Linear(feature_dim, 256 * 2 * 2)
        
        # 5 ConvTranspose layers: 2x2 → 4x4 → 8x8 → 16x16 → 32x32 → 64x64
        # Kernels from paper: 5x5, 5x5, 5x5, 6x6, 6x6
        self.net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=6, stride=2, padding=2, output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(32, output_channels, kernel_size=6, stride=2, padding=2, output_padding=0),
        )
    
    def forward(self, features):
        """Decode latent features to image distribution.
        
        Args:
            features: [batch, feature_dim]
        
        Returns:
            Independent(Normal(mean, std), reinterpreted_batch_ndims=3)
            where mean.shape = [batch, 3, 64, 64]
        """
        x = self.fc(features)
        x = x.view(-1, 256, 2, 2)
        mean = self.net(x)
        
        return Independent(
            Normal(mean, self.std),
            reinterpreted_batch_ndims=3
        )


class MLP(nn.Module):
    """Multi-Layer Perceptron for auxiliary models.
    
    Used for reward predictor, value network, etc.
    """
    
    def __init__(self, input_dim, output_dim, hidden_dim=400, hidden_layers=1):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for _ in range(hidden_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ELU())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)
