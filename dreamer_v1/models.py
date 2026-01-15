import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal, Categorical, kl_divergence


class RSSM(nn.Module):
    """
    Recurrent State Space Model (RSSM) for DreamerV1.
    """
    def __init__(self, action_dim, stoch_dim, deter_dim, hidden_dim, embed_dim):
        super().__init__()
        self.stoch_dim = stoch_dim
        self.deter_dim = deter_dim

        # RNN Cell
        self.cell = nn.GRUCell(hidden_dim, deter_dim)

        # Pre-GRU processing (img1 in TF code)
        self.pre_gru_net = nn.Sequential(
            nn.Linear(stoch_dim + action_dim, hidden_dim),
            nn.ELU()
        )

        # Transition model q(s_t | h_t)
        self.transition_net = nn.Sequential(
            nn.Linear(deter_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, stoch_dim * 2)
        )

        # Representation model p(s_t | h_t, e_t)
        self.representation_net = nn.Sequential(
            nn.Linear(deter_dim + embed_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, stoch_dim * 2)
        )

    def get_mean_std(self, stats):
        mean, std = torch.chunk(stats, 2, dim=-1)
        std = torch.nn.functional.softplus(std) + 0.1
        return mean, std

    def get_dist(self, stats):
        mean, std = self.get_mean_std(stats)
        return Normal(mean, std)

    def step(self, prev_stoch, prev_action, prev_deter):
        x = torch.cat([prev_stoch, prev_action], dim=-1)
        x = self.pre_gru_net(x)
        deter = self.cell(x, prev_deter)
        return deter

    def observe(self, embed, action, state=None):
        batch_size, seq_len, _ = embed.shape
        if state is None:
            deter = torch.zeros(batch_size, self.deter_dim, device=embed.device)
            stoch = torch.zeros(batch_size, self.stoch_dim, device=embed.device)
        else:
            stoch, deter = state

        q_stats_list = []
        p_stats_list = []
        deters_list = []
        stochs_list = []

        for t in range(seq_len):
            deter = self.step(stoch, action[:, t], deter)
            
            # Prior q
            tran_stats = self.transition_net(deter)
            
            # Posterior p
            repr_stats = self.representation_net(torch.cat([deter, embed[:, t]], dim=-1))
            
            # Optimization: Manual reparameterization to avoid creating Distributions in loop
            p_mean, p_std = self.get_mean_std(repr_stats)
            stoch = p_mean + p_std * torch.randn_like(p_std)

            q_stats_list.append(tran_stats)
            p_stats_list.append(repr_stats)
            deters_list.append(deter)
            stochs_list.append(stoch)

        q_stats = torch.stack(q_stats_list, dim=1)
        p_stats = torch.stack(p_stats_list, dim=1)
        stochs = torch.stack(stochs_list, dim=1)
        deters = torch.stack(deters_list, dim=1)

        return q_stats, p_stats, stochs, deters

    def imagine(self, actor, start_state, horizon):
        stoch, deter = start_state
        # Preallocate for efficiency (avoid list append + stack)
        batch_size, stoch_dim = stoch.shape
        deter_dim = deter.shape[-1]
        
        stochs = torch.empty(batch_size, horizon, stoch_dim, device=stoch.device, dtype=stoch.dtype)
        deters = torch.empty(batch_size, horizon, deter_dim, device=deter.device, dtype=deter.dtype)
        
        # Dummy action to get dimension
        test_dist = actor(stoch[:1], deter[:1])
        if actor.discrete:
            action_dim = actor.action_dim
        else:
            action_dim = test_dist.mean.shape[-1]
        actions = torch.empty(batch_size, horizon, action_dim, device=stoch.device, dtype=stoch.dtype)

        for t in range(horizon):
            # No .detach() needed - caller uses torch.no_grad()
            dist = actor(stoch, deter)
            if actor.discrete:
                # Gumbel-Softmax for differentiable discrete actions
                action = torch.nn.functional.gumbel_softmax(dist.logits, tau=1.0, hard=True)
            else:
                # Tanh-Normal for continuous actions
                # Paper: "Normal distribution that is then transformed using tanh"
                action = torch.tanh(dist.rsample())
            
            deter = self.step(stoch, action, deter)
            
            tran_stats = self.transition_net(deter)
            
            # Optimization: Manual reparameterization
            q_mean, q_std = self.get_mean_std(tran_stats)
            stoch = q_mean + q_std * torch.randn_like(q_std)

            stochs[:, t] = stoch
            deters[:, t] = deter
            actions[:, t] = action

        return stochs, deters, actions


class ConvEncoder(nn.Module):
    def __init__(self, input_shape, embed_dim=1024):
        super().__init__()
        # Input: (3, 64, 64)
        self.net = nn.Sequential(
            # 64 -> 31
            nn.Conv2d(input_shape[0], 32, 4, stride=2),
            nn.ReLU(),
            # 31 -> 14
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            # 14 -> 6
            nn.Conv2d(64, 128, 4, stride=2),
            nn.ReLU(),
            # 6 -> 2
            nn.Conv2d(128, 256, 4, stride=2),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        # Output will be 256 * 2 * 2 = 1024
        # We assume the embedding dimension matches this output.
        # If not, we could add a Linear layer, but standard PlaNet/Dreamer 
        # uses the flattened output directly if it matches (1024).
        
    def forward(self, obs):
        # Normalize if input is integer-like (0-255) to range [-0.5, 0.5]
        # Input shape expected by Conv2d is (B, C, H, W).
        # But we receive (Batch, Time, C, H, W) = (50, 50, 3, 64, 64) from RSSM observe() loop usually.
        # Wait, in WorldModel.observe, we embed the whole sequence?
        # self.encoder(obs) -> obs is (B, T, C, H, W).
        # Linear layers handle (B, T, ...) automatically as (..., feature).
        # Conv2d expects exactly 3D (C,H,W) or 4D (B, C, H, W).
        # So we must fold Batch and Time dimensions.
        
        # x shape: [Batch, Sequence, Channels, Height, Width]
        # or [Batch, Channels, Height, Width] (if single step)
        
        is_sequence = False
        if obs.ndim == 5:
            is_sequence = True
            B, T, C, H, W = obs.shape
            x = obs.view(B * T, C, H, W)
        else:
            x = obs
            
        if x.dtype == torch.uint8:
            x = x.float() / 255.0 - 0.5
        elif x.max() > 1.0: # Heuristic
             x = x / 255.0 - 0.5
        
        embed = self.flatten(self.net(x))
        
        if is_sequence:
            embed = embed.view(B, T, -1)
            
        return embed

class ConvDecoder(nn.Module):
    def __init__(self, stoch_dim, deter_dim, output_shape=(3, 64, 64)):
        super().__init__()
        self.linear = nn.Linear(stoch_dim + deter_dim, 1024)
        # Optimized Decoder: Upsample + Conv2d
        # Input reshape: (Batch, 256, 2, 2) = 1024 dimensions
        self.net = nn.Sequential(
            # 2x2 -> 4x4
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),            
            # 4x4 -> 8x8
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),            
            # 8x8 -> 16x16
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),            
            # 16x16 -> 32x32
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),            
            # 32x32 -> 64x64
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(32, output_shape[0], 3, padding=1),
        )
        self.output_shape = output_shape

    def forward(self, stoch, deter):
        x = torch.cat([stoch, deter], dim=-1)
        
        # Flatten B and T if necessary
        is_sequence = x.ndim == 3
        if is_sequence:
            B, T, D = x.shape
            x = x.view(B * T, D)
            
        x = self.linear(x)
        # Reshape to map directly to spatial dimensions
        x = x.view(-1, 256, 2, 2)
        recon = self.net(x)
        
        if is_sequence:
            # recon shape: (B*T, C, H, W)
            _, C, H, W = recon.shape
            recon = recon.view(B, T, C, H, W)
            
        return recon

# Aliases to keep compatibility
Encoder = ConvEncoder
Decoder = ConvDecoder

# Legacy Dense classes if needed for vector envs
class DenseEncoder(nn.Module):
    def __init__(self, input_shape, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_shape[0], 300),
            nn.ELU(),
            nn.Linear(300, 300),
            nn.ELU(),
            nn.Linear(300, embed_dim)
        )
    def forward(self, obs):
        return self.net(obs)

class DenseDecoder(nn.Module):
    def __init__(self, stoch_dim, deter_dim, output_shape):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(stoch_dim + deter_dim, 300),
            nn.ELU(),
            nn.Linear(300, 300),
            nn.ELU(),
            nn.Linear(300, output_shape[0])
        )
    def forward(self, stoch, deter):
        x = torch.cat([stoch, deter], dim=-1)
        return self.net(x)

class RewardModel(nn.Module):
    def __init__(self, stoch_dim, deter_dim, hidden_dim=400):
        super().__init__()
        # Paper: "three dense layers of size 300 with ELU activations" -> Code says num_units=400, 2 layers
        self.net = nn.Sequential(
            nn.Linear(stoch_dim + deter_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, stoch, deter):
        x = torch.cat([stoch, deter], dim=-1)
        return self.net(x)

class ContinueModel(nn.Module):
    def __init__(self, stoch_dim, deter_dim, hidden_dim=400):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(stoch_dim + deter_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, stoch, deter):
        x = torch.cat([stoch, deter], dim=-1)
        return self.net(x)



class WorldModel(nn.Module):
    """
    Encapsulates the components that learn the environment dynamics (parameters theta).
    """
    def __init__(self, obs_shape, action_dim, config):
        super().__init__()
        cfg = config['model']
        r_cfg = cfg['rssm']
        hidden_dim = cfg.get('num_units', 400)
        
        # Determine if visual (3D input) or vector (1D input)
        if len(obs_shape) == 3:
            self.encoder = ConvEncoder(obs_shape, cfg['embed_dim'])
            self.decoder = ConvDecoder(r_cfg['stoch_dim'], r_cfg['deter_dim'], obs_shape)
        else:
            self.encoder = DenseEncoder(obs_shape, cfg['embed_dim'])
            self.decoder = DenseDecoder(r_cfg['stoch_dim'], r_cfg['deter_dim'], obs_shape)

        self.rssm = RSSM(action_dim, r_cfg['stoch_dim'], r_cfg['deter_dim'], r_cfg['hidden_dim'], cfg['embed_dim'])
        self.reward = RewardModel(r_cfg['stoch_dim'], r_cfg['deter_dim'], hidden_dim)
        
        # Optional PCont
        if config.get('use_pcont', False):
            self.pcont = ContinueModel(r_cfg['stoch_dim'], r_cfg['deter_dim'], hidden_dim)
        else:
            self.pcont = None

    def observe(self, obs, action, state=None):
        embed = self.encoder(obs)
        return self.rssm.observe(embed, action, state)

    def imagine(self, actor, start_state, horizon):
        return self.rssm.imagine(actor, start_state, horizon)


class ActionModel(nn.Module):
    def __init__(self, stoch_dim, deter_dim, action_dim, hidden_dim=400, discrete=True):
        super().__init__()
        self.discrete = discrete
        # Paper code: 4 hidden layers of size 400
        self.net = nn.Sequential(
            nn.Linear(stoch_dim + deter_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, action_dim if discrete else action_dim * 2)
        )
        self.mean_scale = 5.0
        # log(exp(5) - 1)
        self.raw_init_std = np.log(np.exp(5.0) - 1)
        self.min_std = 1e-4

    def forward(self, stoch, deter):
        x = torch.cat([stoch, deter], dim=-1)
        out = self.net(x)
        if self.discrete:
            return Categorical(logits=out)
        
        mean, std = torch.chunk(out, 2, dim=-1)
        
        # Paper code: mean = 5 * tanh(mean / 5)
        mean = self.mean_scale * torch.tanh(mean / self.mean_scale)
        
        # Paper code: std = softplus(std + initial_offset) + min_std
        std = torch.nn.functional.softplus(std + self.raw_init_std) + self.min_std
        
        return Normal(mean, std)
