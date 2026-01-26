import torch
import torch.nn as nn
from torch.distributions import Normal
from dreamer_v1.networks import ConvEncoder, ConvDecoder


class RSSM_V1(nn.Module):
    def __init__(self, action_dim, stoch_dim, deter_dim, hidden_dim, embed_dim):
        super().__init__()
        self.stoch_dim = stoch_dim
        self.deter_dim = deter_dim

        self.cell = nn.GRUCell(hidden_dim, deter_dim)

        self.pre_gru_net = nn.Sequential(
            nn.Linear(stoch_dim + action_dim, hidden_dim),
            nn.ELU()
        )

        self.transition_net = nn.Sequential(
            nn.Linear(deter_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, stoch_dim * 2)
        )

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
            
            tran_stats = self.transition_net(deter)
            repr_stats = self.representation_net(torch.cat([deter, embed[:, t]], dim=-1))
            
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
        batch_size, stoch_dim = stoch.shape
        deter_dim = deter.shape[-1]
        
        stochs = torch.empty(batch_size, horizon, stoch_dim, device=stoch.device, dtype=stoch.dtype)
        deters = torch.empty(batch_size, horizon, deter_dim, device=deter.device, dtype=deter.dtype)
        
        test_dist = actor(stoch[:1], deter[:1])
        if actor.discrete:
            action_dim = actor.action_dim
        else:
            action_dim = test_dist.mean.shape[-1]
        actions = torch.empty(batch_size, horizon, action_dim, device=stoch.device, dtype=stoch.dtype)

        for t in range(horizon):
            dist = actor(stoch, deter)
            if actor.discrete:
                action = torch.nn.functional.gumbel_softmax(dist.logits, tau=1.0, hard=True)
            else:
                action = torch.tanh(dist.rsample())
            
            deter = self.step(stoch, action, deter)
            
            tran_stats = self.transition_net(deter)
            
            q_mean, q_std = self.get_mean_std(tran_stats)
            stoch = q_mean + q_std * torch.randn_like(q_std)

            stochs[:, t] = stoch
            deters[:, t] = deter
            actions[:, t] = action

        return stochs, deters, actions


class WorldModel(nn.Module):
    """World Model for Dreamer V1.
    
    Learns a latent dynamics model of the environment using:
    - Encoder: observations → embeddings
    - RSSM: latent dynamics (s_t, a_t) → (s_{t+1}, h_{t+1})
    - Decoder: latent states → reconstructed observations
    - Reward predictor: latent states → predicted rewards
    
    This is trained separately from the actor-critic and represents
    the agent's understanding of how the world works.
    
    Paper: "Dream to Control" (Hafner et al., 2020), Section 2.
    """
    
    def __init__(self, obs_shape, action_dim, config):
        """Initialize World Model.
        
        Args:
            obs_shape: (C, H, W) observation shape
            action_dim: Action dimensionality
            config: Dict with model configuration (YAML loaded)
        """
        super().__init__()
        
        cfg = config['model']
        r_cfg = cfg['rssm']
        embed_dim = cfg['embed_dim']
        stoch_dim = r_cfg['stoch_dim']
        deter_dim = r_cfg['deter_dim']
        rssm_hidden = r_cfg['hidden_dim']
        hidden_dim = cfg.get('num_units', 800)
        
        # Only support image observations (3D: [C, H, W])
        assert len(obs_shape) == 3, f"Only image observations supported, got shape {obs_shape}"
        
        # World model components (all trained together)
        self.encoder = ConvEncoder(input_channels=obs_shape[0], embed_dim=embed_dim)
        self.decoder = ConvDecoder(stoch_dim + deter_dim, output_channels=obs_shape[0])
        self.rssm = RSSM_V1(action_dim, stoch_dim, deter_dim, rssm_hidden, embed_dim)
        self.reward = RewardModel(stoch_dim, deter_dim, hidden_dim)  # Part of world model!

    def observe(self, obs, action, state=None):
        """Observe a sequence of observations and actions.
        
        Args:
            obs: [batch, seq, C, H, W] or [batch, C, H, W]
            action: [batch, seq, action_dim] or [batch, action_dim]
            state: Optional initial state (stoch, deter)
        
        Returns:
            q_stats, p_stats, stochs, deters (all with seq dimension)
        """
        # Handle both [batch, seq, ...] and [batch, ...] inputs
        if obs.dim() == 5:  # [batch, seq, C, H, W]
            batch_size, seq_len = obs.shape[:2]
            obs_flat = obs.view(batch_size * seq_len, *obs.shape[2:])
            embed = self.encoder(obs_flat)
            embed = embed.view(batch_size, seq_len, -1)
        else:  # [batch, C, H, W]
            embed = self.encoder(obs)
            if embed.dim() == 2:
                embed = embed.unsqueeze(1)  # Add seq dimension
        
        return self.rssm.observe(embed, action, state)

    def imagine(self, actor, start_state, horizon):
        return self.rssm.imagine(actor, start_state, horizon)


class RewardModel(nn.Module):
    """Reward predictor model.
    
    Predicts scalar reward from latent state (stoch + deter).
    This IS part of the world model - it learns to predict rewards
    from the environment, not to maximize them.
    
    Trained together with encoder/RSSM/decoder using observed rewards
    from the environment (supervised learning).
    
    Architecture: 3 hidden layers × hidden_dim units (paper Appendix A).
    Paper: Part of world model loss (Equation 10).
    """
    
    def __init__(self, stoch_dim, deter_dim, hidden_dim=300):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(stoch_dim + deter_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, stoch, deter):
        """Predict reward from latent state.
        
        Args:
            stoch: [batch, stoch_dim] stochastic state
            deter: [batch, deter_dim] deterministic state
        
        Returns:
            [batch, 1] predicted reward
        """
        x = torch.cat([stoch, deter], dim=-1)
        return self.net(x)


class ValueModel(nn.Module):
    """Value function model (critic).
    
    Estimates expected cumulative return V(s_t) from latent state.
    This is NOT part of the world model - it's part of the actor-critic
    algorithm for policy learning.
    
    Trained with TD(λ) targets from imagined trajectories to evaluate
    how good states are under the current policy.
    
    Architecture: 3 hidden layers × hidden_dim units (paper Appendix A).
    Paper: Equation 8 (value loss).
    """
    
    def __init__(self, stoch_dim, deter_dim, hidden_dim=300):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(stoch_dim + deter_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, stoch, deter):
        """Predict value from latent state.
        
        Args:
            stoch: [batch, stoch_dim] stochastic state
            deter: [batch, deter_dim] deterministic state
        
        Returns:
            [batch, 1] predicted value (expected cumulative return)
        """
        x = torch.cat([stoch, deter], dim=-1)
        return self.net(x)


class ActionModel(nn.Module):
    """Actor network for policy.
    
    Outputs action distribution given latent state.
    This is NOT part of the world model - it's part of the actor-critic
    algorithm for policy learning.
    
    Trained to maximize value predictions from the critic via policy gradient.
    
    Architecture: 3 hidden layers × hidden_dim units (paper Appendix A).
    Paper: Equation 7 (actor loss).
    
    Outputs:
    - Categorical distribution for discrete actions
    - Gaussian distribution for continuous actions
    """
    
    def __init__(self, stoch_dim, deter_dim, action_dim, hidden_dim=300, discrete=False):
        super().__init__()
        self.discrete = discrete
        self.action_dim = action_dim
        out_dim = action_dim if discrete else action_dim * 2
        
        self.net = nn.Sequential(
            nn.Linear(stoch_dim + deter_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, out_dim)
        )
        
        if not discrete:
            self.mean_scale = 5.0
            self.raw_init_std = torch.log(torch.exp(torch.tensor(5.0)) - 1).item()
            self.min_std = 1e-4

    def forward(self, stoch, deter):
        """Compute action distribution from latent state.
        
        Args:
            stoch: [batch, stoch_dim] stochastic state
            deter: [batch, deter_dim] deterministic state
        
        Returns:
            Categorical distribution (discrete) or Normal distribution (continuous)
        """
        x = torch.cat([stoch, deter], dim=-1)
        out = self.net(x)
        
        if self.discrete:
            from torch.distributions import Categorical
            logits = out
            return Categorical(logits=logits)
        else:
            mean, std = torch.chunk(out, 2, dim=-1)
            mean = self.mean_scale * torch.tanh(mean / self.mean_scale)
            std = torch.nn.functional.softplus(std + self.raw_init_std) + self.min_std
            return Normal(mean, std)
