import torch
import torch.nn as nn
import numpy as np
from torch.distributions import kl_divergence
from common.agent import Agent
from .models import WorldModel, ActionModel, ValueModel

class DreamerV1Agent(Agent):
    def __init__(self, config, obs_shape, action_dim, is_discrete, device):
        self.cfg = config
        self.device = device
        self.action_dim = action_dim
        self.is_discrete = is_discrete
        
        self.world_model = WorldModel(obs_shape, action_dim, config).to(device)
        
        try:
            self.world_model.encoder = torch.compile(self.world_model.encoder, mode='reduce-overhead')
            self.world_model.decoder = torch.compile(self.world_model.decoder, mode='reduce-overhead')
        except:
            pass
        
        hidden_dim = config['model'].get('num_units', 400)
        
        self.actor = ActionModel(
            config['model']['rssm']['stoch_dim'], 
            config['model']['rssm']['deter_dim'], 
            action_dim,
            hidden_dim=hidden_dim,
            discrete=is_discrete
        ).to(device)
        
        self.value = ValueModel(
            config['model']['rssm']['stoch_dim'], 
            config['model']['rssm']['deter_dim'],
            hidden_dim=hidden_dim
        ).to(device)

        model_lr = config['model'].get('lr', 6e-4)
        actor_lr = config['actor'].get('lr', 8e-5)
        
        if 'critic' in config and 'lr' in config['critic']:
            value_lr = config['critic']['lr'] 
        else:
            value_lr = 8e-5

        self.wm_optimizer = torch.optim.Adam(self.world_model.parameters(), lr=model_lr)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=value_lr)
        
        # Mixed precision support (configurable)
        self.use_mixed_precision = config.get('mixed_precision', True)
        if self.use_mixed_precision:
            self.scaler = torch.amp.GradScaler('cuda')
            print("Using mixed precision (FP16). Infinite gradient norms are normal with loss scaling.")
            print("To disable: set 'mixed_precision: false' in config or use --precision 32")
        else:
            self.scaler = None
            print("Mixed precision disabled. Using FP32 for numerical stability.")

    def init_state(self, batch_size):
        return (
            torch.zeros(batch_size, self.world_model.rssm.stoch_dim).to(self.device),
            torch.zeros(batch_size, self.world_model.rssm.deter_dim).to(self.device)
        )

    def policy(self, obs, state, last_action, mode='train'):
        with torch.no_grad():
            # Normalize observations: uint8 [0,255] -> float32 [-0.5, 0.5] (paper-faithful)
            if isinstance(obs, torch.Tensor):
                if obs.dtype == torch.uint8:
                    obs = obs.float() / 255.0 - 0.5
                obs_tensor = obs.to(self.device)
            else:
                if obs.dtype == np.uint8:
                    obs = obs.astype(np.float32) / 255.0 - 0.5
                obs_tensor = torch.from_numpy(np.ascontiguousarray(obs)).to(self.device)
            
            if len(obs_tensor.shape) == 3:
                obs_tensor = obs_tensor.unsqueeze(0)
            elif len(obs_tensor.shape) == 1:
                obs_tensor = obs_tensor.float().unsqueeze(0)
            
            embed = self.world_model.encoder(obs_tensor)
            deter = self.world_model.rssm.step(state[0], last_action, state[1])
            stats = self.world_model.rssm.representation_net(torch.cat([deter, embed], dim=-1))
            stoch = self.world_model.rssm.get_dist(stats).sample()
            next_state = (stoch, deter)
            
            action_dist = self.actor(stoch, deter)
            
            if mode == 'train':
                if not self.is_discrete:
                     action = action_dist.mean + torch.randn_like(action_dist.mean) * 0.3
                else:
                     action = action_dist.sample()
            else:
                 if not self.is_discrete:
                     action = action_dist.mean
                 elif hasattr(action_dist, 'mode'):
                     action = action_dist.mode()
                 else:
                     action = action_dist.sample()

            if not self.is_discrete:
                 action = torch.tanh(action)

            if self.is_discrete:
                if len(obs_tensor) > 1:
                     action_idx = action.cpu().numpy()
                     act_data = np.zeros((len(obs_tensor), self.action_dim))
                     act_data[np.arange(len(obs_tensor)), action_idx] = 1.0
                     env_action = action_idx
                else:
                    env_action = action.item()
                    act_data = np.zeros(self.action_dim)
                    act_data[env_action] = 1.0
            else:
                if len(obs_tensor) > 1:
                    env_action = action.cpu().numpy()
                else:
                    env_action = action.cpu().numpy().flatten()
                act_data = env_action
                
        return act_data, next_state, env_action

    def train_step(self, obs, action, reward, terminal):
        if obs.dtype == torch.uint8:
            obs = obs.float() / 255.0 - 0.5
        
        # Handle 5D obs: [batch, seq_len, C, H, W] -> [batch*seq_len, C, H, W] for encoder
        # But keep original shape for RSSM.observe which needs [batch, seq_len, ...]
        if len(obs.shape) == 5:
            batch_size, seq_len = obs.shape[0], obs.shape[1]
            obs_flat = obs.reshape(batch_size * seq_len, obs.shape[2], obs.shape[3], obs.shape[4])
        else:
            batch_size, seq_len = obs.shape[0], 1
            obs_flat = obs
        
        obs_target = obs_flat
        
        device_str = self.device if isinstance(self.device, str) else str(self.device)
        device_type = 'cuda' if 'cuda' in device_str else 'cpu'
        if self.use_mixed_precision:
            autocast_ctx = torch.amp.autocast(device_type)
        else:
            from contextlib import nullcontext
            autocast_ctx = nullcontext()
        
        # ========================================
        # FUSED TRAINING: All 3 phases in single autocast
        # ========================================
        with autocast_ctx:
            # ========================================
            # 1. Train World Model
            # ========================================
            # Encode flattened obs
            embed_flat = self.world_model.encoder(obs_flat)
            # Reshape back for RSSM: [batch*seq_len, embed_dim] -> [batch, seq_len, embed_dim]
            embed = embed_flat.reshape(batch_size, seq_len, -1)
            
            # Pass sequential data to RSSM.observe
            tran_stats, repr_stats, stoch, deter = self.world_model.rssm.observe(embed, action)
            
            # Flatten stoch and deter for decoder: [batch, seq_len, dim] -> [batch*seq_len, dim]
            stoch_flat = stoch.reshape(-1, stoch.shape[-1])
            deter_flat = deter.reshape(-1, deter.shape[-1])
            features_flat = torch.cat([stoch_flat, deter_flat], dim=-1)
            
            recon_dist = self.world_model.decoder(features_flat)
            loss_obs = -recon_dist.log_prob(obs_target).mean()
            
            pred_rew = self.world_model.reward(stoch_flat, deter_flat)
            
            # Reshape reward for loss calculation
            if reward.dim() == 3:  # [batch, seq_len, 1]
                reward_flat = reward.reshape(-1, reward.shape[-1])
            elif reward.dim() == 2:  # [batch, seq_len]
                reward_flat = reward.reshape(-1, 1)
            else:
                reward_flat = reward
            
            loss_rew = 0.5 * nn.functional.mse_loss(pred_rew, reward_flat).mean()

            # Dreamer V1 does not have pcont (discount predictor)
            model_loss = loss_obs + loss_rew
            
            dist_q = self.world_model.rssm.get_dist(tran_stats)
            dist_p = self.world_model.rssm.get_dist(repr_stats)
            kl_val = kl_divergence(dist_p, dist_q).sum(-1)
            kl_free_nats = self.cfg['model'].get('kl_free_nats', 3.0)
            loss_kl = torch.clamp(kl_val, min=kl_free_nats).mean()
            
            kl_scale = self.cfg['model'].get('kl_scale', 1.0)
            model_loss = model_loss + kl_scale * loss_kl
        
        # WORLD MODEL BACKWARD PASS (outside autocast for mixed precision)
        self.wm_optimizer.zero_grad(set_to_none=True)
        if self.use_mixed_precision:
            self.scaler.scale(model_loss).backward()
            self.scaler.unscale_(self.wm_optimizer)
            nn.utils.clip_grad_norm_(self.world_model.parameters(), 100.0)
            self.scaler.step(self.wm_optimizer)
        else:
            model_loss.backward()
            nn.utils.clip_grad_norm_(self.world_model.parameters(), 100.0)
            self.wm_optimizer.step()
        
        # Continue actor and critic in SAME autocast context
        with autocast_ctx:
            # ========================================
            # 2. Train Actor (imagination-based)
            # ========================================
            
            # Use original batch_size and seq_len from input reshaping
            # stoch/deter at this point are [batch, seq_len, dim]
            start_stoch = stoch.detach().flatten(0, 1)  # Faster than view(-1, ...)
            start_deter = deter.detach().flatten(0, 1)
            
            imag_stoch, imag_deter, _ = self.world_model.imagine(self.actor, (start_stoch, start_deter), 
                                                                  self.cfg['actor'].get('horizon', 15))
            
            reward_imag = self.world_model.reward(imag_stoch, imag_deter).squeeze(-1)
            value_imag = self.value(imag_stoch, imag_deter).squeeze(-1)
            
            # Dreamer V1 uses fixed discount gamma (no pcont predictor like V2/V3)
            gamma_val = self.cfg['critic'].get('gamma', 0.99)
            
            returns = self.compute_lambda_returns(reward_imag, value_imag, value_imag[:, -1], 
                                                  self.cfg['critic'].get('lambda', 0.95), gamma_val)
            
            # Cache timestep slices (used in both actor and critic)
            imag_stoch_t = imag_stoch[:, :-1]
            imag_deter_t = imag_deter[:, :-1]
            returns_t = returns[:, :-1]
            
            # Use already computed value_imag as baseline (no extra forward pass needed)
            baseline = value_imag[:, :-1].detach()
            advantage = returns_t - baseline
            loss_actor = -advantage.mean()
            
            # ========================================
            # 3. Train Value Function (critic)
            # ========================================
            # Continue in same autocast context
            
            # IMPORTANT: Detach features to avoid double backward through imagination
            value_pred = self.value(imag_stoch_t.detach(), imag_deter_t.detach()).squeeze(-1)
            returns_target = returns_t.detach()
            
            # Normalize returns using running statistics (in-place EMA)
            if not hasattr(self, 'return_mean'):
                self.return_mean = returns_target.mean()
                self.return_std = returns_target.std() + 1e-8
            else:
                # EMA with momentum 0.99 (in-place for speed)
                self.return_mean = self.return_mean.mul_(0.99).add_(returns_target.mean(), alpha=0.01)
                self.return_std = self.return_std.mul_(0.99).add_(returns_target.std() + 1e-8, alpha=0.01)
            
            returns_normalized = ((returns_target - self.return_mean) / self.return_std).clamp_(-10, 10)
            loss_value = 0.5 * nn.functional.mse_loss(value_pred, returns_normalized)
        
        # End of fused autocast context
        
        self.actor_optimizer.zero_grad(set_to_none=True)
        if self.use_mixed_precision:
            self.scaler.scale(loss_actor).backward()
            self.scaler.unscale_(self.actor_optimizer)
            nn.utils.clip_grad_norm_(self.actor.parameters(), 100.0)
            self.scaler.step(self.actor_optimizer)
        else:
            loss_actor.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 100.0)
            self.actor_optimizer.step()
        
        self.value_optimizer.zero_grad(set_to_none=True)
        if self.use_mixed_precision:
            self.scaler.scale(loss_value).backward()
            self.scaler.unscale_(self.value_optimizer)
            nn.utils.clip_grad_norm_(self.value.parameters(), 100.0)
            self.scaler.step(self.value_optimizer)
            self.scaler.update()
        else:
            loss_value.backward()
            nn.utils.clip_grad_norm_(self.value.parameters(), 100.0)
            self.value_optimizer.step()
        
        return {
            "wm_loss": model_loss.detach(), 
            "obs_loss": loss_obs.detach(),
            "reward_loss": loss_rew.detach(),
            "kl": loss_kl.detach(), 
            "actor_loss": loss_actor.detach(), 
            "value_loss": loss_value.detach(),
        }

    def compute_lambda_returns(self, reward, value, bootstrap, lambda_, gamma):
        """Compute lambda-returns (optimized: in-place ops avoid allocation)."""
        B, T = reward.shape
        returns = reward.clone()  # Start with rewards (avoids zeros_like allocation)
        last_v = bootstrap
        
        # Backward pass (must be sequential due to dependency)
        for t in reversed(range(T)):
            disc = gamma[:, t] if isinstance(gamma, torch.Tensor) else gamma
            # TD(λ): r_t + γ_t * ((1-λ)*V + λ*last_v)
            # Use in-place add_ (reward already in returns)
            td_target = disc * ((1 - lambda_) * value[:, t] + lambda_ * last_v)
            returns[:, t].add_(td_target)
            last_v = returns[:, t]
        
        return returns

    def save(self, path, logs=None):
        data = {
            'world_model': self.world_model.state_dict(),
            'actor': self.actor.state_dict(),
            'value': self.value.state_dict(),
            'wm_opt': self.wm_optimizer.state_dict(),
            'actor_opt': self.actor_optimizer.state_dict(),
            'value_opt': self.value_optimizer.state_dict(),
        }
        if logs is not None:
            data['logs'] = logs
        torch.save(data, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.world_model.load_state_dict(checkpoint['world_model'])
        self.actor.load_state_dict(checkpoint['actor'])
        self.value.load_state_dict(checkpoint['value'])
        self.wm_optimizer.load_state_dict(checkpoint['wm_opt'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_opt'])
        self.value_optimizer.load_state_dict(checkpoint['value_opt'])
        return checkpoint.get('logs', None)
