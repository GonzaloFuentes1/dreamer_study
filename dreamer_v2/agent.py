import torch
import torch.nn as nn
import numpy as np
from torch.distributions import kl_divergence
from common.agent import Agent
from .models import WorldModel, ActorModel, ValueModel


class DreamerV2Agent(Agent):
    def __init__(self, config, obs_shape, action_dim, is_discrete, device):
        self.cfg = config
        self.device = device
        self.action_dim = action_dim
        self.is_discrete = is_discrete
        
        # Create World Model (encapsulates encoder, decoder, RSSM, reward, discount)
        self.world_model = WorldModel(obs_shape, action_dim, config).to(device)
        
        # Create Actor-Critic
        feature_dim = config['model']['rssm']['deter_dim'] + (config['model']['rssm'].get('stoch_dim', 32) * config['model']['rssm'].get('num_classes', 32))
        hidden = config.get('network', {}).get('mlp_units', 400)
        
        self.actor = ActorModel(
            feature_dim,
            action_dim,
            hidden=hidden,
            layers=4,
            discrete=is_discrete
        ).to(device)
        
        self.value = ValueModel(
            feature_dim,
            hidden=hidden,
            layers=4
        ).to(device)
        
        # Target critic for stable value learning
        self.target_value = ValueModel(
            feature_dim,
            hidden=hidden,
            layers=4
        ).to(device)
        self.target_value.load_state_dict(self.value.state_dict())
        self._updates = 0
        
        # Optimizers
        model_lr = float(config['model'].get('lr', 0.0003))
        actor_lr = float(config['actor'].get('lr', 0.00008))
        value_lr = float(config.get('critic', {}).get('lr', 0.00008))
        
        self.wm_optimizer = torch.optim.Adam(self.world_model.parameters(), lr=model_lr, eps=1e-5)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, eps=1e-5)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=value_lr, eps=1e-5)
        
        # Mixed precision support
        self.use_mixed_precision = config.get('mixed_precision', True)
        if self.use_mixed_precision:
            self.scaler = torch.amp.GradScaler('cuda')
            print("✓ Using mixed precision (FP16)")
        else:
            self.scaler = None
            print("✓ Using FP32 precision")

    def init_state(self, batch_size):
        stoch_flat_dim = self.world_model.rssm.stoch_dim * self.world_model.rssm.stoch_classes
        return (
            torch.zeros(batch_size, stoch_flat_dim).to(self.device),
            torch.zeros(batch_size, self.world_model.rssm.deter_dim).to(self.device)
        )

    def policy(self, obs, state, last_action, mode='train'):
        with torch.no_grad():
            # Normalize observations
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
            
            # Encode observation
            embed = self.world_model.encoder(obs_tensor)
            stoch_flat, deter = state
            
            # Step RSSM
            deter = self.world_model.rssm.step(stoch_flat, last_action, deter)
            post_logits = self.world_model.rssm.posterior_net(torch.cat([deter, embed], dim=-1))
            _, _, stoch_flat = self.world_model.rssm.sample_stoch(post_logits)
            
            next_state = (stoch_flat, deter)
            
            # Get action from actor
            features = torch.cat([deter, stoch_flat], dim=-1)
            
            if self.is_discrete:
                logits = self.actor(features)
                action_dist = torch.distributions.Categorical(logits=logits)
                if mode == 'train':
                    action_idx = action_dist.sample()
                else:
                    action_idx = logits.argmax(dim=-1)
                # Return one-hot for environment
                action = torch.zeros_like(logits).scatter_(-1, action_idx.unsqueeze(-1), 1.0)
                env_action = action_idx.cpu().numpy().item() if action_idx.numel() == 1 else action_idx.cpu().numpy()
            else:
                mean = self.actor(features)
                std = torch.exp(self.actor.log_std).expand_as(mean)
                action_dist = torch.distributions.Normal(mean, std)
                if mode == 'train':
                    action = action_dist.sample()
                else:
                    action = mean
                action = torch.tanh(action)
                env_action = action.cpu().numpy().flatten() if action.numel() > 1 else action.cpu().numpy()
            
            # Convert to numpy for buffer
            act_data = action.cpu().numpy().flatten() if action.numel() > 1 else action.cpu().numpy()
                
        return act_data, next_state, env_action

    def train_step(self, obs, action, reward, terminal):
        # Normalize observations
        if torch.isnan(obs).any():
            obs = torch.nan_to_num(obs)
        
        if obs.dtype == torch.uint8:
            obs = obs.float() / 255.0 - 0.5
        
        obs_target = obs
        
        device_str = self.device if isinstance(self.device, str) else str(self.device)
        device_type = 'cuda' if 'cuda' in device_str else 'cpu'
        
        if self.use_mixed_precision:
            autocast_ctx = torch.amp.autocast(device_type)
        else:
            from contextlib import nullcontext
            autocast_ctx = nullcontext()
        
        # ========================================
        # FUSED TRAINING: All 3 phases in single autocast context
        # Reduces overhead from context switching
        # ========================================
        with autocast_ctx:
            # ========================================
            # 1. Train World Model
            # ========================================
            B, T, C, H, W = obs.shape
            obs_flat = obs.view(B*T, C, H, W)
            embed = self.world_model.encoder(obs_flat)
            embed = embed.view(B, T, -1)
            
            # RSSM observe
            prior_logits, post_logits, stoch_flat, deters = self.world_model.rssm.observe(embed, action)
            
            # Reconstruct observations
            features = torch.cat([deters, stoch_flat], dim=-1)
            features_flat = features.view(B*T, -1)
            
            recon_dist = self.world_model.decoder(features_flat)
            loss_recon = -recon_dist.log_prob(obs_target.view(B*T, C, H, W)).mean()
            
            # Predict rewards
            pred_reward = self.world_model.reward_model(features).squeeze(-1)
            reward_target = reward if reward.dim() == pred_reward.dim() else reward.squeeze(-1)
            loss_reward = 0.5 * nn.functional.mse_loss(pred_reward, reward_target)
            
            # Predict discount (continuation probability)
            pred_discount_logits = self.world_model.discount_model(features)
            target_discount = (1.0 - terminal.float())
            if target_discount.dim() < pred_discount_logits.dim():
                target_discount = target_discount.unsqueeze(-1)
            loss_discount = nn.functional.binary_cross_entropy_with_logits(pred_discount_logits, target_discount)
            
            # KL divergence loss
            loss_kl = self.world_model.rssm.kl_loss(
                post_logits, prior_logits,
                alpha=self.cfg['model'].get('kl_alpha', 0.8),
                free_nats=self.cfg['model'].get('kl_free_nats', 3.0)
            )
            
            kl_scale = self.cfg['model'].get('kl_scale', 1.0)
            model_loss = loss_recon + loss_reward + loss_discount + kl_scale * loss_kl
        
        # set_to_none=True is faster than zero_grad() and frees memory
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
        
        # ========================================
        # 2. Train Actor (imagination-based)
        # ========================================
        with autocast_ctx:
            # Start from encoded states (detached from world model gradients)
            start_stoch = stoch_flat.detach().flatten(0, 1)
            start_deter = deters.detach().flatten(0, 1)
            
            # Imagine trajectories
            img_deter, img_stoch, img_actions = self.world_model.rssm.imagine(
                self.actor,
                (start_stoch, start_deter),
                horizon=self.cfg['actor'].get('horizon', 15),
                is_discrete=self.is_discrete
            )
            
            # Predict rewards and values in imagination
            img_features = torch.cat([img_deter, img_stoch], dim=-1)
            img_rewards = self.world_model.reward_model(img_features).squeeze(-1)
            img_values = self.target_value(img_features).squeeze(-1)
            img_discounts = self.world_model.discount_model(img_features).squeeze(-1).sigmoid_()
            
            # Compute λ-returns
            returns = self.compute_lambda_returns(
                img_rewards, img_values, img_values[:, -1],
                lambda_=self.cfg.get('critic', {}).get('lambda_', 0.95),
                gamma=img_discounts
            )
            
            # Actor loss (Equation 6 from paper)
            # For Atari: ρ = 1 (only Reinforce), η = 1e-3 (entropy)
            # For Continuous: ρ = 0 (only dynamics backprop), η = 1e-4 (entropy)
            # L(ψ) = -ρ ln pψ(ât|ẑt) sg(V^λ - vξ) - (1-ρ) V^λ - η H[at|ẑt]
            
            rho = self.cfg.get('actor', {}).get('rho', 1.0)  # 1.0 for Atari, 0.0 for continuous
            entropy_scale = self.cfg.get('actor', {}).get('entropy_scale', 1e-3)  # 1e-3 for Atari, 1e-4 for continuous
            
            # Cache features without last timestep (used multiple times)
            img_features_t = img_features[:, :-1]
            img_actions_t = img_actions[:, :-1]
            returns_t = returns[:, :-1]
            
            # Compute policy distribution (reuse for log_prob and entropy)
            actor_output = self.actor(img_features_t)
            
            if self.is_discrete:
                # Categorical distribution for discrete actions
                from torch.distributions import Categorical
                actor_dist = Categorical(logits=actor_output)
                log_probs = actor_dist.log_prob(img_actions_t.argmax(dim=-1))
                entropy = actor_dist.entropy()
            else:
                # For continuous actions, actor outputs mean only
                # std is a learned parameter (self.actor.log_std)
                from torch.distributions import Normal
                mean = actor_output
                std = torch.exp(self.actor.log_std).expand_as(mean)
                actor_dist = Normal(mean, std)
                log_probs = actor_dist.log_prob(img_actions_t).sum(dim=-1)
                entropy = actor_dist.entropy().sum(dim=-1)
            
            # Reuse img_values as baseline (already computed with target network)
            # This avoids redundant forward pass through value network
            baseline = img_values[:, :-1].detach()
            advantage = (returns_t - baseline).detach()
            
            # Actor loss with entropy regularization (paper Eq 6)
            reinforce_loss = -rho * (log_probs * advantage).mean()
            dynamics_loss = -(1.0 - rho) * returns_t.mean()
            entropy_loss = -entropy_scale * entropy.mean()
            
            loss_actor = reinforce_loss + dynamics_loss + entropy_loss
        
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
        
        # ========================================
        # 3. Train Value Function (critic)
        # ========================================
        with autocast_ctx:
            # IMPORTANT: Detach features to avoid double backward through imagination
            value_pred = self.value(img_features_t.detach()).squeeze(-1)
            returns_target = returns_t.detach()
            
            # Normalize returns for stability using running mean/std (faster than quantiles)
            # Update running statistics with in-place operations
            if not hasattr(self, 'return_mean'):
                self.return_mean = returns_target.mean()
                self.return_std = returns_target.std() + 1e-8
            else:
                # EMA with momentum 0.99 (in-place for speed)
                self.return_mean = self.return_mean.mul_(0.99).add_(returns_target.mean(), alpha=0.01)
                self.return_std = self.return_std.mul_(0.99).add_(returns_target.std() + 1e-8, alpha=0.01)
            
            # Normalize using running statistics (fused operations)
            returns_normalized = ((returns_target - self.return_mean) / self.return_std).clamp_(-10, 10)
            loss_value = 0.5 * nn.functional.mse_loss(value_pred, returns_normalized)
        
        # End of fused autocast context
        
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
        
        # Update target network
        self._update_target_network()
        
        return {
            "wm_loss": model_loss.detach(),
            "kl": loss_kl.detach(),
            "actor_loss": loss_actor.detach(),
            "reinforce": reinforce_loss.detach(),
            "dynamics": dynamics_loss.detach(),
            "entropy": -entropy_loss.detach(),  # Negative to show positive entropy
            "value_loss": loss_value.detach()
        }

    def compute_lambda_returns(self, reward, value, bootstrap, lambda_, gamma):
        """Compute λ-returns (optimized: in-place ops avoid allocation)."""
        B, T = reward.shape
        returns = reward.clone()  # Start with rewards (avoids zeros_like allocation)
        last_v = bootstrap
        
        # Backward pass (must be sequential due to dependency)
        for t in reversed(range(T)):
            disc = gamma[:, t] if isinstance(gamma, torch.Tensor) else gamma
            # TD(λ): r_t + γ_t * ((1-λ)*V(s_{t+1}) + λ*V^λ_{t+1})
            # Use in-place add_ (reward already in returns)
            td_target = disc * ((1 - lambda_) * value[:, t] + lambda_ * last_v)
            returns[:, t].add_(td_target)
            last_v = returns[:, t]
        
        return returns
    
    def _update_target_network(self):
        """Soft update of target value network."""
        self._updates += 1
        if self._updates % self.cfg.get('critic', {}).get('slow_target_update', 100) == 0:
            state_dict = self.value.state_dict()
            
            # Check for NaNs
            if any(torch.isnan(v).any() for v in state_dict.values()):
                return
            
            # Handle torch.compile artifacts
            if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
                state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
            
            self.target_value.load_state_dict(state_dict)

    def save(self, path, logs=None):
        data = {
            'world_model': self.world_model.state_dict(),
            'actor': self.actor.state_dict(),
            'value': self.value.state_dict(),
            'target_value': self.target_value.state_dict(),
            'wm_opt': self.wm_optimizer.state_dict(),
            'actor_opt': self.actor_optimizer.state_dict(),
            'value_opt': self.value_optimizer.state_dict(),
            'updates': self._updates,
        }
        if logs is not None:
            data['logs'] = logs
        torch.save(data, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.world_model.load_state_dict(checkpoint['world_model'])
        self.actor.load_state_dict(checkpoint['actor'])
        self.value.load_state_dict(checkpoint['value'])
        self.target_value.load_state_dict(checkpoint['target_value'])
        self.wm_optimizer.load_state_dict(checkpoint['wm_opt'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_opt'])
        self.value_optimizer.load_state_dict(checkpoint['value_opt'])
        self._updates = checkpoint.get('updates', 0)
        return checkpoint.get('logs', None)
