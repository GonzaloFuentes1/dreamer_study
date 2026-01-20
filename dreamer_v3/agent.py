import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal, kl_divergence
from common.agent import Agent
from .models import RSSM_V3, ConvEncoder, ConvDecoder, MLP, SymexpTwohotMLP, symlog, symexp


class DreamerV3Agent(Agent):
    def __init__(self, config, obs_shape, action_dim, is_discrete, device):
        super().__init__()
        self.cfg = config
        self.device = device
        self.action_dim = action_dim
        self.is_discrete = is_discrete
        
        # Architecture parameters
        self.cnn_depth = config['model'].get('cnn_depth', 48)
        self.mlp_width = config['model'].get('mlp_width', 512)
        
        self.encoder = ConvEncoder(input_channels=obs_shape[0], depth=self.cnn_depth).to(device)
        
        # ConvEncoder output calculation:
        # Depth doubles at each stage: d -> 2d -> 4d -> 8d
        # Resolution halves 4 times: 64 -> 32 -> 16 -> 8 -> 4 (if input 64x64)
        # Flattened: (8 * depth) * 2 * 2 = 32 * depth.
        embed_dim = 32 * self.cnn_depth
        
        # Add projection layer to get to target embed_dim
        target_embed_dim = config['model'].get('embed_dim', 1024)
        self.embed_proj = nn.Linear(embed_dim, target_embed_dim).to(device)
        
        # V3 uses larger RSSM (512 units default)
        self.rssm = RSSM_V3(
            action_dim=action_dim,
            stoch_dim=config['model']['rssm']['stoch_dim'],
            stoch_classes=config['model']['rssm']['stoch_classes'],
            deter_dim=config['model']['rssm']['deter_dim'],
            hidden_dim=config['model']['rssm']['hidden_dim'],
            embed_dim=target_embed_dim,
            blocks=8 # Standard BlockGRU blocks
        ).to(device)
        
        feature_dim = config['model']['rssm']['deter_dim'] + (
            config['model']['rssm']['stoch_dim'] * config['model']['rssm']['stoch_classes']
        )
        
        self.decoder = ConvDecoder(input_dim=feature_dim, output_channels=obs_shape[0], depth=self.cnn_depth).to(device)
        self.reward_model = SymexpTwohotMLP(feature_dim, num_bins=255, hidden=self.mlp_width, layers=3).to(device)
        self.continue_model = MLP(feature_dim, 1, hidden=self.mlp_width, layers=3).to(device)
        
        # Actor outputs mean and log_std for continuous, or logits for discrete
        if is_discrete:
            self.actor = MLP(feature_dim, action_dim, hidden=self.mlp_width, layers=3).to(device)
        else:
            self.actor_mean = MLP(feature_dim, action_dim, hidden=self.mlp_width, layers=3).to(device)
            self.actor_log_std = nn.Parameter(torch.zeros(action_dim).to(device))
        
        self.critic = SymexpTwohotMLP(feature_dim, num_bins=255, hidden=self.mlp_width, layers=3).to(device)
        self.target_critic = SymexpTwohotMLP(feature_dim, num_bins=255, hidden=self.mlp_width, layers=3).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self._updates = 0
        
        # V3 unified learning rate & LaProp Optimizer
        lr = config['model'].get('lr', 4e-5)
        eps = 1e-20 # LaProp epsilon
        
        self.wm_optimizer = LaProp(
            list(self.encoder.parameters()) + 
            list(self.embed_proj.parameters()) +
            list(self.rssm.parameters()) +
            list(self.decoder.parameters()) +
            list(self.reward_model.parameters()) +
            list(self.continue_model.parameters()),
            lr=lr, eps=eps
        )
        
        if is_discrete:
            self.actor_optimizer = LaProp(self.actor.parameters(), lr=lr, eps=eps)
        else:
            self.actor_optimizer = LaProp(
                list(self.actor_mean.parameters()) + [self.actor_log_std], 
                lr=lr, eps=eps
            )
        self.critic_optimizer = LaProp(self.critic.parameters(), lr=lr, eps=eps)
        
        self.scaler = torch.amp.GradScaler('cuda')
        
        # Return normalization (paper Eq 7) - initialize as simple tensor
        self.return_scale = torch.ones(1, device=device)
        
        # Advantage normalization (paper p.6) - initialize with momentum
        self.adv_mean = torch.zeros(1, device=device)
        # Initialize std to 1 to avoid division by zero
        self.adv_std = torch.ones(1, device=device)

    def init_state(self, batch_size):
        stoch_flat_dim = self.rssm.stoch_dim * self.rssm.stoch_classes
        return (
            torch.zeros(batch_size, stoch_flat_dim).to(self.device),
            torch.zeros(batch_size, self.rssm.deter_dim).to(self.device)
        )

    def policy(self, obs, state, last_action, mode='train'):
        with torch.no_grad():
            if isinstance(obs, torch.Tensor):
                obs_tensor = obs.to(self.device)
            else:
                obs_tensor = torch.from_numpy(np.ascontiguousarray(obs)).to(self.device)
            
            if obs_tensor.dtype == torch.uint8:
                obs_tensor = obs_tensor.float() / 255.0 - 0.5
            else:
                obs_tensor = torch.sign(obs_tensor) * torch.log(torch.abs(obs_tensor) + 1.0)
            
            if len(obs_tensor.shape) == 3:
                obs_tensor = obs_tensor.unsqueeze(0)
            elif len(obs_tensor.shape) == 1:
                obs_tensor = obs_tensor.float().unsqueeze(0)
            
            embed = self.encoder(obs_tensor)
            embed = self.embed_proj(embed)
            stoch_flat, deter = state
            
            # Embed action before passing to step
            x_action = self.rssm.action_in(last_action)
            deter = self.rssm.step(stoch_flat, x_action, deter)
            post_logits = self.rssm.posterior_net(torch.cat([deter, embed], dim=-1))
            _, _, stoch_flat = self.rssm.get_stoch_state(post_logits)
            
            next_state = (stoch_flat, deter)
            
            features = torch.cat([deter, stoch_flat], dim=-1)
            
            # Get action distribution
            if self.is_discrete:
                logits = self.actor(features)
                dist = torch.distributions.Categorical(logits=logits)
                if mode == 'train':
                    action_idx = dist.sample()
                else:
                    action_idx = logits.argmax(dim=-1)
                action = torch.zeros_like(logits).scatter_(-1, action_idx.unsqueeze(-1), 1.0)
            else:
                mean = self.actor_mean(features)
                std = torch.exp(self.actor_log_std).expand_as(mean)
                dist = torch.distributions.Normal(mean, std)
                if mode == 'train':
                    action = dist.sample()
                else:
                    action = mean
                action = torch.tanh(action)
            
            if self.is_discrete:
                action_idx = action.argmax(dim=-1).cpu().numpy()
                if len(obs_tensor) > 1:
                    act_data = np.zeros((len(obs_tensor), self.action_dim))
                    act_data[np.arange(len(obs_tensor)), action_idx] = 1.0
                    env_action = action_idx
                else:
                    env_action = int(action_idx.item())
                    act_data = np.zeros(self.action_dim)
                    act_data[env_action] = 1.0
            else:
                if len(obs_tensor) > 1:
                    env_action = action.cpu().numpy()
                else:
                    env_action = action.cpu().numpy().squeeze(0)
                act_data = env_action
        
        return act_data, next_state, env_action

    def train_step(self, obs, action, reward, terminal):
        # DreamerV3 normalization (paper Section 4.1 & JAX implementation)
        if obs.dtype == torch.uint8:
            # Input to encoder: [-0.5, 0.5]
            obs_input = obs.float() / 255.0 - 0.5
            # Target for decoder: [0, 1] (asymmetric as in JAX code)
            obs_target = obs.float() / 255.0
        else:
            # Vector observations: Symlogged for both
            obs_input = torch.sign(obs) * torch.log(torch.abs(obs) + 1.0)
            obs_target = obs_input
        
        obs = obs_input
        
        # PROFILING
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t2 = torch.cuda.Event(enable_timing=True)
        t3 = torch.cuda.Event(enable_timing=True)
        t_end = torch.cuda.Event(enable_timing=True)
        t0.record()
        
        with torch.amp.autocast('cuda'):
            B, T, C, H, W = obs.shape
            obs_flat = obs.view(B*T, C, H, W)
            embed = self.encoder(obs_flat)
            embed = self.embed_proj(embed)
            embed = embed.view(B, T, -1)
            
            prior_logits, post_logits, stoch_flat, deters = self.rssm.observe(embed, action)
            
            features = torch.cat([deters, stoch_flat], dim=-1)
            
            recon = self.decoder(features.view(B*T, -1))
            recon = recon.view(B, T, C, H, W)
            # MSE loss with 0.5 factor (standard for Gaussian log-likelihood)
            # Sum over pixels (C, H, W), mean over batch/time
            loss_recon = 0.5 * nn.functional.mse_loss(recon, obs_target, reduction='none').sum(dim=[-3,-2,-1]).mean()
            
            # V3: Reward prediction via twohot categorical (paper Eq 11)
            reward_flat = (reward if reward.dim() == 2 else reward.squeeze(-1)).view(-1)
            loss_reward = self.reward_model.loss(features.view(B*T, -1), reward_flat)
            
            # V3: Continue predictor (1 = episode continues, 0 = terminal)
            pred_continue_logits = self.continue_model(features).squeeze(-1)
            target_continue = (1.0 - terminal.float())
            # Ensure target has same shape as pred: [B, T]
            if target_continue.dim() == 3:
                target_continue = target_continue.squeeze(-1)
            loss_continue = nn.functional.binary_cross_entropy_with_logits(pred_continue_logits, target_continue).mean()
            
            # V3: KL loss with dynamics and representation components (paper Eq 3)
            # L_dyn = max(1, KL[sg(post)||prior]) and L_rep = max(1, KL[post||sg(prior)])
            loss_kl = self.rssm.kl_loss(
                post_logits, prior_logits,
                free_nats=self.cfg['model'].get('kl_free_nats', 1.0),
                beta_dyn=self.cfg['model'].get('beta_dyn', 1.0),
                beta_rep=self.cfg['model'].get('beta_rep', 0.1)
            )
            
            # Paper Eq 2: βpred=1, βdyn=1, βrep=0.1
            beta_pred = self.cfg['model'].get('beta_pred', 1.0)
            loss_wm = beta_pred * (loss_recon + loss_reward + loss_continue) + loss_kl
        
        t1.record() # End WM Fwd

        self.wm_optimizer.zero_grad()
        self.scaler.scale(loss_wm).backward()
        self.scaler.unscale_(self.wm_optimizer)
        
        # AGC (Adaptive Gradient Clipping)
        self.adaptive_gradient_clipping(
            list(self.encoder.parameters()) +
            list(self.embed_proj.parameters()) +
            list(self.rssm.parameters()) +
            list(self.decoder.parameters()) +
            list(self.reward_model.parameters()) +
            list(self.continue_model.parameters()),
            threshold=0.3
        )
        
        self.scaler.step(self.wm_optimizer)
        t2.record() # End WM Bwd
        
        # Actor-Critic training
        with torch.amp.autocast('cuda'):
            start_stoch = stoch_flat.detach().view(-1, stoch_flat.shape[-1])
            start_deter = deters.detach().view(-1, deters.shape[-1])
            
            # Create actor function for imagine
            if self.is_discrete:
                actor_fn = self.actor
            else:
                actor_fn = self.actor_mean
            
            img_deter, img_stoch, img_actions = self.rssm.imagine(
                actor_fn, (start_stoch, start_deter),
                horizon=self.cfg['actor'].get('horizon', 15),
                is_discrete=self.is_discrete
            )
            
            img_features = torch.cat([img_deter, img_stoch], dim=-1)
            t3.record() # End Imagine
            
            # V3 Behavior: Stop gradients from AC to World Model (JAX sg(imgfeat))
            img_features_ac = img_features.detach()
            
            img_rewards = self.reward_model.predict(img_features_ac)
            img_values = self.target_critic.predict(img_features_ac)
            img_continues = torch.sigmoid(self.continue_model(img_features_ac).squeeze(-1))
            
            returns = self.compute_lambda_returns(
                img_rewards, img_values, img_values[:, -1],
                lambda_=self.cfg['critic'].get('lambda', 0.95),
                gamma=img_continues
            )
            
            # V3: Return normalization (paper Eq 6-7)
            return_scale = self.update_return_scale(returns)
            eta = self.cfg['actor'].get('entropy_coef', 3e-4)
            
            # Paper Eq 6: Normalize RETURNS
            # Only scale down when return_scale > 1
            normalized_returns = returns / torch.maximum(torch.ones_like(return_scale), return_scale)
            
            # Get entropy for regularization
            if self.is_discrete:
                logits = self.actor(img_features_ac[:, :-1])
                action_dist = torch.distributions.Categorical(logits=logits)
                entropy = action_dist.entropy().mean()
                
                # Compute advantage baseline for REINFORCE
                normalized_values = img_values / torch.maximum(torch.ones_like(return_scale), return_scale)
                advantage = normalized_returns[:, :-1] - normalized_values[:, :-1]
                
                img_actions_slice = img_actions[:, :-1]
                if img_actions_slice.dim() == 3 and img_actions_slice.shape[-1] == 1:
                    img_actions_slice = img_actions_slice.squeeze(-1)
                log_probs = action_dist.log_prob(img_actions_slice.detach())
                
                loss_actor = -(advantage.detach() * log_probs).mean() - eta * entropy
            else:
                mean = self.actor_mean(img_features_ac[:, :-1])
                std = torch.exp(self.actor_log_std).expand_as(mean)
                action_dist = torch.distributions.Normal(mean, std)
                entropy = action_dist.entropy().sum(dim=-1).mean()
                
                # V3: REINFORCE for continuous actions (paper Eq 6)
                # This replaces the Dynamics Backprop used in DreamerV2
                normalized_values = img_values / torch.maximum(torch.ones_like(return_scale), return_scale)
                advantage = normalized_returns[:, :-1] - normalized_values[:, :-1]
                
                # Advantage Normalization (Crucial for V3 stability)
                self.update_adv_stats(advantage)
                normalized_advantage = (advantage - self.adv_mean) / torch.maximum(torch.ones_like(self.adv_std), self.adv_std)
                
                log_probs = action_dist.log_prob(img_actions[:, :-1].detach()).sum(dim=-1)
                loss_actor = -(normalized_advantage.detach() * log_probs).mean() - eta * entropy
        
        t4.record() # End Act Loss

        self.actor_optimizer.zero_grad()
        self.scaler.scale(loss_actor).backward()
        self.scaler.unscale_(self.actor_optimizer)
        if self.is_discrete:
            self.adaptive_gradient_clipping(self.actor.parameters(), threshold=0.3)
        else:
            self.adaptive_gradient_clipping(
                list(self.actor_mean.parameters()) + [self.actor_log_std], threshold=0.3
            )
        self.scaler.step(self.actor_optimizer)
        t5.record() # End Act Upd
        
        # Critic learning with twohot loss and replay (paper Table 4: βval=1, βrepval=0.3)
        with torch.amp.autocast('cuda'):
            # Imagination loss
            loss_critic_imag = self.critic.loss(img_features[:, :-1].detach(), returns[:, :-1].detach())
            
            # Replay loss (paper p.6)
            replay_features = features[:, :-1].detach().reshape(-1, features.shape[-1])
            # Ensure rewards have shape [B, T] not [B, T, 1]
            reward_replay = reward[:, :-1]
            if reward_replay.dim() == 3:
                reward_replay = reward_replay.squeeze(-1)
            replay_returns = self.compute_lambda_returns(
                reward_replay, 
                self.critic.predict(features[:, 1:].detach()),
                self.critic.predict(features[:, -1:].detach()).squeeze(-1),
                lambda_=self.cfg['critic'].get('lambda', 0.95),
                gamma=1.0 - terminal[:, :-1].float()
            ).detach()
            loss_critic_replay = self.critic.loss(replay_features, replay_returns.reshape(-1))
            
            # EMA regularization (paper p.6)
            loss_critic_reg = self.critic_ema_regularization(img_features[:, :-1].detach())
            
            loss_critic = loss_critic_imag + 0.3 * loss_critic_replay + loss_critic_reg
        
        self.critic_optimizer.zero_grad()
        self.scaler.scale(loss_critic).backward()
        self.scaler.unscale_(self.critic_optimizer)
        self.adaptive_gradient_clipping(self.critic.parameters(), threshold=0.3)
        self.scaler.step(self.critic_optimizer)
        
        self.scaler.update()
        
        # Update target network (EMA every step)
        self._update_target_network()
        self._updates += 1
        
        # Print profiling stats every 100 steps
        t_end.record()
        if self._updates % 100 == 0:
            torch.cuda.synchronize()
            print(f"Step {self._updates}: "
                  f"Total: {t0.elapsed_time(t_end):.2f} ms | "
                  f"WM Fwd: {t0.elapsed_time(t1):.2f} ms | "
                  f"WM Bwd: {t1.elapsed_time(t2):.2f} ms | "
                  f"Imagine: {t2.elapsed_time(t3):.2f} ms | "
                  f"Act Loss: {t3.elapsed_time(t4):.2f} ms | "
                  f"Act Upd: {t4.elapsed_time(t5):.2f} ms | "
                  f"Crit: {t5.elapsed_time(t_end):.2f} ms")
        
        return {
            'wm_loss': loss_wm.item(),
            'recon': loss_recon.item(),
            'reward': loss_reward.item(),
            'continue': loss_continue.item(),
            'kl': loss_kl.item(),
            'actor_loss': loss_actor.item(),
            'value_loss': loss_critic.item()
        }

    def _update_target_network(self):
        # EMA update (decay=0.98 from paper Table 4)
        mix = 0.98
        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(mix * target_param.data + (1 - mix) * param.data)
    def update_return_scale(self, returns):
        """Update return scale using percentiles (paper Eq 7)."""
        with torch.no_grad():
            flat_returns = returns.flatten()
            percentile_95 = torch.quantile(flat_returns, 0.95)
            percentile_05 = torch.quantile(flat_returns, 0.05)
            new_scale = percentile_95 - percentile_05
            
            # Update EMA and detach to prevent graph accumulation
            decay = 0.9
            self.return_scale = decay * self.return_scale + (1 - decay) * new_scale.unsqueeze(0)
            self.return_scale = torch.clamp(self.return_scale, 0.1, 10.0).detach()
        
        return self.return_scale
    
    def update_adv_stats(self, advantage):
        """Update advantage statistics using EMA (paper p.6)."""
        with torch.no_grad():
            # Batch mean and std
            batch_mean = advantage.mean()
            batch_std = advantage.std()
            
            # EMA updates (decay=0.99 from JAX code)
            decay = 0.99
            self.adv_mean = decay * self.adv_mean + (1 - decay) * batch_mean
            self.adv_std = decay * self.adv_std + (1 - decay) * batch_std
            
            # Detach to prevent graph accumulation
            self.adv_mean = self.adv_mean.detach()
            self.adv_std = torch.clamp(self.adv_std, 0.1, 10.0).detach()

    def critic_ema_regularization(self, features):
        """Regularize critic towards EMA of its own parameters (paper p.6)."""
        # Get predictions from current critic
        current_pred = self.critic.predict(features)
        # Get predictions from target (EMA) critic
        with torch.no_grad():
            target_pred = self.target_critic.predict(features)
        # Regularize to match
        return 0.5 * ((current_pred - target_pred) ** 2).mean()

    def compute_lambda_returns(self, rewards, values, bootstrap, lambda_, gamma):
        # Ensure all inputs are [B, T] shape
        if rewards.dim() == 3:
            rewards = rewards.squeeze(-1)
        if values.dim() == 3:
            values = values.squeeze(-1)
        if gamma.dim() == 3:
            gamma = gamma.squeeze(-1)
            
        B, T = rewards.shape
        returns = torch.zeros_like(rewards)
        last_value = bootstrap
        
        for t in reversed(range(T)):
            if t == T - 1:
                returns[:, t] = rewards[:, t] + gamma[:, t] * last_value
            else:
                returns[:, t] = rewards[:, t] + gamma[:, t] * (
                    (1 - lambda_) * values[:, t+1] + lambda_ * returns[:, t+1]
                )
        
        return returns

    def save(self, path, logs=None):
        data = {
            'encoder': self.encoder.state_dict(),
            'embed_proj': self.embed_proj.state_dict(),
            'rssm': self.rssm.state_dict(),
            'decoder': self.decoder.state_dict(),
            'reward': self.reward_model.state_dict(),
            'continue': self.continue_model.state_dict(),
            'critic': self.critic.state_dict(),
            'target_critic': self.target_critic.state_dict(),
            'wm_opt': self.wm_optimizer.state_dict(),
            'actor_opt': self.actor_optimizer.state_dict(),
            'critic_opt': self.critic_optimizer.state_dict(),
            'updates': self._updates,
        }
        if self.is_discrete:
            data['actor'] = self.actor.state_dict()
        else:
            data['actor_mean'] = self.actor_mean.state_dict()
            data['actor_log_std'] = self.actor_log_std
        if logs is not None:
            data['logs'] = logs
        torch.save(data, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.embed_proj.load_state_dict(checkpoint['embed_proj'])
        self.rssm.load_state_dict(checkpoint['rssm'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        self.reward_model.load_state_dict(checkpoint['reward'])
        self.continue_model.load_state_dict(checkpoint['continue'])
        if self.is_discrete:
            self.actor.load_state_dict(checkpoint['actor'])
        else:
            self.actor_mean.load_state_dict(checkpoint['actor_mean'])
            self.actor_log_std.data.copy_(checkpoint['actor_log_std'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.target_critic.load_state_dict(checkpoint['target_critic'])
        self.wm_optimizer.load_state_dict(checkpoint['wm_opt'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_opt'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_opt'])
        self._updates = checkpoint.get('updates', 0)
        return checkpoint.get('logs', None)
    
    def adaptive_gradient_clipping(self, parameters, threshold=0.3, eps=1e-3):
        """Adaptive Gradient Clipping (AGC) as per DreamerV3 paper (Vectorized)."""
        # Filter parameters with gradients
        params = [p for p in parameters if p.grad is not None]
        if not params:
            return
            
        grads = [p.grad for p in params]
        
        # Compute norms efficiently using foreach (returns list of scalar tensors)
        p_norms = torch._foreach_norm(params)
        g_norms = torch._foreach_norm(grads)
        
        # Batch computation of clipping coefficients on device
        device = params[0].device
        p_norms_stack = torch.stack(p_norms)
        g_norms_stack = torch.stack(g_norms)
        
        max_norms = torch.maximum(p_norms_stack, torch.tensor(eps, device=device)) * threshold
        
        # Calculate clip coefficients: min(1, max_norm / grad_norm)
        clip_coefs = torch.clamp(max_norms / (g_norms_stack + 1e-6), max=1.0)
        
        # Apply clipping batched
        # unbind converts 1D tensor back to list of scalar tensors for foreach
        torch._foreach_mul_(grads, torch.unbind(clip_coefs))


class LaProp(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('LaProp does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                # 1. Update second moment (RMSProp style)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # 2. Normalize gradient
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                norm_grad = grad / denom

                # 3. Update momentum (on normalized gradient)
                exp_avg.mul_(beta1).add_(norm_grad, alpha=1 - beta1)

                # 4. Update parameters
                p.add_(exp_avg, alpha=-group['lr'])

        return loss
