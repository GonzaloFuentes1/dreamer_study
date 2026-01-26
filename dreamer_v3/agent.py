import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
import torch.optim as optim
import numpy as np
import copy

try:
    import pytorch_optimizer as optim_plus
except ImportError:
    optim_plus = None

from .models import WorldModel, ActorModel, CriticModel
from .utils import adaptive_gradient_clip, twohot_loss


class DreamerV3Agent(nn.Module):
    def __init__(self, config, obs_shape, action_dim, is_discrete, device):
        super().__init__()
        self.cfg = config
        self.device = device
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.is_discrete = is_discrete
        self.num_bins = 255

        # Add device to config for WorldModel
        config['device'] = device
        
        # World Model (contains encoder, decoder, RSSM, reward, continue)
        self.world_model = WorldModel(obs_shape, action_dim, config).to(self.device)
        
        # Get feature dimension from world model's RSSM
        feat_dim = self.world_model.rssm.deter + self.world_model.rssm.stoch * self.world_model.rssm.classes
        
        # Actor Critic (Paper: 5 layers, 640 units, critic uses twohot)
        model_cfg = config['model']
        mlp_hidden = model_cfg.get('mlp_hidden', 640)
        mlp_layers = model_cfg.get('mlp_layers', 5)
        
        self.actor = ActorModel(feat_dim, action_dim, hidden=mlp_hidden, layers=mlp_layers, is_discrete=is_discrete).to(self.device)
        self.critic = CriticModel(feat_dim, hidden=mlp_hidden, layers=mlp_layers, device=self.device).to(self.device)
        self.slow_critic = copy.deepcopy(self.critic).to(self.device)
        
        # Optimizer settings: V3 paper uses LaProp with eps=1e-20
        lr = float(config.get('model', {}).get('lr', config['training'].get('lr', 3e-4)))
        
        # Try LaProp first (V3 default), fallback to Adam
        if optim_plus is not None:
            try:
                print("Using LaProp optimizer (V3 default)")
                self.opt = optim_plus.LaProp(self.parameters(), lr=lr, eps=1e-20)
            except AttributeError:
                print("LaProp not available, using Adam")
                self.opt = optim.Adam(self.parameters(), lr=lr, eps=1e-8)
        else:
            print("pytorch_optimizer not installed, using Adam")
            self.opt = optim.Adam(self.parameters(), lr=lr, eps=1e-8)
        
        # EMA decay for slow critic
        self.ema_decay = config.get('critic', {}).get('ema_decay', 0.98)
        
        # Return normalization scale (EMA, paper eq. 7)
        self.register_buffer('return_scale', torch.ones(1, device=self.device))

    def policy(self, obs, state, last_action, mode='train'):
        # obs: tensor or numpy
        # state: (stoch_flat, deter) tuple
        # last_action: tensor [B, A]
        
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).to(self.device)
        
        # Ensure obs is (B, C, H, W)
        if obs.ndim == 3: # (C, H, W) -> (1, C, H, W)
            obs = obs.unsqueeze(0)
            
        B = obs.shape[0]
        # Handle initial state
        if state is None or (isinstance(state, tuple) and len(state) == 0):
            stoch_flat, deter = self.world_model.rssm.initial(B, obs.device)
        else:
            stoch_flat, deter = state
            # ALWAYS force to match current batch size B (no conditionals)
            # This handles any size mismatch from initialization or previous steps
            stoch_flat = stoch_flat[:B].contiguous()
            deter = deter[:B].contiguous()
        
        # Ensure last_action also matches B
        last_action = last_action[:B].contiguous()
        
        # Encoder expect (B, T, ...) so unsqueeze time
        embed = self.world_model.encoder(dict(image=obs.unsqueeze(1))) # (B, 1, E)
        
        # RSSM Step
        # 1. Deterministic step
        inp = torch.cat([stoch_flat, last_action], dim=-1)
        x = self.world_model.rssm.img_in(inp)
        deter = self.world_model.rssm.cell(x, deter)
        
        # 2. Posterior (Observe)
        obs_inp = torch.cat([deter, embed.squeeze(1)], dim=-1)
        post_logit = self.world_model.rssm.obs_out(obs_inp)  # Keep flat for get_dist
        
        # Sample
        if mode == 'train':
            dist = self.world_model.rssm.get_dist(post_logit)
            stoch = dist.sample() + dist.probs - dist.probs.detach()
        else:
            # For eval, reshape manually then argmax
            post_logit_shaped = post_logit.view(B, self.world_model.rssm.stoch, self.world_model.rssm.classes)
            stoch = F.one_hot(torch.argmax(post_logit_shaped, dim=-1), self.world_model.rssm.classes).float()
            
        stoch_flat = stoch.view(B, -1)
        
        # Feature for Actor
        feat = torch.cat([deter, stoch_flat], dim=-1)
        
        # Actor - now uses ActorModel
        dist = self.actor.get_distribution(feat)
        
        if not self.is_discrete:
            # Continuous actions
            if mode == 'train':
                action = dist.rsample()
            else:
                action = torch.tanh(dist.mean)  # No exploration in eval
            env_action = action.detach().cpu().numpy()
            
        else:
            # Discrete actions
            if mode == 'train':
                action = dist.sample()
            else:
                action_idx = torch.argmax(dist.logits, dim=-1)
                action = F.one_hot(action_idx, self.action_dim).float()
            
            env_action = np.argmax(action.detach().cpu().numpy(), axis=-1)
            
        next_state = (stoch_flat, deter)
        
        # Returns: act_data (for buffer), next_state, env_action (for env)
        return action.detach().cpu().numpy(), next_state, env_action

    def train_step(self, obs, actions, rewards, dones):
        # obs: (B, T, C, H, W)
        # actions: (B, T, A)
        
        # 1. Encode
        embed = self.world_model.encoder(dict(image=obs))
        
        # 2. Observe
        state = self.world_model.rssm.initial(obs.shape[0], self.device)
        post = self.world_model.rssm.observe(embed, actions, state)
        
        feat = torch.cat([post['deter'], post['stoch'].flatten(2)], dim=-1)
        
        # 3. WM Loss (Paper eq. 2-3)
        recon = self.world_model.decoder(feat)
        # Decoder: V3 uses sigmoid [0,1] output, MSE loss
        target_img = obs.float() / 255.0
        loss_img = F.mse_loss(recon['image'], target_img)
        
        # Reward: Symexp twohot loss (paper eq. 10-11)
        reward_logits = self.world_model.reward_model(feat)
        loss_rew = twohot_loss(reward_logits, rewards.squeeze(-1).float(), self.world_model.reward_model.bins)
        
        # Continue: Binary classification via logistic regression
        cont_logits = self.world_model.continue_model(feat).squeeze(-1)
        loss_cont = F.binary_cross_entropy_with_logits(cont_logits, (1.0 - dones.squeeze(-1).float()))
        
        # KL
        prior_logits = post['prior_logit']
        post_logits = post['logit']
        
        def kl_cat_sum(p_logits, q_logits):
            p = F.softmax(p_logits, dim=-1)
            log_p = F.log_softmax(p_logits, dim=-1)
            log_q = F.log_softmax(q_logits, dim=-1)
            return torch.sum(p * (log_p - log_q), dim=-1)
            
        kl_value = kl_cat_sum(post_logits.detach(), prior_logits)
        loss_dyn = torch.mean(torch.maximum(kl_value, torch.tensor(1.0, device=self.device)))
        
        kl_value_rep = kl_cat_sum(post_logits, prior_logits.detach())
        loss_rep = torch.mean(torch.maximum(kl_value_rep, torch.tensor(1.0, device=self.device)))
        
        # Paper eq. 2: L = βpred*Lpred + βdyn*Ldyn + βrep*Lrep
        # βpred=1, βdyn=1, βrep=0.1
        # Lpred = loss_img + loss_rew + loss_cont
        loss_wm = loss_img + loss_rew + loss_cont + loss_dyn + 0.1 * loss_rep
        
        # Check for NaN in world model losses before proceeding to actor
        if torch.isnan(loss_wm) or torch.isinf(loss_wm):
            print(f"WARNING: NaN/Inf in world model loss. Components:")
            print(f"  loss_img={loss_img.item()}, loss_rew={loss_rew.item()}, loss_cont={loss_cont.item()}")
            print(f"  loss_dyn={loss_dyn.item()}, loss_rep={loss_rep.item()}")
            return {k: 0.0 for k in ['wm_loss', 'actor_loss', 'critic_loss', 'recon_loss', 'reward_loss', 'cont_loss', 'kl_dyn_loss', 'kl_rep_loss']}
        
        # 4. Actor Critic
        if not self.is_discrete:
            # Continuous Control (Dynamics Backprop)
            B, T, _ = feat.shape
            flat_deter = post['deter'].detach().reshape(B*T, -1)
            flat_stoch = post['stoch'].detach().reshape(B*T, self.world_model.rssm.stoch, self.world_model.rssm.classes)
            start_state = {'deter': flat_deter, 'stoch': flat_stoch}
            
            def imag_policy(feat):
                dist = self.actor.get_distribution(feat)
                return dist.rsample()
            
            imag_outs = self.world_model.rssm.imagine(imag_policy, start_state, horizon=16)
            imag_feat = torch.cat([imag_outs['deter'], imag_outs['stoch'].flatten(2)], dim=-1)
            
            # Predict rewards and values using twohot models
            with torch.no_grad():
                # Reward prediction (expected value from categorical distribution)
                imag_rew = self.world_model.reward_model.predict(imag_feat)
                
                # Continue prediction
                imag_cont_logits = self.world_model.continue_model(imag_feat).squeeze(-1)
                imag_cont = torch.sigmoid(imag_cont_logits)
                
                # Value prediction from slow critic (expected value from categorical)
                imag_val = self.slow_critic.predict(imag_feat)
            
            # Lambda return parameters from config (or paper defaults)
            lambda_ = self.cfg.get('critic', {}).get('lambda', 0.95)
            discount = self.cfg.get('critic', {}).get('gamma', 0.997)
            
            # Lambda-return calculation
            returns_list = []
            next_val = imag_val[:, -1]
            
            for t in reversed(range(imag_rew.shape[1])):
                disc = discount * imag_cont[:, t]
                ret_t = imag_rew[:, t] + disc * (1 - lambda_) * imag_val[:, t] + disc * lambda_ * next_val
                returns_list.append(ret_t)
                next_val = ret_t
            
            returns_list.reverse()
            returns = torch.stack(returns_list, dim=1)
            
            # Actor loss with return normalization (paper eq. 6-7)
            dist = self.actor.get_distribution(imag_feat)
            log_prob = dist.log_prob(imag_outs['action'])
            entropy = dist.entropy() if hasattr(dist, 'entropy') else torch.zeros_like(log_prob)
            
            # EMA return normalization scale (paper eq. 7)
            return_flat = returns.reshape(-1)
            with torch.no_grad():
                p95 = torch.quantile(return_flat, 0.95)
                p05 = torch.quantile(return_flat, 0.05)
                batch_scale = torch.maximum(p95 - p05, torch.tensor(1.0, device=self.device))
                # Update EMA: S = EMA[Per95 - Per05, 0.99]
                self.return_scale.mul_(0.99).add_(batch_scale * 0.01)
            
            # Normalize returns with max(1, S) to avoid amplifying noise under sparse rewards
            scale = torch.maximum(torch.tensor(1.0, device=self.device), self.return_scale)
            norm_returns = returns / scale
            
            # Paper eq. 6: entropy scale η = 3e-4
            loss_actor = -torch.mean(norm_returns * log_prob + 3e-4 * entropy)
            
            # Critic loss: twohot categorical cross-entropy (paper page 5-6)
            critic_logits = self.critic(imag_feat)
            loss_critic = twohot_loss(critic_logits, returns.detach(), self.critic.bins)

        else:
            # Discrete / Binning (REINFORCE / Straight-Through)
            with torch.no_grad():
                B, T, _ = feat.shape
                flat_deter = post['deter'].reshape(B*T, -1)
                flat_stoch = post['stoch'].reshape(B*T, self.world_model.rssm.stoch, self.world_model.rssm.classes)
                start_state = {'deter': flat_deter, 'stoch': flat_stoch}

            def imag_policy(feat):
                dist = self.actor.get_distribution(feat)
                sample = dist.sample()
                probs = dist.probs
                # Straight-through estimator
                action = (sample - probs).detach() + probs
                return action

            imag_outs = self.world_model.rssm.imagine(imag_policy, start_state, horizon=16)
            imag_feat = torch.cat([imag_outs['deter'], imag_outs['stoch'].flatten(2)], dim=-1)
            
            with torch.no_grad():
                # Predict rewards and values using twohot models
                imag_rew = self.world_model.reward_model.predict(imag_feat)
                imag_cont_logits = self.world_model.continue_model(imag_feat).squeeze(-1)
                imag_cont = torch.sigmoid(imag_cont_logits)
                imag_val = self.slow_critic.predict(imag_feat)
                 
                # Lambda return parameters from config (or paper defaults)
                lambda_ = self.cfg.get('critic', {}).get('lambda', 0.95)
                discount = self.cfg.get('critic', {}).get('gamma', 0.997)
                 
                returns = torch.zeros_like(imag_rew)
                next_val = imag_val[:, -1]
                 
                for t in reversed(range(imag_rew.shape[1])):
                    disc = discount * imag_cont[:, t]
                    returns[:, t] = imag_rew[:, t] + disc * (1 - lambda_) * imag_val[:, t] + disc * lambda_ * next_val
                    next_val = returns[:, t]
                    
            # Actor loss with return normalization
            dist = self.actor.get_distribution(imag_feat.detach())
            entropy = dist.entropy()
            log_prob = dist.log_prob(imag_outs['action'])
            
            # EMA return normalization scale (paper eq. 7)
            return_flat = returns.reshape(-1)
            with torch.no_grad():
                p95 = torch.quantile(return_flat, 0.95)
                p05 = torch.quantile(return_flat, 0.05)
                batch_scale = torch.maximum(p95 - p05, torch.tensor(1.0, device=self.device))
                # Update EMA
                self.return_scale.mul_(0.99).add_(batch_scale * 0.01)
            
            scale = torch.maximum(torch.tensor(1.0, device=self.device), self.return_scale)
            norm_returns = returns / scale
            
            # Actor loss: maximize return + entropy bonus (η = 3e-4)
            loss_actor = -torch.mean(norm_returns * log_prob + 3e-4 * entropy)
            
            # Critic loss: twohot categorical cross-entropy
            critic_logits = self.critic(imag_feat.detach())
            loss_critic = twohot_loss(critic_logits, returns, self.critic.bins)
        
        loss_total = loss_wm + loss_actor + loss_critic
        
        # Check for NaN before backprop
        if torch.isnan(loss_total) or torch.isinf(loss_total):
            print(f"WARNING: NaN/Inf detected in loss. Skipping update.")
            print(f"  loss_wm={loss_wm.item()}, loss_actor={loss_actor.item()}, loss_critic={loss_critic.item()}")
            return {
                'wm_loss': 0.0,
                'actor_loss': 0.0,
                'critic_loss': 0.0,
                'recon_loss': 0.0,
                'reward_loss': 0.0,
                'cont_loss': 0.0,
                'kl_dyn_loss': 0.0,
                'kl_rep_loss': 0.0
            }
        
        self.opt.zero_grad()
        loss_total.backward()
        # Adaptive Gradient Clipping (AGC) as in paper
        agc_stats = adaptive_gradient_clip(self.parameters(), clip=0.3, pmin=1e-3)
        self.opt.step()
        
        with torch.no_grad():
            for p, sp in zip(self.critic.parameters(), self.slow_critic.parameters()):
                sp.data = self.ema_decay * sp.data + (1 - self.ema_decay) * p.data
        
        return {
            'wm_loss': loss_wm.item(),
            'actor_loss': loss_actor.item(),
            'critic_loss': loss_critic.item(),
            'recon_loss': loss_img.item(),
            'reward_loss': loss_rew.item(),
            'cont_loss': loss_cont.item(),
            'kl_dyn_loss': loss_dyn.item(),
            'kl_rep_loss': loss_rep.item(),
            'grad_norm': agc_stats['avg_norm'],
            'grad_max': agc_stats['max_norm'],
            'grad_clipped': agc_stats['num_clipped']
        }
        
    def save(self, path, logs):
        torch.save({
            'agent_state_dict': self.state_dict(),
            'opt_state_dict': self.opt.state_dict(),
            'logs': logs
        }, path)
        
    def load(self, path):
        if not torch.cuda.is_available():
            checkpoint = torch.load(path, map_location='cpu')
        else:
            checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['agent_state_dict'])
        self.opt.load_state_dict(checkpoint['opt_state_dict'])
        return checkpoint['logs']
