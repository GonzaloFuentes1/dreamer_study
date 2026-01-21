import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
import torch.optim as optim
import numpy as np
import copy
from . import models

def symlog(x):
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)

def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)

def adaptive_gradient_clip(parameters, clip=0.3, pmin=1e-3):
    """
    Adaptive Gradient Clipping (AGC) as used in DreamerV3.
    Clips gradients based on parameter norms rather than global gradient norm.
    
    For each parameter:
        clip_value = clip * max(pmin, ||param||)
        if ||grad|| > clip_value:
            grad = grad * (clip_value / ||grad||)
    
    Args:
        parameters: Model parameters with gradients
        clip: Clipping factor (default 0.3 as in paper)
        pmin: Minimum parameter norm (default 1e-3)
    
    Returns:
        dict with statistics: avg_norm, max_norm, num_clipped
    """
    if clip <= 0:
        return {'avg_norm': 0.0, 'max_norm': 0.0, 'num_clipped': 0}
    
    grad_norms = []
    num_clipped = 0
    
    for param in parameters:
        if param.grad is None:
            continue
        
        # Flatten for norm computation
        grad_flat = param.grad.detach().flatten()
        param_flat = param.detach().flatten()
        
        # Compute norms
        grad_norm = torch.linalg.norm(grad_flat, ord=2)
        param_norm = torch.linalg.norm(param_flat, ord=2)
        
        grad_norms.append(grad_norm.item())
        
        # Compute adaptive clip threshold
        max_norm = clip * torch.maximum(param_norm, torch.tensor(pmin, device=param.device))
        
        # Clip if necessary
        if grad_norm > max_norm:
            param.grad.mul_(max_norm / grad_norm)
            num_clipped += 1
    
    return {
        'avg_norm': sum(grad_norms) / max(len(grad_norms), 1),
        'max_norm': max(grad_norms) if grad_norms else 0.0,
        'num_clipped': num_clipped
    }

class DreamerV3Agent(nn.Module):
    def __init__(self, config, obs_shape, action_dim, is_discrete, device):
        super().__init__()
        self.cfg = config
        self.device = device
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.is_discrete = is_discrete
        self.continuous_actions = config.get('continuous_actions', False)
        self.num_bins = 255

        # Prepare configs
        model_cfg = config['model']
        rssm_cfg = model_cfg['rssm'].copy()
        if 'embed_dim' in model_cfg:
            rssm_cfg['embed_dim'] = model_cfg['embed_dim']
            
        enc_cfg = model_cfg.get('encoder', model_cfg.copy())
        if 'cnn_depth' in model_cfg:
            enc_cfg['depth'] = model_cfg['cnn_depth']
            
        dec_cfg = model_cfg.get('decoder', model_cfg.copy())
        if 'cnn_depth' in model_cfg:
            dec_cfg['depth'] = model_cfg['cnn_depth']
        
        # Effective action dim for the model
        if self.continuous_actions:
             model_action_dim = self.action_dim
        elif self.is_discrete:
             model_action_dim = self.action_dim
        else:
             model_action_dim = self.action_dim * self.num_bins
        
        # Initialize RSSM
        self.rssm = models.RSSM(model_action_dim, config=rssm_cfg).to(self.device)
        
        # Initialize Encoder
        self.encoder = models.Encoder(obs_shape, config=enc_cfg).to(self.device)
        
        # Initialize Decoder
        feat_dim = self.rssm.deter + self.rssm.stoch * self.rssm.classes
        self.decoder = models.Decoder(feat_dim, shape=obs_shape, config=dec_cfg).to(self.device)
        
        # Heads
        self.reward_head = models.MLP(feat_dim, 1, self.rssm.hidden, layers=2).to(self.device)
        self.cont_head = models.MLP(feat_dim, 1, self.rssm.hidden, layers=2).to(self.device)
        
        # Actor Critic
        if self.continuous_actions:
             actor_output_dim = self.action_dim * 2
        elif self.is_discrete:
             actor_output_dim = self.action_dim
        else:
             actor_output_dim = self.action_dim * self.num_bins

        self.actor = models.MLP(feat_dim, actor_output_dim, self.rssm.hidden, layers=2).to(self.device)
        self.critic = models.MLP(feat_dim, 1, self.rssm.hidden, layers=2).to(self.device)
        self.slow_critic = copy.deepcopy(self.critic).to(self.device)
        
        # Optimizer settings: use config lr if provided, else paper default 3e-4
        # Note: walker_walk.yaml specifies 4e-5 which is also valid
        lr = config.get('model', {}).get('lr', config['training'].get('lr', 3e-4))
        self.opt = optim.Adam(self.parameters(), lr=lr, eps=1e-8)
        
        # EMA decay for slow critic
        self.ema_decay = config.get('critic', {}).get('ema_decay', 0.98)

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
            stoch_flat, deter = self.rssm.initial(B, obs.device)
        else:
            stoch_flat, deter = state
            # ALWAYS force to match current batch size B (no conditionals)
            # This handles any size mismatch from initialization or previous steps
            stoch_flat = stoch_flat[:B].contiguous()
            deter = deter[:B].contiguous()
        
        # Ensure last_action also matches B
        last_action = last_action[:B].contiguous()
        
        # Encoder expect (B, T, ...) so unsqueeze time
        embed = self.encoder(dict(image=obs.unsqueeze(1))) # (B, 1, E)
        
        # RSSM Step
        # 1. Deterministic step
        inp = torch.cat([stoch_flat, last_action], dim=-1)
        x = self.rssm.img_in(inp)
        deter = self.rssm.cell(x, deter)
        
        # 2. Posterior (Observe)
        obs_inp = torch.cat([deter, embed.squeeze(1)], dim=-1)
        post_logit = self.rssm.obs_out(obs_inp)  # Keep flat for get_dist
        
        # Sample
        if mode == 'train':
            dist = self.rssm.get_dist(post_logit)
            stoch = dist.sample() + dist.probs - dist.probs.detach()
        else:
            # For eval, reshape manually then argmax
            post_logit_shaped = post_logit.view(B, self.rssm.stoch, self.rssm.classes)
            stoch = F.one_hot(torch.argmax(post_logit_shaped, dim=-1), self.rssm.classes).float()
            
        stoch_flat = stoch.view(B, -1)
        
        # Feature for Actor
        feat = torch.cat([deter, stoch_flat], dim=-1)
        
        # Actor
        logits = self.actor(feat)
        
        if self.continuous_actions:
            mean, std = torch.chunk(logits, 2, dim=-1)
            mean = torch.tanh(mean)
            std = F.softplus(std) + 0.1
            base_dist = td.Normal(mean, std)
            dist = td.Independent(td.TransformedDistribution(base_dist, td.TanhTransform(cache_size=1)), 1)
            if mode == 'train':
                action = dist.rsample()
            else:
                action = torch.tanh(mean)  # No exploration in eval
            env_action = action.detach().cpu().numpy()
            
        else:
            # Discrete actions - add exploration noise in training
            if mode == 'train':
                # Add small uniform noise for exploration (as in paper)
                logits_with_noise = logits
                dist = torch.distributions.OneHotCategorical(logits=logits_with_noise)
                action = dist.sample()
            else:
                action_idx = torch.argmax(logits, dim=-1)
                action = F.one_hot(action_idx, logits.shape[-1]).float()
            
            env_action = action.detach().cpu().numpy()
            if self.is_discrete:
                env_action = np.argmax(env_action, axis=-1)
            
        next_state = (stoch_flat, deter)
        
        # Returns: act_data (for buffer), next_state, env_action (for env)
        return action.detach().cpu().numpy(), next_state, env_action

    def train_step(self, obs, actions, rewards, dones):
        # obs: (B, T, C, H, W)
        # actions: (B, T, A)
        
        # 1. Encode
        embed = self.encoder(dict(image=obs))
        
        # 2. Observe
        state = self.rssm.initial(obs.shape[0], self.device)
        post = self.rssm.observe(embed, actions, state)
        
        feat = torch.cat([post['deter'], post['stoch'].flatten(2)], dim=-1)
        
        # 3. WM Loss
        recon = self.decoder(feat)
        target_img = obs.float() / 255.0 - 0.5
        
        loss_img = F.mse_loss(recon['image'], target_img)
        
        pred_rew = self.reward_head(feat).squeeze(-1)
        loss_rew = F.mse_loss(pred_rew, symlog(rewards.squeeze(-1).float()))
        
        pred_cont = self.cont_head(feat).squeeze(-1)
        loss_cont = F.binary_cross_entropy_with_logits(pred_cont, (1.0 - dones.squeeze(-1).float()))
        
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
        
        loss_wm = loss_img + loss_rew + loss_cont + 0.5 * loss_dyn + 0.1 * loss_rep
        
        # Check for NaN in world model losses before proceeding to actor
        if torch.isnan(loss_wm) or torch.isinf(loss_wm):
            print(f"WARNING: NaN/Inf in world model loss. Components:")
            print(f"  loss_img={loss_img.item()}, loss_rew={loss_rew.item()}, loss_cont={loss_cont.item()}")
            print(f"  loss_dyn={loss_dyn.item()}, loss_rep={loss_rep.item()}")
            return {k: 0.0 for k in ['wm_loss', 'actor_loss', 'critic_loss', 'recon_loss', 'reward_loss', 'cont_loss', 'kl_dyn_loss', 'kl_rep_loss']}
        
        # 4. Actor Critic
        if self.continuous_actions:
            # Continuous Control (Dynamics Backprop)
            B, T, _ = feat.shape
            flat_deter = post['deter'].detach().reshape(B*T, -1)
            flat_stoch = post['stoch'].detach().reshape(B*T, self.rssm.stoch, self.rssm.classes)
            start_state = {'deter': flat_deter, 'stoch': flat_stoch}

            def imag_policy_cont(feat):
                logits = self.actor(feat)
                mean, std = torch.chunk(logits, 2, dim=-1)
                mean = torch.tanh(mean)
                std = F.softplus(std) + 0.1
                base_dist = td.Normal(mean, std)
                dist = td.Independent(td.TransformedDistribution(base_dist, td.TanhTransform(cache_size=1)), 1)
                return dist.rsample()

            imag_outs = self.rssm.imagine(imag_policy_cont, start_state, horizon=15)
            imag_feat = torch.cat([imag_outs['deter'], imag_outs['stoch'].flatten(2)], dim=-1)
            
            # Predict in Symlog space
            imag_rew_sym = self.reward_head(imag_feat).squeeze(-1)
            imag_val_sym = self.critic(imag_feat).squeeze(-1) # Online critic
            
            # Denormalize for Return Calculation (Use slow critic for bootstrapping if desired, or online)
            # Typically Dreamer uses slow_critic for the target calculation
            with torch.no_grad():
                 imag_val_slow_sym = self.slow_critic(imag_feat).squeeze(-1)
                 
                 imag_rew = symexp(imag_rew_sym)
                 imag_val = symexp(imag_val_slow_sym)
            
            imag_cont_logits = self.cont_head(imag_feat).squeeze(-1)
            imag_cont = torch.sigmoid(imag_cont_logits)
            
            # Lambda return parameters from config (or paper defaults)
            lambda_ = self.cfg.get('critic', {}).get('lambda', 0.95)
            discount = self.cfg.get('critic', {}).get('gamma', 0.997)
            
            # Lambda-return calculation (in Domain Space)
            returns_list = []
            next_val = imag_val[:, -1]
            
            for t in reversed(range(imag_rew.shape[1])):
                disc = discount * imag_cont[:, t]
                ret_t = imag_rew[:, t] + disc * (1 - lambda_) * imag_val[:, t] + disc * lambda_ * next_val
                returns_list.append(ret_t)
                next_val = ret_t
            
            returns_list.reverse()
            returns = torch.stack(returns_list, dim=1)
            
            # Re-normalize returns for Critic Loss (Target)
            returns_sym = symlog(returns)
            
            # Entropy
            actor_logits = self.actor(imag_feat)
            mean, std = torch.chunk(actor_logits, 2, dim=-1)
            mean = torch.tanh(mean)
            std = F.softplus(std) + 0.1
            base_dist = td.Normal(mean, std)
            dist = td.Independent(td.TransformedDistribution(base_dist, td.TanhTransform(cache_size=1)), 1)
            log_prob = dist.log_prob(imag_outs['action'])
            
            # Loss
            # DreamerV3: Return Normalization (Shift/Scale)
            # Use percentiles stats from the current batch for stability
            # Scale = P95 - P05
            return_flat = returns.reshape(-1)
            # Using torch.quantile requires a moderately large batch. With 16x64=1024 we are safe.
            p95 = torch.quantile(return_flat, 0.95)
            p05 = torch.quantile(return_flat, 0.05)
            scale = torch.maximum(p95 - p05, torch.tensor(1.0, device=self.device))
            offset = p05
            
            # Normalize returns for Actor (keeping linearity/unbiased)
            norm_returns = (returns - offset) / scale
            
            loss_actor = -torch.mean(norm_returns + 3e-4 * log_prob) 
            
            loss_critic = F.mse_loss(imag_val_sym, returns_sym.detach())

        else:
            # Discrete / Binning (REINFORCE / Straight-Through)
            with torch.no_grad():
                B, T, _ = feat.shape
                flat_deter = post['deter'].reshape(B*T, -1)
                flat_stoch = post['stoch'].reshape(B*T, self.rssm.stoch, self.rssm.classes)
                start_state = {'deter': flat_deter, 'stoch': flat_stoch}

            def imag_policy(feat):
                logits = self.actor(feat)
                dist = torch.distributions.OneHotCategorical(logits=logits)
                sample = dist.sample()
                probs = dist.probs
                # Straight-through estimator
                action = (sample - probs).detach() + probs
                return action

            imag_outs = self.rssm.imagine(imag_policy, start_state, horizon=15)
            imag_feat = torch.cat([imag_outs['deter'], imag_outs['stoch'].flatten(2)], dim=-1)
            
            with torch.no_grad():
                 imag_rew = self.reward_head(imag_feat).squeeze(-1)
                 imag_cont_logits = self.cont_head(imag_feat).squeeze(-1)
                 imag_cont = torch.sigmoid(imag_cont_logits)
                 imag_val = self.slow_critic(imag_feat).squeeze(-1)
                 
                 # Lambda return parameters from config (or paper defaults)
                 lambda_ = self.cfg.get('critic', {}).get('lambda', 0.95)
                 discount = self.cfg.get('critic', {}).get('gamma', 0.997)
                 
                 returns = torch.zeros_like(imag_rew)
                 next_val = imag_val[:, -1]
                 
                 for t in reversed(range(imag_rew.shape[1])):
                     # DreamerV2/V3 style return calculation
                     # Based on Hafner's code: ret = rew + disc * (1-lambda) * val + disc * lambda * next_val
                     # disc is gamma * pcont
                     disc = discount * imag_cont[:, t]
                     returns[:, t] = imag_rew[:, t] + disc * (1 - lambda_) * imag_val[:, t] + disc * lambda_ * next_val
                     next_val = returns[:, t]
                     
            actor_logits = self.actor(imag_feat.detach())
            dist = torch.distributions.OneHotCategorical(logits=actor_logits)
            
            # Entropy regularization (critical for preventing collapse)
            entropy = dist.entropy()
            log_prob = dist.log_prob(imag_outs['action'])
            
            # Actor loss: maximize return + entropy bonus (from config or paper default 3e-4)
            entropy_coef = self.cfg.get('actor', {}).get('entropy_coef', 3e-4)
            loss_actor = -torch.mean(returns * log_prob + entropy_coef * entropy)
            
            pred_val = self.critic(imag_feat.detach()).squeeze(-1)
            loss_critic = F.mse_loss(pred_val, returns)
        
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
