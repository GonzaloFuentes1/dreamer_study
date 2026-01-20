import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal, kl_divergence
from common.agent import Agent
from .models import RSSM_V2, ConvEncoder, ConvDecoder, MLP


class DreamerV2Agent(Agent):
    def __init__(self, config, obs_shape, action_dim, is_discrete, device):
        super().__init__()
        self.cfg = config
        self.device = device
        self.action_dim = action_dim
        self.is_discrete = is_discrete
        
        self.encoder = ConvEncoder(input_channels=obs_shape[0]).to(device)
        embed_dim = 1024
        
        self.rssm = RSSM_V2(
            action_dim=action_dim,
            stoch_dim=config['model']['rssm']['stoch_dim'],
            stoch_classes=config['model']['rssm']['stoch_classes'],
            deter_dim=config['model']['rssm']['deter_dim'],
            hidden_dim=config['model']['rssm']['hidden_dim'],
            embed_dim=embed_dim
        ).to(device)
        
        feature_dim = config['model']['rssm']['deter_dim'] + (
            config['model']['rssm']['stoch_dim'] * config['model']['rssm']['stoch_classes']
        )
        
        self.decoder = ConvDecoder(input_dim=feature_dim, output_channels=obs_shape[0]).to(device)
        self.reward_model = MLP(feature_dim, 1, hidden=400).to(device)
        self.discount_model = MLP(feature_dim, 1, hidden=400).to(device)
        
        # Actor outputs mean and log_std for continuous, or logits for discrete
        if is_discrete:
            self.actor = MLP(feature_dim, action_dim, hidden=400).to(device)
        else:
            self.actor_mean = MLP(feature_dim, action_dim, hidden=400).to(device)
            self.actor_log_std = nn.Parameter(torch.zeros(action_dim).to(device))
        
        self.critic = MLP(feature_dim, 1, hidden=400).to(device)
        self.target_critic = MLP(feature_dim, 1, hidden=400).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self._updates = 0
        
        model_lr = config['model'].get('lr', 3e-4)
        actor_lr = config['actor'].get('lr', 8e-5)
        value_lr = config.get('critic', {}).get('lr', 8e-5)
        
        self.wm_optimizer = optim.Adam(
            list(self.encoder.parameters()) + 
            list(self.rssm.parameters()) +
            list(self.decoder.parameters()) +
            list(self.reward_model.parameters()) +
            list(self.discount_model.parameters()),
            lr=model_lr
        )
        
        if is_discrete:
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        else:
            self.actor_optimizer = optim.Adam(
                list(self.actor_mean.parameters()) + [self.actor_log_std], 
                lr=actor_lr
            )
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=value_lr)
        
        self.scaler = torch.amp.GradScaler('cuda')

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
                obs_tensor = obs_tensor.float() / 255.0
            
            if len(obs_tensor.shape) == 3:
                obs_tensor = obs_tensor.unsqueeze(0)
            elif len(obs_tensor.shape) == 1:
                obs_tensor = obs_tensor.float().unsqueeze(0)
            
            embed = self.encoder(obs_tensor)
            stoch_flat, deter = state
            
            deter = self.rssm.step(stoch_flat, last_action, deter)
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
            obs = obs.float() / 255.0
            obs_target = obs - 0.5
        elif obs.max() > 1.0:
            obs = obs / 255.0
            obs_target = obs - 0.5
        else:
            obs_target = obs
        
        with torch.amp.autocast('cuda'):
            B, T, C, H, W = obs.shape
            obs_flat = obs.view(B*T, C, H, W)
            embed = self.encoder(obs_flat)
            embed = embed.view(B, T, -1)
            
            prior_logits, post_logits, stoch_flat, deters = self.rssm.observe(embed, action)
            
            features = torch.cat([deters, stoch_flat], dim=-1)
            
            recon = self.decoder(features.view(B*T, -1))
            recon = recon.view(B, T, C, H, W)
            loss_recon = 0.5 * nn.functional.mse_loss(recon, obs_target, reduction='none').sum(dim=[-3,-2,-1]).mean()
            
            pred_reward = self.reward_model(features).squeeze(-1)
            reward_target = reward if reward.dim() == pred_reward.dim() else reward.squeeze(-1)
            loss_reward = 0.5 * nn.functional.mse_loss(pred_reward, reward_target).mean()
            
            pred_discount_logits = self.discount_model(features)
            target_discount = (1.0 - terminal.float())
            if target_discount.dim() < pred_discount_logits.dim():
                target_discount = target_discount.unsqueeze(-1)
            loss_discount = nn.functional.binary_cross_entropy_with_logits(pred_discount_logits, target_discount)
            
            loss_kl = self.rssm.kl_loss(
                post_logits, prior_logits, 
                alpha=self.cfg['model'].get('kl_alpha', 0.8),
                free_nats=self.cfg['model'].get('kl_free_nats', 1.0)
            )
            
            kl_scale = self.cfg['model'].get('kl_scale', 1.0)
            model_loss = loss_recon + loss_reward + loss_discount + kl_scale * loss_kl
        
        self.wm_optimizer.zero_grad()
        self.scaler.scale(model_loss).backward()
        self.scaler.unscale_(self.wm_optimizer)
        nn.utils.clip_grad_norm_(list(self.encoder.parameters()) + list(self.rssm.parameters()), 100.0)
        self.scaler.step(self.wm_optimizer)
        
        with torch.amp.autocast('cuda'):
            start_stoch = stoch_flat.detach().view(-1, stoch_flat.shape[-1])
            start_deter = deters.detach().view(-1, deters.shape[-1])
            
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
            img_rewards = self.reward_model(img_features).squeeze(-1)
            img_values = self.target_critic(img_features).squeeze(-1)
            img_discounts = torch.sigmoid(self.discount_model(img_features).squeeze(-1))
            
            returns = self.compute_lambda_returns(
                img_rewards, img_values, img_values[:, -1],
                lambda_=self.cfg['critic'].get('lambda', 0.95),
                gamma=img_discounts
            )
            
            rho = self.cfg['actor'].get('rho', 0.0)
            eta = self.cfg['actor'].get('entropy_coef', 1e-4)
            
            advantage = returns[:, :-1] - img_values[:, :-1]
            
            if rho > 0:
                if self.is_discrete:
                    logits = self.actor(img_features[:, :-1])
                    action_dist = torch.distributions.Categorical(logits=logits)
                    img_actions_slice = img_actions[:, :-1]
                    if img_actions_slice.dim() == 3 and img_actions_slice.shape[-1] > 1:
                        action_idx = img_actions_slice.argmax(dim=-1)
                    else:
                        action_idx = img_actions_slice.squeeze(-1).long()
                    log_probs = action_dist.log_prob(action_idx)
                    entropy = action_dist.entropy().mean()
                else:
                    mean = self.actor_mean(img_features[:, :-1])
                    std = torch.exp(self.actor_log_std).expand_as(mean)
                    action_dist = torch.distributions.Normal(mean, std)
                    entropy = action_dist.entropy().sum(dim=-1).mean()
                    log_probs = torch.zeros_like(advantage)
                
                advantage_clipped = torch.clamp(advantage, -10, 10)
                loss_reinforce = -(rho * log_probs * advantage_clipped.detach()).mean()
            else:
                loss_reinforce = 0.0
                if self.is_discrete:
                    logits = self.actor(img_features[:, :-1])
                    action_dist = torch.distributions.Categorical(logits=logits)
                    entropy = action_dist.entropy().mean()
                else:
                    mean = self.actor_mean(img_features[:, :-1])
                    std = torch.exp(self.actor_log_std).expand_as(mean)
                    action_dist = torch.distributions.Normal(mean, std)
                    entropy = action_dist.entropy().sum(dim=-1).mean()
            
            loss_dynamics = -(1 - rho) * returns[:, :-1].mean()
            
            loss_entropy = -eta * entropy
            
            loss_actor = loss_reinforce + loss_dynamics + loss_entropy
        
        self.actor_optimizer.zero_grad()
        self.scaler.scale(loss_actor).backward()
        self.scaler.unscale_(self.actor_optimizer)
        if self.is_discrete:
            nn.utils.clip_grad_norm_(self.actor.parameters(), 100.0)
        else:
            nn.utils.clip_grad_norm_(
                list(self.actor_mean.parameters()) + [self.actor_log_std], 100.0
            )
        self.scaler.step(self.actor_optimizer)
        
        with torch.amp.autocast('cuda'):
            value_pred = self.critic(img_features[:, :-1].detach()).squeeze(-1)
            loss_critic = 0.5 * nn.functional.mse_loss(value_pred, returns[:, :-1].detach())
        
        self.critic_optimizer.zero_grad()
        self.scaler.scale(loss_critic).backward()
        self.scaler.unscale_(self.critic_optimizer)
        nn.utils.clip_grad_norm_(self.critic.parameters(), 100.0)
        self.scaler.step(self.critic_optimizer)
        
        self.scaler.update()
        self._update_target_network()
        
        return {
            "wm_loss": model_loss.detach(),
            "kl": loss_kl.detach(),
            "actor_loss": loss_actor.detach(),
            "value_loss": loss_critic.detach()
        }

    def compute_lambda_returns(self, reward, value, bootstrap, lambda_, gamma):
        v_lambda = torch.zeros_like(reward).to(self.device)
        last_v = bootstrap
        for t in reversed(range(reward.shape[1])):
            if isinstance(gamma, torch.Tensor):
                disc = gamma[:, t]
            else:
                disc = gamma
            v_lambda[:, t] = reward[:, t] + disc * ((1 - lambda_) * value[:, t] + lambda_ * last_v)
            last_v = v_lambda[:, t]
        return v_lambda
    
    def _update_target_network(self):
        """Update target critic every 100 gradient steps"""
        self._updates += 1
        if self._updates % 100 == 0:
            self.target_critic.load_state_dict(self.critic.state_dict())

    def save(self, path, logs=None):
        data = {
            'encoder': self.encoder.state_dict(),
            'rssm': self.rssm.state_dict(),
            'decoder': self.decoder.state_dict(),
            'reward': self.reward_model.state_dict(),
            'discount': self.discount_model.state_dict(),
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
        self.rssm.load_state_dict(checkpoint['rssm'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        self.reward_model.load_state_dict(checkpoint['reward'])
        self.discount_model.load_state_dict(checkpoint['discount'])
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

