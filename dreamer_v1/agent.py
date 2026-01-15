import torch
import torch.nn as nn
import numpy as np
from torch.distributions import kl_divergence
from common.agent import Agent
from .models import WorldModel, ActionModel

class ValueModel(nn.Module):
    def __init__(self, stoch_dim, deter_dim, hidden_dim=400):
        super().__init__()
        # Paper code: 3 hidden layers of size 400, then output
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
        x = torch.cat([stoch, deter], dim=-1)
        return self.net(x)

class DreamerV1Agent(Agent):
    def __init__(self, config, obs_shape, action_dim, is_discrete, device):
        self.cfg = config
        self.device = device
        self.action_dim = action_dim
        self.is_discrete = is_discrete
        
        # 1. World Model (Theta)
        self.world_model = WorldModel(obs_shape, action_dim, config).to(device)
        
        # Compile encoder/decoder for faster inference (PyTorch 2.0+)
        try:
            self.world_model.encoder = torch.compile(self.world_model.encoder, mode='reduce-overhead')
            self.world_model.decoder = torch.compile(self.world_model.decoder, mode='reduce-overhead')
        except:
            pass  # PyTorch < 2.0
        
        hidden_dim = config['model'].get('num_units', 400)
        
        # 2. Actor (Phi)
        self.actor = ActionModel(
            config['model']['rssm']['stoch_dim'], 
            config['model']['rssm']['deter_dim'], 
            action_dim,
            hidden_dim=hidden_dim,
            discrete=is_discrete
        ).to(device)
        
        # 3. Critic (Psi)
        self.value = ValueModel(
            config['model']['rssm']['stoch_dim'], 
            config['model']['rssm']['deter_dim'],
            hidden_dim=hidden_dim
        ).to(device)

        # Optimizers
        # Check configs for LR, default to paper values if missing
        model_lr = config['model'].get('lr', 6e-4)
        actor_lr = config['actor'].get('lr', 8e-5)
        
        # Critic LR sometimes under 'critic', sometimes 'value_lr'
        if 'critic' in config and 'lr' in config['critic']:
            value_lr = config['critic']['lr'] 
        else:
            value_lr = 8e-5

        self.wm_optimizer = torch.optim.Adam(self.world_model.parameters(), lr=model_lr)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=value_lr)
        
        # Mixed Precision Scaler
        self.scaler = torch.amp.GradScaler('cuda')

    def init_state(self, batch_size):
        return (
            torch.zeros(batch_size, self.world_model.rssm.stoch_dim).to(self.device),
            torch.zeros(batch_size, self.world_model.rssm.deter_dim).to(self.device)
        )

    def policy(self, obs, state, last_action, mode='train'):
        with torch.no_grad():
            # Handle obs normalization for policy inference
            if isinstance(obs, torch.Tensor):
                obs_tensor = obs.to(self.device)
            else:
                # Copy to avoid negative stride issues
                obs_tensor = torch.from_numpy(np.ascontiguousarray(obs)).to(self.device)
            
            if obs_tensor.dtype == torch.uint8:
                pass # Encoder deals with it
            
            # Since policy sends single observation, unsqueeze dim 0
            # Obs shape (3, 64, 64) -> (1, 3, 64, 64)
            if len(obs_tensor.shape) == 3:
                obs_tensor = obs_tensor.unsqueeze(0)
            elif len(obs_tensor.shape) == 1: # Vector env
                obs_tensor = obs_tensor.float().unsqueeze(0)
            
            embed = self.world_model.encoder(obs_tensor)
            deter = self.world_model.rssm.step(state[0], last_action, state[1])
            stats = self.world_model.rssm.representation_net(torch.cat([deter, embed], dim=-1))
            stoch = self.world_model.rssm.get_dist(stats).sample()
            next_state = (stoch, deter)
            
            action_dist = self.actor(stoch, deter)
            
            if mode == 'train':
                if not self.is_discrete:
                     # Paper: "executing the predicted mode action with Normal(0,0.3) exploration noise"
                     # We ignore the learned standard deviation during data collection
                     action = action_dist.mean + torch.randn_like(action_dist.mean) * 0.3
                else:
                     action = action_dist.sample()
            else:
                 # For eval, create video, etc.
                 # If continuous, mean is usually better. 
                 if not self.is_discrete:
                     # For TanhNormal (Normal), mean is the mode of the underlying Gaussian
                     # Our ActionModel returns Normal(mean, std), where mean is already 5*tanh(x).
                     # So we just take the mean.
                     action = action_dist.mean
                 elif hasattr(action_dist, 'mode'):
                     # For Categorical, mode() usually works if available or argmax probs
                     # PyTorch Categorical doesn't always strictly implement .mode() depending on version/mixin
                     # So let's be safe:
                     try:
                        action = action_dist.mode()
                     except:
                        action = action_dist.probs.argmax(dim=-1)
                 else:
                     action = action_dist.sample()

            if not self.is_discrete:
                 # Squash action to [-1, 1] for continuous control tasks
                 # This matches the behavior in WorldModel.imagine()
                 action = torch.tanh(action)

            if self.is_discrete:
                if mode == 'train' and np.random.rand() < 0.1:
                    # Epsilon greedy? batch logic tricky here without more code.
                    # Assuming continuous for now or single env.
                    pass
                
                if len(obs_tensor) > 1:
                     action_idx = action.cpu().numpy() # (B,)? No, Categorical.sample returns (B,)
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
        # Observation normalization for reconstruction target
        # ConvEncoder handles uint8->float internally, so we need matching target
        if obs.dtype == torch.uint8:
            obs_target = obs.float() / 255.0 - 0.5
        elif obs.max() > 1.0:
            obs_target = obs / 255.0 - 0.5
        else:
            obs_target = obs
        
        # --- 1. World Model Learning ---
        with torch.amp.autocast('cuda'):
            tran_stats, repr_stats, stoch, deter = self.world_model.observe(obs, action)
            
            recon = self.world_model.decoder(stoch, deter)
            pred_rew = self.world_model.reward(stoch, deter)
            
            # Reconstruction loss (sum over spatial dims, mean over batch/time)
            loss_obs = 0.5 * nn.functional.mse_loss(recon, obs_target, reduction='none').sum(dim=[-3, -2, -1]).mean()
            
            # Reward loss
            reward_target = reward if reward.dim() == pred_rew.dim() else reward.unsqueeze(-1)
            loss_rew = 0.5 * nn.functional.mse_loss(pred_rew, reward_target).mean()
            
            model_loss = loss_obs + loss_rew
            
            # PCont loss (Check if enabled)
            if self.world_model.pcont is not None:
                pred_pcont = self.world_model.pcont(stoch, deter)
                target_pcont = 1.0 - terminal.float()
                if target_pcont.dim() < pred_pcont.dim():
                    target_pcont = target_pcont.unsqueeze(-1)
                loss_pcont = nn.functional.mse_loss(pred_pcont, target_pcont)
                model_loss += loss_pcont
            else:
                loss_pcont = torch.tensor(0.0, device=self.device)

            dist_q, dist_p = self.world_model.rssm.get_dist(tran_stats), self.world_model.rssm.get_dist(repr_stats)
            kl_val = kl_divergence(dist_p, dist_q).sum(-1).mean()
            loss_kl = torch.max(kl_val, torch.tensor(self.cfg['model']['kl_free_nats']).to(self.device))
            
            model_loss += loss_kl
        
        self.wm_optimizer.zero_grad()
        self.scaler.scale(model_loss).backward()
        self.scaler.unscale_(self.wm_optimizer)
        nn.utils.clip_grad_norm_(self.world_model.parameters(), 100.0)
        self.scaler.step(self.wm_optimizer)
        # Don't update scaler yet

        # --- 2. Behavior Learning ---
        # Imagine from all states in the batch (flatten batch and time)
        with torch.amp.autocast('cuda'):
            batch_size, seq_len = stoch.shape[:2]
            start_stoch = stoch.detach().view(-1, stoch.shape[-1])
            start_deter = deter.detach().view(-1, deter.shape[-1])
            
            imag_stoch, imag_deter, _ = self.world_model.imagine(self.actor, (start_stoch, start_deter), self.cfg['actor']['horizon'])
            
            reward_imag = self.world_model.reward(imag_stoch, imag_deter).squeeze(-1)
            value_imag = self.value(imag_stoch, imag_deter).squeeze(-1)
            
            if self.world_model.pcont is not None:
                pcont_imag = self.world_model.pcont(imag_stoch, imag_deter).squeeze(-1)
                gamma_input = pcont_imag
            else:
                gamma_input = self.cfg['critic']['gamma']
            
            returns = self.compute_lambda_returns(reward_imag, value_imag, value_imag[:, -1], 
                                                self.cfg['critic']['lambda'], gamma_input)

            # Actor and Critic updates
            loss_actor = -returns.mean()
        
        self.actor_optimizer.zero_grad()
        self.scaler.scale(loss_actor).backward()
        self.scaler.unscale_(self.actor_optimizer)
        nn.utils.clip_grad_norm_(self.actor.parameters(), 100.0)
        self.scaler.step(self.actor_optimizer)
        
        with torch.amp.autocast('cuda'):
            value_pred = self.value(imag_stoch[:, :-1].detach(), imag_deter[:, :-1].detach()).squeeze(-1)
            loss_value = 0.5 * nn.functional.mse_loss(value_pred, returns[:, :-1].detach())
            
        self.value_optimizer.zero_grad()
        self.scaler.scale(loss_value).backward()
        self.scaler.unscale_(self.value_optimizer)
        nn.utils.clip_grad_norm_(self.value.parameters(), 100.0)
        self.scaler.step(self.value_optimizer)
        
        # Update scaler once at the end
        self.scaler.update()

        # Optimization: Return detached tensors instead of floats to avoid GPU synchronization at every step.
        # Calling .item() forces a CPU-GPU sync. We delays this until logging.
        return {
            "wm_loss": model_loss.detach(), 
            "kl": loss_kl.detach(), 
            "actor_loss": loss_actor.detach(), 
            "value_loss": loss_value.detach()
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
