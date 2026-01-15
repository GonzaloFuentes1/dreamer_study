"""
Author: Minh Pham-Dinh (adapted for parallel envs)
Created: Jan 26th, 2024
Last Modified: Feb 5th, 2024

Description:
    Parallel ReplayBuffer for vectorized environments.
    Maintains separate pointers per environment to preserve sequence integrity.
"""

import numpy as np
import torch
from addict import Dict

class ParallelReplayBuffer:
    def __init__(self, capacity, obs_size, action_size, num_envs):
        self.obs_size = obs_size
        self.action_size = action_size
        self.num_envs = num_envs
        self.env_capacity = capacity // num_envs
        
        # Memory optimization: uint8 for images
        state_type = np.uint8 if len(self.obs_size) == 3 else np.float32
        
        # Storage: (num_envs, env_capacity, ...)
        self.observation = np.zeros((num_envs, self.env_capacity) + self.obs_size, dtype=state_type)
        self.actions = np.zeros((num_envs, self.env_capacity) + self.action_size, dtype=np.float32)
        self.rewards = np.zeros((num_envs, self.env_capacity, 1), dtype=np.float32)
        self.dones = np.zeros((num_envs, self.env_capacity, 1), dtype=np.float32)
        
        # Separate pointer per env
        self.pointer = np.zeros(num_envs, dtype=np.int32)
        self.full = np.zeros(num_envs, dtype=bool)
        
        print(f'''
-----------initialized parallel memory----------              

num_envs: {num_envs}
env_capacity: {self.env_capacity}
obs_buffer_shape: {self.observation.shape}
actions_buffer_shape: {self.actions.shape}
rewards_buffer_shape: {self.rewards.shape}
dones_buffer_shape: {self.dones.shape}

------------------------------------------------
              ''')

    def add(self, obs, action, reward, done):
        """Add parallel transitions from multiple environments
        
        Args:
            obs: (num_envs, *obs_size)
            action: (num_envs, *action_size)
            reward: (num_envs,) or (num_envs, 1)
            done: (num_envs,) or (num_envs, 1)
        """
        for i in range(self.num_envs):
            idx = self.pointer[i]
            self.observation[i, idx] = obs[i]
            self.actions[i, idx] = action[i]
            self.rewards[i, idx] = reward[i]
            self.dones[i, idx] = done[i]
            
            self.pointer[i] = (idx + 1) % self.env_capacity
            if self.pointer[i] == 0:
                self.full[i] = True

    def sample(self, batch_size, seq_len, device):
        """Sample sequences avoiding wrap-around boundary"""
        
        # Check if any env has enough data
        valid_envs = []
        for i in range(self.num_envs):
            available = self.env_capacity if self.full[i] else self.pointer[i]
            if available >= seq_len:
                valid_envs.append(i)
        
        if len(valid_envs) == 0:
            raise Exception('not enough data to sample')
        
        # Sample from valid environments
        env_indices = np.random.choice(valid_envs, size=batch_size)
        
        obs_list = []
        act_list = []
        rew_list = []
        done_list = []
        
        for env_idx in env_indices:
            # Determine valid range for this env
            if self.full[env_idx]:
                ptr = self.pointer[env_idx]
                if ptr - seq_len < 0:
                    valid_range = np.arange(ptr, self.env_capacity - (seq_len - ptr))
                else:
                    range_1 = np.arange(0, ptr - seq_len + 1)
                    range_2 = np.arange(ptr, self.env_capacity)
                    valid_range = np.concatenate((range_1, range_2))
            else:
                valid_range = np.arange(0, self.pointer[env_idx] - seq_len + 1)
            
            start_idx = np.random.choice(valid_range)
            seq_indices = (start_idx + np.arange(seq_len)) % self.env_capacity
            
            obs_list.append(self.observation[env_idx, seq_indices])
            act_list.append(self.actions[env_idx, seq_indices])
            rew_list.append(self.rewards[env_idx, seq_indices])
            done_list.append(self.dones[env_idx, seq_indices])
        
        batch = Dict()
        batch.obs = torch.from_numpy(np.array(obs_list)).to(device)
        batch.actions = torch.from_numpy(np.array(act_list)).to(device)
        batch.rewards = torch.from_numpy(np.array(rew_list)).to(device)
        batch.dones = torch.from_numpy(np.array(done_list)).to(device)
        
        return batch
    
    def clear(self):
        self.pointer[:] = 0
        self.full[:] = False
    
    def __len__(self):
        return np.sum(np.where(self.full, self.env_capacity, self.pointer))
