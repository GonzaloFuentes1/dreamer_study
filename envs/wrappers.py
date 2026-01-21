"""
Author: Minh Pham-Dinh
Created: Feb 4th, 2024
Last Modified: Feb 7th, 2024
Email: mhpham26@colby.edu

Description:
    File containing wrappers for different environment types.
"""

import gymnasium as gym
from dm_control import suite
from dm_control.suite.wrappers import pixels
import numpy as np
import cv2
import os
import imageio
from dm_control.rl.control import Environment
import multiprocessing as mp
from multiprocessing import Process, Pipe
from typing import Callable, List, Tuple, Any

#wrapper by Hafner et al
class ActionRepeat(gym.Wrapper):
    def __init__(self, env, repeats):
        super().__init__(env)
        self.repeats = repeats

    def step(self, action):
        done = False
        total_reward = 0
        current_step = 0
        while current_step < self.repeats and not done:
            obs, reward, termination, truncation, info = self.env.step(action)
            total_reward += reward
            current_step += 1
            done = termination or truncation
        return obs, total_reward, termination, truncation, info


#wrapper by Hafner et al
class NormalizeActions(gym.Wrapper):
    """
    A wrapper class that normalizes the action space of an environment.
    """

    def __init__(self, env):
        super().__init__(env)
        self._mask = np.logical_and(
            np.isfinite(env.action_space.low),
            np.isfinite(env.action_space.high))
        self._low = np.where(self._mask, env.action_space.low, -1)
        self._high = np.where(self._mask, env.action_space.high, 1)
        
        low = np.where(self._mask, -np.ones_like(self._low), self._low).astype(np.float32)
        high = np.where(self._mask, np.ones_like(self._low), self._high).astype(np.float32)
        self.action_space = gym.spaces.Box(low, high, dtype=np.float32)

    def step(self, action):
        original = (action + 1) / 2 * (self._high - self._low) + self._low
        original = np.where(self._mask, original, action)
        return self.env.step(original)


class DMCtoGymWrapper(gym.Env):
    """
    Wrapper to convert a DeepMind Control Suite environment to a Gymnasium environment with additional features like recording and episode truncation.
    """
    metadata = {'render_modes': ['rgb_array']}
    
    def __init__(self, domain_name, task_name, task_kwargs=None, visualize_reward=False, resize=[64,64], record=False, record_freq=100, record_path='../', max_episode_steps=1000, camera=None, render_mode='rgb_array', action_repeat=1):
        super().__init__()
        self.render_mode = render_mode
        self.env = suite.load(domain_name, task_name, task_kwargs=task_kwargs, visualize_reward=visualize_reward)
        self.episode_count = -1
        self.record = record
        self.record_freq = record_freq
        self.record_path = record_path
        self.max_episode_steps = max_episode_steps
        self.action_repeat = action_repeat
        self.current_step = 0
        self.total_reward = 0
        self.recorder = None
        self.video_path = None
        self.frames = []
        self.episode_actions = []

        # Define action and observation space based on the DMC environment
        action_spec = self.env.action_spec()
        self.action_space = gym.spaces.Box(
            low=action_spec.minimum.astype(np.float32), 
            high=action_spec.maximum.astype(np.float32), 
            dtype=np.float32
        )
        
        self.resize = resize
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(3, *resize), dtype=np.uint8)

        if camera is None:
            camera = dict(quadruped=2).get(domain_name, 0)
        # Force camera 0 (side) for tracking in walker. -1 is static.
        if domain_name == "walker":
            camera = 0
            
        self._camera = camera

    def step(self, action):
        step_reward = 0.0
        termination = False
        truncation = False
        
        # Track joint usage (absolute magnitude)
        if hasattr(self, 'episode_actions'):
            self.episode_actions.append(np.array(action).flatten())
        
        for _ in range(self.action_repeat):
            time_step = self.env.step(action)
            r = time_step.reward if time_step.reward is not None else 0
            step_reward += r
            self.total_reward += (r or 0)
            self.current_step += 1
            
            termination = time_step.last()
            truncation = (self.current_step == self.max_episode_steps)
            
            if termination or truncation:
                break
        
        obs = self.env.physics.render(height=self.resize[0], width=self.resize[1], camera_id=self._camera)
        obs = obs.transpose([2, 0, 1])  # HWC -> CHW
        
        info = {}
        if termination or truncation:
            # Calculate action statistics per joint
            action_stats = {}
            if hasattr(self, 'episode_actions') and len(self.episode_actions) > 0:
                actions_stacked = np.stack(self.episode_actions)
                # Mean Absolute Value per joint
                usage = np.mean(np.abs(actions_stacked), axis=0)
                # Add to stats as lists (for aggregation compatibility)
                action_stats = {f'joint_{i}_usage': [u] for i, u in enumerate(usage)}
                
            info = {
                'episode': {
                    'r': [self.total_reward],
                    'l': self.current_step,
                    **action_stats
                }
            }
            
        if self.record:
            if self.episode_count % self.record_freq == 0:
                frame = self.env.physics.render(camera_id=self._camera, height=480, width=640)
                self.frames.append(frame.copy())
                
                if termination or truncation:
                    self._save_video()
                    info['video_path'] = self.video_path
        
        return obs, step_reward, termination, truncation, info

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.total_reward = 0
        self.episode_count += 1
        self.frames = []
        self.episode_actions = []
        
        # DM Control suite handles seeding at load time usually.
        time_step = self.env.reset()
        obs = self.env.physics.render(height=self.resize[0], width=self.resize[1], camera_id=self._camera)
        obs = obs.transpose([2, 0, 1])  # HWC -> CHW
        
        return obs, {}

    def _save_video(self):
        if not os.path.exists(self.record_path):
            os.makedirs(self.record_path)
            
        self.video_path = os.path.join(self.record_path, f"episode_{self.episode_count}.mp4")
        try:
            imageio.mimwrite(self.video_path, self.frames, fps=30, macro_block_size=None, 
                             quality=8, codec='libx264', pixelformat='yuv420p')
        except Exception as e:
            print(f"Warning: Could not save MP4 with h264 ({e}). Fallback to GIF.")
            self.video_path = os.path.join(self.record_path, f"episode_{self.episode_count}.gif")
            imageio.mimsave(self.video_path, self.frames, fps=30)
            
        self.frames = []
            
    def _get_obs(self):
        # DEPRECATED - now we render directly in step/reset
        obs = self.env.physics.render(*self.resize, camera_id=self._camera)
        return obs.transpose([2, 0, 1])

    def render(self, mode='rgb_array'):
        # Use high-res for external recording/viewing
        return self.env.physics.render(height=480, width=640, camera_id=self._camera)


class AtariPreprocess(gym.Wrapper):
    """
    A custom Gym wrapper that integrates multiple environment processing steps:
    - Records episode statistics and videos.
    - Resizes observations to a specified shape.
    - Scales and reorders observation channels.
    - Scales rewards using the tanh function.

    Parameters:
    - env (gym.Env): The original environment to wrap.
    - new_obs_size (tuple): The target size for observation resizing (height, width).
    - record (bool): If True, enable video recording.
    - record_path (str): The directory path where videos will be saved.
    - record_freq (int): Frequency (in episodes) at which to record videos.
    """
    def __init__(self, env, new_obs_size, record=False, record_path='../videos/', record_freq=100):
        super().__init__(env)
        self.env = gym.wrappers.RecordEpisodeStatistics(env)
        
        if record:
            self.env = gym.wrappers.RecordVideo(self.env, record_path, episode_trigger=lambda episode_id: episode_id % record_freq == 0)
        self.env = gym.wrappers.ResizeObservation(self.env, shape=new_obs_size)
        
        self.new_obs_size = new_obs_size
        self.observation_space = gym.spaces.Box(
            low=-0.5, high=0.5, 
            shape=(3, new_obs_size[0], new_obs_size[1]), 
            dtype=np.float32
        )

    def step(self, action):
        obs, reward, termination, truncation, info = super().step(action)
        obs = self.process_observation(obs)
        reward = np.tanh(reward)  # Scale reward
        return obs, reward, termination, truncation, info

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        obs = self.process_observation(obs)
        return obs, info

    def process_observation(self, observation):
        """
        Process and return the observation from the environment.
        - Scales pixel values to the range [-0.5, 0.5].
        - Reorders channels to CHW format (channels, height, width).

        Parameters:
        - observation (np.ndarray): The original observation from the environment.

        Returns:
        - np.ndarray: The processed observation.
        """
        if 'pixels' in observation:
            observation = observation['pixels']
        observation = observation / 255.0 - 0.5
        observation = np.transpose(observation, (2, 0, 1))
        return observation

def make_env(env_id, action_repeat=2, seed=None, record=False, record_path='videos', record_freq=1):
    if "dm_control" in env_id:
        short_id = env_id.replace("dm_control/", "")
        parts = short_id.split("-")
        domain = parts[0]
        task = parts[1].replace("-v0", "")
        
        env = DMCtoGymWrapper(domain, task, record=record, record_path=record_path, record_freq=record_freq, action_repeat=action_repeat)
        env = NormalizeActions(env)
    else:
        env = gym.make(env_id, render_mode="rgb_array")
        if record:
            env = gym.wrappers.RecordVideo(env, record_path, episode_trigger=lambda x: True)
        env = ActionRepeat(env, action_repeat)
        
    return env


# Async wrapper for parallel environment execution
def _worker(remote: mp.connection.Connection, parent_remote: mp.connection.Connection, env_fn: Callable):
    """Worker process for asynchronous environment execution."""
    parent_remote.close()
    env = env_fn()
    
    try:
        while True:
            cmd, data = remote.recv()
            
            if cmd == 'step':
                obs, reward, terminated, truncated, info = env.step(data)
                if terminated or truncated:
                    # Auto-reset on done
                    obs, _ = env.reset()
                remote.send((obs, reward, terminated, truncated, info))
                
            elif cmd == 'reset':
                obs, info = env.reset()
                remote.send((obs, info))
                
            elif cmd == 'close':
                env.close()
                remote.close()
                break
                
            elif cmd == 'get_spaces':
                remote.send((env.observation_space, env.action_space))
                
            else:
                raise NotImplementedError(f"Command {cmd} not implemented")
                
    except KeyboardInterrupt:
        env.close()
        remote.close()


class AsyncVectorEnv:
    """
    Asynchronous vectorized environment that runs multiple environments in parallel processes.
    
    This allows data collection to happen in parallel with training, similar to the 
    TensorFlow implementation in the DreamerV1 paper.
    """
    
    def __init__(self, env_fns: List[Callable]):
        """
        Args:
            env_fns: List of functions that create environments
        """
        self.num_envs = len(env_fns)
        self.closed = False
        self.waiting = False
        
        # Create pipes for communication
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(self.num_envs)])
        
        # Start worker processes
        self.processes = [
            Process(target=_worker, args=(work_remote, remote, env_fn), daemon=True)
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)
        ]
        
        for p in self.processes:
            p.start()
            
        for work_remote in self.work_remotes:
            work_remote.close()
            
        # Get observation and action spaces
        self.remotes[0].send(('get_spaces', None))
        self.observation_space, self.action_space = self.remotes[0].recv()
        
    def step_async(self, actions):
        """Send step commands to all environments (non-blocking)."""
        if self.waiting:
            raise RuntimeError("step_async called while already waiting")
            
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
            
        self.waiting = True
        
    def step_wait(self):
        """Wait for step results from all environments."""
        if not self.waiting:
            raise RuntimeError("step_wait called without step_async")
            
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        
        obs, rewards, terminateds, truncateds, infos = zip(*results)
        return np.stack(obs), np.array(rewards), np.array(terminateds), np.array(truncateds), list(infos)
        
    def step(self, actions):
        """Step all environments (blocking)."""
        self.step_async(actions)
        return self.step_wait()
        
    def reset(self):
        """Reset all environments."""
        for remote in self.remotes:
            remote.send(('reset', None))
            
        results = [remote.recv() for remote in self.remotes]
        obs, infos = zip(*results)
        return np.stack(obs), list(infos)
        
    def close(self):
        """Close all environments and worker processes."""
        if self.closed:
            return
            
        if self.waiting:
            # Wait for pending operations
            try:
                for remote in self.remotes:
                    remote.recv()
            except:
                pass
                
        for remote in self.remotes:
            try:
                remote.send(('close', None))
            except:
                pass
            
        for p in self.processes:
            p.join(timeout=1.0)
            if p.is_alive():
                p.terminate()
            
        self.closed = True
        
    def __del__(self):
        if not self.closed:
            self.close()
