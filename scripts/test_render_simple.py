"""
Simple rendering benchmark - must run BEFORE importing dm_control
"""
import os
import sys

backend = os.environ.get('MUJOCO_GL', 'egl')
resolution = os.environ.get('RENDER_RES', '64')
num_envs = int(os.environ.get('NUM_ENVS', '10'))

print(f"Testing: {backend} {resolution}x{resolution} with {num_envs} envs")

import time
import numpy as np
import gymnasium as gym

sys.path.append('/workspace1/gonzalo.fuentes/dreamer_study')
from envs.wrappers import make_env, DMCtoGymWrapper

# Patch resolution
resize_dim = int(resolution)
original_init = DMCtoGymWrapper.__init__

def patched_init(self, *args, **kwargs):
    kwargs['resize'] = [resize_dim, resize_dim]
    original_init(self, *args, **kwargs)

DMCtoGymWrapper.__init__ = patched_init

try:
    if num_envs > 1:
        def make_env_fn(rank):
            def _thunk():
                return make_env("dm_control/walker-walk-v0", action_repeat=2)
            return _thunk
        
        env = gym.vector.AsyncVectorEnv([make_env_fn(i) for i in range(num_envs)])
    else:
        env = make_env("dm_control/walker-walk-v0", action_repeat=2)
        num_envs = 1
    
    # Warmup
    obs, _ = env.reset()
    for _ in range(10):
        action = env.action_space.sample() if num_envs == 1 else np.array([env.single_action_space.sample() for _ in range(num_envs)])
        env.step(action)
    
    # Benchmark
    num_steps = 500
    step_times = []
    obs, _ = env.reset()
    
    for i in range(num_steps):
        start = time.time()
        
        if num_envs == 1:
            action = env.action_space.sample()
        else:
            action = np.array([env.single_action_space.sample() for _ in range(num_envs)])
        
        obs, reward, terminated, truncated, _ = env.step(action)
        step_times.append(time.time() - start)
    
    env.close()
    
    avg_ms = np.mean(step_times) * 1000
    std_ms = np.std(step_times) * 1000
    env_steps_per_sec = (num_steps * num_envs) / np.sum(step_times)
    
    print(f"✓ Avg: {avg_ms:.2f}ms ± {std_ms:.2f}ms")
    print(f"✓ Throughput: {env_steps_per_sec:.1f} env_steps/sec")
    print(f"RESULT:{avg_ms:.2f}:{env_steps_per_sec:.1f}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    print(f"RESULT:ERROR:{e}")
