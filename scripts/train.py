import argparse
import pathlib
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--version", type=str, default="v1")
parser.add_argument("--exp", type=str, default="default")
parser.add_argument("--env", type=str, default=None, help="Override env id from config")
parser.add_argument("--gpu", type=str, default="0", help="GPU id to use, or -1 for CPU")
parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint.pt to resume")
args = parser.parse_args()

# Only set CUDA_VISIBLE_DEVICES if not forcing CPU mode
if args.gpu != "-1":
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.environ['MUJOCO_GL'] = 'egl'
os.environ['__NV_PRIME_RENDER_OFFLOAD'] = '1'
os.environ['__GLX_VENDOR_LIBRARY_NAME'] = 'nvidia'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

import torch
import numpy as np
import datetime
import time
import json
from collections import deque
import gymnasium as gym
from ruamel.yaml import YAML
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.wrappers import make_env, AsyncVectorEnv
from common.buffer import ReplayBuffer
from common.buffer_parallel import ParallelReplayBuffer
from common.prefetch_buffer import PrefetchBuffer
from dreamer_v1.agent import DreamerV1Agent
from dreamer_v2.agent import DreamerV2Agent
# from dreamer_v3.agent import DreamerV3Agent  # TODO: Fix imports

def load_config(version, exp_name):
    yaml = YAML(typ='safe')
    config_path = pathlib.Path(f"dreamer_{version}/configs/{exp_name}.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    return yaml.load(config_path)

def get_state_dims(agent):
    """Helper to get state dimensions from agent (works for V1, V2, V3)"""
    if hasattr(agent, 'rssm') and hasattr(agent.rssm, 'stoch_dim'):
        stoch_dim = agent.rssm.stoch_dim
        if hasattr(agent.rssm, 'stoch_classes'):
            stoch_classes = agent.rssm.stoch_classes
            stoch_flat_dim = stoch_dim * stoch_classes
        else:
            stoch_flat_dim = stoch_dim
        deter_dim = agent.rssm.deter_dim
    else:
        stoch_dim = agent.cfg['model']['rssm'].get('stoch_dim', 32)
        stoch_classes = agent.cfg['model']['rssm'].get('stoch_classes', 1)
        stoch_flat_dim = stoch_dim * stoch_classes
        deter_dim = agent.cfg['model']['rssm']['deter_dim']
    
    return stoch_flat_dim, deter_dim

def eval_policy(agent, env_id, device, num_episodes=100, max_steps=1000, action_repeat=2, seed=None):
    """
    Evaluate policy on multiple episodes and return statistics.
    
    Args:
        agent: DreamerAgent
        env_id: Environment ID
        device: Device to use
        num_episodes: Number of evaluation episodes (default 100)
        max_steps: Max steps per episode
        action_repeat: Action repeat factor
        seed: Random seed for reproducibility
    
    Returns:
        dict: {
            'mean_reward': float,
            'std_reward': float,
            'min_reward': float,
            'max_reward': float,
            'mean_length': float,
            'episode_rewards': list
        }
    """
    stoch_flat_dim, deter_dim = get_state_dims(agent)
    action_dim = agent.action_dim
    
    episode_rewards = []
    episode_lengths = []
    
    for ep_idx in range(num_episodes):
        env = make_env(env_id, action_repeat=action_repeat, seed=seed)
        obs, _ = env.reset()
        
        state = (torch.zeros(1, stoch_flat_dim).to(device), torch.zeros(1, deter_dim).to(device))
        last_action = torch.zeros(1, action_dim).to(device)
        
        ep_reward = 0.0
        ep_length = 0
        done = False
        
        while not done and ep_length < max_steps:
            act_data, next_state, env_action = agent.policy(obs, state, last_action, mode='eval')
            state = next_state
            last_action = torch.tensor(act_data, dtype=torch.float32).to(device)
            if last_action.ndim == 1:
                last_action = last_action.unsqueeze(0)
            
            obs, reward, terminated, truncated, _ = env.step(env_action)
            ep_reward += reward
            ep_length += 1
            done = terminated or truncated
        
        env.close()
        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_length)
        
        if (ep_idx + 1) % max(1, num_episodes // 10) == 0:
            print(f"  Eval progress: {ep_idx + 1}/{num_episodes} episodes")
    
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    
    return {
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'episode_rewards': episode_rewards
    }

def eval_and_record(agent, env_id, step, video_dir, device, num_episodes=100, max_steps=1000, action_repeat=2):
    """
    Evaluate policy (100 episodes) and record one video.
    
    Returns:
        dict with eval stats and video path
    """
    print(f"\n{'='*60}")
    print(f"Evaluation at step {step} ({num_episodes} episodes)")
    print(f"{'='*60}")
    
    # Run evaluation
    eval_stats = eval_policy(agent, env_id, device, num_episodes=num_episodes, 
                            max_steps=max_steps, action_repeat=action_repeat)
    
    print(f"Mean reward: {eval_stats['mean_reward']:.1f} ± {eval_stats['std_reward']:.1f}")
    print(f"Min/Max: {eval_stats['min_reward']:.1f} / {eval_stats['max_reward']:.1f}")
    print(f"Mean episode length: {eval_stats['mean_length']:.0f} steps")
    
    # Record one video episode
    stoch_flat_dim, deter_dim = get_state_dims(agent)
    step_video_dir = pathlib.Path(video_dir) / f"step_{step}"
    step_video_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nRecording video to {step_video_dir}...")
    env = make_env(env_id, action_repeat=action_repeat, record=True, 
                  record_path=str(step_video_dir), record_freq=1)
    
    obs, _ = env.reset()
    state = (torch.zeros(1, stoch_flat_dim).to(device), torch.zeros(1, deter_dim).to(device))
    last_action = torch.zeros(1, agent.action_dim).to(device)
    
    done = False
    curr_step = 0
    video_reward = 0
    
    while not done and curr_step < max_steps:
        act_data, next_state, env_action = agent.policy(obs, state, last_action, mode='eval')
        state = next_state
        last_action = torch.tensor(act_data, dtype=torch.float32).to(device)
        if last_action.ndim == 1:
            last_action = last_action.unsqueeze(0)
        
        obs, reward, terminated, truncated, _ = env.step(env_action)
        video_reward += reward
        curr_step += 1
        done = terminated or truncated
    
    env.close()
    eval_stats['video_reward'] = video_reward
    eval_stats['video_path'] = str(step_video_dir)
    
    return eval_stats

def main():
    config = load_config(args.version, args.exp)
    
    if args.env:
        config['env'] = args.env

    action_repeat = config.get('action_repeat', 2)
    
    # Force CPU if --gpu -1, otherwise check CUDA availability
    if args.gpu == "-1":
        device = torch.device("cpu")
        print("✓ Using CPU (forced by --gpu -1)")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() and config['device'] == "cuda" else "cpu")
        if device.type == 'cuda':
            print(f"✓ Using GPU {args.gpu}")
            
            # GPU optimizations
            if config.get('tf32', True):
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                print("✓ TF32 enabled")
            
            if config.get('cudnn_benchmark', True):
                torch.backends.cudnn.benchmark = True
                print("✓ CuDNN benchmark enabled")
    
    num_envs = config.get('num_envs', 1)
    is_vectorized = num_envs > 1
    use_async = config.get('use_async', True)
    
    if is_vectorized:
        print(f"Vectorized Training: {num_envs} parallel environments")
        def make_env_fn(rank):
            def _thunk():
                return make_env(config['env'], action_repeat=action_repeat)
            return _thunk
        
        if use_async:
            print(f"Using AsyncVectorEnv (parallel rendering with multiprocessing)")
            env = AsyncVectorEnv([make_env_fn(i) for i in range(num_envs)])
            obs_shape = env.observation_space.shape
            action_space = env.action_space
        else:
            # Fallback to gym's implementation
            print(f"Using gym.vector.AsyncVectorEnv")
            env = gym.vector.AsyncVectorEnv([make_env_fn(i) for i in range(num_envs)])
            obs_shape = env.single_observation_space.shape
            action_space = env.single_action_space
    else:
        print("Single Environment Training")
        env = make_env(config['env'], action_repeat=action_repeat)
        obs_shape = env.observation_space.shape
        action_space = env.action_space
    
    # Paper uses repeat 2, so the env will have it already via wrapper
    action_dim = action_space.n if hasattr(action_space, 'n') else action_space.shape[0]
    is_discrete = hasattr(action_space, 'n')

    if is_vectorized:
        buffer = ParallelReplayBuffer(
            config['buffer']['capacity'], 
            obs_shape, 
            (action_dim,),
            num_envs
        )
    else:
        buffer = ReplayBuffer(
            config['buffer']['capacity'], 
            obs_shape, 
            (action_dim,)
        )
    
    # Select agent based on version
    if args.version == "v1":
        agent = DreamerV1Agent(config, obs_shape, action_dim, is_discrete, device)
    elif args.version == "v2":
        agent = DreamerV2Agent(config, obs_shape, action_dim, is_discrete, device)
    elif args.version == "v3":
        raise NotImplementedError("V3 agent imports need fixing. Use --version v1 or v2")
        # agent = DreamerV3Agent(config, obs_shape, action_dim, is_discrete, device)
    else:
        raise ValueError(f"Unknown version: {args.version}. Use 'v1' or 'v2'")
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"{args.version}_{args.exp}_{timestamp}"
    
    # Unified structure: everything inside runs/<version>/<run_id>/
    run_dir = pathlib.Path(f"runs/{args.version}/{run_id}")
    log_dir = run_dir / "logs"
    ckpt_dir = run_dir / "checkpoints"
    video_dir = run_dir / "videos"
    
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    
    writer = SummaryWriter(log_dir=str(log_dir))
    json_log_path = run_dir / "metrics.jsonl"
    
    def log_metrics(step, metrics):
        entry = {'step': step, 'timestamp': time.time()}
        # Convert numpy types to native Python types for JSON serialization
        for k, v in metrics.items():
            if hasattr(v, 'item'):  # torch.Tensor or numpy scalar
                entry[k] = v.item()
            elif isinstance(v, np.ndarray):
                entry[k] = v.tolist()
            elif isinstance(v, (np.integer, np.floating)):
                entry[k] = v.item()
            else:
                entry[k] = v
        with open(json_log_path, 'a') as f:
            f.write(json.dumps(entry) + '\n')
    
    # Env created above for shape info
    # env = make_env(config['env']) # Already created
    
    # Optional: Resume from checkpoint
    start_step = 0
    if args.resume:
        print(f"Resuming from {args.resume}...")
        loaded_logs = agent.load(args.resume)
        if loaded_logs and 'step' in loaded_logs:
            start_step = loaded_logs['step'] + 1
            print(f"Resumed at step {start_step}")

    # 2. Initial collection
    if not args.resume:
        print(f"Collecting {config['training']['initial_episodes']} random episodes...")
        episodes_collected = 0
        # For vectorized: collect initial_episodes total (running in parallel)
        # For single env: collect initial_episodes sequentially
        target_episodes = config['training']['initial_episodes']
        
        collection_start = time.time()
        
        if is_vectorized:
            obs, _ = env.reset()
            episode_dones = np.zeros(num_envs, dtype=bool)
            episode_steps = np.zeros(num_envs, dtype=int)
            
            while episodes_collected < target_episodes:
                action = np.array([action_space.sample() for _ in range(num_envs)])
                act_data = np.zeros((num_envs, action_dim))
                
                if is_discrete:
                    for i in range(num_envs):
                        act_data[i, action[i]] = 1.0
                else:
                    act_data = action
                
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated | truncated
                
                buffer.add(obs, act_data, reward, done)
                obs = next_obs
                episode_steps += 1
                
                for i in range(num_envs):
                    if done[i] and not episode_dones[i]:
                        episodes_collected += 1
                        episode_dones[i] = True
                        print(f"  Episode {episodes_collected}/{target_episodes} collected ({episode_steps[i]} steps).")
                        episode_steps[i] = 0
                        episode_dones[i] = False
        else:
            for ep_idx in range(config['training']['initial_episodes']):
                ep_start = time.time()
                obs, _ = env.reset()
                done = False
                step_count = 0
                step_times = []
                
                while not done:
                    step_start = time.time()
                    action = env.action_space.sample()
                    act_data = np.zeros(action_dim)
                    if is_discrete: act_data[action] = 1.0
                    else: act_data = action
                    
                    next_obs, reward, terminated, truncated, _ = env.step(action)
                    buffer.add(obs, act_data, reward, terminated or truncated)
                    obs = next_obs
                    done = terminated or truncated
                    step_count += 1
                    step_times.append(time.time() - step_start)
                
                ep_time = time.time() - ep_start
                avg_step_time = np.mean(step_times) * 1000
                print(f"  Episode {ep_idx+1}/{config['training']['initial_episodes']} collected ({step_count} steps, {ep_time:.1f}s total, {avg_step_time:.1f}ms/step).")
        
        total_collection_time = time.time() - collection_start
        print(f"Initial collection completed in {total_collection_time:.1f}s\n")

    # 3. Main Training Loop
    print(f"--- Training Dreamer {args.version} on {config['env']} (GPU: {args.gpu}) ---")
    obs, _ = env.reset()
    deter_dim = config['model']['rssm']['deter_dim']
    
    # For V2/V3: stoch_dim and stoch_classes are FIXED at 32x32
    if args.version in ["v2", "v3"]:
        stoch_dim = 32
        stoch_classes = 32
        stoch_flat_dim = stoch_dim * stoch_classes  # 1024
    else:
        # V1 uses configurable stoch_dim
        stoch_dim = config['model']['rssm']['stoch_dim']
        stoch_flat_dim = stoch_dim
    
    batch_size_state = num_envs if is_vectorized else 1
    state = (torch.zeros(batch_size_state, stoch_flat_dim).to(device), torch.zeros(batch_size_state, deter_dim).to(device))
    last_action = torch.zeros(batch_size_state, action_dim, device=device, dtype=torch.float32)
    
    episode_reward = np.zeros(num_envs) if is_vectorized else 0
    episode_step = np.zeros(num_envs, dtype=int) if is_vectorized else 0
    num_episodes = 0
    best_reward = -float('inf')
    avg_reward = None
    losses_scalar = {}
    
    # Track últimos 100 episodios para promedio más representativo
    recent_rewards = deque(maxlen=100)
    
    # Initialize prefetch buffer for async data loading (overlaps with training)
    use_prefetch = config.get('use_prefetch', True)
    prefetch_buffer = None
    if use_prefetch:
        prefetch_buffer = PrefetchBuffer(
            replay_buffer=buffer,
            batch_size=config['training']['batch_size'],
            seq_length=config['training']['seq_length'],
            device=device,
            buffer_size=3  # Reduced from 20 to 3 to save GPU memory
        )
        print("Using PrefetchBuffer for async data loading (buffer_size=3)")
    
    # Timing variables
    timer = {'interaction': 0.0, 'train': 0.0, 'eval': 0.0, 'total': 0.0}
    log_interval = 100
    
    # Contador de environment steps REALES (paper uses this, not iterations)
    total_env_steps = start_step * num_envs  # Total env steps collected so far
    last_train_env_steps = 0  # Last env step count when we trained

    # Progress bar should count env steps, not iterations
    # With num_envs parallel: total_steps is in env steps, not iterations
    total_env_steps_target = config['training']['total_steps']
    pbar = tqdm(total=total_env_steps_target, desc="Env Steps", unit="envsteps")
    pbar.update(total_env_steps)  # Start from current position
    
    # Loop in iterations, but stop when we reach total_env_steps target
    max_iterations = (config['training']['total_steps'] + num_envs - 1) // num_envs  # Ceiling division
    for step in range(start_step, max_iterations):
        step_start_time = time.time()
        
        # --- Interaction ---
        t_interact_start = time.time()
        
        t_policy = time.time()
        act_data, next_state, env_action = agent.policy(obs, state, last_action, mode='train')
        timer['policy'] = timer.get('policy', 0) + (time.time() - t_policy)
        state = next_state
        
        if is_vectorized:
            # Reuse preallocated tensor (avoid recreation overhead)
            last_action.copy_(torch.from_numpy(act_data) if isinstance(act_data, np.ndarray) else act_data)
        else:
            last_action[0].copy_(torch.from_numpy(act_data) if isinstance(act_data, np.ndarray) else act_data)

        t_env = time.time()
        next_obs, reward, terminated, truncated, infos = env.step(env_action)
        timer['env_step'] = timer.get('env_step', 0) + (time.time() - t_env)
        
        if is_vectorized:
            done = terminated | truncated
            buffer.add(obs, act_data, reward, done)
            
            episode_reward += reward
            episode_step += 1
            
            for i in range(num_envs):
                if done[i]:
                    ep_rew = episode_reward[i]
                    ep_len = episode_step[i]
                    
                    recent_rewards.append(ep_rew)
                    avg_reward_100 = np.mean(recent_rewards) if recent_rewards else ep_rew
                    
                    writer.add_scalar("episode/reward", ep_rew, num_episodes)
                    writer.add_scalar("episode/length", ep_len, num_episodes)
                    writer.add_scalar("episode/reward_avg100", avg_reward_100, num_episodes)
                    
                    metrics_to_log = {
                        'episode_reward': ep_rew, 
                        'episode_length': ep_len, 
                        'episode': num_episodes, 
                        'reward_avg100': avg_reward_100
                    }
                    
                    # Log joint usage stats if available
                    if 'episode' in infos[i]:
                        for k, v in infos[i]['episode'].items():
                            if k.startswith('joint_'):
                                if isinstance(v, (list, np.ndarray)):
                                    v = np.mean(v)
                                writer.add_scalar(f"episode/{k}", v, num_episodes)
                                metrics_to_log[k] = v
                    
                    # Skip JSON logging to avoid disk I/O overhead
                    # log_metrics(step, metrics_to_log)
                    
                    if avg_reward is None:
                        avg_reward = ep_rew
                    else:
                        avg_reward = 0.95 * avg_reward + 0.05 * ep_rew
                    
                    print(f"  > Episode {num_episodes} finished. Reward: {ep_rew:.1f} | Avg(100ep): {avg_reward_100:.1f} | EMA: {avg_reward:.1f} | Length: {ep_len}")
                    
                    if ep_rew > best_reward:
                        best_reward = ep_rew
                        logs = {'step': step, 'episode_reward': ep_rew, 'num_episodes': num_episodes, 'losses': losses_scalar}
                        agent.save(ckpt_dir / "best.pt", logs)
                    
                    episode_reward[i] = 0
                    episode_step[i] = 0
                    num_episodes += 1
                    
                    # NOTE: Paper does NOT reset RSSM state during training episodes!
                    # This allows the model to learn continuity across episodes.
                    # Only reset for evaluation/video recording.
                    # state[0][i] = 0
                    # state[1][i] = 0
                    # last_action[i] = 0
        else:
            buffer.add(obs, act_data, reward, terminated or truncated)
            
            episode_reward += reward
            episode_step += 1
            
            if terminated or truncated:
                recent_rewards.append(episode_reward)
                avg_reward_100 = np.mean(recent_rewards) if recent_rewards else episode_reward
                
                writer.add_scalar("episode/reward", episode_reward, num_episodes)
                writer.add_scalar("episode/length", episode_step, num_episodes)
                writer.add_scalar("episode/reward_avg100", avg_reward_100, num_episodes)
                
                metrics_to_log = {
                    'episode_reward': episode_reward, 
                    'episode_length': episode_step, 
                    'episode': num_episodes, 
                    'reward_avg100': avg_reward_100
                }
                
                # Log joint usage stats if available
                # In non-vectorized mode, infos is the single info dict
                info = infos 
                if 'episode' in info:
                    for k, v in info['episode'].items():
                        if k.startswith('joint_'):
                            if isinstance(v, (list, np.ndarray)):
                                v = np.mean(v)
                            writer.add_scalar(f"episode/{k}", v, num_episodes)
                            metrics_to_log[k] = v
                
                # Skip JSON logging to avoid disk I/O overhead
                # log_metrics(step, metrics_to_log)
                
                if avg_reward is None:
                    avg_reward = episode_reward
                else:
                    avg_reward = 0.95 * avg_reward + 0.05 * episode_reward
                    
                print(f"  > Episode {num_episodes} finished. Reward: {episode_reward:.1f} | Avg(100ep): {avg_reward_100:.1f} | EMA: {avg_reward:.1f} | Length: {episode_step}")
                
                if episode_reward > best_reward:
                    best_reward = episode_reward
                    logs = {'step': step, 'episode_reward': episode_reward, 'num_episodes': num_episodes, 'losses': losses_scalar}
                    agent.save(ckpt_dir / "best.pt", logs)
                
                obs, _ = env.reset()
                state = (torch.zeros(1, stoch_flat_dim).to(device), torch.zeros(1, deter_dim).to(device))
                last_action = torch.zeros(1, action_dim).to(device)
                episode_reward, episode_step, num_episodes = 0, 0, num_episodes + 1
        
        obs = next_obs
        timer['interaction'] += time.time() - t_interact_start
        
        # Update TOTAL env steps counter: number of agent steps (NOT physics steps)
        # With vectorized envs, each iteration = num_envs steps
        # action_repeat is internal to the environment, we count agent decisions
        total_env_steps += num_envs

        # --- Training ---
        # Paper: train every N env steps. 
        # IMPORTANTE: total_env_steps ahora crece 2x más rápido (si repeat=2)
        env_steps_since_train = total_env_steps - last_train_env_steps
        should_train = (total_env_steps > config['training']['batch_size'] * config['training']['seq_length'] and 
                       env_steps_since_train >= config['training']['train_every'])
        
        if should_train:
            pbar.set_description("Phase: Learn Dynamics & Behavior")
            t0 = time.time()
            
            # Start prefetching if not already started
            if prefetch_buffer is not None and prefetch_buffer.thread is None:
                prefetch_buffer.start()
            
            for train_step_idx in range(config['training']['train_steps']):
                if prefetch_buffer is not None:
                    # Get prefetched batch (already on device)
                    try:
                        batch_dict = prefetch_buffer.get_batch()
                    except RuntimeError:
                        # Fallback to sync sampling if prefetcher is too slow
                        batch = buffer.sample(
                            config['training']['batch_size'],
                            config['training']['seq_length'],
                            device
                        )
                        batch_dict = {
                            'obs': batch.obs,
                            'actions': batch.actions,
                            'rewards': batch.rewards,
                            'dones': batch.dones
                        }
                    
                    losses = agent.train_step(
                        batch_dict['obs'], 
                        batch_dict['actions'], 
                        batch_dict['rewards'], 
                        batch_dict['dones']
                    )
                else:
                    # Original synchronous sampling
                    batch = buffer.sample(
                        config['training']['batch_size'],
                        config['training']['seq_length'],
                        device
                    )
                    losses = agent.train_step(batch.obs, batch.actions, batch.rewards, batch.dones)
                
                # Only log every 50 updates to reduce I/O overhead and GPU sync
                if train_step_idx % 50 == 0:
                    # Convert GPU tensors to scalars for logging
                    losses_scalar = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()}
                    for name, loss in losses_scalar.items():
                        writer.add_scalar(f"train/{name}", loss, step + train_step_idx)
                    # Skip JSON logging during training to avoid disk I/O
                    # log_metrics(step, {f'train/{k}': v for k, v in losses_scalar.items()})
            
            # Use last losses for progress bar (convert to scalars) - only update every 10 steps
            if step % 10 == 0:
                losses_scalar = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()}
                pbar_dict = {k: f"{v:.3f}" for k, v in losses_scalar.items()}
                if recent_rewards:
                    pbar_dict['avg_100ep'] = f"{np.mean(recent_rewards):.1f}"
                if avg_reward is not None:
                    pbar_dict['ema'] = f"{avg_reward:.1f}"
                pbar.set_postfix(pbar_dict)
                # Update progress bar with env steps, not iterations
                pbar.n = total_env_steps
                pbar.refresh()
            timer['train'] += time.time() - t0
            
            # Update last train counter
            last_train_env_steps = total_env_steps

        # Periodic checkpoint and video (use config intervals in env_steps)
        t0 = time.time()
        save_interval = config.get('logging', {}).get('save_interval', 10000)
        eval_interval = config.get('logging', {}).get('eval_interval', 20000)
        
        # Save checkpoint at intervals (in env_steps)
        if total_env_steps % save_interval < num_envs and total_env_steps >= save_interval:
             logs = {'step': step, 'total_env_steps': total_env_steps, 'episode_reward': episode_reward, 'num_episodes': num_episodes, 'losses': losses_scalar}
             agent.save(ckpt_dir / f"step_{total_env_steps}.pt", logs)
             print(f"\n>>> Checkpoint saved at {total_env_steps} env steps <<<\n")

        # Record video at intervals (in env_steps)
        if total_env_steps % eval_interval < num_envs and total_env_steps >= eval_interval:
            pbar.set_description("Phase: Evaluation & Video")
            
            # Evaluate policy over 100 episodes with statistics
            eval_stats = eval_and_record(agent, config['env'], total_env_steps, video_dir, device, 
                                        num_episodes=100, action_repeat=action_repeat)
            
            # Log evaluation statistics
            writer.add_scalar("eval/mean_reward", eval_stats['mean_reward'], total_env_steps)
            writer.add_scalar("eval/std_reward", eval_stats['std_reward'], total_env_steps)
            writer.add_scalar("eval/min_reward", eval_stats['min_reward'], total_env_steps)
            writer.add_scalar("eval/max_reward", eval_stats['max_reward'], total_env_steps)
            writer.add_scalar("eval/mean_length", eval_stats['mean_length'], total_env_steps)
            writer.add_scalar("eval/video_reward", eval_stats['video_reward'], total_env_steps)
            
            log_metrics(total_env_steps, {
                'eval_mean_reward': eval_stats['mean_reward'],
                'eval_std_reward': eval_stats['std_reward'],
                'eval_video_reward': eval_stats['video_reward']
            })
            print(f">>> Evaluation completed at {total_env_steps} env steps <<<")
            print(f"    Mean±Std: {eval_stats['mean_reward']:.1f}±{eval_stats['std_reward']:.1f}")
            print(f"    Video reward: {eval_stats['video_reward']:.1f}")
            print(f"    Video saved to: {eval_stats['video_path']}\n")
        timer['eval'] += time.time() - t0

        timer['total'] += time.time() - step_start_time

        if step % log_interval == 0 and step > 0:
             # Calculate average time per step for each component
             avg_total = timer['total'] / log_interval
             avg_inter = timer['interaction'] / log_interval
             avg_policy = timer.get('policy', 0) / log_interval
             avg_env = timer.get('env_step', 0) / log_interval
             avg_train = timer['train'] / log_interval
             avg_eval = timer['eval'] / log_interval
             
             # Calculate env steps per second
             elapsed = time.time() - pbar.start_t
             env_steps_per_sec = total_env_steps / elapsed if elapsed > 0 else 0
             
             print(f"\nStep {step} (TotalEnvSteps: {total_env_steps}) Timing (ms): Total {avg_total*1000:.1f} | Policy {avg_policy*1000:.1f} | Env {avg_env*1000:.1f} | Train {avg_train*1000:.1f} | Eval {avg_eval*1000:.1f} | EnvSteps/s: {env_steps_per_sec:.1f}")
             # Reiniciar contadores
             timer = {k: 0.0 for k in timer}

        pbar.update(1)
    
    # Final video (if not at checkpoint boundary)
    if config['training']['total_steps'] % 20000 != 0:
        eval_and_record(agent, config['env'], config['training']['total_steps'], video_dir, device, action_repeat=action_repeat)
    
    logs = {'step': config['training']['total_steps'], 'episode_reward': episode_reward, 'num_episodes': num_episodes, 'losses': losses_scalar}
    agent.save(ckpt_dir / "final.pt", logs)
    
    # Cleanup
    if prefetch_buffer is not None:
        prefetch_buffer.stop()
    env.close()
    writer.close()

if __name__ == "__main__":
    main()
