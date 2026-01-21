import sys
import glob
import os
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def analyze_log(log_dir, version_name):
    print(f"--- Analysis for {version_name} ---")
    event_files = glob.glob(os.path.join(log_dir, 'events.out.tfevents.*'))
    if not event_files:
        print(f"No event files found in {log_dir}")
        return

    # Load the event accumulator
    ea = EventAccumulator(event_files[0])
    ea.Reload()
    
    tags = ea.Tags()['scalars']
    
    if 'episode/reward' in tags:
        rewards = ea.Scalars('episode/reward')
        steps = [x.step for x in rewards]
        values = [x.value for x in rewards]
        
        print(f"Total Episodes Logged: {steps[-1] if steps else 0}")
        if values:
            print(f"First Reward: {values[0]:.2f}")
            print(f"Last Reward: {values[-1]:.2f}")
            print(f"Max Reward: {max(values):.2f}")
            print(f"Avg Reward (Last 10): {np.mean(values[-10:]):.2f}")
            
            # Simple ASCII plot
            print("Recent Reward Trend (Last 10):")
            for v in values[-10:]:
                print(f"{v:.2f}", end=" ")
            print("\n")

    if 'train/wm_loss' in tags:
        wm_loss = ea.Scalars('train/wm_loss')
        train_steps = [x.step for x in wm_loss]
        if train_steps:
             print(f"Total Training Steps (Batches): {train_steps[-1]}")
             # Estimate total env steps
             # Assuming 16 envs, action repeat 2, and 1 batch per loop step (roughly)
             # Note: logic in train.py logs at 'step + train_step_idx'. 
             # If train_steps_per_iter is small (e.g. 1 or 2), this is roughly loop_steps * train_ratio
             print(f"Last Training Loss Step: {train_steps[-1]}")
            
    if 'eval/reward' in tags:
        eval_rewards = ea.Scalars('eval/reward')
        eval_values = [x.value for x in eval_rewards]
        if eval_values:
             print(f"Last Eval Reward: {eval_values[-1]:.2f}")
             print(f"Max Eval Reward: {max(eval_values):.2f}")

    if not values and not eval_values:
        print("No reward data found.")
    print("-" * 30)

if __name__ == "__main__":
    # Hardcoded paths based on exploration
    base_path = "/workspace1/gonzalo.fuentes/dreamer_study/runs"
    
    # We found these specific timestamped folders earlier
    # v1
    v1_path = os.path.join(base_path, "v1/v1_walker_walk_20260121-004027/logs")
    analyze_log(v1_path, "Dreamer v1")
    
    # v2
    v2_path = os.path.join(base_path, "v2/v2_walker_walk_20260121-004019/logs")
    analyze_log(v2_path, "Dreamer v2")
    
    # v3
    v3_path = os.path.join(base_path, "v3/v3_walker_walk_20260121-070015/logs")
    analyze_log(v3_path, "Dreamer v3")
