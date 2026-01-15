import argparse
import json
import matplotlib.pyplot as plt
import pandas as pd
import pathlib

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_file", type=str, help="Path to metrics.jsonl file")
    parser.add_argument("--out", type=str, default="plots.png", help="Output image file")
    args = parser.parse_args()

    log_path = pathlib.Path(args.log_file)
    if not log_path.exists():
        print(f"File not found: {log_path}")
        return

    data = []
    with open(log_path, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except:
                pass
    
    if not data:
        print("No data found in log file.")
        return

    df = pd.DataFrame(data)
    
    # Identify unique keys (excluding step and timestamp)
    keys = set()
    for d in data:
        keys.update(d.keys())
    keys = keys - {'step', 'timestamp', 'episode'}
    
    # Group keys by category for cleaner plotting
    categories = {
        'Losses': [k for k in keys if 'loss' in k or 'grad_norm' in k],
        'Rewards': [k for k in keys if 'reward' in k],
        'Others': [k for k in keys if 'loss' not in k and 'reward' not in k]
    }
    
    # Remove empty categories
    categories = {k: v for k, v in categories.items() if v}
    
    fig, axes = plt.subplots(len(categories), 1, figsize=(12, 6 * len(categories)), sharex=True)
    if len(categories) == 1:
        axes = [axes]
        
    for ax, (cat_name, cat_keys) in zip(axes, categories.items()):
        for key in cat_keys:
            # Filter non-null values for this key
            subset = df[['step', key]].dropna()
            if not subset.empty:
                # Smoothing for noisy losses
                if 'loss' in key and len(subset) > 100:
                    subset[key] = subset[key].rolling(window=20, min_periods=1).mean()
                ax.plot(subset['step'], subset[key], label=key)
        
        ax.set_title(cat_name)
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[-1].set_xlabel("Environment Steps")
    plt.tight_layout()
    plt.savefig(args.out)
    print(f"Plots saved to {args.out}")

if __name__ == "__main__":
    main()
