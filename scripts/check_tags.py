from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import sys
import glob
import os

def list_tags(log_dir):
    event_files = glob.glob(os.path.join(log_dir, 'events.out.tfevents.*'))
    if not event_files:
        print(f"No event files found in {log_dir}")
        return

    ea = EventAccumulator(event_files[0])
    ea.Reload()
    print(f"Tags in {log_dir}:")
    print(ea.Tags())

if __name__ == "__main__":
    if len(sys.argv) > 1:
        list_tags(sys.argv[1])
    else:
        print("Please provide a log directory.")