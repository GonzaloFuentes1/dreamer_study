"""
Prefetch buffer for loading training data in parallel with training.

This allows data sampling to happen asynchronously, reducing the overhead
of data loading during training steps.
"""

import torch
import threading
import queue
from typing import Tuple


class PrefetchBuffer:
    """
    Asynchronous data prefetcher that loads batches in a background thread.
    
    Similar to TensorFlow's dataset.prefetch(), this allows data loading
    to overlap with training computation.
    """
    
    def __init__(self, replay_buffer, batch_size: int, seq_length: int, 
                 device: torch.device, buffer_size: int = 3):
        """
        Args:
            replay_buffer: The replay buffer to sample from
            batch_size: Batch size for sampling
            seq_length: Sequence length for sampling
            device: Device to move data to
            buffer_size: Number of batches to prefetch (default: 3)
        """
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.device = device
        self.buffer_size = buffer_size
        
        self.queue = queue.Queue(maxsize=buffer_size)
        self.thread = None
        self.stop_flag = threading.Event()
        
    def _worker(self):
        """Background worker that loads data into the queue."""
        while not self.stop_flag.is_set():
            try:
                # Sample batch from replay buffer
                # ParallelReplayBuffer needs device arg, regular ReplayBuffer doesn't
                try:
                    batch = self.replay_buffer.sample(self.batch_size, self.seq_length, self.device)
                    # ParallelReplayBuffer returns Dict with tensors already on device
                    batch_device = {k: v for k, v in batch.items()}
                except TypeError:
                    # Regular ReplayBuffer doesn't take device argument
                    batch = self.replay_buffer.sample(self.batch_size, self.seq_length)
                    # Move to device
                    batch_device = {}
                    for k, v in batch.items():
                        if isinstance(v, torch.Tensor):
                            batch_device[k] = v.to(self.device, non_blocking=True)
                        else:
                            batch_device[k] = torch.from_numpy(v).to(self.device, non_blocking=True)
                
                # Put in queue (blocks if queue is full)
                self.queue.put(batch_device, timeout=1.0)
                
            except queue.Full:
                # Queue is full, wait a bit
                continue
            except Exception as e:
                if not self.stop_flag.is_set():
                    print(f"Error in prefetch worker: {e}")
                break
                
    def start(self):
        """Start the prefetch worker thread."""
        if self.thread is not None and self.thread.is_alive():
            return
            
        self.stop_flag.clear()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        
    def get_batch(self) -> dict:
        """
        Get a prefetched batch. Blocks if no batch is available.
        
        Returns:
            Dictionary with batch data already on the target device
        """
        try:
            return self.queue.get(timeout=5.0)
        except queue.Empty:
            raise RuntimeError("Prefetch queue is empty - worker may have stopped")
            
    def stop(self):
        """Stop the prefetch worker thread."""
        self.stop_flag.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)
            
        # Clear any remaining items in queue
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break
                
    def __del__(self):
        self.stop()
