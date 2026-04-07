import numpy as np
import asyncio
import os

class AsyncReplayBuffer:
    """
    Asynchronous, thread-safe replay buffer for storing HVAC environment transitions.
    """
    def __init__(self, capacity=100000, obs_dim=15, action_dim=1):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.lock = asyncio.Lock()
        
        # Buffer arrays
        self.states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.int32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.bool_)
        
        # Storage for metadata (logging/stats)
        self.metadata = []

    async def add(self, state, action, reward, next_state, done, metadata: dict):
        async with self.lock:
            # Add to main buffer
            self.states[self.ptr] = state
            self.actions[self.ptr] = action
            self.rewards[self.ptr] = reward
            self.next_states[self.ptr] = next_state
            self.dones[self.ptr] = done
            
            # Update metadata (replaces old metadata if capacity exceeded)
            if self.size < self.capacity:
                self.metadata.append(metadata)
            else:
                self.metadata[self.ptr] = metadata
                
            self.ptr = (self.ptr + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    async def sample(self, batch_size=256):
        async with self.lock:
            if self.size < batch_size:
                return None
            
            indices = np.random.randint(0, self.size, size=batch_size)
            return {
                "states": self.states[indices],
                "actions": self.actions[indices],
                "rewards": self.rewards[indices],
                "next_states": self.next_states[indices],
                "dones": self.dones[indices]
            }

    async def save(self, filepath):
        async with self.lock:
            np.savez(filepath, 
                     states=self.states[:self.size], 
                     actions=self.actions[:self.size],
                     rewards=self.rewards[:self.size],
                     next_states=self.next_states[:self.size],
                     dones=self.dones[:self.size])

    async def load(self, filepath):
        async with self.lock:
            data = np.load(filepath)
            size = len(data['states'])
            self.states[:size] = data['states']
            self.actions[:size] = data['actions']
            self.rewards[:size] = data['rewards']
            self.next_states[:size] = data['next_states']
            self.dones[:size] = data['dones']
            self.size = size
            self.ptr = size % self.capacity

    def __len__(self):
        return self.size

    def is_ready(self, min_samples=1000):
        return self.size >= min_samples

    def get_statistics(self):
        if not self.metadata:
            return {"status": "empty"}
        
        # Basic stats across recent transitions
        recent_meta = self.metadata[-1000:]
        rewards = [m.get('reward', 0) for m in recent_meta]
        pmvs = [m.get('pmv', 0) for m in recent_meta]
        energies = [m.get('hvac_power', 0) for m in recent_meta]
        
        return {
            "mean_reward": float(np.mean(rewards)),
            "mean_pmv": float(np.mean(pmvs)),
            "mean_energy_w": float(np.mean(energies)),
            "total_samples": self.size
        }
