import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, obs_shape, action_shape, capacity: int, device):
        print(f"[ReplayBuffer.__init__] capacity={capacity}")
        self.capacity = capacity
        self.device = device

        obs_dtype = np.float32
        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.terminated_flags = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx
    
    def add(self, obs, action, reward, next_obs, terminated_flag):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.terminated_flags[self.idx], terminated_flag)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.capacity if self.full else self.idx,
                                 size = batch_size)
        
        obs_batch = torch.as_tensor(self.obses[idxs], device=self.device).float()
        action_batch = torch.as_tensor(self.actions[idxs], device=self.device)
        reward_batch = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obs_batch = torch.as_tensor(self.next_obses[idxs], device=self.device)
        terminated_flag_batch = torch.as_tensor(self.terminated_flags[idxs], device=self.device)

        return obs_batch, action_batch, reward_batch, next_obs_batch, terminated_flag_batch
