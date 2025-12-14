import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from agent import Agent
import utils
# import hydra

class TD3Agent(Agent):
    """
    Fully-recursive style: Hydra will instantiate critic/actor/replay first
    and pass REAL OBJECTS into this constructor.
    No hydra.utils.instantiate(...) here.
    """
    def __init__(
        self, obs_dim, action_dim, action_range, device, critic,
        actor, discount, actor_lr, actor_betas, 
        critic_lr, critic_betas, tau, 
        policy_update_frequency, expl_noise, policy_noise,
        noise_clip, batch_size, name):

        self.action_range = action_range
        # ! New added variables
        self.min_action = action_range[0]
        self.max_action = action_range[1]

        self.action_dim = action_dim
        self.device = torch.device(device)
        self.discount = discount
        self.tau = tau
        self.policy_update_frequency = policy_update_frequency
        self.batch_size = batch_size

        self.critic = critic.to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        # freeze & set eval (targets never optimized; no BN updates)
        for p in self.critic_target.parameters():
            p.requires_grad = False
        self.critic_target.eval()

        self.actor = actor.to(self.device)
        self.actor_target = copy.deepcopy(self.actor).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())


        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr,
                                                betas=actor_betas)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr,
                                                 betas=critic_betas)
        
        self.train()
        self.expl_noise = expl_noise * self.max_action
        self.policy_noise = policy_noise * self.max_action
        self.noise_clip = noise_clip * self.max_action
        self.name = name

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()
    
    def get_action(self, obs, sample=False):
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)
        action = self.actor(obs)

        assert action.ndim == 2 and action.shape[0] == 1
        return utils.to_np(action[0])
    
    def update_critic(self, obs, action, reward, next_obs, 
                      terminated_flag, logger, step):
        # Calculate the target Q value
        with torch.no_grad():
            # ! start below
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)	
            next_action = (self.actor_target(next_obs) + noise).clamp(self.min_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1-terminated_flag) * self.discount * target_Q

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) +\
              F.mse_loss(current_Q2, target_Q)
        logger.log('train_critic/loss', critic_loss, step)

        # Optimzer the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update_actor(self, obs, logger, step):
        action = self.actor(obs)
        actor_Q1, actor_Q2 = self.critic(obs, action)

        # !only select Q1 based on the algorithm
        actor_loss = -actor_Q1.mean()

        logger.log('train_actor/loss', actor_loss, step)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def update(self, replay_buffer, logger, step):
        obs, action, reward, next_obs, terminated_flag = replay_buffer.sample(self.batch_size)
        logger.log('train/batch_reward', reward.mean(), step)
        self.update_critic(obs, action, reward, 
                           next_obs, terminated_flag, logger, step)

        if step % self.policy_update_frequency == 0:
            self.update_actor(obs, logger, step)

            # update both target actor and target critic networks
            utils.soft_update_params(self.critic, self.critic_target, self.tau)
            utils.soft_update_params(self.actor, self.actor_target, self.tau)

