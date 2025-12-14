# train.py
import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
import copy
import math
import os
import sys
import time
import pickle as pkl
import gymnasium as gym
from gymnasium.wrappers import RescaleAction
from video import VideoRecorder
from logger import Logger
from  replay_buffer import ReplayBuffer
import utils
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

class WorkSpace():
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        self.logger = Logger(self.work_dir, 
                             log_freq=cfg.log_freq, 
                             agent=cfg.agent.name)
        
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        # !Change here, put arguments to yaml later
        self.env = gym.make(self.cfg.experiment,
                            render_mode='rgb_array',
                            # width=256,
                            # height=256,
                            )
        # self.env = RescaleAction(self.env, min_action=-0.4, max_action=0.4)
        
        self.env_eval = gym.make(self.cfg.experiment,
                            render_mode='rgb_array',
                            # width=256,
                            # height=256,
                            )
        # self.env_eval = RescaleAction(self.env_eval, min_action=-0.4, max_action=0.4)
        # camera_id=0

        cfg.agent.obs_dim = self.env.observation_space.shape[0]
        cfg.agent.action_dim = self.env.action_space.shape[0]
        cfg.agent.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.agent = hydra.utils.instantiate(cfg.agent)

        self.replay_buffer = ReplayBuffer(self.env.observation_space.shape,
                                          self.env.action_space.shape,
                                          int(cfg.replay_buffer_capacity),
                                          self.device)
        
        self.video_recorder = VideoRecorder(self.work_dir if cfg.save_video else None)
        self.step = 0
        self.env.action_space.seed(cfg.seed)
        self.env.reset(seed=self.cfg.seed)
        self.env_eval.reset(seed=self.cfg.seed + 100)
        self.env_eval.action_space.seed(cfg.seed + 100)

        # self.max_action = float(self.env.action_space.high[0])

    def evaluate(self):
        avg_episode_reward_eval = 0
        for episode in range(self.cfg.num_eval_episodes):
            obs_eval, _ = self.env_eval.reset()
            self.agent.reset()
            # !initialize recorder here
            self.video_recorder.init(enabled=episode==0)
            done_eval = False
            episode_reward_eval = 0
            while not done_eval:
                #! Pay attention here
                self.agent.actor.eval()
                with torch.no_grad():
                    action_eval = self.agent.get_action(obs_eval, sample=False)
                self.agent.actor.train()
                obs_eval, reward_eval, terminated_eval, truncated_eval, info_eval = self.env_eval.step(action_eval)
                self.video_recorder.record(self.env_eval)
                episode_reward_eval += reward_eval
                done_eval = bool(terminated_eval or truncated_eval)

            avg_episode_reward_eval += episode_reward_eval
            self.video_recorder.save(f'{self.step}.gif')
        avg_episode_reward_eval /= self.cfg.num_eval_episodes
        self.logger.log('eval/episode_reward', 
                        avg_episode_reward_eval, self.step)
        self.logger.dump(self.step)

    def run(self):
        episode, episode_reward, done = 0, 0, True
        while self.step <= self.cfg.num_train_steps:

            # This reset the environment (done) and evaluate the agent (eval_frequency)
            # if done or self.step % self.cfg.eval_frequency == 0:
                
            # evaluate agent periodically and dump logging info (after first update)
            if self.step > self.cfg.num_seed_steps and self.step % self.cfg.eval_frequency == 0:
                self.logger.log('eval/episode', episode, self.step)
                # This use a separate environment and should not interupt the training loop
                self.evaluate()
                # first dump (save=True) create _csv_writer (to call .flush())
                self.logger.dump(self.step,save=True)

            # reset the environment only when episode is done 
            if done:
                self.logger.log('train/episode_reward', 
                                episode_reward, self.step)
                obs, _ = self.env.reset()
                self.agent.reset() #! what does this do?
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1

                self.logger.log('train/episode', episode, self.step)
            
            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            elif self.agent.name == 'td3':
                # !Need to modify later
                with torch.no_grad():
                    action = (
                        self.agent.get_action(obs)
                        + np.random.normal(0, self.agent.expl_noise, size=self.agent.action_dim)
                    ).clip(self.agent.min_action, self.agent.max_action)
            elif self.agent.name == 'sac':
                with torch.no_grad():
                    action = self.agent.get_action(obs, sample=True)
            else:
                raise ValueError(f"Invalid agent name: {self.agent.name!r}")

            # run training update
            if self.step >= self.cfg.num_seed_steps:
                # This update also create key-value pairs for logger
                self.agent.update(self.replay_buffer, self.logger, self.step)
            
            next_obs, reward, terminated, truncated, info = self.env.step(action)

            # allow infinite bootstrap
            done = bool(terminated or truncated)
            #! should use terminated?
            terminated_flag = float(terminated)
            episode_reward += reward

            self.replay_buffer.add(obs, action, reward, next_obs, terminated_flag)

            obs = next_obs
            episode_step += 1
            self.step += 1


@hydra.main(config_path="config", config_name="train", version_base="1.3")
def main(cfg: DictConfig):

    workspace = WorkSpace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
