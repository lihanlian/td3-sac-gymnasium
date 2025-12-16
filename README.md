
## Overview

This repository provide pytorch implementation of **Twin Delayed (TD3)** and **Soft Actor-Critic (SAC)** RL algorithms using gymnasium (MuJoCo) environments. Code is implemented using **Python 3.10**, **hydra-core 1.3.2** and **gymnasium 1.2.2**. More implementation details can be found at this [blog post](https://lihanlian.github.io/posts/blog9). 

**TD3 Result**
<p align="center">
  <img alt="Image 1" src="./animation/sac-ant.gif" width="20%" />
  <img alt="Image 1" src="./animation/sac-half-cheetah.gif" width="20%" />
  <img alt="Image 1" src="./animation/sac-hopper.gif" width="20%" />
  <img alt="Image 1" src="./animation/sac-walker.gif" width="20%" />
</p>

**SAC Result**
<p align="center">
  <img alt="Image 1" src="./animation/td3-ant.gif" width="20%" />
  <img alt="Image 1" src="./animation/td3-half-cheetah.gif" width="20%" />
  <img alt="Image 1" src="./animation/td3-hopper.gif" width="20%" />
  <img alt="Image 1" src="./animation/td3-walker.gif" width="20%" />
</p>

**Training Curves**

## Run Locally
After clone the repo
```bash
conda env create -n td3sac -f td3.yaml # If you use conda
```
Go to project directory

- run _train.py_ to start training using the rl algorithms specified in the train.yaml.

## References
 1. [Addressing Function Approximation Error in Actor-Critic Methods](https://proceedings.mlr.press/v80/fujimoto18a) (TD3 Paper)
 2. [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://proceedings.mlr.press/v80/haarnoja18b) (SAC Paper)
 3. [TD3](https://spinningup.openai.com/en/latest/algorithms/td3.html), [SAC](https://spinningup.openai.com/en/latest/algorithms/sac.html) (OpenAI Spinning Up) 
 4. [pytorch_sac](https://github.com/denisyarats/pytorch_sac) (PyTorch implementation of Soft Actor-Critic.)