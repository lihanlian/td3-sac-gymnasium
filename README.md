
## Overview

This repository provide pytorch implementation of **Twin Delayed (TD3)** and **Soft Actor-Critic (SAC)** RL algorithms using gymnasium (MuJoCo) environments. Code is implemented using **Python 3.10** and **gymnasium 1.2.2**. More implementation details can be found at this [blog post](https://lihanlian.github.io/posts/blog9). 

**TD3 Result**

**SAC Result**

<p align="center">
  <img alt="Image 1" src="./animation/ant.gif" width="20%" />
  <img alt="Image 1" src="./animation/half-cheetah.gif" width="20%" />
  <img alt="Image 1" src="./animation/hopper.gif" width="20%" />
  <img alt="Image 1" src="./animation/walker.gif" width="20%" />
</p>



## References
 1. [Addressing Function Approximation Error in Actor-Critic Methods](https://proceedings.mlr.press/v80/fujimoto18a) (TD3 Paper)
 2. [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://proceedings.mlr.press/v80/haarnoja18b) (SAC Paper)
 3. [TD3](https://spinningup.openai.com/en/latest/algorithms/td3.html), [SAC](https://spinningup.openai.com/en/latest/algorithms/sac.html) (OpenAI Spinning Up) 
 4. [pytorch_sac](https://github.com/denisyarats/pytorch_sac) (PyTorch implementation of Soft Actor-Critic.)