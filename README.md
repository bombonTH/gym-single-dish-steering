# gym-single-dish-steering

Repository for the implementation of deep reinforcement learning as telescope control for KMITL radio telescope.
Extend OpenAI Gym [https://github.com/openai/gym]. Continous action space, modifiable motor and antenna parameter.

# API
```
import gym
from gym-single-dish-antenna_v2 import SingleDishAntennaV2

env = gym.make("SingleDish-v2")
for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    
    if done:
        observation, info = env.reset(return_info=True)
env.close()
```
