import math

import numpy as np
from numpy import ndarray
from gym import Env, spaces
from cv2 import cv2 as cv2

from antenna import Antenna
from point import Point
from encoder import Encoder

from gym.envs.registration import register


class SingleDishAntennaV2(Env):
    def __init__(self, refresh_rate=1, current_time=0):
        super(SingleDishAntennaV2, self).__init__()
        self.refresh_rate = refresh_rate

        self.min_action = -1.0
        self.max_action = 1.0

        self.observation_shape = 8
        self.canvas_shape = (420, 420, 3)
        self.observation_space = spaces.Box(low=np.full(self.observation_shape, -1),
                                            high=np.ones(self.observation_shape))
        self.action_space = spaces.Box(low=self.min_action, high=self.max_action, shape=(2,))
        self.obs = np.zeros(self.observation_shape)

        self.antenna = Antenna()
        self.targets = []
        self.obstacles = []
        self.current_time = current_time
        self.encoder = Encoder().load_model()
        self.obs = self.encode()
        self.canvas = (np.zeros(self.canvas_shape) * 1).astype('uint8')

        print(f'Antenna simulator started. Location: {self.antenna.location}')
        print(f'Frame rate: {self.refresh_rate}')

    def step(self, actions: ndarray):
        self.current_time += 1 / self.refresh_rate
        self.antenna.move(actions, 1000 / self.refresh_rate)

        reward = 0

        for obstacle in self.obstacles:
            obstacle.update(1 / self.refresh_rate)
            obstacle.hit = obstacle.hit + 1 if self.antenna.equal(obstacle) else 0

        self.antenna.obstructed = any(obstacle.hit > 0 for obstacle in self.obstacles)

        for target in self.targets:
            if not target.done:
                target.update(1 / self.refresh_rate)
                target.hit = target.hit + 1 if self.antenna.equal(target, 2) and self.antenna.same_pace(target, 2) else 0
                target.done = target.hit >= 10
                if target.done:
                    reward += 100
                if target.hit > 0:
                    print(f'Hit: {target.hit}')
        if not self.antenna.obstructed:
            reward += sum([1 for target in self.targets if target.hit > 0])

        self.obs = self.encode()
        reward -= self.calculate_penalty()
        done = all(target.out_of_bound or target.done for target in self.targets)
        self.render()
        return self.obs, reward, done, {}

    def calculate_penalty(self):
        penalty = 0.1
        penalty += 1 if self.antenna.out_of_bound else 0.0
        return penalty

    def add_target(self, target: Point):
        self.targets.append(target)

    def add_obstacle(self, obstacle: Point):
        self.obstacles.append(obstacle)

    def remove_target(self, target: Point):
        self.targets.remove(target)

    def remove_obstacle(self, obstacle: Point):
        self.obstacles.remove(obstacle)

    def reset(self, exploration_start=True):
        self.obs = self.encode()
        self.current_time = 0
        self.antenna.reset(random=exploration_start)
        for target in self.targets:
            target.reset(random=exploration_start)
        for obstacle in self.obstacles:
            obstacle.reset(random=exploration_start)
        return self.obs

    def render(self, mode='human'):
        assert mode in ['human', 'rgb_array'], 'Invalid mode, must be either \'human\' or \'rgb_array\''
        self.draw()  # update the canvas
        if mode == 'human' or True:
            cv2.imshow('Game', self.canvas)
            cv2.waitKey(1)
        return self.canvas

    def encode(self):
        state = np.zeros([1, 21, 5])
        state[0][0] = [1, self.antenna.az, self.antenna.el, self.antenna.az_rate, self.antenna.el_rate]
        for i, target in enumerate(self.targets):
            state[0][1 + i] = [0 , target.az, target.el, target.az_rate, target.el_rate]
            if i >= 9:
                break
        for j, obstacle in enumerate(self.obstacles):
            state[0][11 + j] = [-1, obstacle.az, obstacle.el, obstacle.az_rate, obstacle.el_rate]
            if j >= 9:
                break
        #obs = self.encoder.encode(state)
        if len(self.targets) > 0:
            target = self.targets[0]
            azimuth = target.az
            elevation = target.el
            az_rate = target.az_rate
            el_rate = target.el_rate
        else:
            azimuth = 0
            elevation = 0
            az_rate = 0
            el_rate = 0
        obs = [self.antenna.az, self.antenna.el, self.antenna.az_rate, self.antenna.el_rate, azimuth, elevation, az_rate, el_rate]
        return obs

    def draw(self):
        self.canvas = np.load('canvas.npy')
        self.canvas = cv2.resize(self.canvas, (420, 420))
        self.antenna.draw(self.canvas)

        for obstacle in self.obstacles:
            obstacle.draw(self.canvas)

        for target in self.targets:
            target.draw(self.canvas)


register(
    # unique identifier for the env `name-version`
    id="SingleDish-v2",
    # path to the class for creating the env
    # Note: entry_point also accept a class as input (and not only a string)
    entry_point=SingleDishAntennaV2,
    # Max number of steps per episode, using a `TimeLimitWrapper`
    max_episode_steps=600,
)
