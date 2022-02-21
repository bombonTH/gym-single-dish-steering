import math

import numpy as np
from cv2 import cv2 as cv2
from gym import Env, spaces
from numpy import ndarray

from antenna import Antenna
from point import Point


class SingleDishAntenna(Env):
    def __init__(self, refresh_rate=1, current_time=0):
        super(SingleDishAntenna, self).__init__()
        self.refresh_rate = refresh_rate
        self.episode_length = 0

        self.min_action = -1.0
        self.max_action = 1.0

        self.observation_shape = (84, 84, 3)  # (Height, Width, Channel)
        self.observation_space = spaces.Box(low=np.zeros(self.observation_shape),
                                            high=np.full(self.observation_shape, 255),
                                            dtype=np.uint8)

        self.action_space = spaces.Box(low=self.min_action, high=self.max_action, shape=(2,), dtype=np.float32)

        self.canvas = np.zeros(self.observation_shape) * 1
        self.overlay = np.zeros(self.observation_shape) * 1
        self.font = cv2.FONT_HERSHEY_COMPLEX_SMALL

        self.antenna = Antenna()
        self.antenna.set_position(0, 0)
        self.elements = []
        self.targets = []
        self.obstacles = []
        self.last_action = [0, 0]
        self.current_time = current_time
        print(f'Antenna simulator started. Location: {self.antenna.name} [{self.antenna.location}]')
        print(f'Frame rate: {self.refresh_rate}')

    def step(self, action: ndarray):
        # TODO current time, show the elements posit after refresh, update current time.
        self.episode_length += 1
        self.antenna.move(action, 1000 / self.refresh_rate)

        reward = -1.0 if self.antenna.out_of_bound else 0.0

        for obstacle in self.obstacles:
            obstacle.update(1 / self.refresh_rate)
        self.antenna.obstructed = self.check_obstructed(0.1)

        for target in self.targets:
            target.update(1 / self.refresh_rate)
            target.out_of_bound = not(-math.pi/3 < target.ns < math.pi/3 and -math.pi/3 < target.ew < math.pi/3)

            if not (self.antenna.obstructed and target.done and target.out_of_bound):
                current_reward = self.calculate_reward(target)
                if current_reward > 0.9: print(f'Reward: {current_reward:.2f}')
                target.done = target.hit >= 120
                reward += current_reward

        reward -= self.calculate_penalty(action) / 100
        reward -= 0.001
        #print(f'Time penalty: {self.episode_length * 0.001:.3f}')
        #print(f'Penalty for action: {self.calculate_penalty(action) / 50:.3f}')
        done = all(target.out_of_bound or target.done for target in self.targets)
        self.last_action = action
        self.render(mode='rgb_array')
        return self.canvas, reward, done, {}

    def check_obstructed(self, distance=0.1):
        ant_position = self.antenna.get_position()
        return any([math.dist(ant_position, obs.get_position()) < distance for obs in self.obstacles])

    def calculate_reward(self, target: Point):
        distance = math.dist(self.antenna.get_position(), target.get_position())
        target.hit += 1 if distance < 0.01 else 0
        return 1 - np.tanh(distance*5)

    def calculate_penalty(self, action):
        penalty = abs(action[0] - self.last_action[0]) + abs(action[1] - self.last_action[1])
        return penalty

    def add_target(self, target: Point):
        self.targets.append(target)

    def remove_target(self, target: Point):
        self.targets.remove(target)

    def add_obstacle(self, obstacle: Point):
        self.obstacles.append(obstacle)

    def remove_obstacle(self, obstacle: Point):
        self.obstacles.remove(obstacle)

    def reset(self, exploration_start=True):
        self.episode_length = 0
        self.antenna.reset(random=exploration_start)
        for target in self.targets:
            target.reset(random=exploration_start)
        for obstacle in self.obstacles:
            obstacle.reset(random=exploration_start)
        self.draw()
        return self.canvas

    def render(self, mode='human'):
        assert mode in ['human', 'rgb_array'], 'Invalid mode, must be either \'human\' or \'rgb_array\''
        self.draw()  # update the canvas
        if mode == 'human' or True:
            image = cv2.resize(self.canvas, (420, 420))
            cv2.imshow('Game', image)
            cv2.waitKey(1)
        return self.canvas

    def render_overlay(self):
        image = cv2.resize(self.canvas, (420, 420))
        self.draw_overlay()
        hud = cv2.add(image, self.overlay)
        return hud

    def draw(self):
        self.canvas = (np.zeros(self.observation_shape) * 1).astype('uint8')
        self.antenna.draw(self.canvas)
        for target in self.targets:
            target.draw(self.canvas)
        for obstacle in self.obstacles:
            obstacle.draw(self.canvas)

    def draw_overlay(self):
        self.overlay = (np.zeros(self.observation_shape) * 1).astype('uint8')
        self.overlay = cv2.resize(self.overlay, (420, 420))
        status = self.antenna.get_status()
        self.overlay = cv2.putText(self.overlay, status, (10, 20), self.font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        for index, target in enumerate(self.targets):
            self.overlay = cv2.putText(self.overlay, target.get_status(), (10, (index + 2) * 20), self.font, 0.5,
                                       (255, 255, 255), 1, cv2.LINE_AA)
