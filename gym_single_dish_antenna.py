import numpy as np
from cv2 import cv2 as cv2
from gym import Env, spaces
from numpy import ndarray

from antenna import Antenna
from point import Point


class SingleDishAntenna(Env):
    def __init__(self, refresh_rate=25, current_time=0):
        super(SingleDishAntenna, self).__init__()
        self.refresh_rate = refresh_rate

        self.min_action = -1.0
        self.max_action = 1.0

        self.observation_shape = (900, 900, 3)  # (Height, Width, Channel)
        self.observation_space = spaces.Box(low=np.zeros(self.observation_shape),
                                            high=np.ones(self.observation_shape),
                                            dtype=np.float64)

        self.action_space = spaces.Box(low=self.min_action, high=self.max_action, shape=(2,), dtype=np.float64)

        self.canvas = np.zeros(self.observation_shape) * 1
        self.overlay = np.zeros(self.observation_shape) * 1
        self.font = cv2.FONT_HERSHEY_COMPLEX_SMALL

        self.antenna = Antenna()
        self.antenna.set_position(0, 0)
        self.elements = []
        self.targets = []
        self.current_time = current_time

    def step(self, action: ndarray):
        # TODO current time, show the elements posit after refresh, update current time.
        self.antenna.move(action, 1000 / self.refresh_rate)
        for target in self.targets:
            target.update(1/self.refresh_rate)
        self.render(mode='rgb_array')
        reward = None  # TODO negative reward if max angle is surpassed. preferably call another function.
        done = False
        return self.canvas, reward, done

    def _calculate_reward(self):
        pass

    def add_target(self, target: Point):
        self.targets.append(target)

    def remove_target(self, target: Point):
        self.targets.remove(target)

    def reset(self, exploration_start=False):
        self.antenna = Antenna()
        if exploration_start:
            # TODO random antenna position np.array([y, x]) [-60, 60] deg
            pass

    def render(self, mode='human'):
        assert mode in ['human', 'rgb_array'], 'Invalid mode, must be either \'human\' or \'rgb_array\''
        self.draw()  # update the canvas
        if mode == 'human':
            self.draw_overlay()
            hud = cv2.add(self.canvas, self.overlay)
            #cv2.imshow('Game', hud)
            #cv2.waitKey(1)
            # cv2.waitKey(int(1000 / self.refresh_rate))
            return hud
        return self.canvas

    def draw(self):
        self.canvas = np.zeros(self.observation_shape) * 1
        self.antenna.draw(self.canvas)
        for target in self.targets:
            target.draw(self.canvas)
            # TODO draw each element in their respective layer

    def draw_overlay(self):
        self.overlay = np.zeros(self.observation_shape) * 1
        status = self.antenna.get_status()
        self.overlay = cv2.putText(self.overlay, status, (10, 20), self.font, 0.8, (1, 1, 1), 1, cv2.LINE_AA)
        for target in self.targets:
            self.overlay = cv2.putText(self.overlay, target.get_status(), (10, 40), self.font, 0.8, (1, 1, 1), 1,
                                       cv2.LINE_AA)
            # TODO draw info on the overlay layer.
