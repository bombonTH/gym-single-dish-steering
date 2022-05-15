import math

import numpy as np
from cv2 import cv2 as cv2
from gym import Env, spaces
from numpy import ndarray

from antenna import Antenna
from point import Point

from gym.envs.registration import register


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

        for target in self.targets:
            target.update(1 / self.refresh_rate)
            target.out_of_bound = not (
                    -math.pi / 3 < target.ns < math.pi / 3 and -math.pi / 3 < target.ew < math.pi / 3)

            if not (target.obstructed or target.done or target.out_of_bound):
                current_reward = self.calculate_reward(target)
                target.done = target.hit >= 5
                if target.hit > 0:
                    reward += 10
                    print(f'Hit: {target.hit} times')
                reward += current_reward * 0
                reward += (100 if target.done else 0)

        reward -= self.calculate_penalty(action) / 10
        # reward -= 0.001
        # print(f'Time penalty: {self.episode_length * 0.001:.3f}')
        # print(f'Penalty for action: {self.calculate_penalty(action) / 10:.2f}')
        done = all(target.out_of_bound or target.done for target in self.targets)
        reward += (1000 if all(target.done for target in self.targets) else 0)
        self.last_action = action
        self.render(mode='rgb_array')

        return self.canvas, reward, done, {}

    def calculate_reward(self, target: Point):
        distance = math.dist(self.antenna.get_position(), target.get_position())
        target.hit = (target.hit + 1 if distance < 0.01 else 0)
        return 1 - np.tanh(distance * 90)

    def calculate_penalty(self, action):
        penalty_0 = abs(action[0] - self.last_action[0]) if abs(action[0] - self.last_action[0]) > 1 else 0
        penalty_1 = abs(action[1] - self.last_action[1]) if abs(action[1] - self.last_action[1]) > 1 else 0
        penalty = penalty_0 + penalty_1
        return penalty * 0

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
            target.reset()
        for obstacle in self.obstacles:
            obstacle.reset()
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
        self.draw_speed_bar()

        for obstacle in self.obstacles:
            obstacle.draw(self.canvas)
            if obstacle.get_distance(self.antenna) < math.pi / 18:
                self.draw_bar(obstacle)

        for target in self.targets:
            target.draw(self.canvas)
            if target.get_distance(self.antenna) < math.pi / 18:
                self.draw_bar(target)

    def draw_overlay(self):
        self.overlay = (np.zeros(self.observation_shape) * 1).astype('uint8')
        self.overlay = cv2.resize(self.overlay, (420, 420))
        status = self.antenna.get_status()
        self.overlay = cv2.putText(self.overlay, status, (10, 20), self.font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        for index, target in enumerate(self.targets):
            self.overlay = cv2.putText(self.overlay, target.get_status(), (10, (index + 2) * 20), self.font, 0.5,
                                       (255, 255, 255), 1, cv2.LINE_AA)

    def draw_speed_bar(self):
        canvas = self.canvas[:, :, 0]
        h = self.canvas.shape[0]
        w = self.canvas.shape[1]
        # y = math.floor(-self.antenna.ns_speed / 0.012 * h + h / 2)
        # x = math.floor(self.antenna.ew_speed / 0.012 * w + w / 2)
        y = math.floor(-self.last_action[0] * 40)
        x = math.floor(self.last_action[1] * 40)
        canvas[0, x] = canvas[0, x + 1] = 255
        canvas[y, 0] = canvas[y + 1, 0] = 255

    def draw_bar(self, point: Point):
        canvas = self.canvas[:, :, point.layer]
        bar_lim = 4 * math.pi / 180  # 4 degrees
        amp_factor = math.pi / bar_lim
        radius = point.radius
        h = self.canvas.shape[0]
        w = self.canvas.shape[1]
        x = math.floor((point.ew - self.antenna.ew) / bar_lim * w - 0.5 + w / 2)
        y = math.floor((self.antenna.ns - point.ns) / bar_lim * h - 0.5 + h / 2)
        # if 0 < y < h - 1:
        #     canvas[y, w - 1] = canvas[y + 1, w - 1] = 255
        # if 0 < x < w - 1:
        #     canvas[h - 1, x] = canvas[h - 1, x + 1] = 255
        for k in range(max(0, math.floor(y - radius)), min(h, math.ceil(y + radius))):
            distance = math.dist((k, x), (y, x))
            pixel = int(192 * (1 - distance / radius)) if distance < radius else 0
            pixel = min(pixel, 255 - canvas[k, w - 1])
            canvas[k, w - 1] = pixel + canvas[k, w - 1]
        for j in range(max(0, math.floor(x - radius)), min(w, math.ceil(x + radius))):
            distance = math.dist((y, j), (y, x))
            pixel = int(192 * (1 - distance / radius)) if distance < radius else 0
            pixel = min(pixel, 255 - canvas[h - 1, j])
            canvas[h - 1, j] = pixel + canvas[h - 1, j]


register(
    # unique identifier for the env `name-version`
    id="SingleDish-v1",
    # path to the class for creating the env
    # Note: entry_point also accept a class as input (and not only a string)
    entry_point=SingleDishAntenna,
    # Max number of steps per episode, using a `TimeLimitWrapper`
    max_episode_steps=1800,
)
