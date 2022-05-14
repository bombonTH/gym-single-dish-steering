import math

import numpy as np
from numpy import ndarray
from gym import Env, spaces
from cv2 import cv2 as cv2

from antenna import Antenna
from point import Point
#from encoder import Encoder

from gym.envs.registration import register


class SingleDishAntennaV2(Env):
    def __init__(self, refresh_rate=1, current_time=0, random_start=False, random_torque=False):
        super(SingleDishAntennaV2, self).__init__()
        self.refresh_rate = refresh_rate

        self.min_action = -1.0
        self.max_action = 1.0

        self.observation_shape = (10, 9)
        self.canvas_shape = (420, 420, 3)
        self.observation_space = spaces.Box(low=np.full(self.observation_shape, -1.0),high=np.full(self.observation_shape, 1.0), dtype=np.float64)
        self.action_space = spaces.Box(self.min_action, self.max_action, (2,), dtype=np.float64)
        self.obs = np.zeros(self.observation_shape)

        self.antenna = Antenna(random_torque=random_torque)
        self.targets = []
        self.obstacles = []
        self.current_time = current_time
        self.obs = self.encode()
        self.canvas = (np.zeros(self.canvas_shape) * 1).astype('uint8')
        self.random_start=random_start
        self.zenith = Point(name='zenith', location = self.antenna.location, layer=0).update().draw(self.canvas)
        self.reward = 0
        
        self.font = cv2.FONT_HERSHEY_COMPLEX_SMALL

        print(f'Antenna simulator started. Location: {self.antenna.location}')
        print(f'Frame rate: {self.refresh_rate}')

    def step(self, actions: ndarray):
        self.current_time += 1 / self.refresh_rate

        self.reward = 0

        for obstacle in self.obstacles:
            obstacle.update(1 / self.refresh_rate)
            obstacle.hit = obstacle.hit + 1 if self.antenna.equal(obstacle) else 0


        for target in self.targets:
            if not target.done:
                target.update(1 / self.refresh_rate)
                target.hit = target.hit + 1 if self.antenna.equal(target, 3) else 0
                target.done = target.hit >= target.dwell
                target.distance = self.antenna.distance_to(target)
                
                if target.done:
                    self.reward += 100
                
                if target.hit > 0:
                    self.reward += 1
                    print(f'Hit: {target.hit}/{target.dwell}')      
         
        if all(target.out_of_bound for target in self.targets):
            distance = self.antenna.distance_to(self.zenith)
            if distance < self.zenith.distance:
                self.reward += self.calculate_reward(distance) / 10
            self.zenith.distance = distance
        else:
            distance = min(target.distance for target in self.targets if not target.out_of_bound)
            if distance < self.antenna.distance:
                self.reward += 1 #self.calculate_reward(distance)
            else:
                self.reward -= 1
            self.antenna.distance = distance
        self.reward -= self.calculate_penalty()

        self.antenna.move(actions, 1000 / self.refresh_rate)
        self.antenna.obstructed = any(obstacle.hit > 0 for obstacle in self.obstacles)
        
        self.obs = self.encode()
        done = all(target.done for target in self.targets)
        self.render()
                
        return self.obs, self.reward, done, {'distance': self.antenna.distance}

    def calculate_penalty(self):
        return 1.0 if self.antenna.out_of_bound else 0.0
    
    def calculate_reward(self, distance):
        return 0.1 / (distance*10 + 1)

    def add_target(self, target: Point):
        self.targets.append(target)

    def add_obstacle(self, obstacle: Point):
        self.obstacles.append(obstacle)

    def remove_target(self, target: Point):
        self.targets.remove(target)

    def remove_obstacle(self, obstacle: Point):
        self.obstacles.remove(obstacle)

    def reset(self):
        self.obs = self.encode()
        self.current_time = 0
        self.antenna.reset(random=self.random_start)
        self.zenith.reset()
        for target in self.targets:
            target.reset(random=self.random_start)
        for obstacle in self.obstacles:
            obstacle.reset(random=self.random_start)
        return self.obs

    def render(self, mode='human'):
        assert mode in ['human', 'rgb_array'], 'Invalid mode, must be either \'human\' or \'rgb_array\''
        self.draw()  # update the canvas
        if mode == 'human' or True:
            cv2.imshow('Game', self.canvas)
            cv2.waitKey(1)
        return self.canvas

    def encode(self):
        state = np.zeros([10, 9])
        ant_az = [az / 2 / math.pi - 1 for az in self.antenna.azs]
        ant_el = [el / math.pi * 2 for el in self.antenna.els]
        state[0] = [0, ant_az[0], ant_az[1], ant_az[2], ant_az[3], ant_el[0], ant_el[1], ant_el[2], ant_el[3]]
        for i, target in enumerate(self.targets):
            if target.done:
                continue
            tar_az = [az / 2 / math.pi - 1 for az in target.azs]
            tar_el = [el / math.pi * 2 for el in target.els]
            state[1 + i] = [target.dwell/100, tar_az[0], tar_az[1], tar_az[2], tar_az[3], tar_el[0], tar_el[1], tar_el[2], tar_el[3]]            
            if i >= 9:
                break
            if max(state[1 + i]) > 1.0 or min(state[1 + i]) < -1.0:
                print(state[1 + i], max(state[1 + i]), min(state[1 + i]))

        for j, obstacle in enumerate(self.obstacles):
            obs_az = [az / 2 / math.pi - 1 for az in obstacle.azs]
            obs_el = [el / math.pi * 2 for el in obstacle.els]
            #state[0][1 + j] = [-1, obs_az[0], obs_az[1], obs_az[2], obs_az[3], obs_el[0], obs_el[1], obs_el[2], obs_el[3]]
            if j >= 9:
                break
        return state

    def draw(self):
        self.canvas = np.load('canvas.npy')
        self.canvas = cv2.resize(self.canvas, (420, 420))
        self.antenna.draw(self.canvas)
        status = self.antenna.get_status()
        self.canvas = cv2.putText(self.canvas, status, self.antenna.pixel, self.font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        self.canvas = cv2.putText(self.canvas, f'Reward: {self.reward:+06.3f}', (10, 15), self.font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        self.canvas = cv2.putText(self.canvas, f'Distance: {self.antenna.distance:+06.3f}', (10, 30), self.font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        for obstacle in self.obstacles:
            obstacle.draw(self.canvas)

        for target in self.targets:
            target.draw(self.canvas)
            color = (255, 255, 255) if not target.out_of_bound else (0, 0, 255)
            self.canvas = cv2.putText(self.canvas, f' {target.distance:06.4f}', (target.pixel[0] - 45, target.pixel[1]), self.font, 0.5, color, 1, cv2.LINE_AA)


register(
    # unique identifier for the env `name-version`
    id="SingleDish-v2",
    # path to the class for creating the env
    # Note: entry_point also accept a class as input (and not only a string)
    entry_point=SingleDishAntennaV2,
    # Max number of steps per episode, using a `TimeLimitWrapper`
    max_episode_steps=3000,
)
