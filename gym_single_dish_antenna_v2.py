import math

import numpy as np
from numpy import ndarray
from gym import Env, spaces
from cv2 import cv2 as cv2
from stable_baselines3 import PPO

from motor import Motor, SGMG_13A2A
from antenna import Antenna
from point import Point
from radec import RaDec

from control.PID import PIDController
# from encoder import Encoder

from gym.envs.registration import register


class SingleDishAntennaV2(Env):
    """
    ### Description
    This is an environment created for the implementation of an RL agent in low-level telescope control at KMITL.

    ### Action Space
    Two continuous actions Box(-1, +1) for primary axis motor and secondary axix motor.

    ### Observation Space
    Array of (10, 9) containing current and previous azimuth-elevation of the telescope.
    Support up to 9 targets with dwell time
    [Antenna Status, az0, az1, az2, az3, el0, el1, el2, el3]
    [Target0 dwell, az0, az1, az2, az3, el0, el1, el2, el3]
                            ......
    [Target9 dwell, az0, az1, az2, az3, el0, el1, el2, el3]

    ### Rewards
    +1 when the antenna is closer to the nearest valid target.
    -1 when the antenna is moving away.
    -1 when the antenna is out of bound.

    ### Starting State
    With random_start=True, antenna and targets will start at random positions.

    ### Episode Termination.
    1. All targets are covered.
    2. Max episode step is reached (using a `TimeLimitWrapper`)
    """

    def __init__(self, refresh_rate=1, current_time=0, random_start=False, random_torque=False, auto_render=False):
        super(SingleDishAntennaV2, self).__init__()
        self.refresh_rate = refresh_rate

        self.min_action = -1.0
        self.max_action = 1.0

        self.observation_shape = (10, 9)
        self.canvas_shape = (420, 420, 3)
        self.observation_space = spaces.Box(low=np.full(self.observation_shape, -1.0),
                                            high=np.full(self.observation_shape, 1.0), dtype=np.float64)
        self.action_space = spaces.Box(self.min_action, self.max_action, (2,), dtype=np.float64)
        self.obs = np.zeros(self.observation_shape)

        self.antenna = Antenna(random_torque=random_torque)
        self.antenna.set_x_motor(SGMG_13A2A, 1.0 / 59400)
        self.antenna.set_y_motor(SGMG_13A2A, 1.0 / 30000)

        self.targets = []
        self.obstacles = []
        self.current_time = current_time
        self.obs = self.encode()
        self.canvas = (np.zeros(self.canvas_shape) * 1).astype('uint8')
        self.random_start = random_start
        self.auto_render = auto_render
        self.zenith = Point(name='zenith', location=self.antenna.location, layer=0).update().draw(self.canvas)
        self.reward = 0
        self.rest = False

        self.font = cv2.FONT_HERSHEY_COMPLEX_SMALL

        print(f'Antenna simulator started. Location: {self.antenna.location}')
        print(f'Frame rate: {self.refresh_rate}')

    def step(self, actions: ndarray):
        actions = np.clip(actions, -1, +1).astype(np.float64)
        self.current_time += 1 / self.refresh_rate

        self.reward = 0

        for obstacle in self.obstacles:
            obstacle.update(1 / self.refresh_rate)
            obstacle.hit = obstacle.hit + 1 if self.antenna.equal(obstacle) else 0

        for target in self.targets:
            if not target.done:
                target.update(1 / self.refresh_rate)
                target.hit = target.hit + 1 if self.antenna.equal(target, 4) else 0
                target.done = target.hit >= target.dwell
                target.distance = self.antenna.distance_to(target)

                if target.done:
                    self.reward += 100

                if target.hit > 0:
                    self.reward += 1
                    print(f'Hit: {target.hit}/{target.dwell}')

        self.zenith.distance = self.antenna.distance_to(self.zenith)
        self.rest = all(target.out_of_bound for target in self.targets)

        if self.rest:
            distance = self.zenith.distance
        else:
            distance = min(target.distance for target in self.targets if not target.out_of_bound)
            
        self.reward += 1.0 if distance < self.antenna.distance else -1.0
        self.antenna.distance = distance

        self.reward -= 2.0 if self.antenna.out_of_bound else 1.0

        self.antenna.move(actions, 1000 / self.refresh_rate)
        self.antenna.obstructed = any(obstacle.hit > 0 for obstacle in self.obstacles)

        self.obs = self.encode()
        done = all(target.done for target in self.targets)

        if self.auto_render:
            self.render()

        return self.obs, self.reward, done, {'distance': self.antenna.distance}

    def add_target(self, target: Point):
        self.targets.append(target)

    def add_obstacle(self, obstacle: Point):
        self.obstacles.append(obstacle)

    def remove_target(self, target: Point):
        self.targets.remove(target)

    def remove_obstacle(self, obstacle: Point):
        self.obstacles.remove(obstacle)

    def reset(self, **kwargs):
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
        if mode == 'human':
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
            state[1 + i] = [target.dwell / 10000, tar_az[0], tar_az[1], tar_az[2], tar_az[3], tar_el[0], tar_el[1],
                            tar_el[2], tar_el[3]]
            if i >= 9:
                break
            if max(state[1 + i]) > 1.0 or min(state[1 + i]) < -1.0:
                print(state[1 + i], max(state[1 + i]), min(state[1 + i]))

        for j, obstacle in enumerate(self.obstacles):
            obs_az = [az / 2 / math.pi - 1 for az in obstacle.azs]
            obs_el = [el / math.pi * 2 for el in obstacle.els]
            # state[0][1 + j] = [-1, obs_az[0], obs_az[1], obs_az[2], obs_az[3], obs_el[0], obs_el[1], obs_el[2], obs_el[3]]
            if j >= 9:
                break
        return state

    def draw(self):
        self.canvas = np.load('canvas.npy')
        self.canvas = cv2.resize(self.canvas, (420, 420))
        self.antenna.draw(self.canvas)
        status = self.antenna.get_status()
        self.canvas = cv2.putText(self.canvas, status, self.antenna.pixel, self.font, 0.5, (255, 255, 255), 1,
                                  cv2.LINE_AA)
        self.canvas = cv2.putText(self.canvas, f'Reward: {self.reward:+06.3f}', (10, 15), self.font, 0.5,
                                  (255, 255, 255), 1, cv2.LINE_AA)
        self.canvas = cv2.putText(self.canvas, f'Distance: {self.antenna.distance:07.5f}', (10, 30), self.font, 0.5,
                                  (255, 255, 255), 1, cv2.LINE_AA)

        for obstacle in self.obstacles:
            obstacle.draw(self.canvas)

        for target in self.targets:
            target.draw(self.canvas)
            color = (255, 255, 255) if not target.out_of_bound else (0, 0, 255)
            self.canvas = cv2.putText(self.canvas, f' {target.distance:07.5f}', (target.pixel[0] - 45, target.pixel[1]),
                                      self.font, 0.5, color, 1, cv2.LINE_AA)


if __name__ == '__main__':
    print('demonstrate')
    env = SingleDishAntennaV2(random_torque=True, auto_render=False, random_start=False)
    env.add_target(RaDec(time='2021-08-16 05:00:00', dwell=3000, layer=1, radius=2, ra=190, dec=-50))
    env2 = SingleDishAntennaV2(random_torque=True, auto_render=False, random_start=False)
    env2.add_target(RaDec(time='2021-08-16 05:00:00', dwell=3000, layer=1, radius=2, ra=190, dec=-50))
    pid_x = PIDController(kp=450, ki=30.0, kd=0.5)
    pid_y = PIDController(kp=450, ki=30.0, kd=0.5)
    obs = env.reset()
    obs2 = env2.reset()
    total_reward = 0
    total_reward2 = 0
    model = PPO.load('d50e6')

    for i in range(3000):
        y_action = pid_y.update(env.targets[0].xy.y, env.antenna.xy.y)
        x_action = pid_x.update(env.targets[0].xy.x, env.antenna.xy.x)
        obs, reward, done, info = env.step(np.array([y_action, x_action]))
        if done:
            env.reset()
        actions, _state = model.predict(obs2)
        obs2, reward2, done2, info2 = env2.step(actions)
        total_reward += reward
        total_reward2 += reward2
        canvas = cv2.addWeighted(env.render('rgb_array'), 0.5, env2.render('rgb_array'), 0.5, 0)
        cv2.imshow('Game', canvas)
        cv2.waitKey(1)
register(
    # unique identifier for the env `name-version`
    id="SingleDish-v2",
    # path to the class for creating the env
    # Note: entry_point also accept a class as input (and not only a string)
    entry_point=SingleDishAntennaV2,
    # Max number of steps per episode, using a `TimeLimitWrapper`
    max_episode_steps=3000,
)
