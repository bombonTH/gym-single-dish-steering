import math

import cv2
import numpy as np
from gym import Env, spaces
from numpy import ndarray

font = cv2.FONT_HERSHEY_COMPLEX_SMALL


class Motor(object):
    def __init__(self, max_rpm, induction_time, motor_inertia, motor_torque):
        pass


class Point(object):
    def __init__(self, name: str, layer: int, location: (float, float, float), radius: int = 10):
        self.name = name
        self.layer = layer
        self.location = location
        self.ns = 0
        self.ew = 0
        self.radius = radius

    def set_position(self, ns: float, ew: float):
        self.ns = ns
        self.ew = ew

    def get_position(self):
        return self.ns, self.ew

    def draw(self, canvas):
        canvas = canvas[:, :, self.layer]
        h = canvas.shape[0]
        w = canvas.shape[1]
        y = (math.pi / 2 - self.ns) / math.pi * h - 0.5
        x = (self.ew + math.pi / 2) / math.pi * w - 0.5
        if (-self.radius < y < h + self.radius) and (-self.radius < x < w + self.radius):
            for k in range(max(0, math.floor(y - self.radius)), min(h, math.ceil(y + self.radius))):
                for j in range(max(0, math.floor(x - self.radius)), min(w, math.ceil(x + self.radius))):
                    distance = math.dist((k, j), (y, x))
                    canvas[k, j] = 1 - math.dist((k, j), (y, x)) / self.radius if distance < self.radius else 0


class Antenna(Point):
    def __init__(self, name: str = 'burn_lab', location=(13.7309711, 100.7873937, 0.07), radius=10):
        super(Antenna, self).__init__(name=name, layer=0, location=location, radius=radius)
        self.max_rpm = 1500  # in degree/sec
        self.gear_ratio = 1  #
        self.max_angle = math.pi / 3
        self.min_angle = -self.max_angle
        self.motor_torque = 23.3  # N.m
        self.load_inertia = 103 * 0.0001  # kg/m^2 (never exceed 5 times motor inertia)
        self.motor_inertia = 20.6 * 0.0001  # kg/m^2
        self.ns_speed = 0
        self.ew_speed = 0
        self.ms_per_rpm = 2 * math.pi * (self.motor_inertia + self.load_inertia) * 1000 / (60 * self.motor_torque)
        print(f'ms per rpm: {self.ms_per_rpm}')
        print(f'rpm per second: {1 / self.ms_per_rpm:.1f}')

    def move(self, ordered_speed: ndarray([2])):
        if self.min_angle < self.ns < self.max_angle:
            self.ns += self.get_travel(self.ns_speed, ordered_speed[0] * self.max_rpm, 0.1) * math.pi / 180
            self.ns_speed = ordered_speed[0] * self.max_rpm
        else:
            self.ns_speed = 0
        if self.min_angle < self.ew < self.max_angle:
            self.ew += self.get_travel(self.ew_speed, ordered_speed[1] * self.max_rpm, 0.1) * math.pi / 180
            self.ew_speed = ordered_speed[1] * self.max_rpm
        else:
            self.ew_speed = 0

    def get_travel(self, rpm, ordered_rpm, load):
        if ordered_rpm == 0:
            transition_time = self.get_stopping_time(rpm, load)
            return transition_time * rpm / 2 / 5400000  # in degree
        elif rpm == 0:
            transition_time = self.get_starting_time(ordered_rpm, load)
            return transition_time * ordered_rpm / 2 / 5400000
        elif rpm * ordered_rpm > 0:  # same direction
            if abs(ordered_rpm) > abs(rpm):
                transition_time = self.get_starting_time(abs(ordered_rpm - rpm), load)
            else:
                transition_time = self.get_stopping_time(abs(ordered_rpm - rpm), load)
            cruising_time = 1000 - transition_time
            transition_travel = transition_time * (rpm + ordered_rpm) / 2
            cruising_travel = cruising_time * ordered_rpm
            return (transition_travel + cruising_travel) / 60000 / 90
        else:
            stopping_time = self.get_stopping_time(abs(rpm), load)
            starting_time = self.get_starting_time(abs(ordered_rpm), load)
            cruising_time = 1000 - stopping_time - starting_time
            stopping_travel = stopping_time * rpm / 2
            starting_travel = starting_time * ordered_rpm / 2
            cruising_travel = cruising_time * ordered_rpm
            print(starting_time)
            return (stopping_travel + starting_travel + cruising_travel) / 60000 / 90

    def get_stopping_time(self, delta_rpm, load):
        return self.ms_per_rpm * delta_rpm * (1 - load) / 1000

    def get_starting_time(self, delta_rpm, load):
        return self.ms_per_rpm * delta_rpm * (1 + load) / 1000

    def get_status(self) -> str:
        ns_degree = self.ns * 180 / math.pi
        ns_speed = self.ns_speed / 90 / 60 * 3600
        ew_degree = self.ew * 180 / math.pi
        ew_speed = self.ew_speed / 90 / 60 * 3600
        return f'NS: {ns_degree:.2f} deg ({ns_speed:.2f} arcsec/s) - EW: {ew_degree:.2f} deg ({ew_speed:.2f} arcsec/s)'


class Satellite(Point):
    pass


class Celestial(Point):
    pass


class SingleDishAntenna(Env):
    def __init__(self, refresh_rate=25):
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
        self.antenna = Antenna()
        self.antenna.set_position(0, 0)
        self.elements = []

    def step(self, action: ndarray):
        self.antenna.move(action)

    def reset(self, exploration_start=False):
        self.antenna = Antenna()
        if exploration_start:
            # TODO random antenna position np.array([y, x]) [-60, 60] deg
            pass

    def render(self, mode='human') -> ndarray:
        assert mode in ['human', 'rgb_array'], 'Invalid mode, must be either \'human\' or \'rgb_array\''
        self.draw()
        if mode == 'human':
            ant_status = self.antenna.get_status()
            self.canvas = cv2.putText(self.canvas, ant_status, (10, 20), font, 0.8, (1, 1, 1), 1, cv2.LINE_AA)
            cv2.imshow('Game', self.canvas)
            cv2.waitKey(int(1000 / self.refresh_rate))
        return self.canvas

    def draw(self):
        self.canvas = np.zeros(self.observation_shape) * 1
        self.antenna.draw(self.canvas)
        for elem in self.elements:
            print(elem.name)
            # TODO draw each element in their respective layer


A = SingleDishAntenna()
print(f'Antenna simulator started. Location: {A.antenna.name} [{A.antenna.location}]')
print(f'Frame rate: {A.refresh_rate}')
i = 0
frames = []
out = cv2.VideoWriter('Antenna.mp4', fourcc=cv2.VideoWriter_fourcc(*'mp4v'), fps=A.refresh_rate, frameSize=(900, 900),
                      isColor=1)
while i < 1000:
    A.step(np.array([1, 1]))
    frame = A.render(mode='human')
    if frame is not None:
        out.write((frame * 255).astype('uint8'))
    i += 1
cv2.destroyAllWindows()
cv2.waitKey(0)
out.release()
