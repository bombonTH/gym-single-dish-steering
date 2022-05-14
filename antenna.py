import math

from numpy import ndarray
import numpy as np

from motor import Motor
from point import Point
import coordinate


class Antenna(Point):
    def __init__(self, name: str = 'burn_lab', location=(13.7309711, 100.7873937, 0.07), radius=2, random_torque=False):
        super(Antenna, self).__init__(name=name, layer=0, location=location, radius=radius)
        self.max_rpm = 1500  # in degree/sec
        self.y_motor = None
        self.x_motor = None
        self.y_per_rev = 0
        self.x_per_rev = 0
        self.random_torque = random_torque

    def set_y_motor(self, motor: Motor, gear_ratio: float):
        self.y_motor = motor
        self.y_per_rev = 360 * gear_ratio * math.pi / 180

    def set_x_motor(self, motor: Motor, gear_ratio: float):
        self.x_motor = motor
        self.x_per_rev = 360 * gear_ratio * math.pi / 180

    def move(self, ordered_speed: ndarray, time):
        ordered_speed[0] = abs(ordered_speed[0]) * ordered_speed[0]
        ordered_speed[1] = abs(ordered_speed[1]) * ordered_speed[1]
        if self.y < self.min_y:
            ordered_speed[0] = max(ordered_speed[0], 0)
        if self.y > self.max_y:
            ordered_speed[0] = min(ordered_speed[0], 0)
        if self.x < self.min_x:
            ordered_speed[1] = max(ordered_speed[1], 0)
        if self.x > self.max_x:
            ordered_speed[1] = min(ordered_speed[1], 0)

        self.out_of_bound = not (self.min_x < self.x < self.max_x and self.min_y < self.y < self.max_y)

        y_rev = self.y_motor.spin(ordered_speed[0] * self.max_rpm, time, random_torque=self.random_torque)
        dif_y = y_rev * self.y_per_rev
        y = self.y + dif_y

        x_rev = self.x_motor.spin(ordered_speed[1] * self.max_rpm, time, random_torque=self.random_torque)
        dif_x = x_rev * self.x_per_rev
        x = self.x + dif_x
        
        self.xy = coordinate.XY(y=math.degrees(y), x=math.degrees(x))
        self.x_rate = self.xy.x - self.x
        self.y_rate = self.xy.y - self.y
        self.x = self.xy.x
        self.y = self.xy.y
        
        self.xs[3] = self.xs[2]
        self.xs[2] = self.xs[1]
        self.xs[1] = self.xs[0]
        self.xs[0] = self.x
        self.ys[3] = self.ys[2]
        self.ys[2] = self.ys[1]
        self.ys[1] = self.ys[0]
        self.ys[0] = self.y
        
        self.elaz = self.xy.to_altaz()
        self.az_rate = self.elaz.az - self.az
        self.el_rate = self.elaz.el - self.el
        self.az = self.elaz.az
        self.el = self.elaz.el
        
        self.azs[3] = self.azs[2]
        self.azs[2] = self.azs[1]
        self.azs[1] = self.azs[0]
        self.azs[0] = self.az
        self.els[3] = self.els[2]
        self.els[2] = self.els[1]
        self.els[1] = self.els[0]
        self.els[0] = self.el
        

    def reset(self, random=False):
        self.y = 0
        self.x = 0
        self.out_of_bound = False
        self.obstructed = False
        if random:
            self.y = np.random.uniform(self.min_y, self.max_y)
            self.x = np.random.uniform(self.min_x, self.max_x)

    def get_status(self) -> str:
        return f' Az: {math.degrees(self.az):05.1f}-El: {math.degrees(self.el):04.1f}'
