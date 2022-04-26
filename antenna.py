import math

from numpy import ndarray
import numpy as np

from motor import Motor
from point import Point
import coordinate


class Antenna(Point):
    def __init__(self, name: str = 'burn_lab', location=(13.7309711, 100.7873937, 0.07), radius=2):
        super(Antenna, self).__init__(name=name, layer=0, location=location, radius=radius)
        self.max_rpm = 1500  # in degree/sec
        self.y_motor = None
        self.x_motor = None
        self.y_per_rev = 0
        self.x_per_rev = 0

    def set_y_motor(self, motor: Motor, gear_ratio: float):
        self.y_motor = motor
        self.y_per_rev = 360 * gear_ratio * math.pi / 180

    def set_x_motor(self, motor: Motor, gear_ratio: float):
        self.x_motor = motor
        self.x_per_rev = 360 * gear_ratio * math.pi / 180

    def move(self, ordered_speed: ndarray, time):
        if self.y < self.min_y:
            ordered_speed[0] = max(ordered_speed[0], 0)
        if self.y > self.max_y:
            ordered_speed[0] = min(ordered_speed[0], 0)
        if self.x < self.min_x:
            ordered_speed[1] = max(ordered_speed[1], 0)
        if self.x > self.max_x:
            ordered_speed[1] = min(ordered_speed[1], 0)

        self.out_of_bound = not (self.min_x < self.x < self.max_x and self.min_y < self.y < self.max_y)

        y_rev = self.y_motor.spin(ordered_speed[0] * self.max_rpm, time, random_torque=True)
        dif_y = y_rev * self.y_per_rev
        y = self.y + dif_y

        x_rev = self.x_motor.spin(ordered_speed[1] * self.max_rpm, time, random_torque=True)
        dif_x = x_rev * self.x_per_rev
        x = self.x + dif_x
        
        self.xy = coordinate.XY(y=math.degrees(y), x=math.degrees(x))
        self.x_rate = self.xy.x - self.x
        self.y_rate = self.xy.y - self.y
        self.x = self.xy.x
        self.y = self.xy.y
        
        self.elaz = self.xy.to_altaz()
        self.az_rate = self.elaz.az - self.az
        self.el_rate = self.elaz.el - self.el
        self.az = self.elaz.az
        self.el = self.elaz.el
        

    def reset(self, random=False):
        self.y = 0
        self.x = 0
        self.out_of_bound = False
        self.obstructed = False
        if random:
            self.y = np.random.uniform(self.min_y, self.max_y)
            self.x = np.random.uniform(self.min_x, self.max_x)

    def get_status(self) -> str:
        y = self.y * 180 / math.pi
        y_speed = self.y_rate * 180 / math.pi * 3600  # deg/s
        y = self.x * 180 / math.pi
        x_speed = self.x_rate * 180 / math.pi * 3600  # deg/s
        return f'NS: {y:+06.2f} deg ({y_rate:+08.2f} \"/s) - Y: {x:+06.2f} deg ({y:+07.2f} \"/s)'
