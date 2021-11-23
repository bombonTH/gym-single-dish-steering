import math

from numpy import ndarray

from motor import Motor
from point import Point


class Antenna(Point):
    def __init__(self, name: str = 'burn_lab', location=(13.7309711, 100.7873937, 0.07), radius=10):
        super(Antenna, self).__init__(name=name, layer=0, location=location, radius=radius)
        self.max_rpm = 1500  # in degree/sec
        self.max_angle = math.pi / 3
        self.min_angle = -self.max_angle  # 60 degrees
        self.ns_motor = None
        self.ew_motor = None
        self.ns_speed = 0
        self.ew_speed = 0
        self.ns_per_rev = 0
        self.ew_per_rev = 0

    def set_ns_motor(self, motor: Motor, gear_ratio: float):
        self.ns_motor = motor
        self.ns_per_rev = 360 * gear_ratio * math.pi / 180

    def set_ew_motor(self, motor: Motor, gear_ratio: float):
        self.ew_motor = motor
        self.ew_per_rev = 360 * gear_ratio * math.pi / 180

    def move(self, ordered_speed: ndarray, time):
        if self.min_angle < self.ns < self.max_angle:  # ns within limit
            rev = self.ns_motor.spin(ordered_speed[0] * self.max_rpm, time)
            dif_ns = rev * self.ns_per_rev
            self.ns += dif_ns
            self.ns_speed = dif_ns / time * 1000
        else:
            rev = self.ns_motor.spin(0, time)
            dif_ns = rev * self.ns_per_rev
            self.ns += dif_ns
            self.ns_speed = dif_ns / time * 1000
        if self.min_angle < self.ew < self.max_angle:  # ew within limit
            rev = self.ew_motor.spin(ordered_speed[1] * self.max_rpm, time)
            dif_ew = rev * self.ew_per_rev
            self.ew += dif_ew
            self.ew_speed = dif_ew / time * 1000
        else:
            rev = self.ew_motor.spin(0, time)
            dif_ew = rev * self.ew_per_rev
            self.ew += dif_ew
            self.ew_speed = dif_ew / time * 1000

    def get_status(self) -> str:
        ns_degree = self.ns * 180 / math.pi
        ns_speed = self.ns_speed * 180 / math.pi * 3600  # deg/s
        ew_degree = self.ew * 180 / math.pi
        ew_speed = self.ew_speed * 180 / math.pi * 3600  # deg/s
        return f'NS: {ns_degree:+06.2f} deg ({ns_speed:+08.2f} \"/s) - EW: {ew_degree:+06.2f} deg ({ew_speed:+07.2f} \"/min)'
