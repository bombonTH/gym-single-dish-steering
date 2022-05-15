import math

import numpy as np


class Motor(object):
    def __init__(self, max_rpm=1500, motor_moi=20.5, load_moi=41.0, motor_peak_torque=23.3):
        self.max_rpm = max_rpm
        self.motor_moi = motor_moi
        self.load_moi = load_moi
        self.motor_peak_torque = motor_peak_torque
        self.rpm = 0
        self.ms_per_rpm = 2 * math.pi * (self.motor_moi + self.load_moi) / (600 * self.motor_peak_torque)

    def spin(self, ordered_rpm, time, load_torque=0.0, random_torque=False) -> float:
        load_torque = (np.random.rand() - 0.5) * 1.0 * self.motor_peak_torque if random_torque else load_torque
        dif_rpm = ordered_rpm - self.rpm
        transition_time = abs(dif_rpm * self.ms_per_rpm * (1 - load_torque if dif_rpm > 0 else 1 + load_torque))
        if time >= transition_time:
            transition_rpm = (ordered_rpm + self.rpm) / 2
            cruising_time = time - transition_time
            self.rpm = ordered_rpm
        else:
            ordered_rpm = self.rpm + dif_rpm * time / transition_time
            transition_rpm = (ordered_rpm + self.rpm) / 2
            transition_time = time
            cruising_time = 0
            self.rpm = ordered_rpm
        return (transition_rpm * transition_time + ordered_rpm * cruising_time) / 60000


SGMG_13A2A = Motor(max_rpm=1500, motor_moi=20.5, load_moi=41, motor_peak_torque=23.3)
