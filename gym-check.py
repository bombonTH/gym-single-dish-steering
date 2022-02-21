from gym.utils import env_checker
#from stable_baselines3 import common.env_checker
import gym

from gym_single_dish_antenna import SingleDishAntenna
from motor import Motor
from point import Point
from sun import Sun

sgms_13a = Motor(max_rpm=1500, motor_moi=20.5, load_moi=41, motor_peak_torque=23.3)
env = SingleDishAntenna(refresh_rate=1)
env.antenna.set_ns_motor(Motor(max_rpm=1500, motor_moi=20.5, load_moi=41, motor_peak_torque=23.3),
                         1.0 / 30000)
env.antenna.set_ew_motor(Motor(max_rpm=1500, motor_moi=20.5, load_moi=41, motor_peak_torque=23.3),
                         1.0 / 59400)
env_checker.check_env(env, True, False)
