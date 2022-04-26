import signal
import sys

from cv2 import cv2 as cv2

import gym

from gym_single_dish_antenna_v2 import SingleDishAntennaV2

from motor import Motor
from sun import Sun
from geo import Geo
from airplane import Airplane
from load_geo import load_geos

from stable_baselines3 import TD3, A2C, SAC, DDPG, PPO
from stable_baselines3.common.vec_env import VecFrameStack, SubprocVecEnv, VecMonitor, VecVideoRecorder
from util import linear_schedule


def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    cv2.destroyAllWindows()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

sgms_13a = Motor(max_rpm=1500, motor_moi=20.5, load_moi=41, motor_peak_torque=23.3)


def make_env(_env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param _env_id: (str) the environment ID
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    """

    def _init():
        _env = gym.make(_env_id)
        _env.add_target(Sun(time='2021-08-16 05:00:00', layer=1, radius=1))
        load_geos(_env)
        _env.add_obstacle(Airplane(antenna=(13.7309711, 100.7873937, 0.07), time='2021-08-16 05:00:00'))
        _env.obstacles[-1].set_location(location=(13.7309711, 100.7873937, 34000 / 3280.84), track=5, speed=300).reset(
            random=True)
        _env.add_obstacle(Airplane(antenna=(13.7309711, 100.7873937, 0.07), time='2021-08-16 05:00:00'))
        _env.obstacles[-1].set_location(location=(13.7309711, 100.7873937, 34000 / 3280.84), track=5, speed=300).reset(
            random=True)
        _env.antenna.set_y_motor(Motor(max_rpm=1500, motor_moi=20.5, load_moi=41, motor_peak_torque=23.3),
                                  1.0 / 30000)
        _env.antenna.set_x_motor(Motor(max_rpm=1500, motor_moi=20.5, load_moi=41, motor_peak_torque=23.3),
                                  1.0 / 59400)
        return _env

    return _init


if __name__ == '__main__':
    env_id = "SingleDish-v2"
    num_cpu = 16  # Number of processes to use
    # Create the vectorized environment
    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
    # env = VecFrameStack(env, 4)
    env = VecMonitor(env, 'recording.3')
    # Stable Baselines provides you with make_vec_env() helper
    # which does exactly the previous steps for you:
    # env = make_vec_env(env_id, n_envs=num_cpu, seed=0)
    policy_kwargs = dict(net_arch=[dict(pi=[64, 32, 32], vf=[64, 32, 32])])

    model = PPO('MlpPolicy', env, verbose=1, n_steps=2048, n_epochs=4, batch_size=64,
                learning_rate=3e-4, clip_range=0.2, vf_coef=0.5, ent_coef=0.01,
                tensorboard_log="./ppo_tensorboard/", policy_kwargs=policy_kwargs)
    model.load('4e7.zip', print_system_info=True)
    model.set_parameters('4e7')
    model.learn(total_timesteps=float(1e7), reset_num_timesteps=False)
    model.save('14e6.zip')
    model.learn(total_timesteps=float(1e7), reset_num_timesteps=False)
    model.save('4e7.zip')
    model.learn(total_timesteps=float(1e7), reset_num_timesteps=False)
    model.save('5e7.zip')
    model.learn(total_timesteps=float(1e7), reset_num_timesteps=False)
    model.save('6e7.zip')
    model.learn(total_timesteps=float(1e7), reset_num_timesteps=False)
    model.save('7e7.zip')
    model.learn(total_timesteps=float(1e7), reset_num_timesteps=False)
    model.save('8e7.zip')
    model.learn(total_timesteps=float(1e7), reset_num_timesteps=False)
    model.save('9e7.zip')
    model.learn(total_timesteps=float(1e7), reset_num_timesteps=False)
    model.save('1e8.zip')

    print('Done!')
