import signal
import sys
from typing import Callable

from cv2 import cv2 as cv2

import gym
from gym.envs.registration import register

from gym_single_dish_antenna import SingleDishAntenna

from motor import Motor
from sun import Sun
from geo import Geo

from stable_baselines3 import TD3, A2C, SAC, DDPG, PPO
from stable_baselines3.common.vec_env import VecFrameStack, SubprocVecEnv, VecMonitor, VecVideoRecorder

register(
    # unique identifier for the env `name-version`
    id="SingleDish-v1",
    # path to the class for creating the env
    # Note: entry_point also accept a class as input (and not only a string)
    entry_point=SingleDishAntenna,
    # Max number of steps per episode, using a `TimeLimitWrapper`
    max_episode_steps=1000,
)


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    cv2.destroyAllWindows()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

sgms_13a = Motor(max_rpm=1500, motor_moi=20.5, load_moi=41, motor_peak_torque=23.3)


def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """

    def _init():
        env = gym.make(env_id)
        env.add_target(Sun(time='2021-08-16 05:00:00', layer=1))
        env.add_target(Geo(ns=0.0, ew=0.0, layer=1))
        env.add_obstacle(Geo(ns=0.0, ew=0.3))
        env.add_obstacle(Geo(ns=0.3, ew=0.0))
        env.antenna.set_ns_motor(Motor(max_rpm=1500, motor_moi=20.5, load_moi=41, motor_peak_torque=23.3),
                                 1.0 / 30000)
        env.antenna.set_ew_motor(Motor(max_rpm=1500, motor_moi=20.5, load_moi=41, motor_peak_torque=23.3),
                                 1.0 / 59400)
        return env

    return _init


if __name__ == '__main__':
    env_id = "SingleDish-v1"
    num_cpu = 8  # Number of processes to use
    # Create the vectorized environment
    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
    env = VecFrameStack(env, 4)
    env = VecMonitor(env, 'recording.2')
    # Stable Baselines provides you with make_vec_env() helper
    # which does exactly the previous steps for you:
    # env = make_vec_env(env_id, n_envs=num_cpu, seed=0)

    model = PPO('CnnPolicy', env, verbose=1, n_steps=128, n_epochs=4, batch_size=256,
                learning_rate=linear_schedule(2.5e-4), clip_range=linear_schedule(0.1), vf_coef=0.5, ent_coef=0.01,
                tensorboard_log="./ppo_tensorboard/")
    #model.load('1e7.zip', print_system_info=True)
    #model.set_parameters('1e7')
    model.learn(total_timesteps=float(1e7), reset_num_timesteps=False)
    model.save('1e7.zip')
    model.learn(total_timesteps=float(1e7), reset_num_timesteps=False)
    model.save('2e7.zip')
    model.learn(total_timesteps=float(1e7), reset_num_timesteps=False)
    model.save('3e7.zip')
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
