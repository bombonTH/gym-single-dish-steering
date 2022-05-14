import signal
import sys

from cv2 import cv2 as cv2
import numpy as np
import gym

from gym_single_dish_antenna_v2 import SingleDishAntennaV2

from motor import Motor
from sun import Sun
from geo import Geo
from radec import RaDec
from airplane import Airplane
from load_geo import load_geos

from stable_baselines3 import TD3, A2C, SAC, DDPG, PPO
from stable_baselines3.common.vec_env import VecFrameStack, SubprocVecEnv, VecMonitor, VecVideoRecorder, DummyVecEnv
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
        _env = gym.make(_env_id, random_start=True, random_torque=True)
        #_env.add_target(Sun(time='2021-08-16 05:00:00', dwell=np.random.randint(0, 20), layer=1, radius=2))
        _env.add_target(RaDec(time='2021-08-16 05:00:00', dwell=np.random.randint(1, 20), layer=1, radius=2,ra=140,dec=10))
        #_env.add_target(RaDec(time='2021-08-16 05:00:00', dwell=np.random.randint(0, 20), layer=1, radius=2,ra=140,dec=10))
        #_env.add_target(RaDec(time='2021-08-16 05:00:00', dwell=np.random.randint(0, 20), layer=1, radius=2,ra=140,dec=10))
        #_env.add_target(RaDec(time='2021-08-16 05:00:00', dwell=np.random.randint(0, 20), layer=1, radius=2,ra=140,dec=10))
        #_env.add_target(RaDec(time='2021-08-16 05:00:00', dwell=np.random.randint(0, 20), layer=1, radius=2,ra=140,dec=10))
        #_env.add_target(RaDec(time='2021-08-16 05:00:00', dwell=np.random.randint(0, 20), layer=1, radius=2,ra=140,dec=10))
        #_env.add_target(RaDec(time='2021-08-16 05:00:00', dwell=np.random.randint(0, 20), layer=1, radius=2,ra=140,dec=10))
        #_env.add_target(RaDec(time='2021-08-16 05:00:00', dwell=np.random.randint(0, 20), layer=1, radius=2,ra=140,dec=10))
        #_env.add_target(RaDec(time='2021-08-16 05:00:00', layer=1, radius=2))
        #load_geos(_env)
        #_env.add_obstacle(Airplane(antenna=(13.7309711, 100.7873937, 0.07), time='2021-08-16 05:00:00'))
        #_env.obstacles[-1].set_location(location=(13.7309711, 100.7873937, 34000 / 3280.84), track=5, speed=300).reset(
        #    random=True)
        #_env.add_obstacle(Airplane(antenna=(13.7309711, 100.7873937, 0.07), time='2021-08-16 05:00:00'))
        #_env.obstacles[-1].set_location(location=(13.7309711, 100.7873937, 34000 / 3280.84), track=5, speed=300).reset(
        #    random=True)
        _env.antenna.set_y_motor(Motor(max_rpm=1500, motor_moi=20.5, load_moi=41, motor_peak_torque=23.3),
                                  1.0 / 30000)
        _env.antenna.set_x_motor(Motor(max_rpm=1500, motor_moi=20.5, load_moi=41, motor_peak_torque=23.3),
                                  1.0 / 59400)
        return _env

    return _init


if __name__ == '__main__':
    env_id = "SingleDish-v2"
    num_cpu = 8  # Number of processes to use
    # Create the vectorized environment
    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
    #env = DummyVecEnv([make_env(env_id, i) for i in range(num_cpu)])
    env = VecMonitor(env, 'recording.3')
    # Stable Baselines provides you with make_vec_env() helper
    # which does exactly the previous steps for you:
    # env = make_vec_env(env_id, n_envs=num_cpu, seed=0)
    policy_kwargs = dict(net_arch=[dict(pi=[64, 64, 64], vf=[64, 64, 64])])

    model = PPO('MlpPolicy', env, verbose=1, n_steps=40960, n_epochs=10, batch_size=5120,
                learning_rate=3e-4, clip_range=0.2, vf_coef=0.5, ent_coef=0.01,
                tensorboard_log="./ppo_tensorboard/dummy/", policy_kwargs=policy_kwargs)
    #model.load('4e6.zip', print_system_info=True)
    model.set_parameters('d24e6')
    model.learn(total_timesteps=float(1e6), reset_num_timesteps=False)
    model.save('d25e6.zip')
    model.learn(total_timesteps=float(1e6), reset_num_timesteps=False)
    model.save('d26e6.zip')
    model.learn(total_timesteps=float(1e6), reset_num_timesteps=False)
    model.save('d27e6.zip')
    model.learn(total_timesteps=float(1e6), reset_num_timesteps=False)
    model.save('d28e6.zip')
    model.learn(total_timesteps=float(1e6), reset_num_timesteps=False)
    model.save('d29e6.zip')
    model.learn(total_timesteps=float(1e6), reset_num_timesteps=False)
    model.save('d30e6.zip')
    model.learn(total_timesteps=float(1e6), reset_num_timesteps=False)
    model.save('d31e6.zip')
    model.learn(total_timesteps=float(1e6), reset_num_timesteps=False)
    model.save('d32e6.zip')
    model.learn(total_timesteps=float(1e6), reset_num_timesteps=False)
    model.save('d33e6.zip')
    model.learn(total_timesteps=float(1e6), reset_num_timesteps=False)
    model.save('d34e6.zip')
    model.learn(total_timesteps=float(1e6), reset_num_timesteps=False)
    model.save('d35e6.zip')
    model.learn(total_timesteps=float(1e6), reset_num_timesteps=False)
    model.save('d36e6.zip')


    print('Done!')
