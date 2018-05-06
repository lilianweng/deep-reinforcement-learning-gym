import os
import time

import click
import numpy as np

import gym
from gym.wrappers import Monitor

from playground.configs.manager import ConfigManager
from playground.policies import QlearningPolicy, DqnPolicy, ReinforcePolicy, ActorCriticPolicy
from playground.utils.misc import plot_from_monitor_results
from playground.utils.wrappers import DigitizedObservationWrapper


def run_qlearning(env_name, model_name):
    """Mountain car, reward is -1; only when done the reward is different.
    """
    env = gym.make(env_name)
    if env_name == 'CartPole-v1':
        # env.observation_space.low = [-4.8, -3.4028e+38, -0.4189, -3.4028e+38]
        # env.observation_space.high = [4.8, 3.4028e+38, 0.4189, 3.4028e+38]
        env = DigitizedObservationWrapper(
            env, n_bins=10,
            low=np.array([-2.4, -2., -0.42, -3.5]),
            high=np.array([2.4, 2., 0.42, 3.5]),
        )
        done_reward = -100.
    else:
        env = DigitizedObservationWrapper(env, n_bins=10)
        done_reward = None

    env = Monitor(env, '/tmp/' + model_name, force=True)
    policy = QlearningPolicy(env, model_name,
                             alpha=0.5, alpha_decay=0.999,
                             epsilon=1.0, epsilon_final=0.1)
    policy.build()
    policy.train(3000, annealing_episodes=2000, done_reward=done_reward, every_episode=20)
    env.close()
    plot_from_monitor_results('/tmp/' + model_name, window=50)


@click.command()
@click.argument('config_name')
@click.option('-m', '--model-name', default=None)
def run(config_name, model_name=None):
    cfg = ConfigManager.load(config_name)

    if model_name is None:
        model_name = '-'.join([
            cfg.env_name.lower(),
            cfg.policy_name.replace('_', '-'),
            os.path.splitext(os.path.basename(config_name))[0] if config_name else 'default',
            str(int(time.time()))
        ])

    model_name = model_name.lower()
    cfg.start_training(model_name)


if __name__ == '__main__':
    run()
