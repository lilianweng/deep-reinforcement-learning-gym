import click

import gym
import numpy as np
from gym.wrappers import Monitor
import time

from playground.policies import QlearningPolicy, DqnPolicy, ReinforcePolicy, ActorCriticPolicy
from playground.utils.misc import plot_from_monitor_results
from playground.utils.wrappers import DigitizedObservationWrapper


def cartpole_qlearning(model_name='cartpole-qlearning'):
    """
    In [22]: env.observation_space.low, env.observation_space.high
    Out[22]:
    (array([-4.8000002e+00, -3.4028235e+38, -4.1887903e-01, -3.4028235e+38], dtype=float32),
     array([4.8000002e+00, 3.4028235e+38, 4.1887903e-01, 3.4028235e+38], dtype=float32))
    """
    env = gym.make('CartPole-v1')
    env = DigitizedObservationWrapper(
        env, n_bins=10,
        low=np.array([-2.4, -2., -0.42, -3.5]),
        high=np.array([2.4, 2., 0.42, 3.5]),
    )
    env = Monitor(env, '/tmp/' + model_name, force=True)

    policy = QlearningPolicy(env, model_name,
                             alpha=0.5, gamma=0.9,
                             epsilon=0.1, epsilon_decay=0.98)
    policy.build()
    policy.train(100000, every_step=5000, done_reward=-100, with_monitor=True)

    env.close()
    plot_from_monitor_results('/tmp/' + model_name)


def run_qlearning(env_name, model_name):
    """Mountain car, reward is -1; only when done the reward is different.
    """
    env = gym.make(env_name)
    env = DigitizedObservationWrapper(env, n_bins=10)
    print(env.action_space, env.observation_space)

    policy = QlearningPolicy(env, model_name, alpha=0.5, gamma=0.99, epsilon=0.1,
                             epsilon_decay=0.98, alpha_decay=0.97)
    policy.build()
    policy.train(100000, every_step=1000, with_monitor=True)

    env.close()
    plot_from_monitor_results('/tmp/' + model_name)


def run_dqn(env_name, model_name):
    env = gym.make(env_name)
    env = Monitor(env, '/tmp/' + model_name, force=True)

    policy = DqnPolicy(env, model_name, training=False,
                       lr=0.001, epsilon=1.0, epsilon_final=0.02, batch_size=32,
                       q_model_type='mlp', q_model_params={'layer_sizes': [32, 32]},
                       target_update_type='hard')
    policy.build()

    if policy.load_model():
        policy.evaluate(10)
    else:
        policy.train(500, annealing_episodes=450, every_episode=10)
        env.close()
        plot_from_monitor_results('/tmp/' + model_name)


def run_reinforce(env_name, model_name):
    env = gym.make(env_name)
    env = Monitor(env, '/tmp/' + model_name, force=True)

    policy = ReinforcePolicy(env, model_name, lr=0.1, lr_decay=0.998,
                             batch_size=32, layer_sizes=[32, 32], baseline=True)
    policy.build()
    policy.train(1000, every_episode=50)

    env.close()
    plot_from_monitor_results('/tmp/' + model_name)


def run_actor_critic(env_name, model_name):
    env = gym.make(env_name)
    env = Monitor(env, '/tmp/' + model_name, force=True)

    policy = ActorCriticPolicy(env, model_name,
                               lr_a=0.01, lr_a_decay=0.995,
                               lr_c=0.001, lr_c_decay=0.995,
                               batch_size=32, layer_sizes=[16], grad_clip_norm=5.0)
    policy.build()
    policy.train(500, annealing_episodes=250, every_episode=10)

    env.close()
    plot_from_monitor_results('/tmp/' + model_name)

@click.command()
@click.argument('policy_name')
@click.argument('env_name')
@click.option('-m', '--model-name', default=None)
def main(policy_name, env_name, model_name=None):
    # env_name: 'CartPole-v1', 'MsPacman-v0', 'BipedalWalkerHardcore-v2'
    if model_name is None:
        model_name = "%s-%s-%d" % (env_name.lower(), policy_name.replace('_', '-'), int(time.time()))

    run_fn = {
        'qlearning': run_qlearning,
        'dqn': run_dqn,
        'reinforce': run_reinforce,
        'actor_critic': run_actor_critic,
    }[policy_name]

    run_fn(env_name, model_name)
    print("Training complete:", model_name)


if __name__ == '__main__':
    main()
