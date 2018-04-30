import gym
import numpy as np
from gym.wrappers import Monitor

from playground.policies import QlearningPolicy, DqnPolicy, ReinforcePolicy
from playground.utils.misc import plot_from_monitor_results
from playground.utils.wrappers import DigitizedObservationWrapper


def mountain_car_qlearning(model_name='mountain-car-qlearning'):
    """Mountain car, reward is -1; only when done the reward is different.
    """
    env = gym.make('MountainCar-v0')
    env = DigitizedObservationWrapper(env, n_bins=10)
    print(env.action_space, env.observation_space)

    policy = QlearningPolicy(env, model_name, alpha=0.5, gamma=0.99, epsilon=0.1,
                             epsilon_decay=0.98, alpha_decay=0.97)
    policy.build()
    policy.train(100000, every_step=1000, with_monitor=True)

    env.close()
    plot_from_monitor_results('/tmp/' + model_name)


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


def cartpole_dqn(model_name='cartpole-dqn'):
    env = gym.make('CartPole-v1')
    env = Monitor(env, '/tmp/' + model_name, force=True)

    policy = DqnPolicy(env, model_name,
                       lr=0.001, epsilon=1.0, epsilon_final=0.02, batch_size=32,
                       # q_model_type='rnn', q_model_params={'step_size': 16, 'lstm_size': 32},
                       q_model_type='cnn', q_model_params={'layer_sizes': [64]},
                       target_update_type='hard')
    policy.build()
    policy.train(500, annealing_episodes=450, every_episode=5)

    env.close()
    plot_from_monitor_results('/tmp/' + model_name)


def cartpole_reinforce(model_name='cartpole-reinforce'):
    env = gym.make('CartPole-v1')
    env = Monitor(env, '/tmp/' + model_name, force=True)

    policy = ReinforcePolicy(env, model_name, lr=0.002, lr_decay=0.999,
                             batch_size=32, layer_sizes=[32, 32])
    policy.build()
    policy.train(750, every_episode=10)

    env.close()
    plot_from_monitor_results('/tmp/' + model_name)


def test_cartpole_dqn(model_name='cartpole-dqn'):
    env = gym.make('CartPole-v1')
    policy = DqnPolicy(env, model_name, training=False,
                       lr=0.001, epsilon=1.0, epsilon_final=0.02, batch_size=32,
                       q_model_type='mlp', q_model_params={'layer_sizes': [32, 32]},
                       target_update_type='hard')
    policy.build()
    assert policy.load_model(), "Failed to load a trained model."
    policy.test(10)


def pacman_dqn(model_name='pacman-dqn'):
    env = gym.make('MsPacman-v0')
    env = Monitor(env, '/tmp/' + model_name, force=True)

    policy = DqnPolicy(env, model_name,
                       lr=0.001, epsilon=1.0, epsilon_final=0.02, batch_size=32,
                       q_model_type='cnn',
                       target_update_type='hard', target_update_params={'every_step': 500})
    policy.build()
    policy.train(1000, annealing_episodes=900, every_episode=10)

    env.close()
    plot_from_monitor_results('/tmp/' + model_name)


if __name__ == '__main__':
    # cartpole_dqn('cartpole-dqn-hard-rnn')
    cartpole_reinforce('cartpole-reinforce')
    # pacman_dqn('pacman-dqn-hard')
