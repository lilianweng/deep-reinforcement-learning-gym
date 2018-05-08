import gym
from playground.learn import DiscretizedObservationWrapper


def test_digitized_observation_wrapper():
    env = gym.make('MountainCar-v0')
    env = DiscretizedObservationWrapper(env)
    obs = env.reset()
    print(obs)
