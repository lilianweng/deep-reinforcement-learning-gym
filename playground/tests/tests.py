import gym
from playground.learn import DigitizedObservationWrapper


def test_digitized_observation_wrapper():
    env = gym.make('MountainCar-v0')
    env = DigitizedObservationWrapper(env)
    obs = env.reset()
    print(obs)
