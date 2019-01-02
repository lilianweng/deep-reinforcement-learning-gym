import gym
from playground.utils.wrappers import DiscretizedObservationWrapper
from playground.utils.misc import Config


def test_digitized_observation_wrapper():
    env = gym.make('MountainCar-v0')
    env = DiscretizedObservationWrapper(env)
    obs = env.reset()
    print(obs)


def test_config_class():
    class ParentConfig(Config):
        a = 1
        b = 2

    class ChildConfig(ParentConfig):
        x = 4
        y = 5
        z = 6

    class GrandChildConfig(ChildConfig):
        red = True
        blue = False

    config = GrandChildConfig(a=100, y=200, blue=True)
    assert config.b == 2
    assert config.blue == True
    assert config.as_dict() == dict(a=100, b=2, x=4, y=200, z=6, red=True, blue=True)
