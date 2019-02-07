import os
import click
from playground.configs.manager import ConfigManager, load_policy_class
from playground.utils.misc import REPO_ROOT
from gym.wrappers import Monitor


@click.command()
@click.argument('config_name')
@click.argument('model_name')
@click.option('-r', '--nb-runs', default=1)
def run(config_name, model_name, nb_runs=1):
    cfg = ConfigManager.load(config_name)
    #env = Monitor(cfg.env, '/tmp/' + model_name, force=True)
    policy = load_policy_class(cfg.policy_name)(cfg.env, "", training=False, **cfg.policy_params)
    policy.build()
    path = os.path.join(REPO_ROOT, "checkpoints", model_name)
    policy.load_checkpoint(path)
    policy.evaluate(nb_runs)
    cfg.env.close()


if __name__ == '__main__':
    run()
