from playground.policies.actor_critic import ActorCriticPolicy
from playground.policies.ddpg import DDPGPolicy
from playground.policies.dqn import DqnPolicy
from playground.policies.ppo import PPOPolicy
from playground.policies.qlearning import QlearningPolicy
from playground.policies.reinforce import ReinforcePolicy

ALL_POLICIES = [
    ActorCriticPolicy,
    DDPGPolicy,
    DqnPolicy,
    PPOPolicy,
    QlearningPolicy,
    ReinforcePolicy
]
