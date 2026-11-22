"""Centralized value critic used by MAPPO."""
from harl.algorithms.critics.v_critic import VCritic

CRITIC_REGISTRY = {"mappo": VCritic}
