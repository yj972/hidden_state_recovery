"""
Core module: Abstract Base Classes for the Cognitive Stackelberg RL framework.
- StackelbergEnv: POMDP wrapper (hidden state y*, belief b_t)
- System2Agent: Thought + Action policies
- IntrinsicRewardModule: Entropy / Information Gain
- TrajectoryBuffer: Thought-Action traces
"""

from .env import StackelbergEnv
from .agent import System2Agent
from .rewards import IntrinsicRewardModule
from .memory import TrajectoryBuffer

__all__ = [
    "StackelbergEnv",
    "System2Agent",
    "IntrinsicRewardModule",
    "TrajectoryBuffer",
]
