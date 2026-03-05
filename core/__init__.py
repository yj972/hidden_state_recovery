"""
Core module: Abstract Base Classes for the Cognitive Stackelberg RL framework.
- StackelbergEnv: POMDP wrapper (hidden state y*)
- System2Agent: Thought + Action policies (text-only, no classification head)
- RewardModel: Black-box process reward (LLM-as-Judge or heuristic)
- TrajectoryBuffer: Thought-Action traces
"""

from .env import StackelbergEnv
from .agent import System2Agent
from .rewards import RewardModel, LLMRewardModel, HeuristicRewardModel
from .memory import TrajectoryBuffer

__all__ = [
    "StackelbergEnv",
    "System2Agent",
    "RewardModel",
    "LLMRewardModel",
    "HeuristicRewardModel",
    "TrajectoryBuffer",
]
