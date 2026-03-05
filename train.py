"""
Main entry point for training System 2 Reasoning Agents (Cognitive Stackelberg Game).

Usage (after concrete implementations are added):
  python train.py --config configs/default.yaml
  python train.py --env medical --algorithm ppo_dual --epochs 100

This script wires:
  - StackelbergEnv (from data loader)
  - System2Agent (Thought + Action policies, backed by models/)
  - IntrinsicRewardModule (r_PRM) + task reward (R_task)
  - TrajectoryBuffer + PPODualReward (or GRPO)
"""

import argparse
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train System 2 RL agents (Cognitive Stackelberg)"
    )
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument(
        "--env",
        type=str,
        choices=["medical", "coding"],
        default="medical",
        help="App / env: medical (DDXPlus) or coding (Stackelberg Coder)",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["ppo_dual", "grpo"],
        default="ppo_dual",
        help="RL algorithm",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Checkpoint/log dir")
    # Add more as needed: batch_size, lr, weight_process, weight_task, etc.
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # TODO: load config if args.config
    # TODO: build dataset (data.medical.loader or data.coding.loader)
    # TODO: build StackelbergEnv from dataset
    # TODO: build TransformerBackbone + ValueHead + System2Agent
    # TODO: build IntrinsicRewardModule + TrajectoryBuffer
    # TODO: build PPODualReward or GRPO
    # TODO: training loop: rollout -> add to buffer -> update
    raise NotImplementedError(
        "Training loop not implemented; scaffold only. "
        "Implement concrete Env, Agent, Reward, Buffer, and Algorithm, then wire here."
    )


if __name__ == "__main__":
    main()
