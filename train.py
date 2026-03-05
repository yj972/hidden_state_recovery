"""
Main entry point: orchestrate training for System 2 Reasoning Agents (Cognitive Stackelberg).

Wires: Dataset -> StackelbergPOMDP Env -> System2Agent (HF LM) -> RewardModel -> DualStepPPOTrainer.
Supports --dry_run with dummy model/dataset for debugging.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import torch

# ---------------------------------------------------------------------------
# Imports: core, data, algorithms
# ---------------------------------------------------------------------------
from core.agent import System2Agent
from core.env import StackelbergEnv
from core.rewards import HeuristicRewardModel, LLMRewardModel, RewardModel
from algorithms.ppo_dual import DualStepPPOTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train System 2 RL agents (Cognitive Stackelberg)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["ddxplus", "multiwoz"],
        default="ddxplus",
        help="Dataset: ddxplus (medical) or multiwoz (goal-oriented dialogue)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Hugging Face model ID (e.g. Qwen/Qwen2.5-1.5B-Instruct or meta-llama/Llama-2-7b-chat)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=100,
        help="Total training steps",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="PPO rollout batch size (episodes per collect)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-6,
        help="Learning rate (conservative for LM fine-tuning)",
    )
    parser.add_argument(
        "--reward_model",
        type=str,
        choices=["llm", "heuristic"],
        default="heuristic",
        help="Process reward: llm (LLM-as-Judge) or heuristic (rule-based)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Checkpoint and log directory",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=20,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--max_steps_per_episode",
        type=int,
        default=15,
        help="Max env steps per episode",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Use dummy model and dataset to test loop without downloading large weights",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Log to Weights & Biases (optional)",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="cognitive-stackelberg",
        help="W&B project name",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def get_dataset(name: str, **kwargs: Any) -> Any:
    if name == "ddxplus":
        from data.medical import DDXPlusEnvironment
        return DDXPlusEnvironment(**kwargs)
    if name == "multiwoz":
        from data.multiwoz import MultiWOZLoader
        return MultiWOZLoader(**kwargs)
    raise ValueError(f"Unknown dataset: {name}")


# ---------------------------------------------------------------------------
# Environment: StackelbergPOMDP wrapping data loader
# ---------------------------------------------------------------------------


class StackelbergPOMDP(StackelbergEnv):
    """
    Gymnasium env that wraps a data loader (DDXPlus or MultiWOZ).
    One episode = one sampled task (e.g. one patient or one dialogue).
    """

    def __init__(self, data_loader: Any, max_steps_per_episode: int = 20, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.data_loader = data_loader
        self.max_steps_per_episode = max_steps_per_episode
        self._current_index: int = 0
        self._step_count: int = 0
        self._history: list[Any] = []
        self._current_sample: dict[str, Any] = {}

    def get_hidden_state(self) -> Any:
        return self._current_sample.get("hidden_state") or self._current_sample.get("y_star")

    def get_belief_state(self) -> Any:
        return self._history  # Simplified: history as proxy for belief

    def sample_hidden_state(self) -> Any:
        n = len(self.data_loader)
        self._current_index = random.randint(0, n - 1) if n > 0 else 0
        self._current_sample = self.data_loader[self._current_index]
        return self.get_hidden_state()

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        if seed is not None:
            random.seed(seed)
        self.sample_hidden_state()
        self._step_count = 0
        self._history = []
        obs = self._current_sample.get("initial_complaint") or self._current_sample.get("context") or self._current_sample.get("observation") or ""
        if not obs and "first_user_utterance" in self._current_sample:
            obs = self._current_sample["first_user_utterance"]
        info = {"hidden_state": self.get_hidden_state(), "index": self._current_index}
        return obs, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        self._step_count += 1
        self._history.append((str(self._current_sample.get("context", "")), str(action)))
        # Simulate user response: for DDXPlus use check_symptom if available
        if hasattr(self.data_loader, "check_symptom"):
            try:
                yes = self.data_loader.check_symptom(self._current_index, str(action))
                next_obs = f"User says: {'Yes' if yes else 'No'}"
            except Exception:
                next_obs = "User says: I'm not sure."
        else:
            next_obs = "User says: Okay."
        r_ext = 0.0
        terminated = self._step_count >= self.max_steps_per_episode
        truncated = False
        if "Answer:" in str(action) or "answer:" in str(action).lower():
            terminated = True
        info = {"outcome": None, "step": self._step_count}
        return next_obs, r_ext, terminated, truncated, info


# ---------------------------------------------------------------------------
# Reward model
# ---------------------------------------------------------------------------


def get_reward_model(name: str, **kwargs: Any) -> RewardModel:
    if name == "llm":
        return LLMRewardModel(**kwargs)
    if name == "heuristic":
        return HeuristicRewardModel(per_turn_penalty=-0.1, constant_reward=0.0, **kwargs)
    raise ValueError(f"Unknown reward model: {name}")


# ---------------------------------------------------------------------------
# Dummy agent and env for dry run
# ---------------------------------------------------------------------------


class DummyAgent(System2Agent):
    """Returns fixed thought/action and fake log_probs for dry-run debugging."""

    def think(self, observation: Any, **kwargs: Any) -> str:
        return "Is it flu?"

    def get_thought_logits_or_sample(
        self,
        observation: Any,
        deterministic: bool = False,
        **kwargs: Any,
    ) -> tuple[str, dict[str, Any] | None]:
        return "Is it flu?", {"log_prob": torch.tensor(-0.5)}

    def act(self, observation: Any, thought: Any, **kwargs: Any) -> str:
        return "Do you have fever?"

    def get_action_logits_or_sample(
        self,
        observation: Any,
        thought: Any,
        deterministic: bool = False,
        **kwargs: Any,
    ) -> tuple[str, dict[str, Any] | None]:
        return "Do you have fever?", {"log_prob": torch.tensor(-0.3)}


class DummyEnv(StackelbergEnv):
    """Fixed transitions for dry run."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._step_count = 0

    def get_hidden_state(self) -> Any:
        return "migraine"

    def get_belief_state(self) -> Any:
        return []

    def sample_hidden_state(self) -> Any:
        return "migraine"

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        self._step_count = 0
        return "headache", {}

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        self._step_count += 1
        next_obs = "User says: Yes" if self._step_count == 1 else "User says: No"
        done = self._step_count >= 3
        return next_obs, 0.5 if done else 0.0, done, False, {}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    # Optional wandb
    if args.use_wandb:
        try:
            import wandb
            wandb.init(project=args.wandb_project, config=vars(args))
        except ImportError:
            args.use_wandb = False

    # 1) Dataset
    if args.dry_run:
        data_loader = None  # not used for DummyEnv
        print("[DRY RUN] Using dummy dataset (no download).", flush=True)
    else:
        try:
            data_loader = get_dataset(args.dataset)
            print(f"Loaded dataset: {args.dataset}, size = {len(data_loader)}", flush=True)
        except Exception as e:
            print(f"Failed to load dataset: {e}", file=sys.stderr)
            return 1

    # 2) Environment
    if args.dry_run:
        env = DummyEnv()
        print("[DRY RUN] Using DummyEnv.", flush=True)
    else:
        env = StackelbergPOMDP(
            data_loader,
            max_steps_per_episode=args.max_steps_per_episode,
        )

    # 3) Agent
    if args.dry_run:
        agent = DummyAgent()
        tokenizer = None
        model = None
        optimizer = None
        print("[DRY RUN] Using DummyAgent (no HF download).", flush=True)
    else:
        try:
            from models.agent_lm import HFSystem2Agent
            agent = HFSystem2Agent(args.model_name, device=device)
            tokenizer = agent.tokenizer
            model = agent.model
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        except Exception as e:
            print(f"Failed to load agent: {e}", file=sys.stderr)
            return 1

    # 4) Reward model
    reward_model = get_reward_model(args.reward_model)
    print(f"Reward model: {args.reward_model}", flush=True)

    # 5) Trainer
    trainer = DualStepPPOTrainer(
        agent=agent,
        reward_model=reward_model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        model=model,
        weight_process=1.0,
        weight_task=1.0,
        device=device,
    )

    # Training loop with KeyboardInterrupt handling
    try:
        for step in range(args.max_steps):
            # Rollout (make_experience)
            rollouts = trainer.rollout(
                env,
                num_episodes=args.batch_size,
                max_steps_per_episode=args.max_steps_per_episode,
                seed=args.seed + step if args.seed is not None else None,
                clear_buffer=True,
            )
            if not rollouts:
                continue
            mean_reward = sum(r["reward_total"] for r in rollouts) / len(rollouts)
            mean_episode_length = len(rollouts)  # total steps in buffer; could group by episode
            entropy_bonus = 0.0  # placeholder; could add from agent aux if desired

            # Update
            logs = trainer.train_step(batch_size=min(args.batch_size * 2, len(rollouts)))
            logs["mean_reward"] = mean_reward
            logs["mean_episode_length"] = mean_episode_length
            logs["entropy_bonus"] = entropy_bonus
            logs["step"] = step

            if step % 10 == 0 or step == args.max_steps - 1:
                print(
                    f"Step {step}: mean_reward={mean_reward:.4f} "
                    f"policy_loss={logs.get('policy_loss', 0):.4f} "
                    f"ep_len={mean_episode_length}",
                    flush=True,
                )
            if args.use_wandb:
                try:
                    wandb.log(logs)
                except Exception:
                    pass

            # Dry-run debug print (first episode only)
            if args.dry_run and step == 0 and rollouts:
                r0 = rollouts[0]
                print(
                    "[DRY RUN] Episode 1: User says 'headache' -> "
                    f"Agent thinks '{r0.get('thought', '')}' -> "
                    f"Agent asks '{r0.get('action', '')}' -> "
                    f"Reward {r0.get('reward_total', 0):.2f}",
                    flush=True,
                )

            # Checkpoint
            if (step + 1) % args.save_every == 0 and not args.dry_run and model is not None:
                ckpt_path = Path(args.output_dir) / f"checkpoint_step_{step+1}"
                ckpt_path.mkdir(parents=True, exist_ok=True)
                try:
                    model.save_pretrained(ckpt_path)
                    if tokenizer is not None:
                        tokenizer.save_pretrained(ckpt_path)
                    with open(ckpt_path / "trainer_state.json", "w") as f:
                        json.dump({"step": step + 1, "args": vars(args)}, f, indent=2)
                    print(f"Saved checkpoint to {ckpt_path}", flush=True)
                except Exception as e:
                    print(f"Checkpoint save failed: {e}", flush=True)

    except KeyboardInterrupt:
        print("\nInterrupted; saving checkpoint...", flush=True)
        if not args.dry_run and model is not None:
            ckpt_path = Path(args.output_dir) / "checkpoint_interrupt"
            ckpt_path.mkdir(parents=True, exist_ok=True)
            try:
                model.save_pretrained(ckpt_path)
                if tokenizer is not None:
                    tokenizer.save_pretrained(ckpt_path)
                print(f"Saved interrupt checkpoint to {ckpt_path}", flush=True)
            except Exception as e:
                print(f"Interrupt save failed: {e}", flush=True)
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
