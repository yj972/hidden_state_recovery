# Cognitive Stackelberg RL — System 2 Reasoning Agents

Modular Reinforcement Learning framework for **System 2** agents in a Cognitive Stackelberg Game: the agent performs internal **Thought** (Chain-of-Thought) to reduce entropy about a hidden user intent \(y^*\) before committing to an external **Action**.

## Theoretical Components

| Concept | Symbol | Description |
|--------|--------|-------------|
| Hidden state | \(y^*\) | Ground truth (e.g. true disease, bug root cause) |
| Belief state | \(b_t\) | Agent's distribution over \(y^*\) |
| Process reward | \(r_{PRM}\) | Intrinsic; information gain / entropy reduction \(\Delta H(b_t)\) |
| Task reward | \(R_{task}\) | Sparse; correct identification of \(y^*\) |
| Meta-actions | — | **Ask** (information seeking), **Hypothesize** (reasoning), **Answer** (terminal) |

## Project Structure

```
project_root/
├── core/                   # Abstract Base Classes
│   ├── env.py              # StackelbergEnv (POMDP wrapper)
│   ├── agent.py            # System2Agent (Thought + Action policies)
│   ├── rewards.py          # IntrinsicRewardModule (entropy / info gain)
│   └── memory.py           # TrajectoryBuffer (thought–action traces)
├── data/                   # Data adapters
│   ├── base_dataset.py     # Abstract dataset loader
│   ├── medical/            # DDXPlus (diagnosis)
│   └── coding/             # Stackelberg Coder (ambiguous bugs)
├── algorithms/             # RL algorithms
│   ├── ppo_dual.py         # PPO with dual rewards
│   └── grpo.py             # Group Relative Policy Optimization
├── models/                 # Neural networks
│   ├── backbone.py         # HF Transformer wrapper (Llama/Qwen)
│   └── critic.py           # Value function head
├── utils/
└── train.py                # Main entry point
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Scaffold only; concrete implementations are left for downstream work. Entry point:

```bash
python train.py --env medical --algorithm ppo_dual --epochs 100
```

## Tech Stack

- **Python 3.10+**, **PyTorch**, **Gymnasium**, **Transformers**
