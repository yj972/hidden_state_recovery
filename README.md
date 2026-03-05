# Cognitive Stackelberg RL — System 2 Reasoning Agents

Modular RL framework: agent performs **Thought** (CoT) then **Action** to reduce uncertainty about hidden intent \(y^*\). Process reward \(r_{PRM}\) from **RewardModel** (LLM-as-Judge or heuristic); task reward \(R_{task}\) sparse.

## Structure

```
├── core/           # ABCs: StackelbergEnv, System2Agent, RewardModel, TrajectoryBuffer
├── data/           # base_dataset, medical (DDXPlus), multiwoz, mcts_synthesis
├── algorithms/     # DualStepPPOTrainer, make_experience, GAE, RolloutBuffer; GRPO stub
├── models/         # TransformerBackbone, ValueHead (ABCs); agent_lm (HFSystem2Agent)
├── utils/
└── train.py        # Entry: dataset → StackelbergPOMDP → Agent → RewardModel → Trainer
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Dry run (no large downloads)
python train.py --dry_run --max_steps 5

# Train
python train.py --dataset ddxplus --model_name Qwen/Qwen2.5-1.5B-Instruct --max_steps 100

# MCTS trajectory synthesis for SFT
python data/mcts_synthesis.py   # writes outputs/mcts_sft.jsonl
```

## Tech

Python 3.10+, PyTorch, Gymnasium, Transformers, Hugging Face datasets.
