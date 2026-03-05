"""MCTS for medical cases: expand with SOTA policy, simulate with DDXPlus; save best trajectory as SFT gold."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

@dataclass
class MCTSNode:
    """state = [(thought, action, user_response), ...]; root = [(None, None, initial_complaint)]."""
    state: list[tuple[Any, Any, Any]]  # (thought, action, user_response) per turn
    parent: MCTSNode | None = None
    children: list[MCTSNode] = field(default_factory=list)
    action_from_parent: str | None = None  # the question that led here from parent
    thought_from_parent: str | None = None
    visits: int = 0
    value: float = 0.0  # Q: average return
    depth: int = 0

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def is_root(self) -> bool:
        return self.parent is None

    def uct_score(self, c: float = 1.4) -> float:
        """Upper Confidence Bound: Q/N + c * sqrt(log(parent.N) / N)."""
        if self.visits == 0:
            return float("inf")
        p = self.parent
        if p is None or p.visits == 0:
            return self.value / self.visits
        return self.value / self.visits + c * math.sqrt(math.log(p.visits + 1) / self.visits)


TOP_K_ACTIONS_PROMPT = """You are a clinical diagnostician. Given the conversation history below, propose exactly {k} DISTINCT strategic questions to narrow down the patient's diagnosis. Each question should be a single short sentence (e.g. "Do you have chest pain?"). Output ONLY a JSON array of {k} strings, no other text.

Conversation history:
{history}

JSON array of {k} questions:"""


def get_top_k_actions_openai(
    history: str,
    k: int = 3,
    api_client: Any = None,
    model: str = "gpt-4o-mini",
) -> list[str]:
    """Call OpenAI (or compatible) API to get k distinct diagnostic questions."""
    if api_client is None:
        try:
            import openai
            api_client = openai.OpenAI()
        except Exception:
            return get_top_k_actions_mock(history, k)
    prompt = TOP_K_ACTIONS_PROMPT.format(k=k, history=history[:3000])
    try:
        resp = api_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        text = resp.choices[0].message.content.strip()
        # Parse JSON array
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        actions = json.loads(text)
        if isinstance(actions, list):
            return [str(a).strip() for a in actions[:k]]
        return [str(actions).strip()][:k]
    except Exception:
        return get_top_k_actions_mock(history, k)


def get_top_k_actions_mock(history: str, k: int = 3) -> list[str]:
    """Fallback: return k generic questions when no API available."""
    pool = [
        "Do you have fever?",
        "Do you have chest pain?",
        "Do you have shortness of breath?",
        "Do you have headache?",
        "Do you have nausea?",
        "Do you have fatigue?",
        "Do you have cough?",
        "Do you have abdominal pain?",
    ]
    random.shuffle(pool)
    return pool[:k]


def get_top_k_actions(
    history: str,
    k: int = 3,
    api_client: Any = None,
    use_mock: bool = False,
) -> list[str]:
    """Public API: get k distinct strategic questions. Uses OpenAI if available else mock."""
    if use_mock or api_client is None and not _has_openai():
        return get_top_k_actions_mock(history, k)
    return get_top_k_actions_openai(history, k=k, api_client=api_client)


def _has_openai() -> bool:
    try:
        import openai
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Simulation: User model via DDXPlusEnvironment
# ---------------------------------------------------------------------------


def user_response_for_action(
    env: Any,
    patient_id: int,
    action: str,
) -> str:
    """Deterministic: ask env (DDXPlusEnvironment) whether patient has this symptom -> Yes/No."""
    if not hasattr(env, "check_symptom"):
        return "No"
    try:
        yes = env.check_symptom(patient_id, action)
        return "Yes" if yes else "No"
    except Exception:
        return "No"


# ---------------------------------------------------------------------------
# MCTS Loop for one case
# ---------------------------------------------------------------------------


def history_from_node(node: MCTSNode) -> str:
    """Format node's state as string for policy prompt."""
    lines = []
    for i, (thought, action, resp) in enumerate(node.state):
        if i == 0 and action is None and resp:
            lines.append(f"Initial complaint: {resp}")
            continue
        if thought:
            lines.append(f"Thought: {thought}")
        if action:
            lines.append(f"Question: {action}")
        if resp:
            lines.append(f"Patient: {resp}")
    return "\n".join(lines) if lines else "No history yet."


def run_rollout(
    env: Any,
    patient_id: int,
    start_state: list[tuple[Any, Any, Any]],
    get_actions: Callable[[str], list[str]],
    max_rollout_depth: int,
    ground_truth: Any,
) -> float:
    """
    From start_state, take greedy/random actions until max_rollout_depth.
    Return +1 if we got at least one "Yes" (positive finding), else -1.
    """
    state = list(start_state)
    found_positive = False
    for _ in range(max_rollout_depth):
        hist = history_from_node(MCTSNode(state=state))
        actions = get_actions(hist)
        if not actions:
            break
        action = random.choice(actions)
        resp = user_response_for_action(env, patient_id, action)
        if resp == "Yes":
            found_positive = True
        state.append((None, action, resp))
    return 1.0 if found_positive else -1.0


def mcts_one_case(
    env: Any,
    patient_id: int,
    initial_complaint: str,
    get_actions: Callable[[str], list[str]],
    num_simulations: int = 50,
    max_depth: int = 15,
    max_rollout_depth: int = 5,
    uct_c: float = 1.4,
) -> tuple[list[tuple[Any, Any, Any]], list[dict[str, Any]]]:
    """
    Run MCTS for one medical case. Return the best trajectory and per-step info (visit counts).
    """
    root = MCTSNode(state=[(None, None, initial_complaint)], depth=0)
    trajectory: list[tuple[Any, Any, Any]] = []
    step_infos: list[dict[str, Any]] = []
    current = root

    for step in range(max_depth):
        # Run num_simulations from current node
        for _ in range(num_simulations):
            # Select: UCT to leaf
            node = current
            while not node.is_leaf() and node.children:
                best = max(node.children, key=lambda c: c.uct_score(c=uct_c))
                node = best
            # Expand if leaf and not terminal
            if node.is_leaf() and node.depth < max_depth:
                hist = history_from_node(node)
                actions = get_actions(hist)
                added = []
                for action in actions[:3]:  # top 3 branches
                    resp = user_response_for_action(env, patient_id, action)
                    child_state = node.state + [(None, action, resp)]
                    child = MCTSNode(
                        state=child_state,
                        parent=node,
                        action_from_parent=action,
                        thought_from_parent=None,
                        depth=node.depth + 1,
                    )
                    node.children.append(child)
                    added.append(child)
                # Simulate (rollout) from one of the new children, then backprop
                if added:
                    child = random.choice(added)
                    outcome = run_rollout(
                        env, patient_id, child.state,
                        get_actions, max_rollout_depth,
                        ground_truth=None,
                    )
                    backprop(child, outcome)
            else:
                # Already expanded: rollout from this node and backprop
                outcome = run_rollout(
                    env, patient_id, node.state,
                    get_actions, max_rollout_depth,
                    ground_truth=None,
                )
                backprop(node, outcome)

        if not current.children:
            break
        # Pick child with highest visit count as next action
        best_child = max(current.children, key=lambda c: c.visits)
        thought = best_child.thought_from_parent or ""
        action = best_child.action_from_parent or ""
        resp = best_child.state[-1][2] if best_child.state else ""
        trajectory.append((thought, action, resp))
        step_infos.append({
            "visits": best_child.visits,
            "value": best_child.value,
            "children_visits": [c.visits for c in current.children],
        })
        current = best_child
        if not current.children and current.depth < max_depth:
            hist = history_from_node(current)
            actions = get_actions(hist)
            for action in actions[:3]:
                resp = user_response_for_action(env, patient_id, action)
                child_state = current.state + [(None, action, resp)]
                child = MCTSNode(
                    state=child_state,
                    parent=current,
                    action_from_parent=action,
                    depth=current.depth + 1,
                )
                current.children.append(child)

    return trajectory, step_infos


def backprop(node: MCTSNode, outcome: float) -> None:
    """Propagate outcome (+1 or -1) up the tree; update visits and value."""
    n = node
    while n is not None:
        n.visits += 1
        n.value += outcome
        n = n.parent


# ---------------------------------------------------------------------------
# Thought CoT synthesis (optional): summarize why we chose this action
# ---------------------------------------------------------------------------


def synthesize_thought_cot(
    history: str,
    action: str,
    visit_counts: list[int] | None = None,
) -> str:
    """Produce a short thought_cot string for SFT (e.g. from template or API)."""
    parts = [f"Given the history, I chose to ask: {action}."]
    if visit_counts:
        parts.append(f"(MCTS visit distribution: {visit_counts})")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Output: Save best trajectory as Gold for SFT
# ---------------------------------------------------------------------------


def trajectory_to_sft_format(
    trajectory: list[tuple[Any, Any, Any]],
    step_infos: list[dict[str, Any]],
    initial_complaint: str,
    case_id: str = "",
) -> list[dict[str, Any]]:
    """
    Convert MCTS trajectory to SFT examples. Each step becomes one training example
    with history, thought_cot, action_label. Visit counts can be used as distribution target.
    """
    examples = []
    history_parts = [f"Initial complaint: {initial_complaint}"]
    for i, (thought, action, user_resp) in enumerate(trajectory):
        if not action:
            continue
        info = step_infos[i] if i < len(step_infos) else {}
        thought_cot = thought or synthesize_thought_cot(
            "\n".join(history_parts),
            action,
            info.get("children_visits"),
        )
        examples.append({
            "history": "\n".join(history_parts),
            "thought_cot": thought_cot,
            "action_label": action,
            "user_response": user_resp,
            "visit_count": info.get("visits"),
            "children_visits": info.get("children_visits"),
            "case_id": case_id,
        })
        history_parts.append(f"Question: {action}")
        history_parts.append(f"Patient: {user_resp}")
    return examples


def run_mcts_synthesis(
    env: Any,
    num_cases: int = 10,
    get_actions: Callable[[str], list[str]] | None = None,
    num_simulations: int = 50,
    max_depth: int = 15,
    max_rollout_depth: int = 5,
    output_path: str | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """
    Run MCTS for num_cases; save best trajectories as SFT gold. Returns all SFT examples.
    """
    random.seed(seed)
    if get_actions is None:
        get_actions = lambda h: get_top_k_actions(h, k=3, use_mock=True)
    all_examples = []
    n = len(env)
    indices = random.sample(range(n), min(num_cases, n)) if n else []
    for idx in indices:
        sample = env[idx]
        initial = sample.get("initial_complaint") or sample.get("context") or ""
        if not initial:
            continue
        trajectory, step_infos = mcts_one_case(
            env,
            patient_id=idx,
            initial_complaint=initial,
            get_actions=get_actions,
            num_simulations=num_simulations,
            max_depth=max_depth,
            max_rollout_depth=max_rollout_depth,
        )
        examples = trajectory_to_sft_format(
            trajectory,
            step_infos,
            initial_complaint=initial,
            case_id=str(idx),
        )
        all_examples.extend(examples)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for ex in all_examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    return all_examples


# ---------------------------------------------------------------------------
# CLI / test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    print("MCTS Synthesis (dry run with mock policy)...", flush=True)
    try:
        from data.medical import DDXPlusEnvironment
        env = DDXPlusEnvironment()
    except Exception as e:
        print(f"DDXPlus load failed: {e}. Using dummy.", flush=True)
        env = None
    if env is None or len(env) == 0:
        class DummyEnv:
            def __len__(self): return 2
            def __getitem__(self, i):
                return {"initial_complaint": "I have a headache.", "context": "I have a headache."}
            def check_symptom(self, pid, q): return random.random() > 0.5
        env = DummyEnv()
    examples = run_mcts_synthesis(
        env,
        num_cases=1,
        num_simulations=8,
        max_depth=3,
        max_rollout_depth=2,
        output_path="outputs/mcts_sft.jsonl",
        seed=42,
    )
    print(f"Generated {len(examples)} SFT examples.")
    if examples:
        print("Sample:", json.dumps(examples[0], indent=2, ensure_ascii=False))
