"""
MultiWOZLoader: MultiWOZ 2.2/2.3 as a Goal-Oriented Environment for Cognitive Stackelberg.

- Hidden state y* = User Goal (parsed non-empty constraints per domain).
- Observation o_t = User utterance at each turn.
- Dialogue history = list of (speaker, text).
- get_user_goal(dialogue_id) = Oracle for reward model (slot checking).
"""

from __future__ import annotations

import json
import random
from typing import Any, Iterator

from datasets import load_dataset

from ..base_dataset import BaseStackelbergDataset


# ---------------------------------------------------------------------------
# Dataset source: prefer 2.2 on HF; fallback to alias
# ---------------------------------------------------------------------------
MULTIWOZ_HF_IDS = ["pfb30/multi_woz_v22", "multi_woz_v22"]


def _normalize_slot_name(s: str) -> str:
    """e.g. 'price range' -> 'pricerange', 'book time' -> 'book_time'."""
    return s.strip().lower().replace(" ", "_").replace("-", "_")


def _extract_goal_from_raw(goal_raw: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Parse MultiWOZ-style goal dict into list of {domain, info/constraints}.
    Example raw: {'restaurant': {'food': 'italian', 'pricerange': 'cheap', ...}, 'hotel': {}}
    Or nested:   {'restaurant': {'info': {'food': 'italian'}, 'book': {'time': '12:00'}}, ...}
    Target: [{"domain": "restaurant", "info": {"food": "italian", "pricerange": "cheap"}}, ...]
    """
    result = []
    for domain, constraints in goal_raw.items():
        if not isinstance(constraints, dict):
            continue
        info = {}
        for k, v in constraints.items():
            if k in ("info", "reqt", "book") and isinstance(v, dict):
                for sk, sv in v.items():
                    if sv is not None and str(sv).strip() and str(sv).lower() not in ("?", ""):
                        info[_normalize_slot_name(sk)] = str(sv).strip()
            elif v is not None and str(v).strip() and str(v).lower() not in ("?", ""):
                info[_normalize_slot_name(k)] = str(v).strip()
        if info:
            result.append({"domain": domain.strip().lower(), "info": info})
    return result if result else [{"domain": "unknown", "info": {}}]


def _goal_from_frames(turns: list[Any]) -> list[dict[str, Any]]:
    """
    Infer user goal from user-turn frames (state.slots_values).
    Returns same format as _extract_goal_from_raw: [{"domain": "...", "info": {...}}, ...].
    """
    # domain -> {slot -> value}; merge across turns
    by_domain: dict[str, dict[str, Any]] = {}
    for turn in turns:
        frames = turn.get("frames") or turn.get("frame") or []
        if not isinstance(frames, list):
            frames = [frames] if frames else []
        for fr in frames:
            if not isinstance(fr, dict):
                continue
            services = fr.get("service") or fr.get("services")
            if isinstance(services, str):
                services = [services]
            if not services:
                continue
            state = fr.get("state") or {}
            if isinstance(state, list):
                state = state[0] if state else {}
            slots_values = state.get("slots_values") or state.get("slot_values") or {}
            # Handle different shapes: slots_values can be {"slots_values_list": [[v]], "slots_values_name": [slot]}
            names = slots_values.get("slots_values_name") or slots_values.get("slot_name") or []
            values = slots_values.get("slots_values_list") or slots_values.get("value") or []
            if isinstance(values, list) and values and isinstance(values[0], list):
                values = values[0] if values[0] else []
            if not isinstance(names, list):
                names = [names]
            if not isinstance(values, list):
                values = [values]
            for svc in services:
                if not svc:
                    continue
                d = by_domain.setdefault(svc.strip().lower(), {})
                for i, name in enumerate(names):
                    if name and i < len(values) and values[i]:
                        v = values[i]
                        if isinstance(v, list):
                            v = v[0] if v else None
                        if v is not None and str(v).strip():
                            d[_normalize_slot_name(str(name))] = str(v).strip()
    out = [{"domain": d, "info": info} for d, info in by_domain.items() if info]
    return out if out else [{"domain": "unknown", "info": {}}]


def _turns_to_history(turns: list[Any]) -> list[tuple[str, str]]:
    """Build list of (speaker, text). Speaker in ('user', 'system')."""
    history = []
    for i, t in enumerate(turns):
        text = (t.get("utterance") or t.get("text") or "").strip()
        if not text:
            continue
        speaker = (t.get("speaker") or t.get("role") or "").strip().lower()
        if not speaker:
            # Infer: user turns often have 'frames' with state (HF 2.2)
            has_frames = bool(t.get("frames") or t.get("frame"))
            if has_frames:
                speaker = "user"
            else:
                # Original MultiWOZ log: first turn is user, then alternate
                speaker = "user" if (i % 2) == 0 else "system"
        if speaker not in ("user", "system"):
            speaker = "user" if "user" in speaker else "system"
        history.append((speaker, text))
    return history


def _first_user_utterance(history: list[tuple[str, str]]) -> str:
    for sp, text in history:
        if sp == "user":
            return text
    return ""


class MultiWOZLoader(BaseStackelbergDataset):
    """
    Load MultiWOZ 2.2 (or 2.x) from Hugging Face; parse goals and dialogue history.
    Adheres to BaseStackelbergDataset: hidden_state = user goal, context = first user utterance.
    """

    def __init__(
        self,
        data_path: str | None = None,
        hf_id: str | None = None,
        split: str = "train",
        cache_dir: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(data_path=data_path, **kwargs)
        self.split = split
        self.cache_dir = cache_dir
        self._hf_id = hf_id
        self._data = None
        self._dialogue_id_to_index: dict[str, int] = {}
        self._load_data()

    def _load_data(self) -> None:
        hf_id = self._hf_id
        if not hf_id:
            for candidate in MULTIWOZ_HF_IDS:
                try:
                    ds = load_dataset(candidate, cache_dir=self.cache_dir, trust_remote_code=True)
                    hf_id = candidate
                    break
                except Exception:
                    continue
            if not hf_id:
                raise FileNotFoundError("Could not load MultiWOZ from any of " + str(MULTIWOZ_HF_IDS))
        else:
            ds = load_dataset(hf_id, cache_dir=self.cache_dir, trust_remote_code=True)
        self._hf_id = hf_id
        if self.split not in ds:
            self.split = list(ds.keys())[0]
        self._data = ds[self.split]
        # Build dialogue_id -> index
        for i in range(len(self._data)):
            row = self._data[i]
            did = row.get("id") or row.get("dialogue_id") or row.get("dialog_id")
            if did is None:
                did = f"dialog_{i}"
            if isinstance(did, (list, tuple)):
                did = did[0] if did else f"dialog_{i}"
            self._dialogue_id_to_index[str(did).strip()] = i

    def _parse_goal(self, row: dict[str, Any]) -> list[dict[str, Any]]:
        goal_raw = row.get("goal")
        if goal_raw and isinstance(goal_raw, dict):
            return _extract_goal_from_raw(goal_raw)
        turns = row.get("turns") or row.get("dialogue") or row.get("log") or []
        return _goal_from_frames(turns)

    def _get_turns(self, row: dict[str, Any]) -> list[Any]:
        turns = row.get("turns") or row.get("dialogue") or row.get("log") or []
        if isinstance(turns, dict):
            turns = turns.get("turns", turns.get("dialogue", []))
        return turns if isinstance(turns, list) else []

    def _row_to_sample(self, index: int, row: dict[str, Any]) -> dict[str, Any]:
        turns = self._get_turns(row)
        history = _turns_to_history(turns)
        goal_list = self._parse_goal(row)
        first_utt = _first_user_utterance(history)
        dialogue_id = row.get("id") or row.get("dialogue_id") or row.get("dialog_id") or f"dialog_{index}"
        if isinstance(dialogue_id, (list, tuple)):
            dialogue_id = dialogue_id[0] if dialogue_id else f"dialog_{index}"
        dialogue_id = str(dialogue_id).strip()

        # Single canonical hidden goal for this dialogue (first domain with info, or merged)
        if goal_list:
            hidden_user_goal = goal_list[0]
            if len(goal_list) > 1:
                hidden_user_goal = {"domain": goal_list[0]["domain"], "info": {}}
                for g in goal_list:
                    hidden_user_goal["info"].update(g.get("info") or {})
        else:
            hidden_user_goal = {"domain": "unknown", "info": {}}

        return {
            "dialogue_id": dialogue_id,
            "hidden_state": hidden_user_goal,
            "y_star": hidden_user_goal,
            "hidden_user_goal": hidden_user_goal,
            "goal_per_domain": goal_list,
            "context": first_utt,
            "observation": first_utt,
            "first_user_utterance": first_utt,
            "dialogue_history": history,
            "turns": turns,
            "metadata": {"index": index},
        }

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self._data[index]
        if isinstance(row, dict):
            pass
        else:
            row = dict(self._data[index])
        return self._row_to_sample(index, row)

    def sample(self, batch_size: int = 1, replace: bool = False, **kwargs: Any) -> Iterator[dict[str, Any]]:
        n = len(self)
        if n == 0:
            return
        size = min(batch_size, n) if not replace else batch_size
        indices = random.sample(range(n), size) if not replace else random.choices(range(n), k=size)
        for i in indices:
            yield self[i]

    def get_user_goal(self, dialogue_id: str) -> dict[str, Any]:
        """
        Return the Oracle user goal for this dialogue (for reward model / slot checking).
        dialogue_id: string id of the dialogue (e.g. 'SNG01856.json' or 'dialog_0').
        """
        idx = self._dialogue_id_to_index.get(str(dialogue_id).strip())
        if idx is None:
            raise KeyError(f"Unknown dialogue_id: {dialogue_id}")
        sample = self[idx]
        return sample["hidden_user_goal"]

    def get_dialogue_id_from_index(self, index: int) -> str:
        row = self._data[index]
        if isinstance(row, dict):
            did = row.get("id") or row.get("dialogue_id") or row.get("dialog_id")
        else:
            did = dict(row).get("id") or dict(row).get("dialogue_id") or dict(row).get("dialog_id")
        return str(did).strip() if did is not None else f"dialog_{index}"

    def get_hidden_state_from_sample(self, sample: dict[str, Any]) -> Any:
        return sample.get("hidden_state") or sample.get("hidden_user_goal") or sample.get("y_star")

    def get_initial_context_from_sample(self, sample: dict[str, Any]) -> Any:
        return sample.get("first_user_utterance") or sample.get("context") or sample.get("observation")


# ---------------------------------------------------------------------------
# Test / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("Loading MultiWOZ 2.2...", flush=True)
    try:
        loader = MultiWOZLoader()
    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
        print(f"Load error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Dataset size: {len(loader)} dialogues.\n")

    # Sample case: User Goal (Hidden) vs User Utterance (Visible)
    sample = loader[0]
    out = {
        "dialogue_id": sample["dialogue_id"],
        "hidden_user_goal": sample["hidden_user_goal"],
        "first_user_utterance": sample["first_user_utterance"],
    }
    print("Sample case (Hidden Goal vs Visible Utterance):")
    print(json.dumps(out, indent=2, ensure_ascii=False))

    # Test get_user_goal
    did = sample["dialogue_id"]
    goal = loader.get_user_goal(did)
    print(f"\nget_user_goal({repr(did)}) -> {json.dumps(goal, indent=2, ensure_ascii=False)}")
