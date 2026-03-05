"""
DDXPlusLoader & DDXPlusEnvironment: DDXPlus dataset and User Simulation for Cognitive Stackelberg.

- Loads from Hugging Face (sylvain59/DDXPlus, english).
- Maps symptom/pathology IDs to text via release dictionary.
- Hidden state y* = pathology; Oracle O_all = list of present symptoms.
- check_symptom(patient_id, symptom_name) simulates user answering "Do you have X?" -> Yes/No.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download

from ..base_dataset import BaseStackelbergDataset


# ---------------------------------------------------------------------------
# Default Hugging Face dataset ID and config
# ---------------------------------------------------------------------------
DDXPLUS_HF_ID = "sylvain59/DDXPlus"
DDXPLUS_CONFIG = "english"


class DDXPlusLoader(BaseStackelbergDataset):
    """
    Load DDXPlus from Hugging Face and map IDs to text.
    Each row = one game episode: hidden state y* = pathology, oracle = symptoms list.
    """

    def __init__(
        self,
        data_path: str | None = None,
        hf_id: str = DDXPLUS_HF_ID,
        config: str = DDXPLUS_CONFIG,
        cache_dir: str | None = None,
        mapping_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(data_path=data_path, **kwargs)
        self.hf_id = hf_id
        self.config = config
        self.cache_dir = cache_dir
        self._mapping_path = mapping_path
        self._dataset = None
        self._df: pd.DataFrame | None = None
        self._symptom_id_to_text: dict[int | str, str] = {}
        self._pathology_id_to_text: dict[int | str, str] = {}
        self._text_to_symptom_id: dict[str, int | str] = {}
        self._load_data_and_mapping()

    def _load_mapping(self) -> None:
        """Load release dictionary: symptom_id / pathology_id -> text."""
        mapping: dict[str, Any] = {}
        # 1) Explicit file path
        if self._mapping_path and Path(self._mapping_path).exists():
            with open(self._mapping_path, encoding="utf-8") as f:
                mapping = json.load(f)
        else:
            # 2) From Hugging Face dataset repo (common filenames)
            for filename in ("mapping.json", "release_dictionary.json", "release_dictionary.csv"):
                try:
                    path = hf_hub_download(
                        repo_id=self.hf_id,
                        filename=filename,
                        repo_type="dataset",
                        cache_dir=self.cache_dir,
                    )
                    with open(path, encoding="utf-8") as f:
                        if filename.endswith(".csv"):
                            mapping = {"symptoms": {}, "pathologies": {}}
                            reader = csv.DictReader(f)
                            for r in reader:
                                if "symptom" in str(r).lower():
                                    k = r.get("id") or r.get("symptom_id")
                                    v = r.get("name") or r.get("symptom_name") or r.get("question")
                                    if k is not None: mapping["symptoms"][str(k)] = v
                                if "pathology" in str(r).lower() or "disease" in str(r).lower():
                                    k = r.get("id") or r.get("pathology_id")
                                    v = r.get("name") or r.get("pathology_name") or r.get("disease")
                                    if k is not None: mapping["pathologies"][str(k)] = v
                        else:
                            mapping = json.load(f)
                    break
                except Exception:
                    continue

        # Normalize mapping structure (different repos use different keys)
        if "symptoms" in mapping:
            for k, v in mapping["symptoms"].items():
                key = int(k) if isinstance(k, str) and k.isdigit() else k
                self._symptom_id_to_text[key] = v if isinstance(v, str) else v.get("name", str(v))
        if "symptom_id_to_text" in mapping:
            for k, v in mapping["symptom_id_to_text"].items():
                key = int(k) if isinstance(k, str) and k.isdigit() else k
                self._symptom_id_to_text[key] = v if isinstance(v, str) else v.get("name", str(v))
        if "pathologies" in mapping:
            for k, v in mapping["pathologies"].items():
                key = int(k) if isinstance(k, str) and k.isdigit() else k
                self._pathology_id_to_text[key] = v if isinstance(v, str) else v.get("name", str(v))
        if "pathology_id_to_text" in mapping:
            for k, v in mapping["pathology_id_to_text"].items():
                key = int(k) if isinstance(k, str) and k.isdigit() else k
                self._pathology_id_to_text[key] = v if isinstance(v, str) else v.get("name", str(v))

        # Build reverse map for symptom text -> id (for check_symptom by name)
        for sid, text in self._symptom_id_to_text.items():
            self._text_to_symptom_id[text.strip().lower()] = sid

    def _load_data_and_mapping(self) -> None:
        """Load dataset and mapping; normalize to pandas with consistent column names."""
        # Load mapping first (may be needed to resolve initial_evidence)
        try:
            self._load_mapping()
        except Exception as e:
            # If no mapping file, we still allow loading; IDs will be used as-is
            self._symptom_id_to_text = {}
            self._pathology_id_to_text = {}
            self._text_to_symptom_id = {}

        # Load dataset from Hugging Face (try with config "english", then default)
        try:
            ds = load_dataset(
                self.hf_id,
                self.config,
                cache_dir=self.cache_dir,
                trust_remote_code=True,
            )
        except Exception:
            ds = load_dataset(
                self.hf_id,
                cache_dir=self.cache_dir,
                trust_remote_code=True,
            )
        # Use train split if available, else first split
        split = "train" if "train" in ds else list(ds.keys())[0]
        self._dataset = ds[split]
        self._df = self._dataset.to_pandas()

        # Normalize column names (handle different naming conventions)
        col_lower = {c.lower(): c for c in self._df.columns}
        self._pathology_col = col_lower.get("pathology") or col_lower.get("pathology_id") or next(
            (c for c in self._df.columns if "pathology" in c.lower()), self._df.columns[0]
        )
        self._symptoms_col = col_lower.get("symptoms") or col_lower.get("symptom") or next(
            (c for c in self._df.columns if "symptom" in c.lower() and "initial" not in c.lower()),
            None,
        )
        self._initial_col = col_lower.get("initial_evidence") or col_lower.get("initial_evidence_id") or next(
            (c for c in self._df.columns if "initial" in c.lower()), None
        )
        if self._symptoms_col is None:
            self._symptoms_col = self._df.columns[1] if len(self._df.columns) > 1 else self._df.columns[0]

    def _symptom_id_to_display(self, sid: int | str) -> str:
        return self._symptom_id_to_text.get(sid, str(sid))

    def _pathology_id_to_display(self, pid: int | str) -> str:
        return self._pathology_id_to_text.get(pid, str(pid))

    def _ensure_list(self, x: Any) -> list[Any]:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return []
        if isinstance(x, list):
            return x
        if isinstance(x, str):
            try:
                return json.loads(x) if x.strip().startswith("[") else [x]
            except Exception:
                return [x]
        return [x]

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self._df.iloc[index]
        pathology_id = row[self._pathology_col]
        pathology_text = self._pathology_id_to_display(pathology_id)
        symptom_ids = self._ensure_list(row[self._symptoms_col])
        user_positive_symptoms = [self._symptom_id_to_display(s) for s in symptom_ids]
        symptom_ids_set = set(symptom_ids)

        # Initial observation: from initial_evidence (first complaint)
        if self._initial_col and self._initial_col in row:
            raw_initial = row[self._initial_col]
            initial_ids = self._ensure_list(raw_initial)
            if initial_ids:
                first_id = initial_ids[0]
                initial_complaint = self._symptom_id_to_display(first_id)
            else:
                initial_complaint = user_positive_symptoms[0] if user_positive_symptoms else ""
        else:
            initial_complaint = user_positive_symptoms[0] if user_positive_symptoms else ""

        return {
            "hidden_state": pathology_text,
            "y_star": pathology_text,
            "ground_truth_disease": pathology_text,
            "pathology_id": pathology_id,
            "context": initial_complaint,
            "observation": initial_complaint,
            "initial_complaint": initial_complaint,
            "user_positive_symptoms": user_positive_symptoms,
            "user_positive_symptom_ids": list(symptom_ids_set),
            "oracle_symptom_ids": list(symptom_ids_set),
            "metadata": {"index": index},
        }

    def sample(self, batch_size: int = 1, replace: bool = False, **kwargs: Any) -> Iterator[dict[str, Any]]:
        indices = random.choices(range(len(self)), k=batch_size) if replace else random.sample(range(len(self)), min(batch_size, len(self)))
        for i in indices:
            yield self[i]

    def get_hidden_state_from_sample(self, sample: dict[str, Any]) -> Any:
        return sample.get("hidden_state") or sample.get("y_star")

    def get_initial_context_from_sample(self, sample: dict[str, Any]) -> Any:
        return sample.get("initial_complaint") or sample.get("context") or sample.get("observation")


# ---------------------------------------------------------------------------
# User Simulation Environment: oracle answers to symptom checks
# ---------------------------------------------------------------------------

class DDXPlusEnvironment(DDXPlusLoader):
    """
    DDXPlus as a User Simulation Environment: answers check_symptom(patient_id, symptom_name).
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._patient_symptom_ids: dict[int, set[int | str]] = {}
        for i in range(len(self._df)):
            row = self._df.iloc[i]
            syms = self._ensure_list(row[self._symptoms_col])
            self._patient_symptom_ids[i] = set(syms)

    def check_symptom(self, patient_id: int, symptom_name: str) -> bool:
        """
        Simulate user answer: "Do you have <symptom_name>?" -> True (Yes) / False (No).
        symptom_name can be exact text or normalized (e.g. "chest_pain", "Do you have chest pain?").
        """
        if patient_id < 0 or patient_id >= len(self._df):
            raise IndexError(f"patient_id {patient_id} out of range [0, {len(self)})")
        positive_ids = self._patient_symptom_ids[patient_id]

        # Resolve symptom_name to id: exact text map, or by substring match
        name_lower = symptom_name.strip().lower()
        if name_lower in self._text_to_symptom_id:
            sid = self._text_to_symptom_id[name_lower]
            return sid in positive_ids
        for text, sid in self._text_to_symptom_id.items():
            if name_lower in text or text in name_lower:
                return sid in positive_ids
        # If no mapping: treat as raw id if numeric
        try:
            sid = int(symptom_name.strip())
            return sid in positive_ids
        except ValueError:
            pass
        return False

    def get_positive_symptom_ids(self, patient_id: int) -> set[int | str]:
        """Return the set of symptom IDs that are present for this patient (oracle O_all)."""
        if patient_id < 0 or patient_id >= len(self._df):
            raise IndexError(f"patient_id {patient_id} out of range [0, {len(self)})")
        return set(self._patient_symptom_ids[patient_id])


if __name__ == "__main__":
    import sys
    print("Loading DDXPlus (english)...", flush=True)
    try:
        env = DDXPlusEnvironment()
    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
    print(f"Dataset size: {len(env)} rows.")
    sample = env[0]
    print(json.dumps({
        "initial_complaint": sample["initial_complaint"],
        "ground_truth_disease": sample["hidden_state"],
        "user_positive_symptoms": sample["user_positive_symptoms"],
    }, indent=2, ensure_ascii=False))
    if sample["user_positive_symptoms"]:
        r = env.check_symptom(0, sample["user_positive_symptoms"][0])
        print(f"check_symptom(0, first_symptom) -> {r}")
