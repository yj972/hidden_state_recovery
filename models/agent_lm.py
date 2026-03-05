"""
HFSystem2Agent: Concrete System2Agent backed by a Hugging Face Causal LM.

Loads tokenizer and model from HF; generates Thought then Action with log_probs for PPO.
"""

from __future__ import annotations

from typing import Any

import torch

from core.agent import System2Agent


# Default prefixes for Think / Act (can be overridden)
THINK_PREFIX = "Observation:\n{obs}\n\nThink:"
ACT_PREFIX = "Observation:\n{obs}\n\nThought:\n{thought}\n\nAction (question or answer):"


class HFSystem2Agent(System2Agent):
    """
    System2Agent that uses a causal LM (e.g. Qwen, Llama) for both thought and action.
    Loads tokenizer and model from Hugging Face; returns log_probs for PPO.
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: str | torch.device | None = None,
        max_new_tokens_think: int = 128,
        max_new_tokens_action: int = 256,
        do_sample: bool = True,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name_or_path = model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        self.max_new_tokens_think = max_new_tokens_think
        self.max_new_tokens_action = max_new_tokens_action
        self.do_sample = do_sample
        self.temperature = temperature
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def _generate_with_log_probs(
        self,
        prompt: str,
        max_new_tokens: int,
        deterministic: bool = False,
    ) -> tuple[str, torch.Tensor]:
        """Generate continuation and return (decoded text, sum of log_probs of generated tokens)."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(self.device)
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=not deterministic and self.do_sample,
                temperature=self.temperature if not deterministic else 1.0,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True,
            )
        gen_ids = outputs.sequences[:, input_len:]
        # Re-forward to get logits for each generated position (to compute log_probs)
        full_ids = torch.cat([inputs["input_ids"], gen_ids], dim=1)
        with torch.no_grad():
            logits = self.model(full_ids).logits
        # logits[i] predicts token at i+1
        logits_for_gen = logits[:, input_len - 1 : input_len + gen_ids.shape[1] - 1, :]
        log_probs = torch.nn.functional.log_softmax(logits_for_gen, dim=-1)
        token_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=gen_ids.unsqueeze(-1),
        ).squeeze(-1)
        sum_log_prob = token_log_probs.sum(dim=1).item()
        text = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()
        return text, torch.tensor(sum_log_prob, device=self.device)

    def think(self, observation: Any, **kwargs: Any) -> str:
        prompt = THINK_PREFIX.format(obs=str(observation))
        text, _ = self._generate_with_log_probs(
            prompt,
            max_new_tokens=self.max_new_tokens_think,
            deterministic=True,
        )
        return text

    def get_thought_logits_or_sample(
        self,
        observation: Any,
        deterministic: bool = False,
        **kwargs: Any,
    ) -> tuple[str, dict[str, Any] | None]:
        prompt = THINK_PREFIX.format(obs=str(observation))
        text, sum_log_prob = self._generate_with_log_probs(
            prompt,
            max_new_tokens=self.max_new_tokens_think,
            deterministic=deterministic,
        )
        return text, {"log_prob": sum_log_prob}

    def act(self, observation: Any, thought: Any, **kwargs: Any) -> str:
        prompt = ACT_PREFIX.format(obs=str(observation), thought=str(thought))
        text, _ = self._generate_with_log_probs(
            prompt,
            max_new_tokens=self.max_new_tokens_action,
            deterministic=True,
        )
        return text

    def get_action_logits_or_sample(
        self,
        observation: Any,
        thought: Any,
        deterministic: bool = False,
        **kwargs: Any,
    ) -> tuple[str, dict[str, Any] | None]:
        prompt = ACT_PREFIX.format(obs=str(observation), thought=str(thought))
        text, sum_log_prob = self._generate_with_log_probs(
            prompt,
            max_new_tokens=self.max_new_tokens_action,
            deterministic=deterministic,
        )
        return text, {"log_prob": sum_log_prob}
