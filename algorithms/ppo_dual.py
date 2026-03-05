"""Dual-Step PPO: rollout (Think->Act), GAE, train_step with macro-action (thought+action) gradient."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Rollout: make_experience(env, agent, reward_model, ...)
# ---------------------------------------------------------------------------


def make_experience(
    env: Any,
    agent: Any,
    reward_model: Any,
    tokenizer: Any | None = None,
    num_episodes: int = 1,
    max_steps_per_episode: int = 20,
    weight_process: float = 1.0,
    weight_task: float = 1.0,
    device: str | torch.device = "cpu",
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """
    Generate rollout experience: Think -> Act -> Env -> r_total.

    Step 1 (Think): Sample τ_t ~ π(·|o_t), store log_probs.
    Step 2 (Act):   Sample a_t ~ π(·|o_t, τ_t), store log_probs.
    Step 3 (Env):   Execute a_t -> o_{t+1}, r_extrinsic.
    Step 4 (Reward): r_total = r_extrinsic + r_PRM(history, action, obs).

    Returns a list of experience dicts, each with:
      observation, thought, action, reward_total, value, done,
      old_log_prob (macro: thought + action), [query_ids, response_ids] if tokenizer given.
    """
    experiences: list[dict[str, Any]] = []
    if seed is not None:
        env.reset(seed=seed)

    for _ in range(num_episodes):
        obs, info = env.reset()
        history: list[Any] = []  # for reward model: list of (obs, action) or dialogue

        for step in range(max_steps_per_episode):
            # Step 1 & 2: Think then Act
            thought, action, aux = agent.step(obs, deterministic=False)
            if aux is None:
                aux = {}

            # Old log prob for macro-action (thought + action)
            thought_aux = aux.get("thought") or {}
            action_aux = aux.get("action") or {}
            log_prob_t = thought_aux.get("log_prob")
            log_prob_a = action_aux.get("log_prob")
            if log_prob_t is not None and hasattr(log_prob_t, "item"):
                log_prob_t = log_prob_t.item()
            elif log_prob_t is None:
                log_prob_t = 0.0
            if log_prob_a is not None and hasattr(log_prob_a, "item"):
                log_prob_a = log_prob_a.item()
            elif log_prob_a is None:
                log_prob_a = 0.0
            old_log_prob = float(log_prob_t) + float(log_prob_a)

            # Step 3: Env step (action is the external response, e.g. question or answer)
            next_obs, r_extrinsic, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated

            # Step 4: Process reward (LLM-as-Judge or heuristic)
            r_prm = reward_model.compute_reward(
                history=history,
                action=action,
                observation=obs,
                outcome=step_info.get("outcome") if isinstance(step_info, dict) else None,
            )
            r_total = weight_task * float(r_extrinsic) + weight_process * float(r_prm)

            # Value placeholder (0 if no critic; replace when using value head)
            value = 0.0
            if isinstance(step_info, dict) and "value" in step_info:
                value = float(step_info["value"])

            exp = {
                "observation": obs,
                "thought": thought,
                "action": action,
                "reward_total": r_total,
                "value": value,
                "done": done,
                "old_log_prob": old_log_prob,
                "thought_aux": thought_aux,
                "action_aux": action_aux,
            }
            # Optional: token ids for PPO (concatenate thought + action as response)
            if tokenizer is not None:
                try:
                    query_ids = tokenizer(
                        str(obs),
                        return_tensors="pt",
                        truncation=True,
                        max_length=512,
                        padding="max_length",
                        return_attention_mask=True,
                    )
                    resp_text = f"{thought}\n{action}" if thought else str(action)
                    response_ids = tokenizer(
                        resp_text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=256,
                        add_special_tokens=False,
                    )
                    exp["query_input_ids"] = query_ids["input_ids"]
                    exp["query_attention_mask"] = query_ids.get("attention_mask")
                    exp["response_ids"] = response_ids["input_ids"]
                except Exception:
                    pass
            experiences.append(exp)

            history.append((str(obs), str(action)))
            obs = next_obs
            if done:
                break

    return experiences


# ---------------------------------------------------------------------------
# GAE: compute_advantages(rewards, values, dones, gamma, lam)
# ---------------------------------------------------------------------------


def compute_advantages(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    last_value: float | torch.Tensor = 0.0,
    last_done: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generalized Advantage Estimation.

    rewards, values, dones: 1D tensors of length T.
    Returns (advantages, returns), both shape [T].
    """
    T = rewards.shape[0]
    if rewards.dim() == 0:
        rewards = rewards.unsqueeze(0)
    if values.dim() == 0:
        values = values.unsqueeze(0)
    if dones.dim() == 0:
        dones = dones.unsqueeze(0)
    device = rewards.device
    if isinstance(last_value, (int, float)):
        last_value = torch.tensor(last_value, device=device, dtype=values.dtype)
    last_val = last_value.view(-1)
    last_done_t = torch.tensor(last_done, device=device, dtype=torch.float32)

    advantages = torch.zeros(T, device=device, dtype=rewards.dtype)
    gae = 0.0
    for t in reversed(range(T)):
        if t == T - 1:
            next_value = last_val
            next_done = last_done_t
        else:
            next_value = values[t + 1]
            next_done = dones[t + 1].float()
        delta = rewards[t] + gamma * next_value * (1.0 - next_done) - values[t]
        gae = delta + gamma * gae_lambda * (1.0 - next_done) * gae
        advantages[t] = gae
    returns = advantages + values
    return advantages, returns


# ---------------------------------------------------------------------------
# RolloutBuffer: hold experiences and build batches for PPO
# ---------------------------------------------------------------------------


class RolloutBuffer:
    """Hold (query_ids, response_ids, old_log_probs, rewards, values, dones) and compute GAE; sample batches."""

    def __init__(
        self,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: str | torch.device = "cpu",
    ) -> None:
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device
        self.experiences: list[dict[str, Any]] = []
        self._advantages: torch.Tensor | None = None
        self._returns: torch.Tensor | None = None

    def add_experiences(self, experiences: list[dict[str, Any]]) -> None:
        self.experiences.extend(experiences)

    def clear(self) -> None:
        self.experiences.clear()
        self._advantages = None
        self._returns = None

    def __len__(self) -> int:
        return len(self.experiences)

    def compute_gae_and_returns(self) -> None:
        """Fill advantages and returns using GAE on stored rewards/values/dones."""
        if not self.experiences:
            return
        rewards = torch.tensor(
            [e["reward_total"] for e in self.experiences],
            device=self.device,
            dtype=torch.float32,
        )
        values = torch.tensor(
            [e["value"] for e in self.experiences],
            device=self.device,
            dtype=torch.float32,
        )
        dones = torch.tensor(
            [e["done"] for e in self.experiences],
            device=self.device,
            dtype=torch.float32,
        )
        self._advantages, self._returns = compute_advantages(
            rewards, values, dones,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )
        # Normalize advantages (common practice)
        if self._advantages.numel() > 1:
            self._advantages = (self._advantages - self._advantages.mean()) / (self._advantages.std() + 1e-8)

    def get_batch(
        self,
        batch_size: int | None = None,
        indices: list[int] | None = None,
    ) -> dict[str, Any]:
        """
        Return a batch for PPO: query_input_ids, response_ids, old_log_probs, advantages, returns.
        Requires that experiences were collected with tokenizer (so query_input_ids, response_ids exist).
        """
        if self._advantages is None or self._returns is None:
            self.compute_gae_and_returns()
        n = len(self.experiences)
        if n == 0:
            return {}
        if indices is None:
            batch_size = batch_size or n
            indices = torch.randperm(n, device=self.device)[:batch_size].tolist()
        idx_t = torch.tensor(indices, device=self.device, dtype=torch.long)
        adv = self._advantages.index_select(0, idx_t)
        ret = self._returns.index_select(0, idx_t)
        need_ids = all(
            "query_input_ids" in self.experiences[i] and "response_ids" in self.experiences[i]
            for i in indices
        )
        if not need_ids:
            # Return non-tensor batch (e.g. for debugging or when tokenizer not used in rollout)
            return {
                "old_log_probs": torch.tensor(
                    [self.experiences[i]["old_log_prob"] for i in indices],
                    device=self.device,
                    dtype=torch.float32,
                ),
                "advantages": adv,
                "returns": ret,
                "experiences": [self.experiences[i] for i in indices],
            }
        query_ids = torch.cat([self.experiences[i]["query_input_ids"] for i in indices]).to(self.device)
        response_ids = torch.cat([self.experiences[i]["response_ids"] for i in indices]).to(self.device)
        masks = None
        if self.experiences and "query_attention_mask" in self.experiences[indices[0]]:
            masks = torch.cat([self.experiences[i]["query_attention_mask"] for i in indices]).to(self.device)
        return {
            "query_input_ids": query_ids,
            "response_ids": response_ids,
            "attention_mask": masks,
            "old_log_probs": torch.tensor(
                [self.experiences[i]["old_log_prob"] for i in indices],
                device=self.device,
                dtype=torch.float32,
            ),
            "advantages": adv,
            "returns": ret,
            "experiences": [self.experiences[i] for i in indices],
        }


# ---------------------------------------------------------------------------
# PPO: log prob of response tokens given context (causal LM)
# ---------------------------------------------------------------------------


def get_log_probs_for_response(
    model: Any,
    input_ids: torch.Tensor,
    response_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    For a causal LM: compute sum of log probs of response tokens given context.
    full = [context; response]. Log probs are taken at positions predicting each response token.
    Returns shape [B] (one log_prob per batch item).
    """
    batch_size = input_ids.shape[0]
    device = input_ids.device
    # Concatenate context and response on the sequence dimension
    full_ids = torch.cat([input_ids, response_ids], dim=1)
    if attention_mask is not None:
        resp_len = response_ids.shape[1]
        resp_mask = torch.ones(batch_size, resp_len, dtype=attention_mask.dtype, device=device)
        full_mask = torch.cat([attention_mask, resp_mask], dim=1)
    else:
        full_mask = None
    outputs = model(input_ids=full_ids, attention_mask=full_mask)
    logits = outputs.logits  # [B, L, V]
    ctx_len = input_ids.shape[1]
    resp_len = response_ids.shape[1]
    # Causal LM: logits at position i predict token at i+1. So logits[ctx_len-1] predicts first response token.
    logits_for_response = logits[:, ctx_len - 1 : ctx_len + resp_len - 1, :]  # [B, Lr, V]
    log_probs = F.log_softmax(logits_for_response, dim=-1)
    # Gather log prob of the actual response token at each position
    response_ids_shifted = response_ids.unsqueeze(-1)  # [B, Lr, 1]
    token_log_probs = torch.gather(log_probs, dim=-1, index=response_ids_shifted).squeeze(-1)  # [B, Lr]
    # Mask padding in response if needed (e.g. response_ids pad with 0 or eos)
    pad_id = getattr(model.config, "pad_token_id", 0) or 0
    mask = (response_ids != pad_id).float()
    token_log_probs = token_log_probs * mask
    return token_log_probs.sum(dim=1)  # [B]


# ---------------------------------------------------------------------------
# DualStepPPOTrainer
# ---------------------------------------------------------------------------


class DualStepPPOTrainer:
    """
    PPO trainer for two-step generation (Think -> Act). Rollout collects
    (o_t, τ_t, a_t, r_total, log_probs); GAE for advantages; one PPO update
    treating Thought+Action as a single macro-action (gradient through both).
    """

    def __init__(
        self,
        agent: Any,
        reward_model: Any,
        tokenizer: Any | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        model: Any | None = None,
        ppo_eps: float = 0.2,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        weight_process: float = 1.0,
        weight_task: float = 1.0,
        device: str | torch.device = "cpu",
        **kwargs: Any,
    ) -> None:
        self.agent = agent
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.model = model  # Causal LM for get_log_probs; if None, try agent.model
        self.ppo_eps = ppo_eps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.weight_process = weight_process
        self.weight_task = weight_task
        self.device = torch.device(device) if isinstance(device, str) else device
        self.buffer = RolloutBuffer(gamma=gamma, gae_lambda=gae_lambda, device=self.device)

    def _get_model(self) -> Any:
        if self.model is not None:
            return self.model
        if hasattr(self.agent, "model"):
            return self.agent.model
        return self.agent

    def rollout(
        self,
        env: Any,
        num_episodes: int = 1,
        max_steps_per_episode: int = 20,
        seed: int | None = None,
        clear_buffer: bool = True,
    ) -> list[dict[str, Any]]:
        """Generate experience and store in buffer; compute GAE. Returns list of experiences."""
        if clear_buffer:
            self.buffer.clear()
        experiences = make_experience(
            env=env,
            agent=self.agent,
            reward_model=self.reward_model,
            tokenizer=self.tokenizer,
            num_episodes=num_episodes,
            max_steps_per_episode=max_steps_per_episode,
            weight_process=self.weight_process,
            weight_task=self.weight_task,
            device=self.device,
            seed=seed,
        )
        self.buffer.add_experiences(experiences)
        self.buffer.compute_gae_and_returns()
        return experiences

    def train_step(self, batch: dict[str, Any] | None = None, batch_size: int | None = None) -> dict[str, float]:
        """
        One PPO update. If batch is None, sample a batch from the buffer.
        Maximizes L_PPO = E[min(r(θ)A, clip(r(θ), 1-ε, 1+ε)A)] with r(θ) = π(a|s)/π_old(a|s).
        Gradient flows through both thought and action tokens (macro-action).
        """
        if batch is None:
            batch = self.buffer.get_batch(batch_size=batch_size or len(self.buffer))
        if not batch:
            return {"policy_loss": 0.0, "value_loss": 0.0, "loss": 0.0}

        model = self._get_model()
        if hasattr(model, "train") and callable(getattr(model, "train")):
            model.train()
        old_log_probs = batch["old_log_probs"].to(self.device)
        advantages = batch["advantages"].to(self.device)
        returns = batch["returns"].to(self.device)

        # If we have token batches, do PPO on the policy (LM)
        if "query_input_ids" in batch and "response_ids" in batch:
            query_ids = batch["query_input_ids"].to(self.device)
            response_ids = batch["response_ids"].to(self.device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            new_log_probs = get_log_probs_for_response(
                model, query_ids, response_ids, attention_mask
            )
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.ppo_eps, 1.0 + self.ppo_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = torch.tensor(0.0, device=self.device)
        else:
            # No tokens: placeholder loss (e.g. buffer built without tokenizer)
            policy_loss = torch.tensor(0.0, device=self.device)
            value_loss = torch.tensor(0.0, device=self.device)

        loss = policy_loss + 0.5 * value_loss
        if self.optimizer is not None:
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            self.optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "loss": loss.item(),
        }


