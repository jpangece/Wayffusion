from __future__ import annotations

import numpy as np
import torch
from torch import nn

from algorithms.ppo import PPOBatch, PPOTrainer


class MAPPOTrainer(PPOTrainer):
    """Minimal MAPPO-style trainer with team advantage and agent-wise ratios.

    Rollout collection, GAE, evaluation, checkpointing, and logging are inherited
    from PPOTrainer. The update step is intentionally the only algorithmic
    change: old/new log-probs are [batch, num_agents], and PPO clipping is
    applied per agent before taking a mask-aware mean.
    """

    def _masked_mean(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.float()
        return (values * weights).sum() / torch.clamp(weights.sum(), min=1.0)

    def update(self, batch: PPOBatch) -> dict:
        advantages = batch.advantages.copy()
        if self.train_config.get("advantage_norm", True):
            advantages = (advantages - advantages.mean()) / max(advantages.std(), 1e-8)

        flat_obs = self._flatten_obs(batch.observations)
        flat_actions = self._flatten_time_env(batch.actions)
        flat_logprobs = batch.logprobs.reshape(-1, batch.logprobs.shape[-1])
        flat_returns = batch.returns.reshape(-1)
        flat_advantages = advantages.reshape(-1)
        batch_size = flat_actions.shape[0]
        epochs = int(self.train_config["epochs"])
        minibatch_size = int(self.train_config["minibatch_size"])
        target_kl = float(self.train_config.get("target_kl", 0.0) or 0.0)

        raw_advantages = batch.advantages.reshape(-1)
        value_predictions = batch.values.reshape(-1)
        return_targets = batch.returns.reshape(-1)
        return_var = float(np.var(return_targets))
        explained_variance = 0.0
        if return_var > 1e-8:
            explained_variance = float(1.0 - np.var(return_targets - value_predictions) / return_var)

        stats = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_frac": 0.0,
            "ratio_mean": 0.0,
            "grad_norm": 0.0,
            "reference_action_mse": 0.0,
        }
        step_count = 0
        kl_early_stop = 0.0

        for _ in range(epochs):
            indices = np.random.permutation(batch_size)
            for start in range(0, batch_size, minibatch_size):
                mb_idx = indices[start : start + minibatch_size]
                obs_tensors = {
                    key: torch.as_tensor(value[mb_idx], dtype=torch.float32, device=self.device)
                    for key, value in flat_obs.items()
                }
                actions = torch.as_tensor(flat_actions[mb_idx], dtype=torch.float32, device=self.device)
                old_logprob = torch.as_tensor(flat_logprobs[mb_idx], dtype=torch.float32, device=self.device)
                mb_advantages = torch.as_tensor(flat_advantages[mb_idx], dtype=torch.float32, device=self.device)
                mb_returns = torch.as_tensor(flat_returns[mb_idx], dtype=torch.float32, device=self.device)

                _, new_logprob, entropy, value = self.policy.get_action_and_value(obs_tensors, actions)
                if new_logprob.ndim != 2:
                    raise ValueError(
                        f"MAPPO policy must return per-agent logprob [B, N], got shape {tuple(new_logprob.shape)}"
                    )
                if "agent_mask" in obs_tensors:
                    agent_mask = obs_tensors["agent_mask"].bool()
                else:
                    agent_mask = torch.ones_like(new_logprob, dtype=torch.bool, device=self.device)

                logratio = new_logprob - old_logprob
                ratio = torch.exp(logratio)
                clip_coef = float(self.train_config["clip_coef"])
                team_advantage = mb_advantages.unsqueeze(-1).expand_as(ratio)
                pg_loss1 = -team_advantage * ratio
                pg_loss2 = -team_advantage * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
                policy_loss = self._masked_mean(torch.max(pg_loss1, pg_loss2), agent_mask)
                value_loss = 0.5 * ((value - mb_returns) ** 2).mean()
                entropy_loss = self._masked_mean(entropy, agent_mask)
                loss = policy_loss + float(self.train_config["vf_coef"]) * value_loss - float(self.train_config["ent_coef"]) * entropy_loss

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(self.policy.parameters(), float(self.train_config["max_grad_norm"]))
                self.optimizer.step()
                self._clamp_policy_log_std()

                with torch.no_grad():
                    approx_kl = self._masked_mean((ratio - 1.0) - logratio, agent_mask)
                    clip_frac = self._masked_mean(((ratio - 1.0).abs() > clip_coef).float(), agent_mask)
                    ratio_mean = self._masked_mean(ratio, agent_mask)

                stats["policy_loss"] += float(policy_loss.item())
                stats["value_loss"] += float(value_loss.item())
                stats["entropy"] += float(entropy_loss.item())
                stats["approx_kl"] += float(approx_kl.item())
                stats["clip_frac"] += float(clip_frac.item())
                stats["ratio_mean"] += float(ratio_mean.item())
                stats["grad_norm"] += float(grad_norm.item())
                step_count += 1

                if target_kl > 0.0 and float(approx_kl.item()) > target_kl:
                    kl_early_stop = 1.0
                    break
            if kl_early_stop:
                break

        result = {key: value / max(step_count, 1) for key, value in stats.items()}
        result.update(
            {
                "advantage_mean": float(np.mean(raw_advantages)),
                "advantage_std": float(np.std(raw_advantages)),
                "return_mean": float(np.mean(return_targets)),
                "return_std": float(np.std(return_targets)),
                "value_pred_mean": float(np.mean(value_predictions)),
                "value_pred_std": float(np.std(value_predictions)),
                "explained_variance": explained_variance,
                "kl_early_stop": kl_early_stop,
            }
        )
        if hasattr(self.policy, "log_std"):
            log_std = self.policy.log_std.detach().cpu().numpy()
            result["log_std_mean"] = float(np.mean(log_std))
            result["policy_std_mean"] = float(np.mean(np.exp(log_std)))
        return result
