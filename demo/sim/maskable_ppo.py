"""Lightweight pure-NumPy Policy-Value Network and Maskable PPO trainer."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from demo.sim.rl_env import GakumasBattleEnv


def _orthogonal_init(rng: np.random.Generator, shape: Tuple[int, int], gain: float = 1.0) -> np.ndarray:
    rows, cols = shape
    a = rng.normal(0.0, 1.0, size=(rows, cols))
    u, _, vt = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == (rows, cols) else vt
    return (gain * q[:rows, :cols]).astype(np.float64)


def masked_log_softmax(logits: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute exact masked probabilities, log-probabilities, and entropy."""
    was_1d = logits.ndim == 1
    z = np.atleast_2d(logits).astype(np.float64)
    m = np.atleast_2d(mask).astype(bool)

    masked_z_for_max = np.where(m, z, -np.inf)
    max_z = np.max(masked_z_for_max, axis=-1, keepdims=True)
    shifted = np.where(m, z - max_z, -1e9)
    exp_z = np.where(m, np.exp(shifted), 0.0)
    sum_exp = np.maximum(np.sum(exp_z, axis=-1, keepdims=True), 1e-30)
    probs = np.where(m, exp_z / sum_exp, 0.0)
    log_sum_exp = max_z + np.log(sum_exp)
    log_probs = np.where(m, z - log_sum_exp, 0.0)
    entropy = -np.sum(probs * log_probs, axis=-1)

    if was_1d:
        return probs[0], log_probs[0], entropy[0]
    return probs, log_probs, entropy


class PolicyValueNet:
    """2-hidden-layer LayerNorm+Tanh MLP with masked policy logits and scalar value head."""

    PARAM_NAMES = (
        "W1",
        "b1",
        "ln1_g",
        "ln1_b",
        "W2",
        "b2",
        "ln2_g",
        "ln2_b",
        "Wp",
        "bp",
        "Wv",
        "bv",
    )

    def __init__(self, obs_dim: int, num_actions: int, hidden_dim: int = 128, seed: int = 0):
        self.obs_dim = int(obs_dim)
        self.num_actions = int(num_actions)
        self.hidden_dim = int(hidden_dim)
        self.ln_eps = 1e-5

        rng = np.random.default_rng(seed)
        sqrt2 = math.sqrt(2.0)
        self.params: Dict[str, np.ndarray] = {
            "W1": _orthogonal_init(rng, (self.obs_dim, self.hidden_dim), gain=sqrt2),
            "b1": np.zeros(self.hidden_dim, dtype=np.float64),
            "ln1_g": np.ones(self.hidden_dim, dtype=np.float64),
            "ln1_b": np.zeros(self.hidden_dim, dtype=np.float64),
            "W2": _orthogonal_init(rng, (self.hidden_dim, self.hidden_dim), gain=sqrt2),
            "b2": np.zeros(self.hidden_dim, dtype=np.float64),
            "ln2_g": np.ones(self.hidden_dim, dtype=np.float64),
            "ln2_b": np.zeros(self.hidden_dim, dtype=np.float64),
            "Wp": _orthogonal_init(rng, (self.hidden_dim, self.num_actions), gain=0.01),
            "bp": np.zeros(self.num_actions, dtype=np.float64),
            "Wv": _orthogonal_init(rng, (self.hidden_dim, 1), gain=1.0),
            "bv": np.zeros(1, dtype=np.float64),
        }

    def _layernorm_forward(
        self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray
    ) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        mean = np.mean(x, axis=-1, keepdims=True)
        xc = x - mean
        var = np.mean(xc * xc, axis=-1, keepdims=True)
        inv_std = 1.0 / np.sqrt(var + self.ln_eps)
        x_hat = xc * inv_std
        y = x_hat * gamma + beta
        return y, (x_hat, inv_std, gamma)

    @staticmethod
    def _layernorm_backward(
        dy: np.ndarray, cache: Tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_hat, inv_std, gamma = cache
        dgamma = np.sum(dy * x_hat, axis=0)
        dbeta = np.sum(dy, axis=0)
        dx_hat = dy * gamma
        dx = inv_std * (
            dx_hat
            - np.mean(dx_hat, axis=-1, keepdims=True)
            - x_hat * np.mean(dx_hat * x_hat, axis=-1, keepdims=True)
        )
        return dx, dgamma, dbeta

    def forward_batch(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Forward pass on shape (B, obs_dim) returning (logits, values, cache)."""
        x = np.atleast_2d(obs).astype(np.float64)
        p = self.params

        z1 = x @ p["W1"] + p["b1"]
        ln1, ln1_cache = self._layernorm_forward(z1, p["ln1_g"], p["ln1_b"])
        h1 = np.tanh(ln1)

        z2 = h1 @ p["W2"] + p["b2"]
        ln2, ln2_cache = self._layernorm_forward(z2, p["ln2_g"], p["ln2_b"])
        h2 = np.tanh(ln2)

        logits = h2 @ p["Wp"] + p["bp"]
        values = (h2 @ p["Wv"] + p["bv"])[:, 0]

        cache = {
            "x": x,
            "ln1_cache": ln1_cache,
            "h1": h1,
            "ln2_cache": ln2_cache,
            "h2": h2,
        }
        return logits, values, cache

    def predict_single(self, obs: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, float]:
        """Fast 1D inference returning (masked_probs, value_scalar)."""
        p = self.params
        x = np.asarray(obs, dtype=np.float64)

        z1 = x @ p["W1"] + p["b1"]
        z1_mean = float(np.mean(z1))
        z1_c = z1 - z1_mean
        z1_inv_std = 1.0 / math.sqrt(float(np.mean(z1_c * z1_c)) + self.ln_eps)
        h1 = np.tanh((z1_c * z1_inv_std) * p["ln1_g"] + p["ln1_b"])

        z2 = h1 @ p["W2"] + p["b2"]
        z2_mean = float(np.mean(z2))
        z2_c = z2 - z2_mean
        z2_inv_std = 1.0 / math.sqrt(float(np.mean(z2_c * z2_c)) + self.ln_eps)
        h2 = np.tanh((z2_c * z2_inv_std) * p["ln2_g"] + p["ln2_b"])

        logits = h2 @ p["Wp"] + p["bp"]
        value = float((h2 @ p["Wv"] + p["bv"])[0])
        probs, _, _ = masked_log_softmax(logits, mask)
        return probs, value

    def select_action(
        self,
        obs: np.ndarray,
        mask: np.ndarray,
        rng: Optional[np.random.Generator] = None,
        *,
        greedy: bool = False,
        temperature: float = 1.0,
    ) -> Tuple[int, float, float]:
        """Sample or select greedy masked action, returning (action_id, log_prob, value)."""
        probs, value = self.predict_single(obs, mask)
        legal_indices = np.flatnonzero(mask)
        if len(legal_indices) == 0:
            raise ValueError("Cannot select action when mask has no legal actions")

        if greedy or temperature <= 0.0:
            best_pos = int(np.argmax(probs[legal_indices]))
            action_id = int(legal_indices[best_pos])
            log_prob = math.log(max(float(probs[action_id]), 1e-30))
            return action_id, log_prob, value

        if abs(temperature - 1.0) > 1e-6:
            scaled = np.power(np.maximum(probs[legal_indices], 1e-30), 1.0 / temperature)
            sample_probs = scaled / np.sum(scaled)
        else:
            p_sub = probs[legal_indices]
            sample_probs = p_sub / np.sum(p_sub)

        active_rng = rng if rng is not None else np.random.default_rng()
        action_id = int(active_rng.choice(legal_indices, p=sample_probs))
        log_prob = math.log(max(float(probs[action_id]), 1e-30))
        return action_id, log_prob, value

    def compute_ppo_loss_and_grads(
        self,
        obs: np.ndarray,
        masks: np.ndarray,
        actions: np.ndarray,
        old_log_probs: np.ndarray,
        advantages: np.ndarray,
        returns: np.ndarray,
        *,
        clip_eps: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        huber_delta: float = 1.0,
    ) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
        """Exact analytical forward/backward pass for Maskable PPO + Huber value loss."""
        batch_size = obs.shape[0]
        inv_b = 1.0 / float(batch_size)
        rows = np.arange(batch_size)

        logits, values, cache = self.forward_batch(obs)
        probs, log_probs, entropy = masked_log_softmax(logits, masks)

        new_log_probs = log_probs[rows, actions]
        log_ratio = new_log_probs - old_log_probs
        ratio = np.exp(np.clip(log_ratio, -20.0, 20.0))

        surr1 = ratio * advantages
        clipped_ratio = np.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
        surr2 = clipped_ratio * advantages
        policy_loss = float(np.mean(-np.minimum(surr1, surr2)))

        active = surr1 <= surr2
        d_logp = -inv_b * np.where(active, ratio * advantages, 0.0)

        one_hot = np.zeros_like(probs)
        one_hot[rows, actions] = 1.0
        d_logits_policy = np.where(masks, d_logp[:, None] * (one_hot - probs), 0.0)

        mean_entropy = float(np.mean(entropy))
        entropy_loss = -ent_coef * mean_entropy
        d_logits_ent = np.where(
            masks,
            (ent_coef * inv_b) * probs * (log_probs + entropy[:, None]),
            0.0,
        )
        d_logits = d_logits_policy + d_logits_ent

        err = values - returns
        abs_err = np.abs(err)
        huber_loss_vec = np.where(
            abs_err <= huber_delta,
            0.5 * err * err,
            huber_delta * (abs_err - 0.5 * huber_delta),
        )
        value_loss = vf_coef * float(np.mean(huber_loss_vec))
        d_values = (vf_coef * inv_b) * np.where(
            abs_err <= huber_delta,
            err,
            huber_delta * np.sign(err),
        )

        total_loss = policy_loss + value_loss + entropy_loss

        p = self.params
        h2 = cache["h2"]
        h1 = cache["h1"]
        x = cache["x"]

        d_Wp = h2.T @ d_logits
        d_bp = np.sum(d_logits, axis=0)
        d_Wv = h2.T @ d_values[:, None]
        d_bv = np.sum(d_values, axis=0, keepdims=True)
        d_h2 = d_logits @ p["Wp"].T + d_values[:, None] @ p["Wv"].T

        d_ln2 = d_h2 * (1.0 - h2 * h2)
        d_z2, d_ln2_g, d_ln2_b = self._layernorm_backward(d_ln2, cache["ln2_cache"])
        d_W2 = h1.T @ d_z2
        d_b2 = np.sum(d_z2, axis=0)
        d_h1 = d_z2 @ p["W2"].T

        d_ln1 = d_h1 * (1.0 - h1 * h1)
        d_z1, d_ln1_g, d_ln1_b = self._layernorm_backward(d_ln1, cache["ln1_cache"])
        d_W1 = x.T @ d_z1
        d_b1 = np.sum(d_z1, axis=0)

        grads = {
            "W1": d_W1,
            "b1": d_b1,
            "ln1_g": d_ln1_g,
            "ln1_b": d_ln1_b,
            "W2": d_W2,
            "b2": d_b2,
            "ln2_g": d_ln2_g,
            "ln2_b": d_ln2_b,
            "Wp": d_Wp,
            "bp": d_bp,
            "Wv": d_Wv,
            "bv": d_bv,
        }
        approx_kl = float(np.mean(old_log_probs - new_log_probs))
        clip_frac = float(np.mean(np.abs(ratio - 1.0) > clip_eps))
        stats = {
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": mean_entropy,
            "approx_kl": approx_kl,
            "clip_frac": clip_frac,
        }
        return stats, grads

    def get_weights(self) -> Dict[str, np.ndarray]:
        return {k: v.copy() for k, v in self.params.items()}

    def set_weights(self, weights: Dict[str, np.ndarray]) -> None:
        for k in self.PARAM_NAMES:
            if k not in weights:
                raise KeyError(f"Missing parameter {k!r} in weights dict")
            arr = np.asarray(weights[k], dtype=np.float64)
            if arr.shape != self.params[k].shape:
                raise ValueError(f"Shape mismatch for {k!r}: got {arr.shape}, expected {self.params[k].shape}")
            self.params[k] = arr.copy()

    def save_npz(self, path: Union[str, Path]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        meta = np.array([self.obs_dim, self.num_actions, self.hidden_dim], dtype=np.int64)
        np.savez(p, _meta=meta, **self.params)

    @classmethod
    def load_npz(cls, path: Union[str, Path]) -> "PolicyValueNet":
        with np.load(Path(path), allow_pickle=False) as data:
            meta = data["_meta"]
            net = cls(obs_dim=int(meta[0]), num_actions=int(meta[1]), hidden_dim=int(meta[2]))
            net.set_weights({k: data[k] for k in cls.PARAM_NAMES})
            return net


class AdamOptimizer:
    """Adam optimizer with global L2 gradient norm clipping."""

    def __init__(
        self,
        params: Dict[str, np.ndarray],
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        max_grad_norm: float = 0.5,
    ):
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.max_grad_norm = float(max_grad_norm)
        self.t = 0
        self.m: Dict[str, np.ndarray] = {k: np.zeros_like(v) for k, v in params.items()}
        self.v: Dict[str, np.ndarray] = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray], lr: Optional[float] = None) -> float:
        active_lr = self.lr if lr is None else float(lr)
        total_sq = sum(float(np.sum(g * g)) for g in grads.values())
        grad_norm = math.sqrt(total_sq)
        scale = 1.0
        if self.max_grad_norm > 0.0 and grad_norm > self.max_grad_norm:
            scale = self.max_grad_norm / (grad_norm + 1e-12)

        self.t += 1
        bc1 = 1.0 - (self.beta1 ** self.t)
        bc2 = 1.0 - (self.beta2 ** self.t)
        step_size = active_lr * math.sqrt(bc2) / bc1

        for k, p in params.items():
            g = grads[k] * scale
            m = self.m[k]
            v = self.v[k]
            m *= self.beta1
            m += (1.0 - self.beta1) * g
            v *= self.beta2
            v += (1.0 - self.beta2) * (g * g)
            p -= step_size * (m / (np.sqrt(v) + self.eps))
        return grad_norm


@dataclass
class MaskablePPOConfig:
    """Hyperparameters for local Maskable PPO training."""

    hidden_dim: int = 128
    total_episodes: int = 2400
    episodes_per_update: int = 32
    ppo_epochs: int = 4
    minibatch_size: int = 256
    gamma: float = 1.0
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    lr_start: float = 1.5e-3
    lr_end: float = 2e-4
    vf_coef: float = 0.5
    ent_coef_start: float = 0.02
    ent_coef_end: float = 0.003
    max_grad_norm: float = 0.5
    huber_delta: float = 1.0
    val_interval_updates: int = 10
    val_episodes: int = 16
    val_seed_offset: int = 50000
    seed: int = 2026


class MaskablePPOTrainer:
    """Self-contained on-policy Maskable PPO trainer for GakumasBattleEnv."""

    def __init__(self, env: GakumasBattleEnv, cfg: MaskablePPOConfig):
        self.env = env
        self.cfg = cfg
        self.net = PolicyValueNet(
            obs_dim=env.obs_dim,
            num_actions=env.num_actions,
            hidden_dim=cfg.hidden_dim,
            seed=cfg.seed,
        )
        self.opt = AdamOptimizer(
            self.net.params,
            lr=cfg.lr_start,
            max_grad_norm=cfg.max_grad_norm,
        )
        self.np_rng = np.random.default_rng(cfg.seed)

    def evaluate_greedy(self, seeds: List[int]) -> Dict[str, float]:
        scores: List[int] = []
        for s in seeds:
            obs, mask = self.env.reset(seed=s)
            done = False
            while not done:
                action_id, _, _ = self.net.select_action(obs, mask, greedy=True)
                obs, mask, _, done, info = self.env.step(action_id)
            scores.append(int(info["score"]))
        arr = np.asarray(scores, dtype=np.float64)
        return {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "max": float(np.max(arr)),
            "min": float(np.min(arr)),
        }

    def train(
        self,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Tuple[PolicyValueNet, Dict[str, Any]]:
        cfg = self.cfg
        num_updates = max(1, math.ceil(cfg.total_episodes / cfg.episodes_per_update))
        val_seeds = [cfg.seed + cfg.val_seed_offset + i for i in range(cfg.val_episodes)]

        best_val_mean = -1.0
        best_weights = self.net.get_weights()
        history: List[Dict[str, Any]] = []
        episodes_done = 0

        for upd in range(1, num_updates + 1):
            frac = float(upd - 1) / float(max(1, num_updates - 1))
            lr_now = cfg.lr_start + frac * (cfg.lr_end - cfg.lr_start)
            ent_now = cfg.ent_coef_start + frac * (cfg.ent_coef_end - cfg.ent_coef_start)

            ep_this_upd = min(cfg.episodes_per_update, cfg.total_episodes - episodes_done)
            if ep_this_upd <= 0:
                break

            obs_list: List[np.ndarray] = []
            mask_list: List[np.ndarray] = []
            act_list: List[int] = []
            logp_list: List[float] = []
            adv_list: List[float] = []
            ret_list: List[float] = []
            ep_scores: List[int] = []

            for _ in range(ep_this_upd):
                ep_seed = cfg.seed + episodes_done
                episodes_done += 1
                obs, mask = self.env.reset(seed=ep_seed)

                ep_obs: List[np.ndarray] = []
                ep_masks: List[np.ndarray] = []
                ep_acts: List[int] = []
                ep_logps: List[float] = []
                ep_vals: List[float] = []
                ep_rews: List[float] = []

                done = False
                final_score = 0
                while not done:
                    action_id, logp, val = self.net.select_action(obs, mask, self.np_rng, greedy=False)
                    next_obs, next_mask, rew, done, info = self.env.step(action_id)
                    ep_obs.append(obs)
                    ep_masks.append(mask)
                    ep_acts.append(action_id)
                    ep_logps.append(logp)
                    ep_vals.append(val)
                    ep_rews.append(rew)
                    obs, mask = next_obs, next_mask
                    final_score = int(info["score"])

                ep_scores.append(final_score)
                t_len = len(ep_rews)
                ep_adv = [0.0] * t_len
                ep_ret = [0.0] * t_len
                gae = 0.0
                next_val = 0.0
                for t in range(t_len - 1, -1, -1):
                    delta = ep_rews[t] + cfg.gamma * next_val - ep_vals[t]
                    gae = delta + cfg.gamma * cfg.gae_lambda * gae
                    ep_adv[t] = gae
                    ep_ret[t] = gae + ep_vals[t]
                    next_val = ep_vals[t]

                obs_list.extend(ep_obs)
                mask_list.extend(ep_masks)
                act_list.extend(ep_acts)
                logp_list.extend(ep_logps)
                adv_list.extend(ep_adv)
                ret_list.extend(ep_ret)

            b_obs = np.asarray(obs_list, dtype=np.float64)
            b_masks = np.asarray(mask_list, dtype=bool)
            b_acts = np.asarray(act_list, dtype=np.int64)
            b_old_logps = np.asarray(logp_list, dtype=np.float64)
            b_adv = np.asarray(adv_list, dtype=np.float64)
            b_ret = np.asarray(ret_list, dtype=np.float64)

            adv_mean = float(np.mean(b_adv))
            adv_std = float(np.std(b_adv))
            b_adv = (b_adv - adv_mean) / (adv_std + 1e-8)

            n_samples = b_obs.shape[0]
            mb_size = min(cfg.minibatch_size, n_samples)
            last_stats: Dict[str, float] = {}

            for _ in range(cfg.ppo_epochs):
                perm = self.np_rng.permutation(n_samples)
                for start in range(0, n_samples, mb_size):
                    mb_idx = perm[start : start + mb_size]
                    stats, grads = self.net.compute_ppo_loss_and_grads(
                        b_obs[mb_idx],
                        b_masks[mb_idx],
                        b_acts[mb_idx],
                        b_old_logps[mb_idx],
                        b_adv[mb_idx],
                        b_ret[mb_idx],
                        clip_eps=cfg.clip_eps,
                        vf_coef=cfg.vf_coef,
                        ent_coef=ent_now,
                        huber_delta=cfg.huber_delta,
                    )
                    self.opt.step(self.net.params, grads, lr=lr_now)
                    last_stats = stats

            train_arr = np.asarray(ep_scores, dtype=np.float64)
            rec: Dict[str, Any] = {
                "update": upd,
                "episodes": episodes_done,
                "train_mean": float(np.mean(train_arr)),
                "train_max": float(np.max(train_arr)),
                "policy_loss": last_stats.get("policy_loss", 0.0),
                "value_loss": last_stats.get("value_loss", 0.0),
                "entropy": last_stats.get("entropy", 0.0),
                "approx_kl": last_stats.get("approx_kl", 0.0),
                "val_mean": None,
            }

            if (upd % cfg.val_interval_updates == 0) or (upd == num_updates) or (episodes_done >= cfg.total_episodes):
                val_stats = self.evaluate_greedy(val_seeds)
                rec["val_mean"] = val_stats["mean"]
                rec["val_max"] = val_stats["max"]
                if val_stats["mean"] > best_val_mean:
                    best_val_mean = val_stats["mean"]
                    best_weights = self.net.get_weights()

            history.append(rec)
            if progress_callback is not None:
                progress_callback(rec)

        final_val_stats = self.evaluate_greedy(val_seeds)
        self.net.set_weights(best_weights)
        summary = {
            "episodes_trained": episodes_done,
            "best_val_mean": best_val_mean,
            "final_val_mean": final_val_stats["mean"],
            "history": history,
        }
        return self.net, summary
