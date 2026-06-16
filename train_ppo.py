"""
PPO Actor-Critic for the minimal Monopoly RL environment.

Best hyperparameters from 50-run grid search (replacing Random, 1000 ep):
  lr=3e-4  entropy_coef=0.001  n_steps=256

Usage:
  py train_ppo.py                    # train on all 4 agent slots (best config)
  py train_ppo.py --replace Random   # train on one slot only
  py train_ppo.py --grid-search      # 50-run hyperparameter sweep
"""

from __future__ import annotations
import argparse
import csv
import itertools
import os
import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from rl_environment import (
    MonopolyEnv, make_default_players, Player, Location, Agent, state_vector,
)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
EPISODES      = 3000
AGG_WINDOW    = 50
OUTPUT_ROOT   = "ppo_training_results_ratio_reward"
ALL_AGENT_NAMES = ["Aggressive", "Conservative", "Random", "ValueBuyer"]

# Best config from grid search — used as defaults throughout
LR           = 3e-4
ENTROPY_COEF = 0.001
N_STEPS      = 256

# PPO fixed params
GAMMA      = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS   = 0.2
VALUE_COEF = 0.5
N_EPOCHS   = 4
BATCH_SIZE = 64
HIDDEN_DIM = 64
STATE_DIM  = 10

IMPROVE_THRESHOLD = 0.8
A_SKIP = 0
A_ACT  = 1

# Grid search axes  (5 × 5 × 2 = 50 runs)
GRID_LR           = [3e-5, 1e-4, 3e-4, 1e-3, 3e-3]
GRID_ENTROPY_COEF = [0.0, 0.001, 0.01, 0.05, 0.1]
GRID_N_STEPS      = [256, 1024]


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------
class ActorCritic(nn.Module):
    """Shared backbone + three actor heads + one critic head."""

    def __init__(self, state_dim: int = STATE_DIM, hidden: int = HIDDEN_DIM):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),   nn.Tanh(),
        )
        self.actor_buy     = nn.Linear(hidden, 2)
        self.actor_mort    = nn.Linear(hidden, 2)
        self.actor_improve = nn.Linear(hidden, 2)
        self.critic        = nn.Linear(hidden, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

    def _head(self, feats: torch.Tensor, dtype: str) -> torch.Tensor:
        if dtype == "buy":   return self.actor_buy(feats)
        if dtype == "mort":  return self.actor_mort(feats)
        return self.actor_improve(feats)

    def value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(self.backbone(x)).squeeze(-1)

    def act(self, x: torch.Tensor, dtype: str):
        feats  = self.backbone(x)
        dist   = Categorical(logits=self._head(feats, dtype))
        action = dist.sample()
        return action, dist.log_prob(action), self.critic(feats).squeeze(-1)

    def evaluate(self, x: torch.Tensor, actions: torch.Tensor, dtype: str):
        feats = self.backbone(x)
        dist  = Categorical(logits=self._head(feats, dtype))
        return dist.log_prob(actions), dist.entropy(), self.critic(feats).squeeze(-1)


# ---------------------------------------------------------------------------
# Per-type rollout buffer
# ---------------------------------------------------------------------------
class _TypeBuffer:
    __slots__ = ("states", "actions", "log_probs", "values", "rewards", "dones")

    def __init__(self):
        self.states:    list = []
        self.actions:   list = []
        self.log_probs: list = []
        self.values:    list = []
        self.rewards:   list = []
        self.dones:     list = []

    def add(self, state, action, log_prob, value, reward, done):
        self.states.append(state);    self.actions.append(action)
        self.log_probs.append(log_prob); self.values.append(value)
        self.rewards.append(float(reward)); self.dones.append(bool(done))

    def __len__(self): return len(self.states)

    def clear(self):
        for a in self.__slots__: getattr(self, a).clear()


# ---------------------------------------------------------------------------
# Training Logger
# ---------------------------------------------------------------------------
class TrainingLogger:
    """
    Writes three CSV files and prints 50-episode aggregate summaries.

    Files written to `output_dir/`:
      training_episodes.csv   -- one row per episode
      training_aggregate.csv  -- one row per AGG_WINDOW episodes
      training_updates.csv    -- one row per _ppo_update() call
    """

    def __init__(self, output_dir: str, ppo_agent_name: str,
                 agg_window: int = AGG_WINDOW):
        self.ppo_agent_name = ppo_agent_name
        self.agg_window     = agg_window
        os.makedirs(output_dir, exist_ok=True)

        self._ep_f  = open(os.path.join(output_dir, "training_episodes.csv"),  "w", newline="")
        self._agg_f = open(os.path.join(output_dir, "training_aggregate.csv"), "w", newline="")
        self._upd_f = open(os.path.join(output_dir, "training_updates.csv"),   "w", newline="")

        self._ep_w  = csv.writer(self._ep_f)
        self._agg_w = csv.writer(self._agg_f)
        self._upd_w = csv.writer(self._upd_f)

        self._ep_w.writerow(["episode", "winner", "ppo_agent_reward", "episode_length"])
        self._agg_w.writerow(["episode_range", "avg_win_rate",
                               "avg_episode_return", "avg_episode_length"])
        self._upd_w.writerow(["update_step", "decision_type",
                               "policy_loss", "value_loss", "entropy_loss", "total_loss"])

        self.update_step  = 0
        self._agg_start   = 0
        self._buf_wins:   list[int]   = []
        self._buf_rew:    list[float] = []
        self._buf_lens:   list[int]   = []

    # -- episode logging ---------------------------------------------------
    def log_episode(self, episode: int, winner: Optional[str],
                    ppo_reward: float, episode_length: int):
        won = 1 if winner == self.ppo_agent_name else 0
        self._ep_w.writerow([episode, winner or "none",
                              round(ppo_reward, 4), episode_length])
        self._ep_f.flush()

        self._buf_wins.append(won)
        self._buf_rew.append(ppo_reward)
        self._buf_lens.append(episode_length)

        if len(self._buf_wins) >= self.agg_window:
            self._flush_agg(episode)

    def _flush_agg(self, last_ep: int):
        wr  = float(np.mean(self._buf_wins))
        ret = float(np.mean(self._buf_rew))
        ln  = float(np.mean(self._buf_lens))
        rng = f"{self._agg_start}-{last_ep}"

        self._agg_w.writerow([rng, round(wr, 4), round(ret, 4), round(ln, 4)])
        self._agg_f.flush()

        print(f"    [ep {rng:>9s}]  win%={wr*100:5.1f}%  "
              f"avg_return={ret:+.3f}  avg_len={ln:.1f}")

        self._agg_start = last_ep + 1
        self._buf_wins.clear(); self._buf_rew.clear(); self._buf_lens.clear()

    # -- update logging ----------------------------------------------------
    def log_update(self, dtype: str, policy_loss: float,
                   value_loss: float, entropy_loss: float, total_loss: float):
        self.update_step += 1
        self._upd_w.writerow([self.update_step, dtype,
                               round(policy_loss, 4), round(value_loss, 4),
                               round(entropy_loss, 4), round(total_loss, 4)])
        self._upd_f.flush()

    def close(self):
        if self._buf_wins:
            end = self._agg_start + len(self._buf_wins) - 1
            self._flush_agg(end)
        self._ep_f.close(); self._agg_f.close(); self._upd_f.close()


# ---------------------------------------------------------------------------
# PPO Agent
# ---------------------------------------------------------------------------
class LossScaleTracker:
    """
    Tracks an EMA of |loss| so each loss component can be normalized to
    roughly unit scale before being combined into the total loss.
    """
    def __init__(self, momentum: float = 0.99):
        self._ema:      Optional[float] = None
        self._momentum: float           = momentum

    def scale(self, loss_val: float) -> float:
        """Update EMA with the current absolute loss value; return current scale."""
        v = abs(loss_val)
        if self._ema is None:
            self._ema = v if v > 1e-8 else 1.0
        else:
            self._ema = self._momentum * self._ema + (1 - self._momentum) * v
        return max(self._ema, 1e-8)


class PPOAgent(Agent):
    """
    PPO actor-critic with three binary decision heads sharing one backbone.
    Assign `.logger` after construction to enable CSV logging.
    """

    def __init__(self,
                 improve_threshold: float = IMPROVE_THRESHOLD,
                 lr:           float = LR,
                 hidden:       int   = HIDDEN_DIM,
                 n_steps:      int   = N_STEPS,
                 entropy_coef: float = ENTROPY_COEF,
                 clip_eps:     float = CLIP_EPS):
        self.improve_threshold = improve_threshold
        self.n_steps      = n_steps
        self.entropy_coef = entropy_coef
        self.clip_eps     = clip_eps
        self.logger:   Optional[TrainingLogger] = None

        self.net = ActorCritic(STATE_DIM, hidden)
        self.opt = optim.Adam(self.net.parameters(), lr=lr, eps=1e-5)

        self._buf_buy     = _TypeBuffer()
        self._buf_mort    = _TypeBuffer()
        self._buf_improve = _TypeBuffer()

        self._pending_buy:     dict[int, list] = {}
        self._pending_mort:    dict[int, list] = {}
        self._pending_improve: dict[int, list] = {}

        self.total_steps: int = 0

        self._policy_scale  = LossScaleTracker()
        self._value_scale   = LossScaleTracker()
        self._entropy_scale = LossScaleTracker()

    # -- internal ----------------------------------------------------------
    def _decide(self, state: dict, dtype: str, pid: int, pending: dict) -> int:
        sv = torch.tensor(state_vector(state), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action, lp, val = self.net.act(sv, dtype)
        pending.setdefault(pid, []).append(
            (sv.squeeze(0), action.squeeze(), lp.squeeze(), val.squeeze()))
        return action.item()

    # -- Agent interface ---------------------------------------------------
    def decide_buy(self, player: Player, loc: Location, state: dict) -> bool:
        return self._decide(state, "buy", player.pid, self._pending_buy) == A_ACT

    def decide_mortgage(self, player: Player, candidates: list,
                        state: dict) -> Optional[Location]:
        if not candidates: return None
        deficit  = max(0, -player.cash)
        covering = [c for c in candidates if c.mortgage_value >= deficit]
        target   = (min(covering, key=lambda c: c.mortgage_value) if covering
                    else max(candidates, key=lambda c: c.mortgage_value))
        act = self._decide(state, "mort", player.pid, self._pending_mort)
        return target if act == A_ACT else None

    def decide_improve(self, player: Player, candidates: list,
                       state: dict) -> Optional[Location]:
        if not candidates: return None
        affordable = [c for c in candidates
                      if c.house_cost <= self.improve_threshold * player.cash]
        if not affordable: return None
        target = max(affordable, key=lambda c: c.rents[-1] / c.house_cost)
        act = self._decide(state, "improve", player.pid, self._pending_improve)
        return target if act == A_ACT else None

    # -- training ----------------------------------------------------------
    def pop_pending(self, pid: int):
        return (self._pending_buy.pop(pid, []),
                self._pending_mort.pop(pid, []),
                self._pending_improve.pop(pid, []))

    def store_transitions(self, buys, morts, improves, reward: float, done: bool):
        for sv, a, lp, v in buys:     self._buf_buy.add(sv, a, lp, v, reward, done)
        for sv, a, lp, v in morts:    self._buf_mort.add(sv, a, lp, v, reward, done)
        for sv, a, lp, v in improves: self._buf_improve.add(sv, a, lp, v, reward, done)
        self.total_steps += len(buys) + len(morts) + len(improves)

    def maybe_update(self, last_state: dict) -> bool:
        if self.total_steps < self.n_steps: return False
        sv       = torch.tensor(state_vector(last_state), dtype=torch.float32).unsqueeze(0)
        last_val = self.net.value(sv).item()
        for dtype, buf in (("buy", self._buf_buy), ("mort", self._buf_mort),
                           ("improve", self._buf_improve)):
            if len(buf) >= BATCH_SIZE:
                self._ppo_update(dtype, buf, last_val)
            buf.clear()
        self.total_steps = 0
        return True

    # -- GAE ---------------------------------------------------------------
    @staticmethod
    def _compute_gae(buf: _TypeBuffer, last_val: float):
        n, gae, nv = len(buf), 0.0, last_val
        adv = [0.0] * n
        for i in reversed(range(n)):
            mask  = 1.0 - float(buf.dones[i])
            v     = buf.values[i].item()
            gae   = buf.rewards[i] + GAMMA * nv * mask - v + GAMMA * GAE_LAMBDA * mask * gae
            adv[i] = gae
            nv    = v
        ret = [adv[i] + buf.values[i].item() for i in range(n)]
        return adv, ret

    # -- PPO update --------------------------------------------------------
    def _ppo_update(self, dtype: str, buf: _TypeBuffer, last_val: float):
        adv_list, ret_list = self._compute_gae(buf, last_val)

        states  = torch.stack(buf.states)
        actions = torch.stack(buf.actions)
        old_lps = torch.stack(buf.log_probs)
        adv_t   = torch.tensor(adv_list, dtype=torch.float32)
        ret_t   = torch.tensor(ret_list,  dtype=torch.float32)
        adv_t   = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        n = len(buf)
        sum_policy = sum_value = sum_entropy = sum_total = 0.0
        n_mb = 0

        for _ in range(N_EPOCHS):
            idx = np.random.permutation(n)
            for start in range(0, n, BATCH_SIZE):
                mb             = idx[start: start + BATCH_SIZE]
                lps, ent, vals = self.net.evaluate(states[mb], actions[mb], dtype)
                ratio          = torch.exp(lps - old_lps[mb])
                clipped        = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
                policy_loss    = -torch.min(ratio * adv_t[mb], clipped * adv_t[mb]).mean()
                value_loss     = nn.functional.mse_loss(vals, ret_t[mb])
                entropy_mean   = ent.mean()

                # normalize each component to unit scale before combining
                p_scale = self._policy_scale.scale(policy_loss.item())
                v_scale = self._value_scale.scale(value_loss.item())
                e_scale = self._entropy_scale.scale(entropy_mean.item())
                loss = (policy_loss  / p_scale
                        + VALUE_COEF * (value_loss  / v_scale)
                        - self.entropy_coef * (entropy_mean / e_scale))

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.opt.step()

                sum_policy  += policy_loss.item()
                sum_value   += value_loss.item()
                sum_entropy += entropy_mean.item()
                sum_total   += loss.item()
                n_mb        += 1

        if self.logger and n_mb > 0:
            p = sum_policy  / n_mb
            v = sum_value   / n_mb
            e = sum_entropy / n_mb
            t = sum_total   / n_mb  # normalized total
            self.logger.log_update(dtype, p, v, e, t)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_for(agent_name:        str,
              episodes:          int   = EPISODES,
              improve_threshold: float = IMPROVE_THRESHOLD,
              lr:                float = LR,
              n_steps:           int   = N_STEPS,
              entropy_coef:      float = ENTROPY_COEF,
              clip_eps:          float = CLIP_EPS,
              logger: Optional[TrainingLogger] = None,
              verbose: bool = True) -> tuple[dict, PPOAgent]:

    players   = make_default_players()
    ppo_agent = PPOAgent(improve_threshold=improve_threshold, lr=lr,
                         n_steps=n_steps, entropy_coef=entropy_coef,
                         clip_eps=clip_eps)
    ppo_agent.logger = logger
    ppo_pid = -1

    for p in players:
        if p.name == agent_name:
            p.agent = ppo_agent
            ppo_pid = p.pid

    win_counts: dict[str, int] = {p.name: 0 for p in players}
    env = MonopolyEnv(players, summary=False, seed=None)

    for ep in range(episodes):
        env.seed = ep
        env.reset()
        ep_ppo_reward = 0.0

        while not env.done:
            current_p = env.current_player()
            is_ppo    = current_p.pid == ppo_pid
            obs, reward, done, _ = env.step()
            if is_ppo:
                buys, morts, improves = ppo_agent.pop_pending(ppo_pid)
                ppo_agent.store_transitions(buys, morts, improves, reward, done)
                ep_ppo_reward += reward
                ppo_agent.maybe_update(obs)

        winner_name = env.winner.name if env.winner else None
        if winner_name:
            win_counts[winner_name] += 1

        if logger:
            logger.log_episode(ep, winner_name, ep_ppo_reward, env.total_turns)

    return win_counts, ppo_agent


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def create_figures(output_dir: str, agent_name: str):
    """Read the three CSVs and write a 2x3 figure of training curves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # -- load aggregate CSV ------------------------------------------------
    agg_ep, agg_wr, agg_ret, agg_len = [], [], [], []
    with open(os.path.join(output_dir, "training_aggregate.csv")) as f:
        for row in csv.DictReader(f):
            start = int(row["episode_range"].split("-")[0])
            agg_ep.append(start)
            agg_wr.append(float(row["avg_win_rate"]) * 100)
            agg_ret.append(float(row["avg_episode_return"]))
            agg_len.append(float(row["avg_episode_length"]))

    # -- load update CSV ---------------------------------------------------
    upd_by_type: dict[str, dict] = {
        t: {"step": [], "policy": [], "value": [], "entropy": [], "total": []}
        for t in ("buy", "mort", "improve")
    }
    with open(os.path.join(output_dir, "training_updates.csv")) as f:
        for row in csv.DictReader(f):
            t = row["decision_type"]
            if t not in upd_by_type: continue
            upd_by_type[t]["step"].append(int(row["update_step"]))
            upd_by_type[t]["policy"].append(float(row["policy_loss"]))
            upd_by_type[t]["value"].append(float(row["value_loss"]))
            upd_by_type[t]["entropy"].append(float(row["entropy_loss"]))
            upd_by_type[t]["total"].append(float(row["total_loss"]))

    def smooth(xs, w=10):
        if len(xs) < w: return xs
        return np.convolve(xs, np.ones(w)/w, mode="valid")

    colors = {"buy": "#2196F3", "mort": "#FF9800", "improve": "#4CAF50"}

    # -- layout ------------------------------------------------------------
    fig, axes = plt.subplots(2, 4, figsize=(22, 9))
    axes[0, 3].set_visible(False)  # no 4th episode-level metric
    fig.suptitle(f"PPO Training — replacing '{agent_name}'  "
                 f"(lr={LR:.0e}, entropy={ENTROPY_COEF}, n_steps={N_STEPS})",
                 fontsize=13)

    # Row 1: episode-level metrics (from aggregate)
    ax = axes[0, 0]
    ax.plot(agg_ep, agg_wr, "o-", color="#3F51B5", linewidth=1.8, markersize=4)
    ax.axhline(25, color="gray", linestyle="--", linewidth=0.8, label="random baseline 25%")
    ax.set_title("Win Rate (per 50-ep window)")
    ax.set_xlabel("Episode"); ax.set_ylabel("Win %")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(agg_ep, agg_ret, "o-", color="#E91E63", linewidth=1.8, markersize=4)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title("Avg Episode Return (per 50-ep window)")
    ax.set_xlabel("Episode"); ax.set_ylabel("Return")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    ax.plot(agg_ep, agg_len, "o-", color="#009688", linewidth=1.8, markersize=4)
    ax.set_title("Avg Episode Length (per 50-ep window)")
    ax.set_xlabel("Episode"); ax.set_ylabel("Turns")
    ax.grid(True, alpha=0.3)

    # Row 2: loss curves per decision type (from updates), smoothed
    loss_keys   = ["policy", "value", "entropy", "total"]
    loss_titles = ["Policy Loss", "Value Loss", "Entropy", "Total Loss"]
    for col, (key, title) in enumerate(zip(loss_keys, loss_titles)):
        ax = axes[1, col]
        for dtype, d in upd_by_type.items():
            ys = d[key]
            xs = d["step"]
            if not ys: continue
            ys_s = smooth(ys, w=min(10, max(1, len(ys)//5)))
            xs_s = xs[:len(ys_s)]
            ax.plot(xs_s, ys_s, linewidth=1.5, color=colors[dtype], label=dtype)
        ax.set_title(title)
        ax.set_xlabel("Update step"); ax.set_ylabel("Loss")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "training_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------
def _print_results(agent_name: str, episodes: int, win_counts: dict):
    print(f"\n{'='*58}")
    print(f"Replaced agent : {agent_name}  (PPO actor-critic)")
    print(f"Episodes       : {episodes}")
    print(f"{'Player':<20} {'Wins':>6}  {'Win %':>7}")
    print(f"{'-'*37}")
    for name, wins in sorted(win_counts.items(), key=lambda x: -x[1]):
        pct    = wins / episodes * 100
        marker = "  << PPO" if name == agent_name else ""
        print(f"  {name:<18} {wins:6d}  {pct:6.1f}%{marker}")
    print('='*58)


def _save_win_csv(agent_name: str, episodes: int, win_counts: dict, output_dir: str):
    path = os.path.join(output_dir, "win_counts.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["player", "wins", "win_rate_pct", "episodes", "is_ppo"])
        for name, wins in win_counts.items():
            w.writerow([name, wins, f"{wins/episodes*100:.2f}", episodes, name == agent_name])
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------
def run_grid_search(target: str, episodes: int):
    """Grid search over lr x entropy_coef x n_steps (50 runs)."""
    combos = list(itertools.product(GRID_LR, GRID_ENTROPY_COEF, GRID_N_STEPS))
    total  = len(combos)
    print(f"\nPPO grid search  (replaced: '{target}', episodes: {episodes}, runs: {total})")
    print(f"  lr:           {GRID_LR}")
    print(f"  entropy_coef: {GRID_ENTROPY_COEF}")
    print(f"  n_steps:      {GRID_N_STEPS}\n")

    rows = []
    for i, (lr, ent, ns) in enumerate(combos, 1):
        print(f"  [{i:2d}/{total}] lr={lr:.0e}  entropy={ent:.3f}  n_steps={ns} ...",
              end="", flush=True)
        wc, _ = train_for(target, episodes, lr=lr, entropy_coef=ent,
                          n_steps=ns, verbose=False)
        wins = wc[target]
        pct  = wins / episodes * 100
        rank = sorted(wc, key=lambda n: -wc[n]).index(target) + 1
        rows.append((lr, ent, ns, wins, pct, rank))
        print(f"  win%={pct:5.1f}%  rank={rank}/4")

    rows.sort(key=lambda r: -r[4])

    print(f"\n{'='*72}")
    print(f"Grid search results  (replaced: {target}, episodes: {episodes})")
    print(f"  {'lr':>8}  {'entropy':>8}  {'n_steps':>8}  {'Wins':>6}  {'Win %':>7}  {'Rank':>5}")
    print(f"  {'-'*62}")
    for j, (lr, ent, ns, wins, pct, rank) in enumerate(rows):
        marker = "  <<" if j == 0 else ""
        print(f"  {lr:>8.0e}  {ent:>8.3f}  {ns:>8d}  {wins:6d}  {pct:6.1f}%  {rank:>4}/4{marker}")
    print('='*72)

    bl, be, bn, _, bp, br = rows[0]
    print(f"\nBest config:")
    print(f"  lr={bl:.0e}  entropy_coef={be}  n_steps={bn}")
    print(f"  -> PPO win rate {bp:.1f}%  (rank {br}/4 over {episodes} episodes)")

    fname = f"results_ppo_grid_{target}.csv"
    with open(fname, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["lr", "entropy_coef", "n_steps", "wins", "win_rate_pct", "rank", "episodes"])
        for lr, ent, ns, wins, pct, rank in rows:
            w.writerow([lr, ent, ns, wins, f"{pct:.2f}", rank, episodes])
    print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO actor-critic in Monopoly env")
    parser.add_argument("--replace", default="All",
                        choices=ALL_AGENT_NAMES + ["All"],
                        help="Which agent slot to replace (default: All)")
    parser.add_argument("--episodes", type=int, default=EPISODES)
    parser.add_argument("--lr",   type=float, default=LR)
    parser.add_argument("--grid-search", action="store_true",
                        help="50-run hyperparameter sweep")
    args = parser.parse_args()

    if args.grid_search:
        target = args.replace if args.replace != "All" else "Random"
        run_grid_search(target, args.episodes)
    else:
        targets = ALL_AGENT_NAMES if args.replace == "All" else [args.replace]
        for t in targets:
            out_dir = os.path.join(OUTPUT_ROOT, t)
            logger  = TrainingLogger(out_dir, t)
            print(f"\n{'='*58}")
            print(f"Training PPO replacing '{t}' for {args.episodes} episodes")
            print(f"Config: lr={args.lr:.0e}  entropy={ENTROPY_COEF}  n_steps={N_STEPS}")
            print(f"Output: {out_dir}/")
            print('='*58)
            win_counts, _ = train_for(t, args.episodes, lr=args.lr, logger=logger)
            logger.close()
            _print_results(t, args.episodes, win_counts)
            _save_win_csv(t, args.episodes, win_counts, out_dir)
            print(f"  Generating figures...")
            create_figures(out_dir, t)
