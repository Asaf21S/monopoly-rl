"""
PPO Actor-Critic for the minimal Monopoly RL environment.

State vector (358-dim):
  10  own-player aggregate features
  12  opponent aggregate features  (4 × 3 opponents)
  336 per-property features        (12 × 28 buyable squares, fixed BUYABLE_POSITIONS order)
        per-property: owned_by_agent, owned_by_opp[0-2], unowned,
                      is_mortgaged, house_count, current_rent,
                      mortgage_value, house_cost, full_color_group, color_group_id

Action space:
  buy:     2-way  (skip / buy)            — binary as before
  mort:    29-way (skip + 28 properties)  — policy selects which property to mortgage
  improve: 29-way (skip + 28 properties)  — policy selects which property to improve
  Invalid actions are masked to -inf before sampling and re-applied during PPO update.

Best hyperparameters from phase-2 grid search (replacing Random, 3000 ep):
  normalized:  lr=1e-3  entropy=0.0  n_steps=128  n_epochs=6  gamma=0.95  clip=0.3  (31.33%)
  raw:         lr=3e-3  entropy=0.001 n_steps=128  n_epochs=10 gamma=0.95  clip=0.2  (31.03%)

These were tuned on the 358-dim state — a new grid search will be needed after
any further architectural change.

Designed to be run directly from PyCharm (no CLI args). Edit the
"Run configuration" block above the entry point below, then hit Run.
"""

from __future__ import annotations
import csv
import itertools
import os
import random
from collections import deque
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

import json

from rl_environment import (
    MonopolyEnv, make_default_players, Player, Location, Agent, state_vector,
    BUYABLE_POSITIONS, HOUSE_MULT, HOTEL_MULT, COLOR_SETS, RandomAgent,
)

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
DEVICE = "auto"

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
EPISODES      = 50000  # main training budget
WARMUP_EPISODES = 750  # pre-training against all-Random opponents before main training
AGG_WINDOW    = 100     # widened from 50 — reduces window-to-window noise in the curves
GRID_SEARCH_EPISODES = 3000  # shorter per-run budget used only during hyperparameter sweeps
BASELINE_EPISODES    = 1000  # games used to estimate the no-RL reference win rate

# Optuna 4-stage search settings
# Stage 1: classic hyperparameters (default architecture)
# Stage 2: reward shaping constants (best stage-1 hypers)
# Stage 3: network architecture (best stages 1+2)
# Stage 4: classic hyperparameters again for the found architecture (best stages 2+3)
OPTUNA_AGENT    = "Random"
OPTUNA_TRIALS_1 = 50    # stage 1: classic hyperparameters
OPTUNA_TRIALS_2 = 40    # stage 2: reward shaping constants
OPTUNA_TRIALS_3 = 30    # stage 3: network architecture
OPTUNA_TRIALS_4 = 50    # stage 4: re-tune classic hypers for found architecture
OPTUNA_EP_1     = 9000  # training episodes per trial — stage 1
OPTUNA_EP_2     = 9000  # training episodes per trial — stage 2
OPTUNA_EP_3     = 9000  # training episodes per trial — stage 3
OPTUNA_EP_4     = 9000  # training episodes per trial — stage 4
OPTUNA_TEST_EP  = 5000  # frozen-weight evaluation episodes per trial
OPTUNA_N_WORKERS  = 7     # parallel CPU worker processes — each uses ~450 MB RAM
OPTUNA_GPU_WORKER = not False  # add CUDA as one extra worker on top of CPU workers
OPTUNA_PRUNE_WINDOW = 1000     # rolling window (episodes) for pruning win% checks
# Thresholds are CI-adjusted: original - 95% CI at n=1000 for that win rate
# original: 5%, 10%, 15%, 20%  →  CI: 1.4%, 1.9%, 2.2%, 2.5%
OPTUNA_PRUNE_CHECKPOINTS = [   # (episode, min_win_pct) — max(rolling-1000, cumulative)
    (2000,  3.6),
    (4000,  8.1),
    (6000, 12.8),
    (8000, 17.5),
]
OUTPUT_ROOT   = "outputs"
ALL_AGENT_NAMES = ["Aggressive", "Conservative", "Random", "ValueBuyer"]

# Best config from Optuna 4-stage search (parallel run, 9000 ep/trial, 5000 test ep)
# S1=31.2%  S2=31.2%  S3=31.1%  S4=31.1%
LR           = 2.0719e-4
ENTROPY_COEF = 5.89468e-4
N_STEPS      = 256
N_EPOCHS     = 10

# PPO fixed params (gamma/clip_eps now exposed in run config and grid search)
GAMMA      = 0.981968
GAE_LAMBDA = 0.95
CLIP_EPS   = 0.42941
VALUE_COEF = 0.5
BATCH_SIZE = 64
HIDDEN_DIM = 256

WIN_BONUS        = 12.9322  # added to reward on the episode the agent wins
BANKRUPT_PENALTY = -2.57095 # added to reward on the episode the agent goes bankrupt
BLOCK_BONUS      = 2.0      # reward for buying the last unowned property in an opponent's near-complete color group
ENTROPY_INIT_MULT = 10.0    # entropy_coef starts at this multiple at ep 0, decays linearly to 1× by the final episode
CUTOFF_RANK_BONUS = 1.5     # scale for rank-based terminal bonus in cutoff games (rank 1 → +1.5, rank N → -1.5)

IMPROVE_THRESHOLD = 0.8
A_SKIP = 0
A_ACT  = 1

# Property action space (mortgage / improve heads)
NUM_PROPERTIES  = len(BUYABLE_POSITIONS)            # 28
PROP_ACTION_DIM = NUM_PROPERTIES + 1               # 29: action 0=skip, 1-28=select property
PROPERTY_INDEX  = {pos: idx for idx, pos in enumerate(BUYABLE_POSITIONS)}
STATE_DIM  = 10 + 4 * 3 + 12 * NUM_PROPERTIES  # 10 + 12 + 336 = 358

# Grid search axes  (4 × 5 × 4 × 4 = 320 runs)
GRID_LR           = [1e-4, 3e-4, 1e-3, 3e-3]
GRID_ENTROPY_COEF = [0.0, 0.001, 0.01, 0.05, 0.1]
GRID_N_STEPS      = [128, 256, 512, 1024]
GRID_N_EPOCHS     = [2, 4, 6, 10]

# Phase-2 grid search axes:
#   top-20 (lr, entropy_coef, n_steps, n_epochs) combos from phase-1
#   × 2 gamma values × 4 clip_eps values × 2 normalize variants
#   = 20 × 2 × 4 × 2 = 320 total training runs
#   (160 unique hyperparam combos, each run with / without reward normalization)
GRID2_TOP_COMBOS = [
    # (lr,   entropy_coef, n_steps, n_epochs) — ordered by phase-1 win rate (desc)
    (3e-4,  0.01,   128,  10),   # 30.47 %
    (1e-3,  0.001,  256,   2),   # 29.57 %
    (3e-4,  0.0,    512,   6),   # 29.47 %
    (1e-3,  0.01,   256,   6),   # 29.37 %
    (1e-4,  0.01,   512,  10),   # 29.27 %
    (3e-3,  0.001,  128,   2),   # 29.27 %
    (1e-3,  0.001,  128,   2),   # 29.10 %
    (1e-3,  0.01,   256,   4),   # 29.10 %
    (1e-3,  0.001,  512,   6),   # 28.73 %
    (1e-4,  0.001, 1024,  10),   # 28.27 %
    (3e-3,  0.0,    256,   2),   # 28.27 %
    (3e-4,  0.01,   512,  10),   # 28.23 %
    (3e-3,  0.0,    256,  10),   # 28.13 %
    (3e-4,  0.001,  512,   4),   # 28.10 %
    (1e-3,  0.0,    128,   6),   # 28.07 %
    (1e-3,  0.0,   1024,   2),   # 28.00 %
    (1e-4,  0.01,   256,   2),   # 27.90 %
    (3e-4,  0.0,    512,  10),   # 27.90 %
    (3e-3,  0.001,  128,  10),   # 27.83 %
    (3e-3,  0.01,   512,   4),   # 27.70 %
]
GRID2_GAMMA    = [0.95, 0.99]           # discount factor (was fixed at 0.99)
GRID2_CLIP_EPS = [0.1, 0.15, 0.2, 0.3] # PPO clip epsilon (was fixed at 0.2)


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------
class ActorCritic(nn.Module):
    """Shared backbone + three actor heads + one critic head."""

    def __init__(self, state_dim: int = STATE_DIM, hidden: int = HIDDEN_DIM, n_layers: int = 3):
        super().__init__()
        layers = [nn.Linear(state_dim, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        self.backbone = nn.Sequential(*layers)
        self.actor_buy     = nn.Linear(hidden, 2)               # binary: skip / buy
        self.actor_mort    = nn.Linear(hidden, PROP_ACTION_DIM)  # skip + 28 properties
        self.actor_improve = nn.Linear(hidden, PROP_ACTION_DIM)  # skip + 28 properties
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

    def act(self, x: torch.Tensor, dtype: str,
            mask: Optional[torch.Tensor] = None):
        """mask: broadcastable additive mask (use -inf to block invalid actions)."""
        feats  = self.backbone(x)
        logits = self._head(feats, dtype)
        if mask is not None:
            logits = logits + mask
        dist   = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), self.critic(feats).squeeze(-1)

    def evaluate(self, x: torch.Tensor, actions: torch.Tensor, dtype: str,
                 masks: Optional[torch.Tensor] = None):
        """masks: [batch, action_dim] additive masks, or None."""
        feats  = self.backbone(x)
        logits = self._head(feats, dtype)
        if masks is not None:
            logits = logits + masks
        dist   = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), self.critic(feats).squeeze(-1)


# ---------------------------------------------------------------------------
# Per-type rollout buffer
# ---------------------------------------------------------------------------
class _PruneSignal(Exception):
    """Raised by train_for when an Optuna trial should be pruned early."""


class _TypeBuffer:
    __slots__ = ("states", "actions", "log_probs", "values", "rewards", "dones", "masks")

    def __init__(self):
        self.states:    list = []
        self.actions:   list = []
        self.log_probs: list = []
        self.values:    list = []
        self.rewards:   list = []
        self.dones:     list = []
        self.masks:     list = []  # None for buy; [PROP_ACTION_DIM] tensor for mort/improve

    def add(self, state, action, log_prob, value, reward, done, mask=None):
        self.states.append(state);       self.actions.append(action)
        self.log_probs.append(log_prob); self.values.append(value)
        self.rewards.append(float(reward)); self.dones.append(bool(done))
        self.masks.append(mask)

    def __len__(self): return len(self.states)

    def clear(self):
        for a in self.__slots__: getattr(self, a).clear()


# ---------------------------------------------------------------------------
# Running reward normalizer (Welford's online algorithm)
# ---------------------------------------------------------------------------
class RunningMeanStd:
    """
    Online mean/variance tracker (Welford).  Call normalize(x) to update stats
    and return the z-scored value.  Used optionally during rollout collection
    to keep reward magnitudes stationary without biasing the sign.
    """
    def __init__(self):
        self.mean  = 0.0
        self.M2    = 0.0
        self.count = 0

    def update(self, x: float):
        self.count += 1
        delta       = x - self.mean
        self.mean  += delta / self.count
        self.M2    += delta * (x - self.mean)

    def std(self) -> float:
        return max(np.sqrt(self.M2 / (self.count - 1)), 1e-6) if self.count >= 2 else 1.0

    def normalize(self, x: float) -> float:
        self.update(x)
        return (x - self.mean) / self.std()


# ---------------------------------------------------------------------------
# Training Logger
# ---------------------------------------------------------------------------
class TrainingLogger:
    """
    Writes three CSV files and prints 50-episode aggregate summaries.

    Files written to `output_dir/`:
      training_episodes.csv   -- one row per episode (incl. action counts & end-state)
      training_aggregate.csv  -- one row per AGG_WINDOW episodes
      training_updates.csv    -- one row per _ppo_update() call
    """

    DECISION_TYPES = ("buy", "mort", "improve")

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

        self._ep_w.writerow([
            "episode", "winner", "ppo_agent_reward", "episode_length",
            "buy_acts", "buy_skips", "mort_acts", "mort_skips",
            "improve_acts", "improve_skips",
            "ppo_bankrupt", "ppo_final_cash", "ppo_final_net_worth", "ppo_num_properties",
        ])
        self._agg_w.writerow([
            "episode_range", "avg_win_rate", "avg_episode_return", "avg_episode_length",
            "avg_decisive_length",
            "buy_act_rate", "mort_act_rate", "improve_act_rate",
            "bankruptcy_rate", "avg_final_net_worth",
        ])
        self._upd_w.writerow(["update_step", "decision_type",
                               "policy_loss", "value_loss", "entropy_loss", "total_loss"])

        self.update_step  = 0
        self._agg_start   = 0
        self._buf_wins:          list[int]   = []
        self._buf_rew:           list[float] = []
        self._buf_lens:          list[int]   = []
        self._buf_decisive_lens: list[int]   = []   # only episodes < MAX_TURNS
        self._buf_bankrupt:      list[int]   = []
        self._buf_net_worth:     list[float] = []
        self._buf_actions: dict[str, list[int]] = {dt: [0, 0] for dt in self.DECISION_TYPES}

    # -- episode logging ---------------------------------------------------
    def log_episode(self, episode: int, winner: Optional[str],
                    ppo_reward: float, episode_length: int,
                    action_counts: dict[str, list[int]], bankrupt: int,
                    final_cash: float, final_net_worth: float, num_properties: int):
        won = 1 if winner == self.ppo_agent_name else 0
        b_skip, b_act = action_counts["buy"]
        m_skip, m_act = action_counts["mort"]
        i_skip, i_act = action_counts["improve"]

        self._ep_w.writerow([episode, winner or "none",
                              round(ppo_reward, 4), episode_length,
                              b_act, b_skip, m_act, m_skip, i_act, i_skip,
                              bankrupt, final_cash, round(final_net_worth, 2), num_properties])
        self._ep_f.flush()

        self._buf_wins.append(won)
        self._buf_rew.append(ppo_reward)
        self._buf_lens.append(episode_length)
        if episode_length < 500:
            self._buf_decisive_lens.append(episode_length)
        self._buf_bankrupt.append(bankrupt)
        self._buf_net_worth.append(final_net_worth)
        for dt in self.DECISION_TYPES:
            sk, ac = action_counts[dt]
            self._buf_actions[dt][0] += sk
            self._buf_actions[dt][1] += ac

        if len(self._buf_wins) >= self.agg_window:
            self._flush_agg(episode)

    def _flush_agg(self, last_ep: int):
        wr  = float(np.mean(self._buf_wins))
        ret = float(np.mean(self._buf_rew))
        ln  = float(np.mean(self._buf_lens))
        br  = float(np.mean(self._buf_bankrupt))
        nw  = float(np.mean(self._buf_net_worth))
        rng = f"{self._agg_start}-{last_ep}"
        dec_ln = float(np.mean(self._buf_decisive_lens)) if self._buf_decisive_lens else float("nan")

        def act_rate(dt: str) -> float:
            sk, ac = self._buf_actions[dt]
            total = sk + ac
            return ac / total if total else 0.0

        buy_r, mort_r, imp_r = act_rate("buy"), act_rate("mort"), act_rate("improve")

        dec_ln_csv = round(dec_ln, 4) if self._buf_decisive_lens else ""
        self._agg_w.writerow([rng, round(wr, 4), round(ret, 4), round(ln, 4),
                               dec_ln_csv,
                               round(buy_r, 4), round(mort_r, 4), round(imp_r, 4),
                               round(br, 4), round(nw, 2)])
        self._agg_f.flush()

        dec_str = f"{dec_ln:.1f}" if self._buf_decisive_lens else "---"
        print(f"    [ep {rng:>9s}]  win%={wr*100:5.1f}%  avg_return={ret:+.3f}  "
              f"avg_len={ln:.1f}  decisive_len={dec_str}  "
              f"act%(buy/mort/imp)={buy_r*100:.0f}/{mort_r*100:.0f}/{imp_r*100:.0f}  "
              f"bankrupt%={br*100:.0f}")

        self._agg_start = last_ep + 1
        self._buf_wins.clear(); self._buf_rew.clear(); self._buf_lens.clear()
        self._buf_decisive_lens.clear()
        self._buf_bankrupt.clear(); self._buf_net_worth.clear()
        for dt in self.DECISION_TYPES:
            self._buf_actions[dt] = [0, 0]

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
class PPOAgent(Agent):
    """
    PPO actor-critic with three binary decision heads sharing one backbone.
    Assign `.logger` after construction to enable CSV logging.
    """

    def __init__(self,
                 improve_threshold: float = IMPROVE_THRESHOLD,
                 lr:                float = LR,
                 hidden:            int   = HIDDEN_DIM,
                 n_layers:          int   = 3,
                 n_steps:           int   = N_STEPS,
                 entropy_coef:      float = ENTROPY_COEF,
                 clip_eps:          float = CLIP_EPS,
                 n_epochs:          int   = N_EPOCHS,
                 gamma:             float = GAMMA,
                 normalize_rewards: bool  = False,
                 device: torch.device = DEVICE):
        self.improve_threshold = improve_threshold
        self.n_steps           = n_steps
        self.entropy_coef      = entropy_coef
        self.clip_eps          = clip_eps
        self.n_epochs          = n_epochs
        self.gamma             = gamma
        self.device            = device
        self.reward_rms: Optional[RunningMeanStd] = RunningMeanStd() if normalize_rewards else None
        self.logger:     Optional[TrainingLogger] = None

        self.net = ActorCritic(STATE_DIM, hidden, n_layers).to(self.device)
        self.opt = optim.Adam(self.net.parameters(), lr=lr, eps=1e-5)

        self._buf_buy     = _TypeBuffer()
        self._buf_mort    = _TypeBuffer()
        self._buf_improve = _TypeBuffer()

        self._pending_buy:     dict[int, list] = {}
        self._pending_mort:    dict[int, list] = {}
        self._pending_improve: dict[int, list] = {}

        self.total_steps: int = 0

        # action-firing stats: {dtype: [skip_count, act_count]}, A_SKIP=0 / A_ACT=1
        self.total_action_counts:   dict[str, list[int]] = {dt: [0, 0] for dt in ("buy", "mort", "improve")}
        self._episode_action_counts: dict[str, list[int]] = {dt: [0, 0] for dt in ("buy", "mort", "improve")}

    # -- internal ----------------------------------------------------------
    def _decide(self, state: dict, dtype: str, pid: int, pending: dict) -> int:
        sv = torch.tensor(state_vector(state), dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, lp, val = self.net.act(sv, dtype)
        pending.setdefault(pid, []).append(
            (sv.squeeze(0), action.squeeze(), lp.squeeze(), val.squeeze()))
        a = action.item()
        self._episode_action_counts[dtype][a] += 1
        self.total_action_counts[dtype][a]     += 1
        return a

    # -- action-stat helpers -------------------------------------------------
    def start_episode(self):
        """Reset per-episode action counters; call once at the top of each episode."""
        self._episode_action_counts = {dt: [0, 0] for dt in ("buy", "mort", "improve")}

    def episode_action_counts(self) -> dict[str, list[int]]:
        return self._episode_action_counts

    # -- property-selection decision (mortgage / improve) ------------------
    def _decide_property(self, state: dict, dtype: str, pid: int,
                         pending: dict, candidates: list) -> Optional[Location]:
        """
        Sample a property from `candidates` (or skip) using a masked 29-way head.
        Action 0 = skip; actions 1-28 = property at PROPERTY_INDEX[loc.pos].
        Stores (sv, action, lp, val, mask) in `pending` for later PPO update.
        """
        sv = torch.tensor(state_vector(state), dtype=torch.float32,
                          device=self.device).unsqueeze(0)

        # Build additive mask: 0.0 = valid, -inf = blocked
        mask = torch.full((PROP_ACTION_DIM,), float('-inf'), device=self.device)
        mask[0] = 0.0  # skip is always valid
        for loc in candidates:
            mask[PROPERTY_INDEX[loc.pos] + 1] = 0.0

        with torch.no_grad():
            action, lp, val = self.net.act(sv, dtype, mask=mask.unsqueeze(0))

        pending.setdefault(pid, []).append(
            (sv.squeeze(0), action.squeeze(), lp.squeeze(), val.squeeze(), mask))

        a = action.item()
        cnt = 0 if a == 0 else 1
        self._episode_action_counts[dtype][cnt] += 1
        self.total_action_counts[dtype][cnt]     += 1

        if a == 0:
            return None
        prop_idx = a - 1
        for loc in candidates:
            if PROPERTY_INDEX[loc.pos] == prop_idx:
                return loc
        return None  # shouldn't be reached if mask is correct

    # -- Agent interface ---------------------------------------------------
    def decide_buy(self, player: Player, loc: Location, state: dict) -> bool:
        return self._decide(state, "buy", player.pid, self._pending_buy) == A_ACT

    def decide_mortgage(self, player: Player, candidates: list,
                        state: dict) -> Optional[Location]:
        if not candidates:
            return None
        return self._decide_property(state, "mort", player.pid,
                                     self._pending_mort, candidates)

    def decide_improve(self, player: Player, candidates: list,
                       state: dict) -> Optional[Location]:
        if not candidates:
            return None
        return self._decide_property(state, "improve", player.pid,
                                     self._pending_improve, candidates)

    # -- training ----------------------------------------------------------
    def pop_pending(self, pid: int):
        return (self._pending_buy.pop(pid, []),
                self._pending_mort.pop(pid, []),
                self._pending_improve.pop(pid, []))

    def store_transitions(self, buys, morts, improves, reward: float, done: bool):
        if self.reward_rms is not None:
            reward = self.reward_rms.normalize(reward)
        for sv, a, lp, v in buys:
            self._buf_buy.add(sv, a, lp, v, reward, done)
        for sv, a, lp, v, mask in morts:
            self._buf_mort.add(sv, a, lp, v, reward, done, mask)
        for sv, a, lp, v, mask in improves:
            self._buf_improve.add(sv, a, lp, v, reward, done, mask)
        self.total_steps += len(buys) + len(morts) + len(improves)

    def add_terminal_bonus(self, bonus: float):
        """Retroactively add a terminal win/bankrupt signal to the last stored reward
        in each buffer. Called once per episode after the game ends."""
        if self.reward_rms is not None:
            bonus = self.reward_rms.normalize(bonus)
        for buf in (self._buf_buy, self._buf_mort, self._buf_improve):
            if buf.rewards:
                buf.rewards[-1] = buf.rewards[-1] + bonus

    def maybe_update(self, last_state: dict) -> bool:
        if self.total_steps < self.n_steps: return False
        sv       = torch.tensor(state_vector(last_state), dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
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
    def _compute_gae(buf: _TypeBuffer, last_val: float, gamma: float):
        n, gae, nv = len(buf), 0.0, last_val
        adv = [0.0] * n
        for i in reversed(range(n)):
            mask  = 1.0 - float(buf.dones[i])
            v     = buf.values[i].item()
            gae   = buf.rewards[i] + gamma * nv * mask - v + gamma * GAE_LAMBDA * mask * gae
            adv[i] = gae
            nv    = v
        ret = [adv[i] + buf.values[i].item() for i in range(n)]
        return adv, ret

    # -- PPO update --------------------------------------------------------
    def _ppo_update(self, dtype: str, buf: _TypeBuffer, last_val: float):
        adv_list, ret_list = self._compute_gae(buf, last_val, self.gamma)

        states  = torch.stack(buf.states)
        actions = torch.stack(buf.actions)
        old_lps = torch.stack(buf.log_probs)
        adv_t   = torch.tensor(adv_list, dtype=torch.float32, device=self.device)
        ret_t   = torch.tensor(ret_list,  dtype=torch.float32, device=self.device)
        adv_t   = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        # masks are stored for mort/improve heads; None for buy
        has_masks = len(buf.masks) > 0 and buf.masks[0] is not None
        masks_t   = torch.stack(buf.masks) if has_masks else None

        n = len(buf)
        sum_policy = sum_value = sum_entropy = sum_total = 0.0
        n_mb = 0

        for _ in range(self.n_epochs):
            idx = np.random.permutation(n)
            for start in range(0, n, BATCH_SIZE):
                mb      = idx[start: start + BATCH_SIZE]
                mb_masks = masks_t[mb] if masks_t is not None else None
                lps, ent, vals = self.net.evaluate(states[mb], actions[mb], dtype,
                                                   masks=mb_masks)
                ratio          = torch.exp(lps - old_lps[mb])
                clipped        = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
                policy_loss    = -torch.min(ratio * adv_t[mb], clipped * adv_t[mb]).mean()
                value_loss     = nn.functional.mse_loss(vals, ret_t[mb])
                entropy_mean   = ent.mean()

                loss = (policy_loss
                        + VALUE_COEF * value_loss
                        - self.entropy_coef * entropy_mean)

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
            t = sum_total   / n_mb
            self.logger.log_update(dtype, p, v, e, t)


# ---------------------------------------------------------------------------
# Warm-up pre-training (all-Random opponents)
# ---------------------------------------------------------------------------
def _warmup_train(ppo_agent: "PPOAgent", agent_name: str, episodes: int,
                  win_bonus: float, bankrupt_penalty: float,
                  cutoff_rank_bonus: float, block_bonus: float,
                  entropy_coef: float, entropy_init_mult: float,
                  device: "torch.device", verbose: bool) -> None:
    """Train ppo_agent against all-Random opponents so it learns basic buy-and-build
    strategy before facing the real bots.  Runs at maximum entropy throughout."""
    warmup_players = make_default_players()
    wu_ppo_pid    = -1
    wu_ppo_player = None
    for p in warmup_players:
        if p.name == agent_name:
            p.agent       = ppo_agent
            wu_ppo_pid    = p.pid
            wu_ppo_player = p
        else:
            p.agent = RandomAgent()

    wu_env = MonopolyEnv(warmup_players, summary=False)

    for ep in range(episodes):
        ppo_agent.entropy_coef = entropy_coef * entropy_init_mult  # full exploration
        wu_env.seed = 200_000 + ep  # seeds distinct from main training
        wu_env.reset()
        ppo_agent.start_episode()

        while not wu_env.done:
            cur   = wu_env.current_player()
            is_ppo = cur.pid == wu_ppo_pid
            if is_ppo:
                _bp = _blocking_positions(wu_ppo_player, wu_env.players)
                _ab = {loc.pos for loc in wu_ppo_player.assets}
            obs, reward, done, _ = wu_env.step()
            if is_ppo:
                _new = {loc.pos for loc in wu_ppo_player.assets} - _ab
                reward += sum(block_bonus for pos in _new if pos in _bp)
                buys, morts, improves = ppo_agent.pop_pending(wu_ppo_pid)
                ppo_agent.store_transitions(buys, morts, improves, reward, done)
                ppo_agent.maybe_update(obs)

        w   = wu_env.winner.name if wu_env.winner else None
        won = w == agent_name
        bk  = wu_ppo_player.status == "bankrupt"
        if won:
            bonus = win_bonus
        elif bk:
            bonus = bankrupt_penalty
        else:
            _act = sorted([p for p in warmup_players if p.status != "bankrupt"],
                          key=lambda p: p.net_worth(), reverse=True)
            _n = len(_act)
            _r = next((i + 1 for i, p in enumerate(_act) if p.pid == wu_ppo_pid), _n)
            bonus = cutoff_rank_bonus * (2.0 * (_n - _r) / max(_n - 1, 1) - 1.0)
        ppo_agent.add_terminal_bonus(bonus)

    if verbose:
        print(f"  Warm-up complete: {episodes} episodes vs Random bots", flush=True)


# ---------------------------------------------------------------------------
# Reward helpers
# ---------------------------------------------------------------------------
def _blocking_positions(ppo_player: Player, all_players: list) -> set:
    """Return board positions where PPO buying would block an opponent's near-complete monopoly.

    Fires when an opponent owns all but one property in a color group and that
    missing property is unowned — PPO buying it denies them the monopoly.
    Only considers color-group (street) properties; railroads/utilities are excluded.
    """
    ppo_owned = {loc.pos for loc in ppo_player.assets}
    blocked: set[int] = set()
    for opp in all_players:
        if opp.pid == ppo_player.pid or opp.status != "active":
            continue
        opp_owned = {loc.pos for loc in opp.assets}
        for color, positions in COLOR_SETS.items():
            missing = [p for p in positions if p not in opp_owned]
            if len(missing) == 1 and missing[0] not in ppo_owned:
                blocked.add(missing[0])
    return blocked


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
              n_epochs:          int   = N_EPOCHS,
              gamma:             float = GAMMA,
              normalize_rewards: bool  = False,
              hidden:            int   = HIDDEN_DIM,
              n_layers:          int   = 3,
              win_bonus:         float = WIN_BONUS,
              bankrupt_penalty:  float = BANKRUPT_PENALTY,
              block_bonus:       float = BLOCK_BONUS,
              entropy_init_mult: float = ENTROPY_INIT_MULT,
              cutoff_rank_bonus: float = CUTOFF_RANK_BONUS,
              warmup_episodes:   int   = 0,
              checkpoint_dir:    Optional[str] = None,
              run_label:         str   = "",
              prune_checkpoints: list = (),
              prune_window:      int  = 1000,
              trial_id:          int  = -1,
              logger: Optional[TrainingLogger] = None,
              device: torch.device = DEVICE,
              verbose: bool = True) -> tuple[dict, PPOAgent, dict]:
    """
    prune_checkpoints: sequence of (episode, min_win_pct) pairs.  At each listed
    episode the cumulative win rate is compared to the threshold; if below, raises
    _PruneSignal so the Optuna objective can mark the trial pruned.  Empty by default.
    """

    players   = make_default_players()
    ppo_agent = PPOAgent(improve_threshold=improve_threshold, lr=lr,
                         hidden=hidden, n_layers=n_layers,
                         n_steps=n_steps, entropy_coef=entropy_coef,
                         clip_eps=clip_eps, n_epochs=n_epochs,
                         gamma=gamma, normalize_rewards=normalize_rewards,
                         device=device)
    ppo_agent.logger = logger
    ppo_pid = -1

    for p in players:
        if p.name == agent_name:
            p.agent = ppo_agent
            ppo_pid = p.pid
    ppo_player = next(p for p in players if p.pid == ppo_pid)

    # Warm-up phase: build basic buy-and-build intuition against Random bots
    if warmup_episodes > 0:
        _warmup_train(ppo_agent, agent_name, warmup_episodes,
                      win_bonus, bankrupt_penalty, cutoff_rank_bonus,
                      block_bonus, entropy_coef, entropy_init_mult,
                      device, verbose)

    win_counts: dict[str, int] = {p.name: 0 for p in players}
    env = MonopolyEnv(players, summary=False, seed=None)

    total_bankrupt = 0
    sum_net_worth  = 0.0
    sum_cash       = 0.0
    sum_props      = 0.0
    _recent_wins:     deque = deque(maxlen=prune_window if prune_window > 0 else None)
    _recent_agg_wins: deque = deque(maxlen=prune_window if prune_window > 0 else None)
    _best_rolling_pct:  float    = -1.0
    _best_window_agg_pct: float  = 0.0
    _best_state:  dict | None    = None
    _lucky_window: dict | None   = None   # best-gap window where PPO beats Aggressive (tracked every ep)
    _lucky_next_check: int       = 0      # cooldown: don't *print* until this episode

    for ep in range(episodes):
        # Entropy schedule: convex decay (fast early, tapering) from entropy_coef*entropy_init_mult → entropy_coef
        _frac = ep / max(episodes - 1, 1)
        ppo_agent.entropy_coef = entropy_coef * (1.0 + (entropy_init_mult - 1.0) * (1.0 - _frac) ** 2)

        env.seed = ep
        env.reset()
        ppo_agent.start_episode()
        ep_ppo_reward = 0.0

        while not env.done:
            current_p = env.current_player()
            is_ppo    = current_p.pid == ppo_pid
            if is_ppo:
                _blocking_pos  = _blocking_positions(ppo_player, env.players)
                _assets_before = {loc.pos for loc in ppo_player.assets}
            obs, reward, done, _ = env.step()
            if is_ppo:
                _new_props = {loc.pos for loc in ppo_player.assets} - _assets_before
                reward    += sum(block_bonus for p in _new_props if p in _blocking_pos)
                buys, morts, improves = ppo_agent.pop_pending(ppo_pid)
                ppo_agent.store_transitions(buys, morts, improves, reward, done)
                ep_ppo_reward += reward
                ppo_agent.maybe_update(obs)

        winner_name = env.winner.name if env.winner else None
        if winner_name:
            win_counts[winner_name] += 1

        # Terminal bonus: large sparse signal for winning / going bankrupt
        won_ep      = winner_name == agent_name
        bankrupt_ep = ppo_player.status == "bankrupt"
        if won_ep:
            bonus = win_bonus
        elif bankrupt_ep:
            bonus = bankrupt_penalty
        else:
            # Cutoff game: PPO is neither winner nor bankrupt.
            # Rank active players by net worth; interpolate bonus linearly:
            #   rank 1 → +cutoff_rank_bonus, rank N → -cutoff_rank_bonus
            _active = sorted(
                [p for p in env.players if p.status != "bankrupt"],
                key=lambda p: p.net_worth(), reverse=True,
            )
            _n = len(_active)
            _rank = next((i + 1 for i, p in enumerate(_active) if p.pid == ppo_pid), _n)
            bonus = cutoff_rank_bonus * (2.0 * (_n - _rank) / max(_n - 1, 1) - 1.0)
        if bonus != 0.0:
            ppo_agent.add_terminal_bonus(bonus)
            ep_ppo_reward += bonus

        bankrupt        = 1 if ppo_player.status == "bankrupt" else 0
        final_cash       = ppo_player.cash
        final_net_worth  = ppo_player.net_worth()
        num_properties   = len(ppo_player.assets)

        total_bankrupt  += bankrupt
        sum_net_worth   += final_net_worth
        sum_cash        += final_cash
        sum_props       += num_properties

        _recent_wins.append(1 if won_ep else 0)
        _recent_agg_wins.append(1 if winner_name == "Aggressive" else 0)

        # Track best rolling win rate — only once window is full
        if prune_window > 0 and len(_recent_wins) == prune_window:
            _rolling_pct     = sum(_recent_wins)     / prune_window * 100
            _rolling_agg_pct = sum(_recent_agg_wins) / prune_window * 100

            if _rolling_pct > _best_rolling_pct:
                _best_rolling_pct    = _rolling_pct
                _best_window_agg_pct = _rolling_agg_pct
                _best_state = {k: v.cpu().clone() for k, v in ppo_agent.net.state_dict().items()}

                # Persist best weights to disk (crash recovery)
                if checkpoint_dir:
                    import datetime as _dt
                    _ts   = _dt.datetime.now().strftime("%H%M%S")
                    _name = f"best_ep{ep+1:05d}_ppo{_rolling_pct:.1f}_agg{_rolling_agg_pct:.1f}_{_ts}.pt"
                    torch.save(_best_state, os.path.join(checkpoint_dir, _name))
                    with open(os.path.join(checkpoint_dir, "best_checkpoint.json"), "w") as _fh:
                        json.dump({"episode": ep + 1, "ppo_pct": round(_rolling_pct, 2),
                                   "agg_pct": round(_rolling_agg_pct, 2), "file": _name}, _fh)

            # Lucky window: track best-gap window continuously (no cooldown on recording),
            # but only print once per independent (non-overlapping) window to avoid spam.
            if _rolling_pct > _rolling_agg_pct:
                _gap = _rolling_pct - _rolling_agg_pct
                _prev_gap = _lucky_window["gap"] if _lucky_window else -1
                if _gap > _prev_gap:
                    # Always update the stored best — no cooldown on recording
                    _lucky_window = {"episode": ep + 1, "ppo_pct": round(_rolling_pct, 2),
                                     "agg_pct": round(_rolling_agg_pct, 2),
                                     "gap": round(_gap, 2), "window": prune_window}
                if ep >= _lucky_next_check:
                    # Print only when outside the cooldown period
                    _prefix = f"[{run_label}] " if run_label else ""
                    print(f"  *** {_prefix}Lucky window ep {ep+1} "
                          f"(replacing '{agent_name}'): "
                          f"PPO={_rolling_pct:.1f}% > Agg={_rolling_agg_pct:.1f}% "
                          f"(gap={_gap:.2f}%)", flush=True)
                    _lucky_next_check = ep + prune_window  # cooldown until next independent window

        if logger:
            logger.log_episode(ep, winner_name, ep_ppo_reward, env.total_turns,
                               ppo_agent.episode_action_counts(), bankrupt,
                               final_cash, final_net_worth, num_properties)

        # Optuna milestones: print win% and optionally prune
        for _prune_ep, _prune_min in prune_checkpoints:
            if (ep + 1) == _prune_ep:
                rolling_pct = sum(_recent_wins) / len(_recent_wins) * 100
                cumul_pct   = win_counts[agent_name] / _prune_ep * 100
                eff_pct     = max(rolling_pct, cumul_pct)
                src         = "roll" if rolling_pct >= cumul_pct else "cum"
                t_tag = f" t{trial_id:3d}" if trial_id >= 0 else ""
                print(f"        ep {_prune_ep:>5d}{t_tag}  win%={eff_pct:.1f}%  "
                      f"[roll={rolling_pct:.1f}% cum={cumul_pct:.1f}% using={src}]",
                      flush=True)
                if eff_pct < _prune_min:
                    raise _PruneSignal(
                        f"win%={eff_pct:.1f}% < {_prune_min:.1f}% at ep {_prune_ep}")

    # Restore best checkpoint if we tracked one (guards against late-training collapse)
    if _best_state is not None:
        ppo_agent.net.load_state_dict({k: v.to(device) for k, v in _best_state.items()})
        if verbose:
            print(f"  Restored best checkpoint: {_best_rolling_pct:.1f}% rolling win rate "
                  f"(window={prune_window})", flush=True)

    stats = {
        "bankruptcies":        total_bankrupt,
        "bankruptcy_rate_pct": total_bankrupt / episodes * 100 if episodes else 0.0,
        "avg_final_net_worth": sum_net_worth / episodes if episodes else 0.0,
        "avg_final_cash":      sum_cash / episodes if episodes else 0.0,
        "avg_num_properties":  sum_props / episodes if episodes else 0.0,
        "best_window_ppo_pct": round(_best_rolling_pct, 2),
        "best_window_agg_pct": round(_best_window_agg_pct, 2),
        "lucky_window":        _lucky_window,   # None or dict with episode/ppo_pct/agg_pct/gap
    }
    return win_counts, ppo_agent, stats


def test_agent(agent_name: str, ppo_agent: PPOAgent,
               episodes: int = 1000,
               snapshots: tuple = (),
               device: torch.device = DEVICE) -> tuple[dict[str, int], dict[int, dict[str, int]]]:
    """Run episodes with frozen weights — no gradient updates, no buffer storage.

    Returns (final_win_counts, snapshot_dict) where snapshot_dict maps each
    value in `snapshots` to the cumulative win_counts at that episode.
    Snapshots let you get 1K/3K/5K results from a single 10K run.
    """
    players = make_default_players()
    ppo_pid = -1
    for p in players:
        if p.name == agent_name:
            p.agent = ppo_agent
            ppo_pid = p.pid

    win_counts: dict[str, int] = {p.name: 0 for p in players}
    snap_results: dict[int, dict[str, int]] = {}
    snap_set = set(snapshots)
    env = MonopolyEnv(players, summary=False, seed=None)

    ppo_agent.net.eval()
    with torch.no_grad():
        for ep in range(episodes):
            env.seed = ep + 10_000_000   # seeds disjoint from training
            env.reset()
            ppo_agent.start_episode()

            while not env.done:
                is_ppo = env.current_player().pid == ppo_pid
                env.step()
                if is_ppo:
                    ppo_agent.pop_pending(ppo_pid)  # discard — no learning

            if env.winner:
                win_counts[env.winner.name] += 1

            if (ep + 1) in snap_set:
                snap_results[ep + 1] = dict(win_counts)  # snapshot at this episode count

    ppo_agent.net.train()
    return win_counts, snap_results


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def create_figures(output_dir: str, agent_name: str,
                   baseline_win_rate_pct: Optional[float] = None):
    """Read the three CSVs and write a 2x3 figure of training curves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Read run_config.csv for baseline and actual training hyperparams used
    baseline_win_rate_pct = baseline_win_rate_pct if baseline_win_rate_pct is not None else 25.0
    cfg_lr, cfg_entropy, cfg_n_steps, cfg_normalize = LR, ENTROPY_COEF, N_STEPS, False
    cfg_path = os.path.join(output_dir, "run_config.csv")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            for row in csv.DictReader(f):
                p, v = row["param"], row["value"]
                if p == "baseline_win_rate_pct": baseline_win_rate_pct = float(v)
                elif p == "lr":                  cfg_lr = float(v)
                elif p == "entropy_coef":        cfg_entropy = float(v)
                elif p == "n_steps":             cfg_n_steps = int(v)
                elif p == "normalize_rewards":   cfg_normalize = v == "True"

    # -- load aggregate CSV ------------------------------------------------
    agg_ep, agg_wr, agg_ret, agg_len = [], [], [], []
    agg_buy_r, agg_mort_r, agg_imp_r, agg_bankrupt = [], [], [], []
    with open(os.path.join(output_dir, "training_aggregate.csv")) as f:
        for row in csv.DictReader(f):
            start = int(row["episode_range"].split("-")[0])
            agg_ep.append(start)
            agg_wr.append(float(row["avg_win_rate"]) * 100)
            agg_ret.append(float(row["avg_episode_return"]))
            agg_len.append(float(row["avg_episode_length"]))
            agg_buy_r.append(float(row["buy_act_rate"]) * 100)
            agg_mort_r.append(float(row["mort_act_rate"]) * 100)
            agg_imp_r.append(float(row["improve_act_rate"]) * 100)
            agg_bankrupt.append(float(row["bankruptcy_rate"]) * 100)

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
    norm_tag = "  [reward norm]" if cfg_normalize else ""
    fig, axes = plt.subplots(2, 4, figsize=(22, 9))
    fig.suptitle(f"PPO Training — replacing '{agent_name}'  "
                 f"(lr={cfg_lr:.0e}, entropy={cfg_entropy}, n_steps={cfg_n_steps}{norm_tag})",
                 fontsize=13)

    # Row 1: episode-level metrics (from aggregate)
    ax = axes[0, 0]
    ax.plot(agg_ep, agg_wr, "o-", color="#3F51B5", linewidth=1.8, markersize=4)
    ax.axhline(baseline_win_rate_pct, color="gray", linestyle="--", linewidth=0.8,
               label=f"no-RL baseline ({baseline_win_rate_pct:.1f}%)")
    ax.set_title(f"Win Rate (per {AGG_WINDOW}-ep window)")
    ax.set_xlabel("Episode"); ax.set_ylabel("Win %")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(agg_ep, agg_ret, "o-", color="#E91E63", linewidth=1.8, markersize=4)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title(f"Avg Episode Return (per {AGG_WINDOW}-ep window)")
    ax.set_xlabel("Episode"); ax.set_ylabel("Return")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    ax.plot(agg_ep, agg_len, "o-", color="#009688", linewidth=1.8, markersize=4)
    ax.set_title(f"Avg Episode Length (per {AGG_WINDOW}-ep window)")
    ax.set_xlabel("Episode"); ax.set_ylabel("Turns")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 3]
    ax.plot(agg_ep, agg_buy_r,    "-", color=colors["buy"],     linewidth=1.6, label="buy")
    ax.plot(agg_ep, agg_mort_r,   "-", color=colors["mort"],    linewidth=1.6, label="mortgage")
    ax.plot(agg_ep, agg_imp_r,    "-", color=colors["improve"], linewidth=1.6, label="improve")
    ax.plot(agg_ep, agg_bankrupt, "--", color="#9E9E9E",        linewidth=1.4, label="bankrupt %")
    ax.set_title(f"Action Rates & Bankruptcy (per {AGG_WINDOW}-ep window)")
    ax.set_xlabel("Episode"); ax.set_ylabel("%")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

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
def _print_results(agent_name: str, episodes: int, win_counts: dict, label: str = "Results"):
    print(f"\n{'='*58}")
    print(f"{label}")
    print(f"Replaced agent : {agent_name}  (PPO actor-critic)")
    print(f"Episodes       : {episodes}")
    print(f"{'Player':<20} {'Wins':>6}  {'Win %':>7}")
    print(f"{'-'*37}")
    for name, wins in sorted(win_counts.items(), key=lambda x: -x[1]):
        pct    = wins / episodes * 100
        marker = "  << PPO" if name == agent_name else ""
        print(f"  {name:<18} {wins:6d}  {pct:6.1f}%{marker}")
    print('='*58)


def compute_baseline_win_rate(target: str, episodes: int = BASELINE_EPISODES) -> float:
    """
    Win rate (%) for `target`'s default heuristic agent with NO learner in any
    slot — i.e. what that slot scores entirely on its own. Used as the
    reference line in training_curves.png instead of a naive 1/4 (25%) mark,
    since the 4 baseline heuristics are far from equally strong (e.g. the
    plain RandomAgent wins under 10% of games on its own).
    """
    players = make_default_players()
    env     = MonopolyEnv(players, summary=False, seed=None)
    wins = 0
    for ep in range(episodes):
        env.seed = ep
        env.reset()
        while not env.done:
            env.step()
        if env.winner and env.winner.name == target:
            wins += 1
    return wins / episodes * 100 if episodes else 0.0


def _save_run_config(agent_name: str, episodes: int, lr: float, device: torch.device,
                     baseline_win_rate_pct: float, output_dir: str,
                     entropy_coef: float = ENTROPY_COEF, n_steps: int = N_STEPS,
                     n_epochs: int = N_EPOCHS, gamma: float = GAMMA,
                     clip_eps: float = CLIP_EPS, normalize_rewards: bool = False):
    path = os.path.join(output_dir, "run_config.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["param", "value"])
        w.writerow(["replaced_agent", agent_name])
        w.writerow(["episodes", episodes])
        w.writerow(["lr", lr])
        w.writerow(["entropy_coef", entropy_coef])
        w.writerow(["n_steps", n_steps])
        w.writerow(["n_epochs", n_epochs])
        w.writerow(["gamma", gamma])
        w.writerow(["gae_lambda", GAE_LAMBDA])
        w.writerow(["clip_eps", clip_eps])
        w.writerow(["normalize_rewards", normalize_rewards])
        w.writerow(["value_coef", VALUE_COEF])
        w.writerow(["batch_size", BATCH_SIZE])
        w.writerow(["hidden_dim", HIDDEN_DIM])
        w.writerow(["state_dim", STATE_DIM])
        w.writerow(["improve_threshold", IMPROVE_THRESHOLD])
        w.writerow(["house_mult", HOUSE_MULT])
        w.writerow(["hotel_mult", HOTEL_MULT])
        w.writerow(["win_bonus", WIN_BONUS])
        w.writerow(["bankrupt_penalty", BANKRUPT_PENALTY])
        w.writerow(["device", str(device)])
        w.writerow(["baseline_win_rate_pct", f"{baseline_win_rate_pct:.2f}"])
    print(f"  Saved: {path}")


def _save_win_csv(agent_name: str, episodes: int, win_counts: dict, output_dir: str):
    path = os.path.join(output_dir, "win_counts.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["player", "wins", "win_rate_pct", "episodes", "is_ppo"])
        for name, wins in win_counts.items():
            w.writerow([name, wins, f"{wins/episodes*100:.2f}", episodes, name == agent_name])
    print(f"  Saved: {path}")


def _save_action_summary(ppo_agent: PPOAgent, stats: dict, episodes: int, output_dir: str):
    """Whole-run totals: how many times each decision type fired vs. was skipped,
    plus bankruptcy rate and average end-of-game standing for the PPO agent."""
    path = os.path.join(output_dir, "action_summary.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["decision_type", "acted", "skipped", "total", "act_rate_pct"])
        for dtype, (skips, acts) in ppo_agent.total_action_counts.items():
            total = skips + acts
            rate  = acts / total * 100 if total else 0.0
            w.writerow([dtype, acts, skips, total, f"{rate:.2f}"])
        w.writerow([])
        w.writerow(["metric", "value"])
        w.writerow(["episodes", episodes])
        w.writerow(["bankruptcies", stats["bankruptcies"]])
        w.writerow(["bankruptcy_rate_pct", f"{stats['bankruptcy_rate_pct']:.2f}"])
        w.writerow(["avg_final_net_worth", f"{stats['avg_final_net_worth']:.2f}"])
        w.writerow(["avg_final_cash", f"{stats['avg_final_cash']:.2f}"])
        w.writerow(["avg_num_properties", f"{stats['avg_num_properties']:.2f}"])
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------
def run_grid_search(target: str, episodes: int = GRID_SEARCH_EPISODES,
                    device: torch.device = DEVICE):
    """Grid search over lr x entropy_coef x n_steps x n_epochs (200 runs)."""
    combos = list(itertools.product(GRID_LR, GRID_ENTROPY_COEF, GRID_N_STEPS, GRID_N_EPOCHS))
    total  = len(combos)
    print(f"\nPPO grid search  (replaced: '{target}', episodes: {episodes}, runs: {total})")
    print(f"  device:       {device}")
    print(f"  lr:           {GRID_LR}")
    print(f"  entropy_coef: {GRID_ENTROPY_COEF}")
    print(f"  n_steps:      {GRID_N_STEPS}")
    print(f"  n_epochs:     {GRID_N_EPOCHS}\n")

    rows = []
    for i, (lr, ent, ns, ne) in enumerate(combos, 1):
        print(f"  [{i:3d}/{total}] lr={lr:.0e}  entropy={ent:.3f}  n_steps={ns}  n_epochs={ne} ...",
              end="", flush=True)
        wc, _, _ = train_for(target, episodes, lr=lr, entropy_coef=ent,
                            n_steps=ns, n_epochs=ne, device=device, verbose=False)
        wins = wc[target]
        pct  = wins / episodes * 100
        rank = sorted(wc, key=lambda n: -wc[n]).index(target) + 1
        rows.append((lr, ent, ns, ne, wins, pct, rank))
        print(f"  win%={pct:5.1f}%  rank={rank}/4")

    rows.sort(key=lambda r: -r[5])

    print(f"\n{'='*80}")
    print(f"Grid search results  (replaced: {target}, episodes: {episodes})")
    print(f"  {'lr':>8}  {'entropy':>8}  {'n_steps':>8}  {'n_epochs':>8}  {'Wins':>6}  {'Win %':>7}  {'Rank':>5}")
    print(f"  {'-'*72}")
    for j, (lr, ent, ns, ne, wins, pct, rank) in enumerate(rows):
        marker = "  <<" if j == 0 else ""
        print(f"  {lr:>8.0e}  {ent:>8.3f}  {ns:>8d}  {ne:>8d}  {wins:6d}  {pct:6.1f}%  {rank:>4}/4{marker}")
    print('='*80)

    bl, be, bn, bne, _, bp, br = rows[0]
    print(f"\nBest config:")
    print(f"  lr={bl:.0e}  entropy_coef={be}  n_steps={bn}  n_epochs={bne}")
    print(f"  -> PPO win rate {bp:.1f}%  (rank {br}/4 over {episodes} episodes)")

    fname = f"results_ppo_grid_{target}.csv"
    with open(fname, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["lr", "entropy_coef", "n_steps", "n_epochs",
                     "wins", "win_rate_pct", "rank", "episodes"])
        for lr, ent, ns, ne, wins, pct, rank in rows:
            w.writerow([lr, ent, ns, ne, wins, f"{pct:.2f}", rank, episodes])
    print(f"  Saved: {fname}")


def run_grid_search2(target: str, episodes: int = GRID_SEARCH_EPISODES,
                     device: torch.device = DEVICE):
    """
    Phase-2 grid search: top-20 (lr/entropy/n_steps/n_epochs) combos from phase-1
    × gamma × clip_eps, each run with and without Welford reward normalization.

    Total runs: 20 × 2 × 4 × 2 = 320.
    Saves results to results_ppo_grid2_{target}.csv, sorted by win rate descending.
    """
    normalize_opts = [False, True]
    combos = list(itertools.product(
        GRID2_TOP_COMBOS,   # 20 fixed (lr, ent, n_steps, n_epochs) tuples
        GRID2_GAMMA,        # 2 gamma values
        GRID2_CLIP_EPS,     # 4 clip_eps values
        normalize_opts,     # with / without reward normalization
    ))
    total = len(combos)
    print(f"\nPPO phase-2 grid search  (replaced: '{target}', episodes: {episodes}, "
          f"runs: {total})")
    print(f"  device:       {device}")
    print(f"  gamma:        {GRID2_GAMMA}")
    print(f"  clip_eps:     {GRID2_CLIP_EPS}")
    print(f"  normalize:    {normalize_opts}")
    print(f"  fixed combos: {len(GRID2_TOP_COMBOS)} top configs from phase-1\n")

    rows = []
    for i, ((lr, ent, ns, ne), gamma, clip_eps, normalize) in enumerate(combos, 1):
        norm_tag = "norm" if normalize else "raw"
        print(f"  [{i:3d}/{total}] lr={lr:.0e} ent={ent:.3f} ns={ns} ne={ne} "
              f"γ={gamma} clip={clip_eps} {norm_tag} ...", end="", flush=True)
        wc, _, _ = train_for(target, episodes,
                             lr=lr, entropy_coef=ent, n_steps=ns, n_epochs=ne,
                             gamma=gamma, clip_eps=clip_eps,
                             normalize_rewards=normalize,
                             device=device, verbose=False)
        wins = wc[target]
        pct  = wins / episodes * 100
        rank = sorted(wc, key=lambda n: -wc[n]).index(target) + 1
        rows.append((lr, ent, ns, ne, gamma, clip_eps, normalize, wins, pct, rank))
        print(f"  win%={pct:5.1f}%  rank={rank}/4")

    rows.sort(key=lambda r: -r[8])   # sort by win_rate_pct desc

    # Print summary tables — overall best, then split by normalize flag
    def _print_table(title: str, subset: list):
        print(f"\n{'='*100}")
        print(title)
        hdr = f"  {'lr':>8}  {'entropy':>8}  {'n_steps':>7}  {'n_epochs':>8}  " \
              f"{'gamma':>6}  {'clip':>6}  {'norm':>5}  {'Wins':>6}  {'Win %':>7}  {'Rank':>5}"
        print(hdr)
        print(f"  {'-'*92}")
        for j, (lr, ent, ns, ne, gamma, clip, norm, wins, pct, rank) in enumerate(subset):
            marker = "  <<" if j == 0 else ""
            print(f"  {lr:>8.0e}  {ent:>8.3f}  {ns:>7d}  {ne:>8d}  "
                  f"{gamma:>6.2f}  {clip:>6.2f}  {'yes' if norm else 'no':>5}  "
                  f"{wins:6d}  {pct:6.1f}%  {rank:>4}/4{marker}")
        print('='*100)

    _print_table(f"Phase-2 results — all {total} runs (target: {target})", rows[:20])
    _print_table("Top-10 WITHOUT normalization",
                 [r for r in rows if not r[6]][:10])
    _print_table("Top-10 WITH normalization",
                 [r for r in rows if r[6]][:10])

    best = rows[0]
    print(f"\nOverall best config:")
    print(f"  lr={best[0]:.0e}  entropy_coef={best[1]}  n_steps={best[2]}  n_epochs={best[3]}")
    print(f"  gamma={best[4]}  clip_eps={best[5]}  normalize={best[6]}")
    print(f"  -> PPO win rate {best[8]:.1f}%  (rank {best[9]}/4 over {episodes} episodes)")

    fname = f"results_ppo_grid2_{target}.csv"
    with open(fname, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["lr", "entropy_coef", "n_steps", "n_epochs",
                    "gamma", "clip_eps", "normalize_rewards",
                    "wins", "win_rate_pct", "rank", "episodes"])
        for lr, ent, ns, ne, gamma, clip, norm, wins, pct, rank in rows:
            w.writerow([lr, ent, ns, ne, gamma, clip, norm,
                        wins, f"{pct:.2f}", rank, episodes])
    print(f"  Saved: {fname}")


def _worker_noop(_):
    """Dummy task used to warm up a worker process before real work starts."""
    return True


def _optuna_worker(storage_url: str, study_name: str, stage: int,
                   device_str: str, fixed_params: dict):
    """
    Subprocess entry point for one parallel Optuna worker.
    Workers use dynamic dispatch: each calls study.optimize with a large budget,
    but a shared stop-callback (via SQLite) halts all workers once the target
    trial count is reached.  Fast workers pick up more trials automatically.
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        raise ImportError("pip install optuna sqlalchemy")

    import rl_environment as _rl_env

    torch.set_num_threads(1)          # one trial → one core; prevents oversubscription
    device   = torch.device(device_str)
    target   = fixed_params["_target"]
    n_trials = fixed_params["_n_trials"]
    extra    = {k: v for k, v in fixed_params.items() if not k.startswith("_")}

    def _cb(s, trial):
        val = trial.value if trial.value is not None else float("nan")
        try:
            best = s.best_value
            mark = " *" if val >= best - 1e-9 else "  "
        except ValueError:
            mark = " *"
        print(f"    {mark} trial {trial.number:3d} [{device_str}]  {val:.1f}%", flush=True)
        # Stop all workers once the target trial count is reached.
        finished = sum(1 for t in s.trials
                       if t.state.is_finished())
        if finished >= n_trials:
            s.stop()

    if stage in (1, 4):
        def obj(trial):
            params = dict(
                lr           = trial.suggest_float("lr", 1e-4, 1e-2, log=True),
                entropy_coef = trial.suggest_float("entropy_coef", 1e-4, 0.1, log=True),
                n_steps      = trial.suggest_categorical("n_steps", [64, 128, 256, 512]),
                n_epochs     = trial.suggest_int("n_epochs", 2, 12),
                gamma        = trial.suggest_float("gamma", 0.90, 0.999),
                clip_eps     = trial.suggest_float("clip_eps", 0.05, 0.5),
            )
            print(f"      trial {trial.number:3d} [{device_str}]  "
                  f"lr={params['lr']:.2e}  ent={params['entropy_coef']:.4f}  "
                  f"ns={params['n_steps']}  ne={params['n_epochs']}  "
                  f"gamma={params['gamma']:.4f}  clip={params['clip_eps']:.4f}", flush=True)
            try:
                _, agent, _ = train_for(target, verbose=False, device=device,
                                        normalize_rewards=True,
                                        prune_checkpoints=OPTUNA_PRUNE_CHECKPOINTS,
                                        prune_window=OPTUNA_PRUNE_WINDOW,
                                        trial_id=trial.number,
                                        **params, **extra)
            except _PruneSignal as e:
                print(f"      [pruned: {e}]", flush=True)
                raise optuna.TrialPruned()
            wc, _ = test_agent(target, agent, OPTUNA_TEST_EP, device=device)
            return wc[target] / OPTUNA_TEST_EP * 100

    elif stage == 2:
        def obj(trial):
            house_mult       = trial.suggest_int("house_mult", 1, 6)
            hotel_mult       = trial.suggest_int("hotel_mult", 1, 6)
            win_bonus        = trial.suggest_float("win_bonus", 1.0, 15.0)
            bankrupt_penalty = trial.suggest_float("bankrupt_penalty", -15.0, -0.5)
            _rl_env.HOUSE_MULT = house_mult   # safe: each process has its own module copy
            _rl_env.HOTEL_MULT = hotel_mult
            print(f"      trial {trial.number:3d} [{device_str}]  "
                  f"house={house_mult}  hotel={hotel_mult}  "
                  f"win_bonus={win_bonus:.2f}  bankrupt={bankrupt_penalty:.2f}", flush=True)
            try:
                _, agent, _ = train_for(target, verbose=False, device=device,
                                        normalize_rewards=True,
                                        win_bonus=win_bonus,
                                        bankrupt_penalty=bankrupt_penalty,
                                        prune_checkpoints=OPTUNA_PRUNE_CHECKPOINTS,
                                        prune_window=OPTUNA_PRUNE_WINDOW,
                                        trial_id=trial.number,
                                        **extra)
            except _PruneSignal as e:
                print(f"      [pruned: {e}]", flush=True)
                raise optuna.TrialPruned()
            wc, _ = test_agent(target, agent, OPTUNA_TEST_EP, device=device)
            return wc[target] / OPTUNA_TEST_EP * 100

    elif stage == 3:
        def obj(trial):
            hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256, 512])
            n_layers   = trial.suggest_int("n_layers", 2, 4)
            print(f"      trial {trial.number:3d} [{device_str}]  "
                  f"hidden={hidden_dim}  n_layers={n_layers}", flush=True)
            try:
                _, agent, _ = train_for(target, verbose=False, device=device,
                                        normalize_rewards=True,
                                        hidden=hidden_dim, n_layers=n_layers,
                                        prune_checkpoints=OPTUNA_PRUNE_CHECKPOINTS,
                                        prune_window=OPTUNA_PRUNE_WINDOW,
                                        trial_id=trial.number,
                                        **extra)
            except _PruneSignal as e:
                print(f"      [pruned: {e}]", flush=True)
                raise optuna.TrialPruned()
            wc, _ = test_agent(target, agent, OPTUNA_TEST_EP, device=device)
            return wc[target] / OPTUNA_TEST_EP * 100

    else:
        raise ValueError(f"Unknown stage {stage}")

    study = optuna.load_study(study_name=study_name, storage=storage_url)
    # Large budget: the _cb stop-callback halts the study after n_trials are done.
    study.optimize(obj, n_trials=n_trials * 10, callbacks=[_cb])


def run_optuna_search(target: str = OPTUNA_AGENT, device: torch.device = DEVICE):
    """
    Four-stage coordinate-descent Optuna search, parallelised with multiprocessing.

    Workers are spawned ONCE and reused across all 4 stages (avoids repeated OOM
    from simultaneous module re-imports on Windows spawn).  Workers are warmed up
    one at a time so only one process compiles train_ppo.py at a time.

    Dynamic dispatch: every worker pulls trials from a shared SQLite queue one at
    a time, so faster workers (CPU) naturally take more trials and any GPU worker
    is purely additive — it can never slow the stage down.

    Note: parallel TPE uses independent sampling for concurrent trials, so
    trial→param mapping differs from sequential mode; reconstruct_optuna_stage_csvs
    does not apply to runs produced by this function.
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        raise ImportError("pip install optuna sqlalchemy")

    from concurrent.futures import ProcessPoolExecutor, wait as fut_wait
    import rl_environment as _rl_env

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    db_path     = os.path.abspath(os.path.join(OUTPUT_ROOT, f"optuna_{target}.db"))
    storage_url = f"sqlite:///{db_path}"

    try:
        os.remove(db_path)
    except FileNotFoundError:
        pass

    use_gpu  = OPTUNA_GPU_WORKER and torch.cuda.is_available()
    gpu_label = " + 1 GPU" if use_gpu else ""
    n_workers = OPTUNA_N_WORKERS + (1 if use_gpu else 0)

    # ── Spawn CPU workers one at a time, with CUDA hidden ────────────────────
    # ProcessPoolExecutor(max_workers=N).submit() spawns ALL N workers at once
    # on the first call to _adjust_process_count — our previous "sequential"
    # warmup loop actually triggered N simultaneous spawns on the first submit.
    # Fix: use N separate pools of max_workers=1.  Each pool has exactly 1 worker
    # process, so spawning is truly sequential — no memory spike.
    #
    # CUDA_VISIBLE_DEVICES="" prevents CPU workers from loading cudnn DLLs
    # (~400 MB/worker instead of ~1.5 GB/worker).
    print(f"  Starting {n_workers} worker processes ({OPTUNA_N_WORKERS} CPU{gpu_label})...")
    _saved_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""   # hide CUDA — inherited by CPU workers

    cpu_pools = []
    for i in range(OPTUNA_N_WORKERS):
        pool = ProcessPoolExecutor(max_workers=1)  # exactly 1 worker per pool
        pool.submit(_worker_noop, i).result()       # spawn & wait before next
        cpu_pools.append(pool)
        print(f"    worker {i+1}/{n_workers} ready [cpu]", flush=True)

    # ── Restore CUDA, then spawn GPU worker (if any) ─────────────────────────
    if _saved_cuda is None:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = _saved_cuda

    gpu_pool = None
    if use_gpu:
        gpu_pool = ProcessPoolExecutor(max_workers=1)
        gpu_pool.submit(_worker_noop, 0).result()
        print(f"    worker {n_workers}/{n_workers} ready [cuda]", flush=True)

    def _run_stage(stage, n_trials, study_name, fixed_params):
        optuna.create_study(
            study_name=study_name,
            storage=storage_url,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            load_if_exists=False,
        )
        fp   = {**fixed_params, "_target": target, "_n_trials": n_trials}
        futs = [pool.submit(_optuna_worker, storage_url, study_name, stage, "cpu", fp)
                for pool in cpu_pools]
        if gpu_pool:
            futs.append(gpu_pool.submit(_optuna_worker, storage_url, study_name, stage, "cuda", fp))
        fut_wait(futs)
        for f in futs:
            f.result()
        return optuna.load_study(study_name=study_name, storage=storage_url)

    def _report_best(stage, study):
        t = study.best_trial
        print(f"\n  Stage {stage} best  win%={t.value:.1f}%")
        for k, v in t.params.items():
            print(f"    {k}: {v:.4g}" if isinstance(v, float) else f"    {k}: {v}")

    def _save_stage_trials(study, stage):
        completed = sorted([t for t in study.trials if t.value is not None],
                           key=lambda t: -t.value)
        if not completed:
            return
        param_keys = list(completed[0].params.keys())
        path = os.path.join(OUTPUT_ROOT, f"optuna_stage{stage}_trials_{target}.csv")
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["rank", "win_pct"] + param_keys)
            for rank, trial in enumerate(completed, 1):
                w.writerow([rank, round(trial.value, 2)]
                           + [trial.params.get(k) for k in param_keys])
        print(f"    Saved {len(completed)} trials -> {path}")

    try:
        # ── Stage 1: classic hyperparameters (default architecture) ──────────
        print(f"\n{'='*60}")
        print(f"Optuna Stage 1 — classic hyperparameters  "
              f"({OPTUNA_TRIALS_1} trials × {OPTUNA_EP_1} ep | {n_workers} workers{gpu_label})")
        print(f"{'='*60}")
        s1    = _run_stage(1, OPTUNA_TRIALS_1, f"stage1_{target}",
                           dict(episodes=OPTUNA_EP_1))
        _report_best(1, s1)
        best1 = s1.best_params

        # ── Stage 2: reward shaping ───────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"Optuna Stage 2 — reward shaping  "
              f"({OPTUNA_TRIALS_2} trials × {OPTUNA_EP_2} ep | {n_workers} workers{gpu_label})")
        print(f"{'='*60}")
        s2    = _run_stage(2, OPTUNA_TRIALS_2, f"stage2_{target}",
                           dict(**best1, episodes=OPTUNA_EP_2))
        _report_best(2, s2)
        best2 = s2.best_params
        _rl_env.HOUSE_MULT = best2["house_mult"]
        _rl_env.HOTEL_MULT = best2["hotel_mult"]

        # ── Stage 3: network architecture ────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"Optuna Stage 3 — network architecture  "
              f"({OPTUNA_TRIALS_3} trials × {OPTUNA_EP_3} ep | {n_workers} workers{gpu_label})")
        print(f"{'='*60}")
        s3    = _run_stage(3, OPTUNA_TRIALS_3, f"stage3_{target}",
                           dict(**best1,
                                win_bonus=best2["win_bonus"],
                                bankrupt_penalty=best2["bankrupt_penalty"],
                                episodes=OPTUNA_EP_3))
        _report_best(3, s3)
        best3 = s3.best_params

        # ── Stage 4: re-tune classic hypers for found architecture ───────────
        print(f"\n{'='*60}")
        print(f"Optuna Stage 4 — re-tune classic hypers for found architecture  "
              f"({OPTUNA_TRIALS_4} trials × {OPTUNA_EP_4} ep | {n_workers} workers{gpu_label})")
        print(f"  fixed: hidden_dim={best3['hidden_dim']}  n_layers={best3['n_layers']}")
        print(f"{'='*60}")
        s4    = _run_stage(4, OPTUNA_TRIALS_4, f"stage4_{target}",
                           dict(hidden=best3["hidden_dim"],
                                n_layers=best3["n_layers"],
                                win_bonus=best2["win_bonus"],
                                bankrupt_penalty=best2["bankrupt_penalty"],
                                episodes=OPTUNA_EP_4))
        _report_best(4, s4)
        best4 = s4.best_params

    finally:
        for pool in cpu_pools:
            pool.shutdown(wait=False)
        if gpu_pool:
            gpu_pool.shutdown(wait=False)

    # ── Save results ──────────────────────────────────────────────────────────
    _save_stage_trials(s1, 1)
    _save_stage_trials(s2, 2)
    _save_stage_trials(s3, 3)
    _save_stage_trials(s4, 4)

    out_csv = os.path.join(OUTPUT_ROOT, f"optuna_results_{target}.csv")
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["stage", "param", "best_value", "stage_win_pct"])
        for k, v in best1.items():
            w.writerow([1, k, v, round(s1.best_value, 2)])
        for k, v in best2.items():
            w.writerow([2, k, v, round(s2.best_value, 2)])
        for k, v in best3.items():
            w.writerow([3, k, v, round(s3.best_value, 2)])
        for k, v in best4.items():
            w.writerow([4, k, v, round(s4.best_value, 2)])
    print(f"\n  Summary saved -> {out_csv}")

    print(f"\n{'='*60}")
    print("Final best configuration")
    print(f"{'='*60}")
    all_best = {**best4, **best2, **best3}
    for k, v in all_best.items():
        print(f"  {k}: {v:.6g}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"\n  Stage win rates:  S1={s1.best_value:.1f}%  S2={s2.best_value:.1f}%  "
          f"S3={s3.best_value:.1f}%  S4={s4.best_value:.1f}%")

    return best1, best2, best3, best4


def reconstruct_optuna_stage_csvs(target: str = OPTUNA_AGENT):
    """
    Reconstruct per-stage trial CSVs from the logged win percentages.
    Optuna's TPESampler is deterministic: given the same seed=42 and the same
    observation history (values fed in order), study.ask() reproduces the exact
    same parameter suggestions as the original run.  No training needed.
    Run once, then set RUN_FINAL_COMBO = True.
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        raise ImportError("Run: pip install optuna")

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # ── Observed win percentages (trial order matches the Optuna output) ──────
    S1_VALS = [11.0,  3.0,  3.0, 25.5, 13.0, 29.0, 22.5,  7.0, 28.5, 19.0,
                0.0, 28.0, 23.5, 17.0, 11.5, 28.0,  0.0, 24.5,  0.0, 28.0,
               23.5, 28.5, 23.5, 23.0, 27.5, 27.0, 26.0, 25.0, 28.0, 19.5,
               26.5, 28.0, 27.5, 24.0, 22.5, 28.5, 25.5, 27.5, 27.0, 14.5,
               27.0, 28.5, 25.0, 26.5, 29.5, 28.0, 27.0, 27.0, 27.5, 26.0]

    S2_VALS = [28.0, 25.5, 28.0, 26.5, 27.0, 27.0, 26.0, 24.5, 26.5, 27.5,
               29.0, 26.5, 26.5, 26.0, 27.0, 25.5, 28.0, 26.0, 28.5, 26.5,
               27.0, 27.0, 25.5, 27.5, 28.0, 25.5, 25.5, 28.0, 29.0, 29.5,
               24.5, 28.5, 27.5, 26.5, 27.0, 29.5, 25.0, 29.5, 27.0, 25.5]

    S3_VALS = [27.5, 30.0, 28.5, 27.5, 29.0, 29.5, 27.0, 25.0, 28.5, 28.0,
               27.0, 28.0, 25.5, 25.0, 25.5, 29.0, 26.5, 27.0, 26.5, 26.0,
               29.0, 27.5, 27.5, 27.0, 28.0, 27.5, 31.0, 28.0, 28.5, 27.5]

    S4_VALS = [23.5, 13.0,  4.0,  0.0, 22.5, 28.0, 22.5, 11.0, 28.0, 23.5,
                0.0, 25.5, 29.0, 27.5,  0.0, 21.5, 29.0, 27.0, 28.0, 30.5,
               27.0, 28.0, 28.5, 26.0, 26.0, 27.5, 11.5, 24.5, 29.5, 26.5,
               25.0, 25.0, 25.5, 27.0, 22.5, 28.5, 26.5, 26.5, 29.5, 23.5,
               29.0, 27.5, 27.0, 14.5, 23.5, 16.0, 26.5, 23.0,  4.5, 27.5]

    # ── Suggestion functions matching run_optuna_search exactly ───────────────
    def _suggest_s1(trial):
        return dict(
            lr           = trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            entropy_coef = trial.suggest_float("entropy_coef", 1e-4, 0.1, log=True),
            n_steps      = trial.suggest_categorical("n_steps", [64, 128, 256, 512]),
            n_epochs     = trial.suggest_int("n_epochs", 2, 12),
            gamma        = trial.suggest_float("gamma", 0.90, 0.999),
            clip_eps     = trial.suggest_float("clip_eps", 0.05, 0.5),
        )

    def _suggest_s2(trial):
        return dict(
            house_mult       = trial.suggest_int("house_mult", 1, 6),
            hotel_mult       = trial.suggest_int("hotel_mult", 1, 6),
            win_bonus        = trial.suggest_float("win_bonus", 1.0, 15.0),
            bankrupt_penalty = trial.suggest_float("bankrupt_penalty", -15.0, -0.5),
        )

    def _suggest_s3(trial):
        return dict(
            hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256, 512]),
            n_layers   = trial.suggest_int("n_layers", 2, 4),
        )

    # Stage 4 has the same search space as stage 1
    _suggest_s4 = _suggest_s1

    def _replay(vals, suggest_fn, stage):
        study = optuna.create_study(direction="maximize",
                                     sampler=optuna.samplers.TPESampler(seed=42))
        records = []
        for val in vals:
            trial = study.ask()
            params = suggest_fn(trial)
            study.tell(trial, val)
            records.append((val, params))

        # Sort by win_pct descending for the rank-based combo search
        records.sort(key=lambda x: -x[0])
        param_keys = list(records[0][1].keys())
        path = os.path.join(OUTPUT_ROOT, f"optuna_stage{stage}_trials_{target}.csv")
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["rank", "win_pct"] + param_keys)
            for rank, (val, params) in enumerate(records, 1):
                w.writerow([rank, val] + [params[k] for k in param_keys])
        # Sanity-check: best params should match the printed stage summary
        best_val, best_params = records[0]
        print(f"  Stage {stage}: {len(vals)} trials reconstructed -> {path}")
        print(f"    best win%={best_val:.1f}%  params={best_params}")

    print("Reconstructing Optuna trial parameters from logged win percentages …")
    _replay(S1_VALS, _suggest_s1, 1)
    _replay(S2_VALS, _suggest_s2, 2)
    _replay(S3_VALS, _suggest_s3, 3)
    _replay(S4_VALS, _suggest_s4, 4)
    print("\nDone — set RUN_FINAL_COMBO = True to launch the combo search.")


def run_final_combo_search(target: str = OPTUNA_AGENT, n_combos: int = 20,
                            episodes: int = 2000, device: torch.device = DEVICE):
    """
    Final grid search: zip the top-N rows from the per-stage trial CSVs produced by
    run_optuna_search, so that combo[i] = (rank-i stage-2 reward params,
    rank-i stage-3 architecture, rank-i stage-4 classic hypers).
    Requires optuna_stage{2,3,4}_trials_{target}.csv to exist in OUTPUT_ROOT.
    Run run_optuna_search first to generate them.
    """
    import rl_environment as _rl_env

    def _read_stage(stage: int) -> list[dict]:
        path = os.path.join(OUTPUT_ROOT, f"optuna_stage{stage}_trials_{target}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found — run run_optuna_search first to generate trial CSVs.")
        rows = []
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                rows.append(row)
        return rows  # already sorted by rank (win_pct desc)

    s2_rows = _read_stage(2)
    s3_rows = _read_stage(3)
    s4_rows = _read_stage(4)

    available = min(len(s2_rows), len(s3_rows), len(s4_rows))
    if available < n_combos:
        print(f"  Warning: only {available} combos available (fewer trials than {n_combos})")
        n_combos = available

    print(f"\n{'='*72}")
    print(f"Final combo search  target={target}  combos={n_combos}  ep={episodes}")
    print(f"  Stage 2 (reward):  {len(s2_rows)} trials available")
    print(f"  Stage 3 (arch):    {len(s3_rows)} trials available")
    print(f"  Stage 4 (hypers):  {len(s4_rows)} trials available")
    print(f"{'='*72}")

    results = []
    for i in range(n_combos):
        s2 = s2_rows[i]; s3 = s3_rows[i]; s4 = s4_rows[i]

        house_mult = int(float(s2["house_mult"]))
        hotel_mult = int(float(s2["hotel_mult"]))
        win_bonus  = float(s2["win_bonus"])
        bankrupt_p = float(s2["bankrupt_penalty"])
        hidden_dim = int(float(s3["hidden_dim"]))
        n_layers   = int(float(s3["n_layers"]))
        lr         = float(s4["lr"])
        ent        = float(s4["entropy_coef"])
        n_steps    = int(float(s4["n_steps"]))
        n_epochs   = int(float(s4["n_epochs"]))
        gamma      = float(s4["gamma"])
        clip_eps   = float(s4["clip_eps"])

        _rl_env.HOUSE_MULT = house_mult
        _rl_env.HOTEL_MULT = hotel_mult

        print(f"  [{i+1:2d}/{n_combos}] "
              f"h={hidden_dim} L={n_layers} lr={lr:.2e} ent={ent:.3f} "
              f"ns={n_steps} ne={n_epochs} γ={gamma:.3f} clip={clip_eps:.3f} "
              f"hm={house_mult} htm={hotel_mult} wb={win_bonus:.1f} bp={bankrupt_p:.1f}",
              end="  ", flush=True)

        wc, _, _ = train_for(target, episodes,
                             lr=lr, entropy_coef=ent, n_steps=n_steps, n_epochs=n_epochs,
                             gamma=gamma, clip_eps=clip_eps, normalize_rewards=True,
                             hidden=hidden_dim, n_layers=n_layers,
                             win_bonus=win_bonus, bankrupt_penalty=bankrupt_p,
                             verbose=False, device=device)
        wr  = wc[target] / episodes * 100
        pos = sorted(wc.values(), reverse=True).index(wc[target]) + 1
        print(f"win%={wr:.1f}%  rank={pos}/4")
        results.append(dict(
            combo=i + 1, win_pct=wr, rank_vs_others=pos,
            lr=lr, entropy_coef=ent, n_steps=n_steps, n_epochs=n_epochs,
            gamma=gamma, clip_eps=clip_eps,
            house_mult=house_mult, hotel_mult=hotel_mult,
            win_bonus=win_bonus, bankrupt_penalty=bankrupt_p,
            hidden_dim=hidden_dim, n_layers=n_layers,
        ))

    results.sort(key=lambda r: -r["win_pct"])

    print(f"\n{'='*72}")
    print(f"Final combo results — sorted by win %")
    print(f"{'='*72}")
    for j, r in enumerate(results):
        mark = "  <<" if j == 0 else ""
        print(f"  #{j+1:2d}  combo={r['combo']:2d}  win%={r['win_pct']:.1f}%  "
              f"h={r['hidden_dim']} L={r['n_layers']} lr={r['lr']:.2e} "
              f"ent={r['entropy_coef']:.3f}{mark}")

    fname = os.path.join(OUTPUT_ROOT, f"optuna_final_combo_{target}.csv")
    fieldnames = ["combo", "win_pct", "rank_vs_others",
                  "lr", "entropy_coef", "n_steps", "n_epochs", "gamma", "clip_eps",
                  "house_mult", "hotel_mult", "win_bonus", "bankrupt_penalty",
                  "hidden_dim", "n_layers"]
    with open(fname, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow({k: round(r[k], 6) if isinstance(r[k], float) else r[k]
                        for k in fieldnames})
    print(f"\n  Saved: {fname}")
    return results


def _benchmark_device(hidden: int = HIDDEN_DIM, n_layers: int = 3,
                       episodes: int = 100) -> torch.device:
    """
    Time a short training run on CPU and (if available) CUDA, return the faster one.
    Pass the same hidden/n_layers as the actual run — the GPU/CPU crossover depends
    on network size (large nets favour CUDA; small nets may lose to CPU due to
    per-step transfer overhead from the Python game loop).
    """
    import time
    candidates = [torch.device("cpu")]
    if torch.cuda.is_available():
        candidates.append(torch.device("cuda"))
    else:
        print("  CUDA not available — using cpu")
        return torch.device("cpu")

    print(f"  Benchmarking devices  (hidden={hidden}, n_layers={n_layers}, "
          f"{episodes} episodes each) …")
    best_dev, best_time = None, float("inf")
    for dev in candidates:
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        train_for("Random", episodes=episodes, hidden=hidden, n_layers=n_layers,
                  verbose=False, device=dev)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        print(f"    {str(dev):6s}  {elapsed:.2f}s  ({elapsed/episodes*1000:.1f} ms/ep)")
        if elapsed < best_time:
            best_time = elapsed
            best_dev = dev

    print(f"  → auto-selected: {best_dev}")
    return best_dev


def _make_output_dir(agent_name: str, tag: str = "") -> str:
    """
    Build a fresh output directory for this run so results are never overwritten.
    Pass a descriptive `tag` (e.g. "_expanded_state") to label a substantial change;
    otherwise an auto-incrementing numeric suffix is used.
    """
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    base = f"{agent_name}{tag}" if tag else agent_name
    candidate = os.path.join(OUTPUT_ROOT, base)
    if tag and not os.path.exists(candidate):
        return candidate
    i = 1
    while os.path.exists(os.path.join(OUTPUT_ROOT, f"{base}_{i}")):
        i += 1
    return os.path.join(OUTPUT_ROOT, f"{base}_{i}")


# ---------------------------------------------------------------------------
# Run configuration — edit these and click Run in PyCharm (no CLI args needed)
# ---------------------------------------------------------------------------
RUN_REPLACE    = "Random"   # one of ALL_AGENT_NAMES, or "All" to run all 4 sequentially
RUN_EPISODES   = EPISODES
RUN_DEVICE     = "auto"     # "cpu", "cuda", or "auto" (benchmarks at startup)

# ── Normalization toggle ──────────────────────────────────────────────────
# Flipping this one variable selects the phase-2 grid-search best for each variant:
#   True  → Welford reward normalization — best: 31.33% win rate
#   False → raw rewards                 — best: 31.03% win rate
RUN_NORMALIZE  = True

# RUN_* always mirrors the top-level constants so updates only need to happen once.
# _RAW_CFG kept for the non-normalized path (no normalization layer → lower LR).
_RAW_CFG = dict(lr=3e-3, entropy_coef=0.005, n_steps=128, n_epochs=10, gamma=0.95, clip_eps=0.2)

if RUN_NORMALIZE:
    RUN_LR           = LR
    RUN_ENTROPY_COEF = ENTROPY_COEF
    RUN_N_STEPS      = N_STEPS
    RUN_N_EPOCHS     = N_EPOCHS
    RUN_GAMMA        = GAMMA
    RUN_CLIP_EPS     = CLIP_EPS
else:
    RUN_LR           = _RAW_CFG["lr"]
    RUN_ENTROPY_COEF = _RAW_CFG["entropy_coef"]
    RUN_N_STEPS      = _RAW_CFG["n_steps"]
    RUN_N_EPOCHS     = _RAW_CFG["n_epochs"]
    RUN_GAMMA        = _RAW_CFG["gamma"]
    RUN_CLIP_EPS     = _RAW_CFG["clip_eps"]
# ─────────────────────────────────────────────────────────────────────────

RUN_HIDDEN_DIM = HIDDEN_DIM   # backbone hidden size  (Optuna best: 256)
RUN_N_LAYERS   = 4            # backbone depth        (Optuna best: 4)

RUN_GRID_SEARCH   = False   # phase-1: sweep over lr/entropy/n_steps/n_epochs (done)
RUN_GRID_SEARCH_2 = False   # phase-2: top-20 × gamma × clip_eps × normalize (done)
RUN_OPTUNA             = False  # 4-stage Optuna search  (run once to get per-stage trial CSVs)
RUN_RECONSTRUCT_OPTUNA = False  # reconstruct trial CSVs from logged win percentages (no training)
RUN_FINAL_COMBO        = False  # final 20-combo search using top-N from each Optuna stage CSV
RUN_PARALLEL      = True   # repeated parallel train+test until PPO beats Aggressive
RUN_TAG           = "_full_state"  # descriptive folder suffix; "" → numbered
# Optuna settings are in the constants block above (OPTUNA_AGENT, OPTUNA_TRIALS_*, etc.)

# Parallel search settings (used only when RUN_PARALLEL = True)
PARALLEL_TOP_K       = 3    # keep this many best run directories on disk at once
PARALLEL_SUCCESS_GAP = 0.4  # stop when PPO win% > Aggressive win% by more than this


# ---------------------------------------------------------------------------
# Parallel search worker + orchestrator
# ---------------------------------------------------------------------------
def _run_single_worker(run_id: int, out_base: str, device_str: str) -> dict:
    """Full warm-up + train + multi-test cycle in a subprocess.
    Uses module-level RUN_* / WARMUP_EPISODES constants."""
    torch.set_num_threads(1)
    device = torch.device(device_str)

    t       = RUN_REPLACE
    run_dir = os.path.join(out_base, f"run_{run_id:04d}")
    os.makedirs(run_dir, exist_ok=True)

    logger = TrainingLogger(run_dir, t)
    try:
        win_counts, ppo_agent, stats = train_for(
            t, RUN_EPISODES,
            lr=RUN_LR, entropy_coef=RUN_ENTROPY_COEF,
            n_steps=RUN_N_STEPS, n_epochs=RUN_N_EPOCHS,
            gamma=RUN_GAMMA, clip_eps=RUN_CLIP_EPS,
            normalize_rewards=RUN_NORMALIZE,
            hidden=RUN_HIDDEN_DIM, n_layers=RUN_N_LAYERS,
            prune_window=1000,
            warmup_episodes=WARMUP_EPISODES,
            checkpoint_dir=run_dir,
            run_label=f"run_{run_id:04d}",
            logger=logger, device=device, verbose=True,
        )
        logger.close()

        _save_run_config(t, RUN_EPISODES, RUN_LR, device, 25.0, run_dir,
                         entropy_coef=RUN_ENTROPY_COEF, n_steps=RUN_N_STEPS,
                         n_epochs=RUN_N_EPOCHS, gamma=RUN_GAMMA, clip_eps=RUN_CLIP_EPS,
                         normalize_rewards=RUN_NORMALIZE)
        _save_action_summary(ppo_agent, stats, RUN_EPISODES, run_dir)
        create_figures(run_dir, t, baseline_win_rate_pct=25.0)

        # --- multi-scale tests (single 10K run; smaller counts are prefix slices) ---
        _snap_sizes = [1_000, 3_000, 5_000]
        tc_10k, snaps = test_agent(t, ppo_agent, episodes=10_000,
                                   snapshots=tuple(_snap_sizes), device=device)
        test_results = {}
        for n_ep, tc in {**snaps, 10_000: tc_10k}.items():
            ppo_p = tc.get(t, 0)            / n_ep * 100
            agg_p = tc.get("Aggressive", 0) / n_ep * 100
            test_results[n_ep] = {"counts": tc, "ppo_pct": round(ppo_p, 2),
                                  "agg_pct": round(agg_p, 2)}
            _save_win_csv(t, n_ep, tc, run_dir)
            print(f"  [run_{run_id:04d}] test {n_ep//1000}K: "
                  f"PPO={ppo_p:.1f}%  Agg={agg_p:.1f}%  gap={ppo_p-agg_p:+.2f}%",
                  flush=True)

        # Primary metric: 10K test
        ppo_pct = test_results[10_000]["ppo_pct"]
        agg_pct = test_results[10_000]["agg_pct"]

        lucky = stats.get("lucky_window")

        result = dict(
            run_id=run_id, run_dir=run_dir,
            ppo_pct=ppo_pct, agg_pct=agg_pct,
            test_results=test_results,
            lucky_window=lucky,
            best_window_ppo_pct=stats.get("best_window_ppo_pct", 0.0),
            best_window_agg_pct=stats.get("best_window_agg_pct", 0.0),
        )
        with open(os.path.join(run_dir, "result.json"), "w") as fh:
            json.dump(result, fh, indent=2)
        return result

    except Exception as exc:
        logger.close()
        return dict(run_id=run_id, run_dir=run_dir,
                    ppo_pct=0.0, agg_pct=0.0, error=str(exc))


def run_parallel_search(agent_name: str):
    """
    Repeatedly spawns OPTUNA_N_WORKERS parallel train+test runs in batches.
    Keeps PARALLEL_TOP_K best result directories; evicts the rest.
    Stops when PPO win% > Aggressive win% + PARALLEL_SUCCESS_GAP, or on Ctrl+C.
    """
    import json as _json, shutil as _shutil
    from concurrent.futures import ProcessPoolExecutor, wait as fut_wait

    out_base = os.path.join(OUTPUT_ROOT, f"parallel_{agent_name}")
    os.makedirs(out_base, exist_ok=True)
    top_k_path = os.path.join(out_base, "top_k.json")

    n_workers = OPTUNA_N_WORKERS
    top_k: list[dict] = []
    # Start run_id past any existing runs so we never reuse a directory
    _existing = [
        int(d[4:]) for d in os.listdir(out_base)
        if d.startswith("run_") and d[4:].isdigit()
        and os.path.isdir(os.path.join(out_base, d))
    ]
    run_id  = (max(_existing) + 1) if _existing else 0
    success = False

    print(f"\n{'='*64}")
    print(f"Parallel search  —  {n_workers} workers  —  replacing '{agent_name}'")
    print(f"Goal : PPO win% > Aggressive win% + {PARALLEL_SUCCESS_GAP}%")
    print(f"Keep : top {PARALLEL_TOP_K} runs in  {out_base}/")
    print('='*64, flush=True)

    saved_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    pools = []
    for i in range(n_workers):
        pool = ProcessPoolExecutor(max_workers=1)
        pool.submit(_worker_noop, i).result()
        pools.append(pool)
        print(f"    worker {i+1}/{n_workers} ready", flush=True)
    if saved_cuda is None:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = saved_cuda

    def _update_top_k(r: dict) -> bool:
        """Insert r into top-k if it qualifies; evict displaced run. Returns True if kept."""
        if r.get("error"):
            return False
        if len(top_k) < PARALLEL_TOP_K:
            top_k.append(r)
        elif r["ppo_pct"] > top_k[-1]["ppo_pct"]:
            evicted = top_k.pop()
            _shutil.rmtree(evicted["run_dir"], ignore_errors=True)
            print(f"    evicted  run_{evicted['run_id']:04d}  "
                  f"(PPO={evicted['ppo_pct']:.1f}%)", flush=True)
            top_k.append(r)
        else:
            _shutil.rmtree(r["run_dir"], ignore_errors=True)
            return False
        top_k.sort(key=lambda x: -x["ppo_pct"])
        with open(top_k_path, "w") as fh:
            _json.dump(top_k, fh, indent=2)
        return True

    try:
        batch = 0
        while True:
            batch += 1
            ids = list(range(run_id, run_id + n_workers))
            run_id += n_workers
            print(f"\n  Batch {batch}  (runs {ids[0]:04d}–{ids[-1]:04d})", flush=True)

            futs = [pool.submit(_run_single_worker, rid, out_base, "cpu")
                    for pool, rid in zip(pools, ids)]
            fut_wait(futs)

            for fut in futs:
                try:
                    r = fut.result()
                except Exception as exc:
                    print(f"    worker error: {exc}", flush=True)
                    continue

                if r.get("error"):
                    print(f"    run_{r['run_id']:04d}  FAILED: {r['error']}", flush=True)
                    continue

                gap   = r["ppo_pct"] - r["agg_pct"]
                lucky = r.get("lucky_window")
                kept  = _update_top_k(r)
                label = "KEPT   " if kept else "evicted"
                lw_tag = f"  [lucky window ep{lucky['episode']} gap={lucky['gap']:+.2f}%]" if lucky else ""
                print(f"    run_{r['run_id']:04d}  [{label}]  "
                      f"PPO={r['ppo_pct']:.1f}%  Agg={r['agg_pct']:.1f}%  "
                      f"gap={gap:+.2f}%{lw_tag}", flush=True)

                if gap > PARALLEL_SUCCESS_GAP:
                    print(f"\n{'*'*64}")
                    print(f"  SUCCESS (test)  run_{r['run_id']:04d}  —  "
                          f"PPO beats Aggressive by {gap:.2f}% on 10K test")
                    print(f"  Saved in: {r['run_dir']}")
                    print('*'*64, flush=True)
                    success = True

                if lucky and not success:
                    print(f"\n{'*'*64}")
                    print(f"  SUCCESS (lucky window)  run_{r['run_id']:04d}")
                    print(f"  ep {lucky['episode']}: PPO={lucky['ppo_pct']:.1f}% > "
                          f"Agg={lucky['agg_pct']:.1f}%  gap={lucky['gap']:+.2f}%")
                    print(f"  Best weights saved in: {r['run_dir']}/best_checkpoint.pt")
                    print('*'*64, flush=True)
                    success = True

            if success:
                break

    except KeyboardInterrupt:
        print("\n  Stopped by user.", flush=True)
    finally:
        for pool in pools:
            pool.shutdown(wait=False)

    print(f"\n  Top {PARALLEL_TOP_K} at stop:")
    for j, r in enumerate(top_k, 1):
        gap = r["ppo_pct"] - r["agg_pct"]
        print(f"    #{j}  run_{r['run_id']:04d}  PPO={r['ppo_pct']:.1f}%  "
              f"Agg={r['agg_pct']:.1f}%  gap={gap:+.2f}%  dir={r['run_dir']}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if RUN_OPTUNA or RUN_PARALLEL:
        # Workers each have their own fixed device; no need to benchmark the main process.
        device = torch.device("cpu")
    elif RUN_DEVICE == "auto":
        device = _benchmark_device(hidden=RUN_HIDDEN_DIM, n_layers=RUN_N_LAYERS)
    else:
        device = torch.device(RUN_DEVICE)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"RUN_DEVICE='{RUN_DEVICE}' but CUDA is not available")

    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else None
    print(f"Using device: {device}" + (f"  ({gpu_name})" if gpu_name else ""))

    if RUN_PARALLEL:
        target = RUN_REPLACE if RUN_REPLACE != "All" else "Random"
        run_parallel_search(target)
    elif RUN_RECONSTRUCT_OPTUNA:
        reconstruct_optuna_stage_csvs(OPTUNA_AGENT)
    elif RUN_FINAL_COMBO:
        run_final_combo_search(OPTUNA_AGENT, n_combos=20, episodes=9000, device=device)
    elif RUN_OPTUNA:
        run_optuna_search(OPTUNA_AGENT, device=device)
    elif RUN_GRID_SEARCH_2:
        target = RUN_REPLACE if RUN_REPLACE != "All" else "Random"
        run_grid_search2(target, GRID_SEARCH_EPISODES, device=device)
    elif RUN_GRID_SEARCH:
        target = RUN_REPLACE if RUN_REPLACE != "All" else "Random"
        run_grid_search(target, GRID_SEARCH_EPISODES, device=device)
    else:
        targets = ALL_AGENT_NAMES if RUN_REPLACE == "All" else [RUN_REPLACE]
        for t in targets:
            baseline_pct = 25.0

            out_dir = _make_output_dir(t, RUN_TAG)
            logger  = TrainingLogger(out_dir, t)
            norm_label = "normalized" if RUN_NORMALIZE else "raw"
            print(f"\n{'='*58}")
            print(f"Training PPO replacing '{t}' for {RUN_EPISODES} episodes")
            print(f"Config: lr={RUN_LR:.0e}  entropy={RUN_ENTROPY_COEF}  n_steps={RUN_N_STEPS}  "
                  f"n_epochs={RUN_N_EPOCHS}  gamma={RUN_GAMMA}  clip_eps={RUN_CLIP_EPS}  "
                  f"hidden={RUN_HIDDEN_DIM}  n_layers={RUN_N_LAYERS}  "
                  f"reward={norm_label}  device={device}")
            print(f"Output: {out_dir}/")
            print('='*58)
            win_counts, ppo_agent, stats = train_for(
                t, RUN_EPISODES,
                lr=RUN_LR, entropy_coef=RUN_ENTROPY_COEF,
                n_steps=RUN_N_STEPS, n_epochs=RUN_N_EPOCHS,
                gamma=RUN_GAMMA, clip_eps=RUN_CLIP_EPS,
                normalize_rewards=RUN_NORMALIZE,
                hidden=RUN_HIDDEN_DIM, n_layers=RUN_N_LAYERS,
                logger=logger, device=device)
            logger.close()
            _test_episodes = 5000
            print(f"  Running test evaluation ({_test_episodes} episodes, frozen weights)...")
            test_counts, _ = test_agent(t, ppo_agent, episodes=_test_episodes, device=device)
            _print_results(t, _test_episodes, test_counts, label=f"Test Results (frozen weights, {_test_episodes} ep)")
            _save_win_csv(t, _test_episodes, test_counts, out_dir)
            _save_run_config(t, RUN_EPISODES, RUN_LR, device, baseline_pct, out_dir,
                             entropy_coef=RUN_ENTROPY_COEF, n_steps=RUN_N_STEPS,
                             n_epochs=RUN_N_EPOCHS, gamma=RUN_GAMMA, clip_eps=RUN_CLIP_EPS,
                             normalize_rewards=RUN_NORMALIZE)
            _save_action_summary(ppo_agent, stats, RUN_EPISODES, out_dir)
            print(f"  Generating figures...")
            create_figures(out_dir, t, baseline_win_rate_pct=baseline_pct)
