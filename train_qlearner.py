"""
Q-learning baseline for the minimal Monopoly RL environment.
Three separate Q-tables learn the buy/skip, mortgage/skip, and improve/skip
decisions.  Which property to act on (when the agent decides to act) is
handled by a heuristic, keeping the action space binary for each decision.
"""

from __future__ import annotations
import argparse
import csv
import random
from collections import defaultdict
from typing import Optional

from rl_environment import (
    MonopolyEnv,
    make_default_players,
    Player,
    extract_state,
    Location,
    Agent,
)

# ---------------------------------------------------------------------------
# Top-level config  (overridable via argparse)
# ---------------------------------------------------------------------------
REPLACE_AGENT = "Random"   # "Aggressive" | "Conservative" | "Random" | "ValueBuyer" | "All"
EPISODES      = 1500

ALL_AGENT_NAMES = ["Aggressive", "Conservative", "Random", "ValueBuyer"]

# Q-learning hyperparameters
ALPHA     = 0.1
GAMMA     = 0.99
EPS_START = 0.2
EPS_END   = 0.02

# Frozen improve-threshold (best from sweep)
IMPROVE_THRESHOLD = 0.8

# Default bucket count for state discretization
NUM_BUCKETS = 4

# Shared action indices (all three decisions are binary)
A_SKIP = 0
A_ACT  = 1


# ---------------------------------------------------------------------------
# State discretizer
# ---------------------------------------------------------------------------
# Feature ranges used for uniform bucketing
_CASH_MAX  = 3000   # typical upper bound for cash during a game
_POS_MAX   = 40     # board squares (0-39)
_PROPS_MAX = 29     # max buyable properties + 1 to avoid out-of-range
_WORTH_MAX = 6000   # typical upper bound for net worth

def _bucket(value: float, max_val: float, n: int) -> int:
    """Map value in [0, max_val] to a bucket index in [0, n-1]."""
    return min(max(int(value * n / max_val), 0), n - 1)

def discretize(state: dict, n_buckets: int) -> tuple:
    """
    Compress a state dict to a hashable key using uniform linear bucketing.
    Maximum possible keys: n_buckets^4 * 2  (the *2 is for in_jail).
    """
    return (
        _bucket(max(state["cash"], 0),            _CASH_MAX,  n_buckets),
        _bucket(state["position"],                 _POS_MAX,   n_buckets),
        _bucket(state["num_properties"],           _PROPS_MAX, n_buckets),
        _bucket(max(state["net_worth"], 0),        _WORTH_MAX, n_buckets),
        state["in_jail"],
    )


# ---------------------------------------------------------------------------
# Q-learner agent
# ---------------------------------------------------------------------------
class QLearnerAgent(Agent):
    """
    Tabular epsilon-greedy Q-learner with three independent Q-tables:
      q_buy     -- buy vs. skip when landing on an unowned property
      q_mort    -- mortgage vs. skip when cash is negative
      q_improve -- improve vs. skip when owning an eligible monopoly property

    For each decision the agent learns WHETHER to act; which specific property
    to act on is determined by a heuristic so the action space stays binary.

    After each decision the (state_key, action) pair is appended to a per-pid
    pending list so the training loop can retrieve all decisions made in a
    single env.step() call and apply Q-updates uniformly.
    """

    def __init__(self, improve_threshold: float = IMPROVE_THRESHOLD,
                 n_buckets: int = NUM_BUCKETS,
                 alpha: float = ALPHA):
        self.improve_threshold = improve_threshold
        self.n_buckets         = n_buckets  # controls Q-table granularity; max keys = n^4 * 2
        self.alpha             = alpha

        self.q_buy:     defaultdict = defaultdict(lambda: [0.0, 0.0])
        self.q_mort:    defaultdict = defaultdict(lambda: [0.0, 0.0])
        self.q_improve: defaultdict = defaultdict(lambda: [0.0, 0.0])

        self.eps: float = EPS_START

        # Pending (state_key, action) lists per pid, one list per decision type
        self._pending_buy:     dict[int, list] = {}
        self._pending_mort:    dict[int, list] = {}
        self._pending_improve: dict[int, list] = {}

    # -- internal helper ----------------------------------------------------
    def _choose(self, q: list) -> int:
        """Epsilon-greedy action selection over a two-element Q list."""
        if random.random() < self.eps:
            return random.randint(0, 1)
        return A_ACT if q[A_ACT] > q[A_SKIP] else A_SKIP

    # -- learned decisions --------------------------------------------------
    def decide_buy(self, player: Player, loc: Location, state: dict) -> bool:
        key    = discretize(state, self.n_buckets)
        action = self._choose(self.q_buy[key])
        self._pending_buy.setdefault(player.pid, []).append((key, action))
        return action == A_ACT

    def decide_mortgage(self, player: Player, candidates: list, state: dict) -> Optional[Location]:
        if not candidates:
            return None

        # Heuristic: pick the cheapest property whose mortgage proceeds alone
        # cover the deficit; if none can, pick the most valuable to make the
        # most progress and let the loop call again.
        deficit  = max(0, -player.cash)
        covering = [c for c in candidates if c.mortgage_value >= deficit]
        target   = (
            min(covering,    key=lambda c: c.mortgage_value) if covering
            else max(candidates, key=lambda c: c.mortgage_value)
        )

        key    = discretize(state, self.n_buckets)
        action = self._choose(self.q_mort[key])
        self._pending_mort.setdefault(player.pid, []).append((key, action))
        return target if action == A_ACT else None

    def decide_improve(self, player: Player, candidates: list, state: dict) -> Optional[Location]:
        if not candidates:
            return None

        # Only consider properties whose house cost is within the configured fraction of cash.
        affordable = [c for c in candidates if c.house_cost <= self.improve_threshold * player.cash]
        if not affordable:
            return None  # nothing affordable; skip without recording a Q decision

        # Heuristic: among affordable options, pick the highest rent-to-cost ratio.
        target = max(affordable, key=lambda c: c.rents[-1] / c.house_cost)

        key    = discretize(state, self.n_buckets)
        action = self._choose(self.q_improve[key])
        self._pending_improve.setdefault(player.pid, []).append((key, action))
        return target if action == A_ACT else None

    # -- training helpers ---------------------------------------------------
    def pop_pending(self, pid: int) -> tuple[list, list, list]:
        """
        Return and clear all pending (state_key, action) tuples for player pid,
        grouped by decision type: (buys, mortgages, improves).
        """
        return (
            self._pending_buy.pop(pid, []),
            self._pending_mort.pop(pid, []),
            self._pending_improve.pop(pid, []),
        )

    def _q_update(self, q_table: defaultdict, state_key: tuple, action: int,
                  reward: float, next_state: dict):
        """
        One-step Q update:
          Q[s,a] += alpha * (reward + gamma * max_a'(Q[s',a']) - Q[s,a])
        """
        next_key   = discretize(next_state, self.n_buckets)
        q          = q_table[state_key]
        td_target  = reward + GAMMA * max(q_table[next_key])
        q[action] += self.alpha * (td_target - q[action])

    def update_all(self, buys: list, morts: list, improves: list,
                   reward: float, next_state: dict):
        """Apply Q updates for every decision recorded during a single step."""
        for s_key, action in buys:
            self._q_update(self.q_buy, s_key, action, reward, next_state)
        for s_key, action in morts:
            self._q_update(self.q_mort, s_key, action, reward, next_state)
        for s_key, action in improves:
            self._q_update(self.q_improve, s_key, action, reward, next_state)

    def total_q_states(self) -> int:
        return len(self.q_buy) + len(self.q_mort) + len(self.q_improve)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_for(agent_name: str, episodes: int = EPISODES,
              improve_threshold: float = IMPROVE_THRESHOLD,
              n_buckets: int = NUM_BUCKETS,
              alpha: float = ALPHA) -> tuple[dict, QLearnerAgent]:
    """
    Run `episodes` games, replacing the primitive agent named `agent_name`
    with a QLearnerAgent.  Returns (win_counts, trained_agent).
    """
    players  = make_default_players()
    ql_agent = QLearnerAgent(improve_threshold=improve_threshold,
                             n_buckets=n_buckets, alpha=alpha)
    ql_pid   = -1

    win_counts: dict[str, int] = {p.name: 0 for p in players}

    for p in players:
        if p.name == agent_name:
            p.agent = ql_agent
            ql_pid  = p.pid

    env = MonopolyEnv(players, summary=False, seed=None)

    for ep in range(episodes):
        # linear epsilon decay from EPS_START down to EPS_END
        ql_agent.eps = EPS_START - (EPS_START - EPS_END) * ep / max(episodes - 1, 1)

        env.seed = ep   # deterministic seed per episode for reproducibility
        env.reset()

        while not env.done:
            current_p = env.current_player()
            is_ql     = current_p.pid == ql_pid

            obs, reward, done, info = env.step()

            # Q update: retrieve every decision the Q-learner made this turn
            if is_ql:
                buys, morts, improves = ql_agent.pop_pending(ql_pid)
                ql_agent.update_all(buys, morts, improves, reward, obs)

        if env.winner:
            win_counts[env.winner.name] += 1

        if (ep + 1) % 100 == 0:
            buy_s, mort_s, imp_s = (
                len(ql_agent.q_buy),
                len(ql_agent.q_mort),
                len(ql_agent.q_improve),
            )
            print(f"  [{agent_name:12s}] ep {ep+1:4d}/{episodes}  "
                  f"eps={ql_agent.eps:.3f}  "
                  f"Q-states buy={buy_s} mort={mort_s} imp={imp_s}")

    return win_counts, ql_agent


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _print_results(agent_name: str, episodes: int, win_counts: dict,
                   ql: QLearnerAgent):
    b, m, i = len(ql.q_buy), len(ql.q_mort), len(ql.q_improve)
    max_keys = ql.n_buckets ** 4 * 2
    print(f"\n{'='*58}")
    print(f"Replaced agent : {agent_name}  (Q-learner)")
    print(f"Episodes       : {episodes}")
    print(f"Buckets        : {ql.n_buckets}  (max keys={max_keys}  threshold={ql.improve_threshold})")
    print(f"Q-table sizes  : buy={b}  mortgage={m}  improve={i}  total={b+m+i}")
    print(f"{'Player':<20} {'Wins':>6}  {'Win %':>7}")
    print(f"{'-'*37}")
    for name, wins in sorted(win_counts.items(), key=lambda x: -x[1]):
        pct    = wins / episodes * 100
        marker = "  << Q-learner" if name == agent_name else ""
        print(f"  {name:<18} {wins:6d}  {pct:6.1f}%{marker}")
    print('='*58)


def _save_csv(agent_name: str, episodes: int, win_counts: dict):
    filename = f"results_{agent_name}.csv"
    with open(filename, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["player", "wins", "win_rate_pct", "episodes", "is_qlearner"])
        for name, wins in win_counts.items():
            writer.writerow([
                name,
                wins,
                f"{wins / episodes * 100:.2f}",
                episodes,
                name == agent_name,
            ])
    print(f"  Saved: {filename}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
THRESHOLD_SWEEP = [0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0]
BUCKET_SWEEP    = [2, 3, 4, 5, 6]

# Grid search: 3 × 5 × 2 = 30 runs
GRID_THRESHOLDS = [0.4, 0.8, 1.0]
GRID_BUCKETS    = [2, 3, 4, 5, 6]
GRID_ALPHAS     = [0.05, 0.2]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train tabular Q-learner in Monopoly env")
    parser.add_argument(
        "--replace", default=REPLACE_AGENT,
        choices=ALL_AGENT_NAMES + ["All"],
        help="Which primitive agent slot to replace with the Q-learner",
    )
    parser.add_argument(
        "--episodes", type=int, default=EPISODES,
        help="Training episodes per agent slot",
    )
    parser.add_argument(
        "--sweep", action="store_true",
        help="Sweep improve_threshold over predefined values (threshold frozen at 0.8 for other runs)",
    )
    parser.add_argument(
        "--sweep-buckets", action="store_true",
        help="Sweep n_buckets over predefined values (threshold frozen at IMPROVE_THRESHOLD=0.8)",
    )
    parser.add_argument(
        "--grid-search", action="store_true",
        help="Grid search over improve_threshold x n_buckets x alpha (30 runs)",
    )
    args = parser.parse_args()

    def _sweep_summary(label: str, param_name: str, rows: list, target: str, episodes: int):
        print(f"\n{'='*50}")
        print(f"{param_name} sweep summary  (replaced: {target}, episodes: {episodes})")
        print(f"  {label:>10}  {'Wins':>6}  {'Win %':>7}  {'Rank':>5}  {'Max keys':>9}")
        print(f"  {'-'*44}")
        for val, wins, pct, rank, max_keys in rows:
            print(f"  {val:>10}  {wins:6d}  {pct:6.1f}%  {rank:>4}/4  {max_keys:>9}")
        print('='*50)

    target = args.replace if args.replace != "All" else REPLACE_AGENT

    if args.sweep:
        print(f"\nThreshold sweep  (replaced: '{target}', "
              f"n_buckets={NUM_BUCKETS}, episodes={args.episodes})\n")
        rows: list[tuple] = []
        for thresh in THRESHOLD_SWEEP:
            print(f"  threshold={thresh:.1f} ...")
            wc, ql = train_for(target, args.episodes, improve_threshold=thresh)
            rank    = sorted(wc, key=lambda n: -wc[n]).index(target) + 1
            max_k   = ql.n_buckets ** 4 * 2
            rows.append((thresh, wc[target], wc[target] / args.episodes * 100, rank, max_k))
        _sweep_summary("Threshold", "improve_threshold", rows, target, args.episodes)

    elif args.grid_search:
        total_runs = len(GRID_THRESHOLDS) * len(GRID_BUCKETS) * len(GRID_ALPHAS)
        print(f"\nGrid search  (replaced: '{target}', episodes={args.episodes}, "
              f"runs={total_runs})\n"
              f"  improve_threshold: {GRID_THRESHOLDS}\n"
              f"  n_buckets:         {GRID_BUCKETS}\n"
              f"  alpha:             {GRID_ALPHAS}\n")

        grid_rows: list[tuple] = []
        run = 0
        for thresh in GRID_THRESHOLDS:
            for nb in GRID_BUCKETS:
                for al in GRID_ALPHAS:
                    run += 1
                    print(f"  [{run:2d}/{total_runs}] thresh={thresh}  buckets={nb}  alpha={al} ...")
                    wc, ql = train_for(target, args.episodes,
                                       improve_threshold=thresh,
                                       n_buckets=nb, alpha=al)
                    rank  = sorted(wc, key=lambda n: -wc[n]).index(target) + 1
                    pct   = wc[target] / args.episodes * 100
                    grid_rows.append((thresh, nb, al, wc[target], pct, rank))

        # sort by win% descending
        grid_rows.sort(key=lambda r: -r[4])

        print(f"\n{'='*66}")
        print(f"Grid search results  (replaced: {target}, episodes: {args.episodes})")
        print(f"  {'thresh':>7}  {'buckets':>7}  {'alpha':>6}  {'Wins':>6}  {'Win %':>7}  {'Rank':>5}")
        print(f"  {'-'*57}")
        for thresh, nb, al, wins, pct, rank in grid_rows:
            print(f"  {thresh:>7.1f}  {nb:>7d}  {al:>6.2f}  {wins:6d}  {pct:6.1f}%  {rank:>4}/4")
        print('='*66)

        # save CSV
        fname = f"results_grid_{target}.csv"
        with open(fname, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["improve_threshold", "n_buckets", "alpha",
                        "wins", "win_rate_pct", "rank", "episodes"])
            for thresh, nb, al, wins, pct, rank in grid_rows:
                w.writerow([thresh, nb, al, wins, f"{pct:.2f}", rank, args.episodes])
        print(f"  Saved: {fname}")

    elif args.sweep_buckets:
        print(f"\nBucket sweep  (replaced: '{target}', "
              f"threshold={IMPROVE_THRESHOLD}, episodes={args.episodes})\n")
        rows = []
        for nb in BUCKET_SWEEP:
            print(f"  n_buckets={nb} ...")
            wc, ql = train_for(target, args.episodes, n_buckets=nb)
            rank   = sorted(wc, key=lambda n: -wc[n]).index(target) + 1
            max_k  = nb ** 4 * 2
            rows.append((nb, wc[target], wc[target] / args.episodes * 100, rank, max_k))
        _sweep_summary("n_buckets", "n_buckets", rows, target, args.episodes)

    else:
        targets = ALL_AGENT_NAMES if args.replace == "All" else [args.replace]
        for t in targets:
            print(f"\nTraining Q-learner replacing '{t}' for {args.episodes} episodes...")
            win_counts, ql = train_for(t, args.episodes)
            _print_results(t, args.episodes, win_counts, ql)
            _save_csv(t, args.episodes, win_counts)
