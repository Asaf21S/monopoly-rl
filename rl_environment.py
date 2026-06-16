"""
Minimal Monopoly RL Environment
Inspired by GNOME-p3 design patterns; stripped to core mechanics only.
"""

from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GO_REWARD = 200
JAIL_POSITION = 10
GO_TO_JAIL_POSITION = 30
INCOME_TAX = 200
LUXURY_TAX = 100
STARTING_CASH = 1500
BANK_HOUSE_SUPPLY = 32
BANK_HOTEL_SUPPLY = 12
HOUSE_HOTEL_THRESHOLD = 4  # 4 houses → can buy a hotel
MORTGAGE_RATE = 0.5        # mortgage value = price * 0.5
UNMORTGAGE_INTEREST = 0.1  # pay back mortgage * 1.1

# ---------------------------------------------------------------------------
# Board definition (40 squares, simplified)
# ---------------------------------------------------------------------------
# Each property: (name, color, price, rent_0, rent_1h, rent_2h, rent_3h, rent_4h, rent_hotel, house_cost)
# Non-property squares use color=None
BOARD = [
    {"pos": 0,  "type": "go",       "name": "GO"},
    {"pos": 1,  "type": "property", "name": "Mediterranean Ave",  "color": "brown",  "price": 60,  "rents": [2,10,30,90,160,250],   "house_cost": 50},
    {"pos": 2,  "type": "tax",      "name": "Community Chest"},
    {"pos": 3,  "type": "property", "name": "Baltic Ave",         "color": "brown",  "price": 60,  "rents": [4,20,60,180,320,450],   "house_cost": 50},
    {"pos": 4,  "type": "tax",      "name": "Income Tax",         "amount": INCOME_TAX},
    {"pos": 5,  "type": "railroad", "name": "Reading Railroad",   "price": 200},
    {"pos": 6,  "type": "property", "name": "Oriental Ave",       "color": "light_blue", "price": 100, "rents": [6,30,90,270,400,550], "house_cost": 50},
    {"pos": 7,  "type": "tax",      "name": "Chance"},
    {"pos": 8,  "type": "property", "name": "Vermont Ave",        "color": "light_blue", "price": 100, "rents": [6,30,90,270,400,550], "house_cost": 50},
    {"pos": 9,  "type": "property", "name": "Connecticut Ave",    "color": "light_blue", "price": 120, "rents": [8,40,100,300,450,600],"house_cost": 50},
    {"pos": 10, "type": "jail",     "name": "Jail / Just Visiting"},
    {"pos": 11, "type": "property", "name": "St. Charles Place",  "color": "pink",   "price": 140, "rents": [10,50,150,450,625,750], "house_cost": 100},
    {"pos": 12, "type": "utility",  "name": "Electric Company",   "price": 150},
    {"pos": 13, "type": "property", "name": "States Ave",         "color": "pink",   "price": 140, "rents": [10,50,150,450,625,750], "house_cost": 100},
    {"pos": 14, "type": "property", "name": "Virginia Ave",       "color": "pink",   "price": 160, "rents": [12,60,180,500,700,900], "house_cost": 100},
    {"pos": 15, "type": "railroad", "name": "Pennsylvania Railroad", "price": 200},
    {"pos": 16, "type": "property", "name": "St. James Place",    "color": "orange", "price": 180, "rents": [14,70,200,550,750,950], "house_cost": 100},
    {"pos": 17, "type": "tax",      "name": "Community Chest"},
    {"pos": 18, "type": "property", "name": "Tennessee Ave",      "color": "orange", "price": 180, "rents": [14,70,200,550,750,950], "house_cost": 100},
    {"pos": 19, "type": "property", "name": "New York Ave",       "color": "orange", "price": 200, "rents": [16,80,220,600,800,1000],"house_cost": 100},
    {"pos": 20, "type": "free_parking", "name": "Free Parking"},
    {"pos": 21, "type": "property", "name": "Kentucky Ave",       "color": "red",    "price": 220, "rents": [18,90,250,700,875,1050],"house_cost": 150},
    {"pos": 22, "type": "tax",      "name": "Chance"},
    {"pos": 23, "type": "property", "name": "Indiana Ave",        "color": "red",    "price": 220, "rents": [18,90,250,700,875,1050],"house_cost": 150},
    {"pos": 24, "type": "property", "name": "Illinois Ave",       "color": "red",    "price": 240, "rents": [20,100,300,750,925,1100],"house_cost": 150},
    {"pos": 25, "type": "railroad", "name": "B&O Railroad",       "price": 200},
    {"pos": 26, "type": "property", "name": "Atlantic Ave",       "color": "yellow", "price": 260, "rents": [22,110,330,800,975,1150],"house_cost": 150},
    {"pos": 27, "type": "property", "name": "Ventnor Ave",        "color": "yellow", "price": 260, "rents": [22,110,330,800,975,1150],"house_cost": 150},
    {"pos": 28, "type": "utility",  "name": "Water Works",        "price": 150},
    {"pos": 29, "type": "property", "name": "Marvin Gardens",     "color": "yellow", "price": 280, "rents": [24,120,360,850,1025,1200],"house_cost": 150},
    {"pos": 30, "type": "go_to_jail","name": "Go To Jail"},
    {"pos": 31, "type": "property", "name": "Pacific Ave",        "color": "green",  "price": 300, "rents": [26,130,390,900,1100,1275],"house_cost": 200},
    {"pos": 32, "type": "property", "name": "North Carolina Ave", "color": "green",  "price": 300, "rents": [26,130,390,900,1100,1275],"house_cost": 200},
    {"pos": 33, "type": "tax",      "name": "Community Chest"},
    {"pos": 34, "type": "property", "name": "Pennsylvania Ave",   "color": "green",  "price": 320, "rents": [28,150,450,1000,1200,1400],"house_cost": 200},
    {"pos": 35, "type": "railroad", "name": "Short Line Railroad","price": 200},
    {"pos": 36, "type": "tax",      "name": "Chance"},
    {"pos": 37, "type": "property", "name": "Park Place",         "color": "dark_blue","price": 350,"rents": [35,175,500,1100,1300,1500],"house_cost": 200},
    {"pos": 38, "type": "tax",      "name": "Luxury Tax",         "amount": LUXURY_TAX},
    {"pos": 39, "type": "property", "name": "Boardwalk",          "color": "dark_blue","price": 400,"rents": [50,200,600,1400,1700,2000],"house_cost": 200},
]

COLOR_SETS: dict[str, list[int]] = {}
for sq in BOARD:
    if sq["type"] == "property":
        COLOR_SETS.setdefault(sq["color"], []).append(sq["pos"])

RAILROAD_RENTS = [25, 50, 100, 200]  # rent by number of railroads owned
UTILITY_MULTIPLIERS = [4, 10]         # multiply dice roll by this


# ---------------------------------------------------------------------------
# Location (property wrapper)
# ---------------------------------------------------------------------------
class Location:
    def __init__(self, data: dict):
        self.pos        = data["pos"]
        self.type       = data["type"]
        self.name       = data["name"]
        self.color      = data.get("color")
        self.price      = data.get("price", 0)
        self.rents      = data.get("rents", [])
        self.house_cost = data.get("house_cost", 0)
        self.amount     = data.get("amount", 0)
        self.mortgage_value = int(self.price * MORTGAGE_RATE)
        self.unmortgage_cost = int(self.price * MORTGAGE_RATE * (1 + UNMORTGAGE_INTEREST))

        self.owned_by:    Optional[Player] = None
        self.is_mortgaged: bool = False
        self.num_houses:   int  = 0
        self.num_hotels:   int  = 0

    def is_purchasable(self) -> bool:
        return self.type in ("property", "railroad", "utility") and self.owned_by is None

    def current_rent(self, dice_roll: int, owner: Player) -> int:
        if self.is_mortgaged:
            return 0
        if self.type == "railroad":
            return RAILROAD_RENTS[min(owner.num_railroads - 1, 3)]
        if self.type == "utility":
            mult_idx = 0 if owner.num_utilities == 1 else 1
            return dice_roll * UTILITY_MULTIPLIERS[mult_idx]
        # property
        if self.num_hotels > 0:
            return self.rents[5]
        if self.num_houses > 0:
            return self.rents[self.num_houses]
        # unimproved: double rent if owner has full color set and no houses anywhere in set
        if self.color in owner.monopolies:
            # check no houses exist on this color group
            return self.rents[0] * 2
        return self.rents[0]

    def reset(self):
        self.owned_by     = None
        self.is_mortgaged = False
        self.num_houses   = 0
        self.num_hotels   = 0


# ---------------------------------------------------------------------------
# Bank
# ---------------------------------------------------------------------------
class Bank:
    def __init__(self):
        self.houses = BANK_HOUSE_SUPPLY
        self.hotels = BANK_HOTEL_SUPPLY
        self.cash   = 50_000  # effectively unlimited

    def reset(self):
        self.houses = BANK_HOUSE_SUPPLY
        self.hotels = BANK_HOTEL_SUPPLY
        self.cash   = 50_000


# ---------------------------------------------------------------------------
# Player
# ---------------------------------------------------------------------------
class Player:
    def __init__(self, pid: int, name: str, agent):
        self.pid    = pid
        self.name   = name
        self.agent  = agent

        # state reset on game start
        self.cash:           int  = STARTING_CASH
        self.position:       int  = 0
        self.assets:         list[Location] = []
        self.monopolies:     set[str]  = set()
        self.num_railroads:  int  = 0
        self.num_utilities:  int  = 0
        self.num_houses:     int  = 0
        self.num_hotels:     int  = 0
        self.status:         str  = "active"  # active | bankrupt | won
        self.in_jail:        bool = False
        self.jail_turns:     int  = 0
        self.turns_played:   int  = 0

    def reset(self):
        self.cash          = STARTING_CASH
        self.position      = 0
        self.assets        = []
        self.monopolies    = set()
        self.num_railroads = 0
        self.num_utilities = 0
        self.num_houses    = 0
        self.num_hotels    = 0
        self.status        = "active"
        self.in_jail       = False
        self.jail_turns    = 0
        self.turns_played  = 0

    # --- helpers -----------------------------------------------------------
    def net_worth(self) -> int:
        prop_value = sum(
            (loc.mortgage_value if loc.is_mortgaged else loc.price)
            + loc.num_houses * loc.house_cost
            + loc.num_hotels * loc.house_cost * HOUSE_HOTEL_THRESHOLD
            for loc in self.assets
        )
        return self.cash + prop_value

    def unmortgaged_assets(self) -> list[Location]:
        return [a for a in self.assets if not a.is_mortgaged and a.type in ("property","railroad","utility")]

    def mortgaged_assets(self) -> list[Location]:
        return [a for a in self.assets if a.is_mortgaged]

    def improvable_properties(self, bank: Bank) -> list[Location]:
        """Properties in a monopoly that can receive a house/hotel."""
        result = []
        for loc in self.assets:
            if loc.type != "property" or loc.is_mortgaged:
                continue
            if loc.color not in self.monopolies:
                continue
            if loc.num_hotels > 0:
                continue
            if loc.num_houses < HOUSE_HOTEL_THRESHOLD and bank.houses > 0:
                result.append(loc)
            elif loc.num_houses == HOUSE_HOTEL_THRESHOLD and bank.hotels > 0:
                result.append(loc)
        return result

    def _update_monopolies(self):
        owned_pos = {loc.pos for loc in self.assets}
        self.monopolies = {
            color for color, positions in COLOR_SETS.items()
            if all(p in owned_pos for p in positions)
        }

    def _update_counts(self):
        self.num_railroads = sum(1 for a in self.assets if a.type == "railroad")
        self.num_utilities = sum(1 for a in self.assets if a.type == "utility")
        self.num_houses    = sum(a.num_houses for a in self.assets)
        self.num_hotels    = sum(a.num_hotels for a in self.assets)

    def __repr__(self):
        return f"Player({self.name}, cash={self.cash}, pos={self.position}, props={len(self.assets)})"


# ---------------------------------------------------------------------------
# State extraction  (numeric feature vector)
# ---------------------------------------------------------------------------
def extract_state(player: Player, bank: Bank) -> dict:
    return {
        "cash":             player.cash,
        "position":         player.position,
        "num_properties":   len(player.assets),
        "num_monopolies":   len(player.monopolies),
        "num_houses":       player.num_houses,
        "num_hotels":       player.num_hotels,
        "bank_houses":      bank.houses,
        "bank_hotels":      bank.hotels,
        "net_worth":        player.net_worth(),
        "in_jail":          int(player.in_jail),
    }


def state_vector(state: dict) -> list[float]:
    """Flat numeric list suitable for an RL policy network."""
    return [
        state["cash"] / 1000.0,
        state["position"] / 39.0,
        state["num_properties"] / 28.0,
        state["num_monopolies"] / 8.0,
        state["num_houses"] / 32.0,
        state["num_hotels"] / 12.0,
        state["bank_houses"] / 32.0,
        state["bank_hotels"] / 12.0,
        state["net_worth"] / 5000.0,
        float(state["in_jail"]),
    ]


def enhanced_net_worth(player: "Player") -> float:
    """
    Net worth with monopoly bonus per Bonjour et al. (2022), Eq. 2-3:
      pa = (price - mv) * b + houses * house_cost + hotels * house_cost
    where b = 2.0 if the property completes a monopoly, else 1.5,
    and mv = unmortgage_cost if mortgaged, else 0.
    """
    nw = float(player.cash)
    for loc in player.assets:
        b  = 2.0 if (loc.color is not None and loc.color in player.monopolies) else 1.5
        mv = loc.unmortgage_cost if loc.is_mortgaged else 0
        pa = (loc.price - mv) * b + loc.num_houses * loc.house_cost + loc.num_hotels * loc.house_cost
        nw += pa
    return nw


def ratio_reward(player: "Player", all_players: list) -> float:
    """
    rx = nwx / sum(nwy for y != x, y active)  (Bonjour et al. 2022, Eq. 4, c=0).
    Returns 1.0 when all opponents are bankrupt.
    """
    my_nw  = enhanced_net_worth(player)
    opp_nw = sum(
        enhanced_net_worth(p) for p in all_players
        if p.pid != player.pid and p.status == "active"
    )
    if opp_nw <= 0:
        return 1.0
    return my_nw / opp_nw


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------
ACTION_BUY          = "buy_property"
ACTION_SKIP_BUY     = "skip_buy"
ACTION_MORTGAGE     = "mortgage_property"
ACTION_SKIP_MORT    = "skip_mortgage"
ACTION_IMPROVE      = "improve_property"
ACTION_SKIP_IMPROVE = "skip_improve"
ACTION_NONE         = "none"


def _apply_buy(player: Player, loc: Location, bank: Bank) -> bool:
    if player.cash < loc.price:
        return False
    player.cash -= loc.price
    bank.cash   += loc.price
    loc.owned_by = player
    player.assets.append(loc)
    player._update_counts()
    player._update_monopolies()
    return True


def _apply_mortgage(player: Player, loc: Location, bank: Bank) -> bool:
    if loc.is_mortgaged or loc.owned_by is not player:
        return False
    if loc.num_houses > 0 or loc.num_hotels > 0:
        return False  # must sell improvements first (simplified: just block)
    loc.is_mortgaged = True
    player.cash     += loc.mortgage_value
    bank.cash       -= loc.mortgage_value
    return True


def _apply_unmortgage(player: Player, loc: Location, bank: Bank) -> bool:
    if not loc.is_mortgaged or loc.owned_by is not player:
        return False
    if player.cash < loc.unmortgage_cost:
        return False
    player.cash -= loc.unmortgage_cost
    bank.cash   += loc.unmortgage_cost
    loc.is_mortgaged = False
    return True


def _apply_improve(player: Player, loc: Location, bank: Bank) -> bool:
    if loc.color not in player.monopolies:
        return False
    if loc.is_mortgaged:
        return False
    if loc.num_hotels > 0:
        return False
    if loc.num_houses < HOUSE_HOTEL_THRESHOLD:
        if bank.houses <= 0 or player.cash < loc.house_cost:
            return False
        loc.num_houses += 1
        bank.houses    -= 1
        player.cash    -= loc.house_cost
        bank.cash      += loc.house_cost
        player._update_counts()
        return True
    # upgrade to hotel
    if bank.hotels <= 0 or player.cash < loc.house_cost:
        return False
    bank.houses    += loc.num_houses  # return houses to bank
    loc.num_houses  = 0
    loc.num_hotels  = 1
    bank.hotels    -= 1
    player.cash    -= loc.house_cost
    bank.cash      += loc.house_cost
    player._update_counts()
    return True


# ---------------------------------------------------------------------------
# Agent interface  (base class)
# ---------------------------------------------------------------------------
class Agent:
    """
    Override the three decision methods.
    All return True to act, False to skip.
    For mortgage/improve, also return which Location to act on.
    """
    def decide_buy(self, player: Player, loc: Location, state: dict) -> bool:
        raise NotImplementedError

    def decide_mortgage(self, player: Player, candidates: list[Location], state: dict) -> Optional[Location]:
        """Return a Location to mortgage, or None to skip."""
        raise NotImplementedError

    def decide_improve(self, player: Player, candidates: list[Location], state: dict) -> Optional[Location]:
        """Return a Location to improve, or None to skip."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Baseline agents
# ---------------------------------------------------------------------------
class AggressiveBuyer(Agent):
    """Buys everything affordable; improves any eligible monopoly property."""
    def decide_buy(self, player, loc, state):
        return player.cash >= loc.price + GO_REWARD  # keep a small cushion

    def decide_mortgage(self, player, candidates, state):
        if not candidates:
            return None
        # mortgage cheapest non-monopoly first
        non_mono = [c for c in candidates if c.color not in player.monopolies and c.type=="property"]
        pool = non_mono if non_mono else candidates
        return min(pool, key=lambda c: c.mortgage_value)

    def decide_improve(self, player, candidates, state):
        if not candidates:
            return None
        # improve most expensive (highest rent potential)
        return max(candidates, key=lambda c: c.rents[-1])


class ConservativeBuyer(Agent):
    """Buys only when cash is comfortable; rarely improves."""
    CASH_BUFFER = 500

    def decide_buy(self, player, loc, state):
        return player.cash - loc.price >= self.CASH_BUFFER

    def decide_mortgage(self, player, candidates, state):
        if not candidates:
            return None
        return min(candidates, key=lambda c: c.mortgage_value)

    def decide_improve(self, player, candidates, state):
        # only improve if very cash-rich
        if player.cash < 800:
            return None
        if not candidates:
            return None
        return min(candidates, key=lambda c: c.house_cost)


class RandomAgent(Agent):
    """Makes all decisions randomly."""
    def __init__(self, buy_prob=0.5, mortgage_prob=0.4, improve_prob=0.4):
        self.buy_prob      = buy_prob
        self.mortgage_prob = mortgage_prob
        self.improve_prob  = improve_prob

    def decide_buy(self, player, loc, state):
        return random.random() < self.buy_prob

    def decide_mortgage(self, player, candidates, state):
        if not candidates or random.random() > self.mortgage_prob:
            return None
        return random.choice(candidates)

    def decide_improve(self, player, candidates, state):
        if not candidates or random.random() > self.improve_prob:
            return None
        return random.choice(candidates)


class ValueBuyer(Agent):
    """Buys when expected rent-to-price ratio is above threshold."""
    def __init__(self, rent_ratio_threshold=0.08, improve_ratio_threshold=0.15):
        self.buy_thresh     = rent_ratio_threshold
        self.improve_thresh = improve_ratio_threshold

    def decide_buy(self, player, loc, state):
        if loc.type == "railroad":
            # railroads have good value when you own multiple
            return player.cash >= loc.price + GO_REWARD
        if loc.type == "utility":
            return player.cash >= loc.price + GO_REWARD
        expected_rent = loc.rents[0]
        ratio = expected_rent / loc.price if loc.price > 0 else 0
        return ratio >= self.buy_thresh and player.cash >= loc.price + GO_REWARD

    def decide_mortgage(self, player, candidates, state):
        if not candidates:
            return None
        # mortgage lowest-value-ratio first
        def ratio(c):
            return c.rents[0] / c.price if c.type == "property" and c.price > 0 else 0.0
        return min(candidates, key=ratio)

    def decide_improve(self, player, candidates, state):
        # only improve if hotel rent ratio is compelling
        good = [
            c for c in candidates
            if c.house_cost > 0 and c.rents[-1] / (c.price + c.house_cost * 5) >= self.improve_thresh
        ]
        if not good:
            return None
        return max(good, key=lambda c: c.rents[-1])


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
class MonopolyEnv:
    """
    Minimal Monopoly RL environment.

    Usage (single step-through):
        env = MonopolyEnv(players)
        obs = env.reset()
        while not env.done:
            obs, reward, done, info = env.step()

    Usage (full game):
        env = MonopolyEnv(players)
        winner, stats = env.run_game()
    """

    MAX_TURNS = 500  # prevent infinite games

    def __init__(self, players: list[Player], summary: bool = False, seed: Optional[int] = None):
        self.players  = players
        self.summary  = summary
        self.seed     = seed
        self.bank     = Bank()
        self.board    = [Location(sq) for sq in BOARD]
        self._pos_map: dict[int, Location] = {loc.pos: loc for loc in self.board}

        self.turn_idx:     int  = 0   # index into active players
        self.total_turns:  int  = 0
        self.done:         bool = False
        self.winner:       Optional[Player] = None
        self.elimination_order: list[Player] = []

    # ------------------------------------------------------------------
    def reset(self) -> dict:
        if self.seed is not None:
            random.seed(self.seed)
        self.bank.reset()
        for loc in self.board:
            loc.reset()
        for p in self.players:
            p.reset()
        self.turn_idx         = 0
        self.total_turns      = 0
        self.done             = False
        self.winner           = None
        self.elimination_order = []
        return extract_state(self.current_player(), self.bank)

    # ------------------------------------------------------------------
    def current_player(self) -> Player:
        active = self._active_players()
        if not active:
            return self.players[0]
        return active[self.turn_idx % len(active)]

    def _active_players(self) -> list[Player]:
        return [p for p in self.players if p.status == "active"]

    # ------------------------------------------------------------------
    def step(self) -> tuple[dict, float, bool, dict]:
        """Advance one full player turn. Returns (obs, reward, done, info)."""
        if self.done:
            p = self.winner or self.players[0]
            return extract_state(p, self.bank), 0.0, True, {}

        active = self._active_players()
        if len(active) <= 1:
            self._finalize()
            p = self.winner or active[0] if active else self.players[0]
            return extract_state(p, self.bank), 0.0, True, {}

        player = active[self.turn_idx % len(active)]
        reward = self._play_turn(player)
        player.turns_played += 1
        self.total_turns    += 1

        # advance turn index over currently-active players
        active_after = self._active_players()
        if active_after:
            self.turn_idx = (self.turn_idx + 1) % len(active_after)
        else:
            self.turn_idx = 0

        if len(active_after) <= 1 or self.total_turns >= self.MAX_TURNS:
            self._finalize()

        obs  = extract_state(player, self.bank)
        info = {"player": player.name, "turn": self.total_turns}
        return obs, reward, self.done, info

    # ------------------------------------------------------------------
    def _play_turn(self, player: Player) -> float:
        # --- jail handling (simplified: pay fine after 3 turns) ---
        if player.in_jail:
            player.jail_turns += 1
            if player.jail_turns >= 3:
                fine = 50
                player.cash     -= fine
                self.bank.cash  += fine
                player.in_jail   = False
                player.jail_turns = 0
            else:
                # still in jail; try to improve/mortgage while waiting
                self._improvement_phase(player)
                self._mortgage_phase_if_negative(player)
                return ratio_reward(player, self.players)

        # --- roll dice ---
        d1  = random.randint(1, 6)
        d2  = random.randint(1, 6)
        roll = d1 + d2

        old_pos      = player.position
        new_pos      = (old_pos + roll) % 40
        passed_go    = new_pos < old_pos and old_pos != 0

        if passed_go:
            player.cash += GO_REWARD
            self.bank.cash -= GO_REWARD

        player.position = new_pos
        square = self._pos_map[new_pos]

        # --- go to jail ---
        if square.type == "go_to_jail":
            player.position  = JAIL_POSITION
            player.in_jail   = True
            player.jail_turns = 0
            return ratio_reward(player, self.players)

        # --- taxes ---
        if square.type == "tax":
            tax = square.amount
            player.cash    -= tax
            self.bank.cash += tax
            if player.cash < 0:
                self._handle_bankruptcy(player, None, abs(player.cash))

        # --- purchasable square ---
        elif square.type in ("property", "railroad", "utility"):
            if square.owned_by is None:
                # buy decision
                state = extract_state(player, self.bank)
                if player.agent.decide_buy(player, square, state):
                    _apply_buy(player, square, self.bank)
            elif square.owned_by is not player and not square.is_mortgaged:
                rent = square.current_rent(roll, square.owned_by)
                player.cash              -= rent
                square.owned_by.cash     += rent
                if player.cash < 0:
                    self._handle_bankruptcy(player, square.owned_by, rent)

        # --- post-move decisions ---
        self._improvement_phase(player)
        self._mortgage_phase_if_negative(player)

        return ratio_reward(player, self.players)

    # ------------------------------------------------------------------
    def _improvement_phase(self, player: Player):
        while True:
            candidates = player.improvable_properties(self.bank)
            state      = extract_state(player, self.bank)
            choice     = player.agent.decide_improve(player, candidates, state)
            if choice is None:
                break
            if not _apply_improve(player, choice, self.bank):
                break

    def _mortgage_phase_if_negative(self, player: Player):
        """Agent may mortgage to escape negative cash."""
        while player.cash < 0:
            candidates = [a for a in player.assets if not a.is_mortgaged
                          and a.num_houses == 0 and a.num_hotels == 0]
            if not candidates:
                break
            state  = extract_state(player, self.bank)
            choice = player.agent.decide_mortgage(player, candidates, state)
            if choice is None:
                break
            if not _apply_mortgage(player, choice, self.bank):
                break

    # ------------------------------------------------------------------
    def _handle_bankruptcy(self, debtor: Player, creditor: Optional[Player], amount: int) -> float:
        """Attempt to cover debt via mortgaging; declare bankrupt if unable."""
        self._mortgage_phase_if_negative(debtor)
        if debtor.cash >= 0:
            return 0.0  # recovered

        # truly bankrupt — liquidate assets to creditor or bank
        for loc in list(debtor.assets):
            # sell houses/hotels back to bank
            self.bank.houses += loc.num_houses
            self.bank.hotels += loc.num_hotels
            loc.num_houses    = 0
            loc.num_hotels    = 0
            if creditor:
                loc.owned_by = creditor
                creditor.assets.append(loc)
            else:
                loc.owned_by     = None
                loc.is_mortgaged = False
            debtor.assets.remove(loc)

        if creditor:
            creditor._update_counts()
            creditor._update_monopolies()

        debtor.assets     = []
        debtor.monopolies = set()
        debtor.cash       = 0
        debtor.status     = "bankrupt"
        self.elimination_order.append(debtor)
        return 0.0

    # ------------------------------------------------------------------
    def _finalize(self):
        self.done = True
        active = self._active_players()
        if active:
            winner = max(active, key=lambda p: p.net_worth())
            winner.status = "won"
            self.winner   = winner
        if self.summary:
            self._print_summary()

    # ------------------------------------------------------------------
    def run_game(self) -> tuple[Optional[Player], dict]:
        """Run until done. Returns (winner, stats_dict)."""
        self.reset()
        while not self.done:
            self.step()
        stats = self._build_stats()
        return self.winner, stats

    # ------------------------------------------------------------------
    def _build_stats(self) -> dict:
        return {
            "winner":            self.winner.name if self.winner else None,
            "total_turns":       self.total_turns,
            "elimination_order": [p.name for p in self.elimination_order],
            "final_net_worth":   {p.name: p.net_worth() for p in self.players},
            "final_cash":        {p.name: p.cash for p in self.players},
            "properties_owned":  {p.name: [a.name for a in p.assets] for p in self.players},
        }

    def _print_summary(self):
        print(f"\n{'='*50}")
        print(f"Game over after {self.total_turns} turns")
        print(f"Winner: {self.winner.name if self.winner else 'None'}")
        print(f"Elimination order: {[p.name for p in self.elimination_order]}")
        print("\nFinal standings:")
        for p in sorted(self.players, key=lambda x: x.net_worth(), reverse=True):
            props       = [a.name for a in p.assets]
            mortgaged   = [a.name for a in p.assets if a.is_mortgaged]
            print(f"  {p.name:20s} cash={p.cash:5d}  net={p.net_worth():6d}  "
                  f"props={len(props)}  mortgaged={len(mortgaged)}  status={p.status}")
        print('='*50)


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------
def make_default_players() -> list[Player]:
    return [
        Player(0, "Aggressive", AggressiveBuyer()),
        Player(1, "Conservative", ConservativeBuyer()),
        Player(2, "Random",      RandomAgent()),
        Player(3, "ValueBuyer",  ValueBuyer()),
    ]


# ---------------------------------------------------------------------------
# Quick demo / smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import collections

    NUM_GAMES = 5
    win_counts = collections.Counter()

    for seed in range(NUM_GAMES):
        players = make_default_players()
        env     = MonopolyEnv(players, summary=True, seed=seed)
        winner, stats = env.run_game()
        if winner:
            win_counts[winner.name] += 1

    print(f"Win rates over {NUM_GAMES} games:")
    for name, wins in win_counts.most_common():
        print(f"  {name:20s}  {wins:4d} wins  ({wins/NUM_GAMES*100:.1f}%)")

    # Single verbose game
    print("\n--- Verbose single game ---")
    players = make_default_players()
    env     = MonopolyEnv(players, summary=True, seed=42)
    winner, stats = env.run_game()
    print(f"\nStats dict keys: {list(stats.keys())}")
    print(f"State vector length: {len(state_vector(extract_state(players[0], env.bank)))}")
