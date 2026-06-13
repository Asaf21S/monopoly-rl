from monopoly_simulator import agent_helper_functions
from monopoly_simulator.agent import Agent
from monopoly_simulator.flag_config import flag_config_dict
import logging

logger = logging.getLogger('monopoly_simulator.logging_info.simple_agents')


def _sorted_assets(assets):
    if not assets:
        return []
    d = {a.name: a for a in assets}
    return [d[k] for k in sorted(d)]


def _ensure_memory(player):
    mem = player.agent._agent_memory
    if 'prev_action' not in mem:
        mem['prev_action'] = None


def _pre_roll_jail(player, current_gameboard, allowable_moves, code):
    """Leave jail if affordable, otherwise skip turn."""
    if player.currently_in_jail and player.current_cash >= current_gameboard['go_increment']:
        param = {'player': player.player_name, 'current_gameboard': 'current_gameboard'}
        if 'use_get_out_of_jail_card' in allowable_moves:
            return ('use_get_out_of_jail_card', param)
        if 'pay_jail_fine' in allowable_moves:
            return ('pay_jail_fine', param)
    if 'skip_turn' in allowable_moves:
        return ('skip_turn', dict())
    return ('concluded_actions', dict())


def _handle_neg_cash_simple(player, current_gameboard):
    """Mortgage cheapest eligible asset, then sell houses/hotels, then sell properties."""
    if player.current_cash >= 0:
        return (None, flag_config_dict['successful_action'])

    assets = _sorted_assets(player.assets) if player.assets else []

    # Mortgage unimproved, unmortgaged assets (cheapest first by name sort)
    for a in assets:
        if not a.is_mortgaged and not (a.loc_class == 'real_estate' and (a.num_houses > 0 or a.num_hotels > 0)):
            return ('mortgage_property', {'player': player.player_name, 'asset': a.name, 'current_gameboard': 'current_gameboard'})

    # Sell improvements
    for a in assets:
        if a.loc_class == 'real_estate' and a.num_houses > 0:
            return ('sell_house_hotel', {'player': player.player_name, 'asset': a.name,
                                         'current_gameboard': 'current_gameboard', 'sell_house': True, 'sell_hotel': False})
    for a in assets:
        if a.loc_class == 'real_estate' and a.num_hotels > 0:
            return ('sell_house_hotel', {'player': player.player_name, 'asset': a.name,
                                         'current_gameboard': 'current_gameboard', 'sell_house': False, 'sell_hotel': True})

    # Sell any asset
    for a in assets:
        return ('sell_property', {'player': player.player_name, 'asset': a.name, 'current_gameboard': 'current_gameboard'})

    return (None, flag_config_dict['successful_action'])


# ═══════════════════════════════════════════════════════════════════════════════
# AlwaysBuy — buys whenever it can afford to, improves monopolies out-of-turn
# ═══════════════════════════════════════════════════════════════════════════════

def _ab_pre_roll(player, current_gameboard, allowable_moves, code):
    return _pre_roll_jail(player, current_gameboard, allowable_moves, code)


def _ab_out_of_turn(player, current_gameboard, allowable_moves, code):
    _ensure_memory(player)
    mem = player.agent._agent_memory

    # Bail out if the previous improvement attempt failed
    if mem['prev_action'] == 'improve_property' and code == flag_config_dict['failure_code']:
        mem['prev_action'] = None
        return ('skip_turn', dict()) if 'skip_turn' in allowable_moves else ('concluded_actions', dict())

    if player.status != 'current_move' and 'improve_property' in allowable_moves:
        param = agent_helper_functions.identify_improvement_opportunity(player, current_gameboard)
        if param:
            mem['prev_action'] = 'improve_property'
            return ('improve_property', {
                'player': param['player'].player_name,
                'asset': param['asset'].name,
                'current_gameboard': 'current_gameboard',
            })

    mem['prev_action'] = 'skip_turn'
    return ('skip_turn', dict()) if 'skip_turn' in allowable_moves else ('concluded_actions', dict())


def _ab_post_roll(player, current_gameboard, allowable_moves, code):
    if 'buy_property' in allowable_moves and code != flag_config_dict['failure_code']:
        loc = current_gameboard['location_sequence'][player.current_position]
        if _ab_buy_decision(player, current_gameboard, loc):
            return ('buy_property', {'player': player.player_name, 'asset': loc.name, 'current_gameboard': 'current_gameboard'})
    return ('concluded_actions', dict())


def _ab_buy_decision(player, current_gameboard, asset):
    return player.current_cash - asset.price >= current_gameboard['go_increment']


def _ab_bid(player, current_gameboard, asset, current_bid):
    return 0


always_buy_agent_methods = {
    'handle_negative_cash_balance': _handle_neg_cash_simple,
    'make_pre_roll_move': _ab_pre_roll,
    'make_out_of_turn_move': _ab_out_of_turn,
    'make_post_roll_move': _ab_post_roll,
    'make_buy_property_decision': _ab_buy_decision,
    'make_bid': _ab_bid,
    'type': 'decision_agent_methods',
}


# ═══════════════════════════════════════════════════════════════════════════════
# NeverBuy — never purchases or improves, just pays rent and survives
# ═══════════════════════════════════════════════════════════════════════════════

def _nb_pre_roll(player, current_gameboard, allowable_moves, code):
    return _pre_roll_jail(player, current_gameboard, allowable_moves, code)


def _nb_out_of_turn(player, current_gameboard, allowable_moves, code):
    return ('skip_turn', dict()) if 'skip_turn' in allowable_moves else ('concluded_actions', dict())


def _nb_post_roll(player, current_gameboard, allowable_moves, code):
    return ('concluded_actions', dict())


def _nb_buy_decision(player, current_gameboard, asset):
    return False


def _nb_bid(player, current_gameboard, asset, current_bid):
    return 0


never_buy_agent_methods = {
    'handle_negative_cash_balance': _handle_neg_cash_simple,
    'make_pre_roll_move': _nb_pre_roll,
    'make_out_of_turn_move': _nb_out_of_turn,
    'make_post_roll_move': _nb_post_roll,
    'make_buy_property_decision': _nb_buy_decision,
    'make_bid': _nb_bid,
    'type': 'decision_agent_methods',
}


# ═══════════════════════════════════════════════════════════════════════════════
# Cautious — buys only when cash after purchase > 2× go_increment, no improvements
# ═══════════════════════════════════════════════════════════════════════════════

_CAUTIOUS_BUFFER = 2


def _ca_pre_roll(player, current_gameboard, allowable_moves, code):
    return _pre_roll_jail(player, current_gameboard, allowable_moves, code)


def _ca_out_of_turn(player, current_gameboard, allowable_moves, code):
    _ensure_memory(player)
    mem = player.agent._agent_memory

    if 'free_mortgage' in allowable_moves and player.mortgaged_assets:
        if not (mem['prev_action'] == 'free_mortgage' and code == flag_config_dict['failure_code']):
            threshold = current_gameboard['go_increment'] * _CAUTIOUS_BUFFER
            for m in sorted(player.mortgaged_assets, key=lambda x: x.mortgage):
                cost = m.mortgage * (1 + current_gameboard['bank'].mortgage_percentage)
                if player.current_cash - cost >= threshold:
                    mem['prev_action'] = 'free_mortgage'
                    return ('free_mortgage', {'player': player.player_name, 'asset': m.name, 'current_gameboard': 'current_gameboard'})

    mem['prev_action'] = 'skip_turn'
    return ('skip_turn', dict()) if 'skip_turn' in allowable_moves else ('concluded_actions', dict())


def _ca_post_roll(player, current_gameboard, allowable_moves, code):
    if 'buy_property' in allowable_moves and code != flag_config_dict['failure_code']:
        loc = current_gameboard['location_sequence'][player.current_position]
        if _ca_buy_decision(player, current_gameboard, loc):
            return ('buy_property', {'player': player.player_name, 'asset': loc.name, 'current_gameboard': 'current_gameboard'})
    return ('concluded_actions', dict())


def _ca_buy_decision(player, current_gameboard, asset):
    return player.current_cash - asset.price >= current_gameboard['go_increment'] * _CAUTIOUS_BUFFER


def _ca_bid(player, current_gameboard, asset, current_bid):
    return 0


cautious_agent_methods = {
    'handle_negative_cash_balance': _handle_neg_cash_simple,
    'make_pre_roll_move': _ca_pre_roll,
    'make_out_of_turn_move': _ca_out_of_turn,
    'make_post_roll_move': _ca_post_roll,
    'make_buy_property_decision': _ca_buy_decision,
    'make_bid': _ca_bid,
    'type': 'decision_agent_methods',
}


# ═══════════════════════════════════════════════════════════════════════════════
# Aggressive — buys whenever cash covers price, mortgages to fund monopoly
# completions, improves aggressively out-of-turn
# ═══════════════════════════════════════════════════════════════════════════════

def _ag_pre_roll(player, current_gameboard, allowable_moves, code):
    return _pre_roll_jail(player, current_gameboard, allowable_moves, code)


def _ag_out_of_turn(player, current_gameboard, allowable_moves, code):
    _ensure_memory(player)
    mem = player.agent._agent_memory

    if mem['prev_action'] == 'improve_property' and code == flag_config_dict['failure_code']:
        mem['prev_action'] = None
        return ('skip_turn', dict()) if 'skip_turn' in allowable_moves else ('concluded_actions', dict())

    if player.status != 'current_move' and 'improve_property' in allowable_moves:
        param = agent_helper_functions.identify_improvement_opportunity(player, current_gameboard)
        if param:
            mem['prev_action'] = 'improve_property'
            return ('improve_property', {
                'player': param['player'].player_name,
                'asset': param['asset'].name,
                'current_gameboard': 'current_gameboard',
            })

    mem['prev_action'] = 'skip_turn'
    return ('skip_turn', dict()) if 'skip_turn' in allowable_moves else ('concluded_actions', dict())


def _ag_post_roll(player, current_gameboard, allowable_moves, code):
    _ensure_memory(player)
    mem = player.agent._agent_memory

    if 'buy_property' in allowable_moves and code != flag_config_dict['failure_code']:
        loc = current_gameboard['location_sequence'][player.current_position]
        if _ag_buy_decision(player, current_gameboard, loc):
            mem['prev_action'] = 'buy_property'
            return ('buy_property', {'player': player.player_name, 'asset': loc.name, 'current_gameboard': 'current_gameboard'})

        # Mortgage to fund a monopoly-completing purchase
        if 'mortgage_property' in allowable_moves and mem['prev_action'] != 'mortgage_property':
            if agent_helper_functions.will_property_complete_set(player, loc, current_gameboard):
                to_mortgage = agent_helper_functions.identify_potential_mortgage(player, loc.price, True)
                if to_mortgage:
                    mem['prev_action'] = 'mortgage_property'
                    return ('mortgage_property', {'player': player.player_name, 'asset': to_mortgage.name, 'current_gameboard': 'current_gameboard'})

    return ('concluded_actions', dict())


def _ag_buy_decision(player, current_gameboard, asset):
    return player.current_cash >= asset.price


def _ag_bid(player, current_gameboard, asset, current_bid):
    return 0


aggressive_agent_methods = {
    'handle_negative_cash_balance': _handle_neg_cash_simple,
    'make_pre_roll_move': _ag_pre_roll,
    'make_out_of_turn_move': _ag_out_of_turn,
    'make_post_roll_move': _ag_post_roll,
    'make_buy_property_decision': _ag_buy_decision,
    'make_bid': _ag_bid,
    'type': 'decision_agent_methods',
}


# ═══════════════════════════════════════════════════════════════════════════════
# Registry & factory
# ═══════════════════════════════════════════════════════════════════════════════

AGENT_REGISTRY = {
    'always_buy': always_buy_agent_methods,
    'never_buy': never_buy_agent_methods,
    'cautious': cautious_agent_methods,
    'aggressive': aggressive_agent_methods,
}


def make_agent(agent_type):
    """Return an Agent instance for the given strategy name."""
    if agent_type not in AGENT_REGISTRY:
        raise ValueError(
            f"Unknown agent type '{agent_type}'. Choose from: {sorted(AGENT_REGISTRY)}"
        )
    return Agent(**AGENT_REGISTRY[agent_type])
