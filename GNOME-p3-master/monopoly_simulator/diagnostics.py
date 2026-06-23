import logging
logger = logging.getLogger('monopoly_simulator.logging_info.diagnostics')
"""
This file is imported into gameplay and primarily used for printing diagnostics. Expand as necessary for your own
use cases.
"""

def print_asset_owners(game_elements):
    """
    Print a list of all purchaseable assets, and who owns them.
    :param game_elements: A dict. Specifies global gameboard data structure
    :return: None
    """
    for k,v in game_elements['location_objects'].items():
        if v.loc_class == 'railroad' or v.loc_class == 'utility' or v.loc_class == 'real_estate':
            if v.owned_by == game_elements['bank']:
                logger.debug('Owner of '+ k+ ' is bank')
            else:
                logger.debug('Owner of '+ k+ ' is '+v.owned_by.player_name)


def print_player_cash_balances(game_elements):
    """
    Print cash balances of the players. Insert additional code as necessary (e.g., to check if cash balance is
    exceeding a certain amount for some player etc.)
    :param game_elements: A dict. Specifies global gameboard data structure
    :return: None
    """

    for p in game_elements['players']:
        logger.debug(p.player_name+ ' has cash balance '+str(p.current_cash))


def max_cash_balance(game_elements):
    """
    Return the maximum cash balance out of all players
    :param game_elements: A dict. Specifies global gameboard data structure
    :return: An integer. The maximum cash balance out of all players
    """
    max = -1
    for p in game_elements['players']:
        if max < p.current_cash:
            max = p.current_cash
    return max


def print_player_net_worths_and_cash_bal(game_elements):
    """
    Print net worth and cash bal of the players. Calculated by liquidating properties (based on then prices) and adding cash balance.
    :param game_elements: A dict. Specifies global gameboard data structure
    :return: None
    """
    for pl in game_elements['players']:
        networth_p1ayer = 0
        networth_p1ayer += pl.current_cash
        if pl.assets:
            for prop in pl.assets:
                if prop.loc_class == 'real_estate':
                    networth_p1ayer += prop.price
                    networth_p1ayer += prop.num_houses*prop.price_per_house
                    networth_p1ayer += prop.num_hotels*prop.price_per_house*(game_elements['bank'].house_limit_before_hotel + 1)
                elif prop.loc_class == 'railroad':
                    networth_p1ayer += prop.price
                elif prop.loc_class == 'utility':
                    networth_p1ayer += prop.price
        logger.debug(pl.player_name + ' has a cash balance of $' + str(pl.current_cash) + ' and a net worth of $' + str(networth_p1ayer))


def print_player_net_worths(game_elements):
    """
    Print only net worth of the players. Calculated by liquidating properties (based on then prices) and adding cash balance.
    :param game_elements: A dict. Specifies global gameboard data structure
    :return: None
    """
    for pl in game_elements['players']:
        networth_p1ayer = 0
        networth_p1ayer += pl.current_cash
        if pl.assets:
            for prop in pl.assets:
                if prop.loc_class == 'real_estate':
                    networth_p1ayer += prop.price
                    networth_p1ayer += prop.num_houses*prop.price_per_house
                    networth_p1ayer += prop.num_hotels*prop.price_per_house*(game_elements['bank'].house_limit_before_hotel + 1)
                elif prop.loc_class == 'railroad':
                    networth_p1ayer += prop.price
                elif prop.loc_class == 'utility':
                    networth_p1ayer += prop.price
        logger.debug(pl.player_name + ' has a net worth of ' + str(networth_p1ayer))


# ─── End-game summary helpers ─────────────────────────────────────────────────

def _net_worth(player, game_elements):
    nw = player.current_cash
    if player.assets:
        for prop in player.assets:
            if prop.loc_class == 'real_estate':
                nw += prop.price
                nw += prop.num_houses * prop.price_per_house
                nw += prop.num_hotels * prop.price_per_house * (game_elements['bank'].house_limit_before_hotel + 1)
            else:
                nw += prop.price
    return nw


def print_player_summary_table(game_elements, elimination_order=None):
    """Print a formatted table of each player's final state."""
    if elimination_order is None:
        elimination_order = []

    col_w = (12, 10, 9, 11, 7, 22, 11)
    header = (f"{'Player':<{col_w[0]}} {'Status':<{col_w[1]}} {'Cash':>{col_w[2]}} "
              f"{'Net Worth':>{col_w[3]}} {'Props':>{col_w[4]}} {'Monopolies':<{col_w[5]}} {'Elim Order':>{col_w[6]}}")
    sep = '=' * len(header)

    print('\n' + sep)
    print('PLAYER SUMMARY')
    print(sep)
    print(header)
    print('-' * len(header))

    for p in game_elements['players']:
        nw = _net_worth(p, game_elements)
        num_props = len(p.assets) if p.assets else 0
        monopolies = ', '.join(sorted(p.full_color_sets_possessed)) if p.full_color_sets_possessed else '-'
        elim_str = str(elimination_order.index(p.player_name) + 1) if p.player_name in elimination_order else '-'
        print(f"{p.player_name:<{col_w[0]}} {p.status:<{col_w[1]}} ${p.current_cash:>{col_w[2]-1}.0f} "
              f"${nw:>{col_w[3]-1}.0f} {num_props:>{col_w[4]}} {monopolies:<{col_w[5]}} {elim_str:>{col_w[6]}}")

    print(sep)


def print_property_ownership_table(game_elements):
    """Print a formatted table of all purchaseable properties and their status."""
    col_w = (26, 10, 12, 10, 8, 8)
    header = (f"{'Property':<{col_w[0]}} {'Color':<{col_w[1]}} {'Owner':<{col_w[2]}} "
              f"{'Mortgaged':>{col_w[3]}} {'Houses':>{col_w[4]}} {'Hotels':>{col_w[5]}}")
    sep = '=' * len(header)

    print('\n' + sep)
    print('PROPERTY OWNERSHIP')
    print(sep)
    print(header)
    print('-' * len(header))

    for name, loc in sorted(game_elements['location_objects'].items()):
        if loc.loc_class not in ('real_estate', 'railroad', 'utility'):
            continue
        owner = loc.owned_by.player_name if hasattr(loc.owned_by, 'player_name') else 'Bank'
        color = getattr(loc, 'color', None) or '-'
        mortgaged = 'Yes' if loc.is_mortgaged else 'No'
        houses = str(getattr(loc, 'num_houses', '-'))
        hotels = str(getattr(loc, 'num_hotels', '-'))
        print(f"{name:<{col_w[0]}} {color:<{col_w[1]}} {owner:<{col_w[2]}} "
              f"{mortgaged:>{col_w[3]}} {houses:>{col_w[4]}} {hotels:>{col_w[5]}}")

    print(sep)


def print_game_summary(game_elements, elimination_order=None, turn_count=0):
    """Print a complete end-of-game summary (player stats + property ownership)."""
    if elimination_order is None:
        elimination_order = []

    winner = next((p.player_name for p in game_elements['players'] if p.status == 'won'), None)
    border = '#' * 60
    print(f'\n{border}')
    print(f'  GAME OVER  |  Turns played: {turn_count}')
    if winner:
        print(f'  WINNER: {winner}')
    elim_str = ' -> '.join(elimination_order) if elimination_order else 'N/A'
    print(f'  Elimination order: {elim_str}')
    print(border)

    print_player_summary_table(game_elements, elimination_order)
    print_property_ownership_table(game_elements)
