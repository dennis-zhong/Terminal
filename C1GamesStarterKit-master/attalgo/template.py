import numpy


def create_spawnable_tile_dict():
    spawnables = {}
    for i in range(14):
        spawnables[str(i) + "-" + str(14 + i)] = [0 for i in range(3)]
        spawnables[str(14 + i) + "-" + str(27 - i)] = [0 for i in range(3)]
    return spawnables


def create_health_tile_dict():
    health_tiles = {}
    for i in range(14):
        health_tiles[str(i) + "-" + str(13 - i)] = 0
        health_tiles[str(14 + i) + "-" + str(i)] = 0
    return health_tiles


template_core_walls = [
    [0, 13, "Wall"],
    [1, 13, "Wall"],
    [2, 13, "Wall"],
    [3, 13, "Wall"],
    [4, 12, "Wall"],
    [5, 11, "Wall"],
    [27, 13, "Wall"],
    [26, 13, "Wall"],
    [25, 13, "Wall"],
    [24, 13, "Wall"],
    [7, 12, "Wall"],
    [7, 11, "Wall"],
    [7, 10, "Wall"],
    [8, 9, "Wall"],
    [9, 8, "Wall"],
    [10, 7, "Wall"],
    [11, 7, "Wall"],
    [12, 7, "Wall"],
    [13, 7, "Wall"],
    [14, 7, "Wall"],
    [15, 7, "Wall"],
    [16, 7, "Wall"],
    [17, 7, "Wall"],
    [18, 7, "Wall"],
    [19, 7, "Wall"],
    [20, 8, "Wall"],
    [21, 9, "Wall"],
    [22, 10, "Wall"],
    [22, 11, "Wall"],
    [23, 12, "Wall"],
]

template_core_turrets = [
    [4, 11, "Turret"],
    [24, 12, "Turret"],
    [2, 12, "Turret"],
    [23, 11, "Turret"],
    [3, 12, "Turret"],
    [7, 9, "Turret"],
    [26, 12, "Turret"],
    [5, 10, "Turret"],
    [1, 12, "Turret"],
    [5, 9, "Turret"],
]

template_wall_upgrades = [
    [2, 13, "UP"],
    [3, 13, "UP"],
    [24, 13, "UP"],
    [25, 13, "UP"],
    [7, 12, "UP"],
    [7, 11, "UP"],
    [0, 13, "UP"],
    [1, 13, "UP"],
    [27, 13, "UP"],
    [26, 13, "UP"],
    [23, 12, "UP"],
    [7, 10, "UP"],
    [22, 11, "UP"],
    [8, 9, "UP"],
    [22, 10, "UP"],
    [9, 8, "UP"],
    [21, 9, "UP"],
    [20, 8, "UP"],
]

template_turret_upgrades = [
    [4, 11, "UP"],
    [24, 12, "UP"],
    [2, 12, "UP"],
    [23, 11, "UP"],
    [3, 12, "UP"],
    [7, 9, "UP"],
    [26, 12, "UP"],
    [5, 10, "UP"],
    [1, 12, "UP"],
    [5, 9, "UP"],
]

template_core_supports = [
    [9, 7, "Support"],
    [8, 8, "Support"],
    [14, 6, "Support"],
    [3, 11, "Support"],
    [2, 11, "Support"],
    [10, 6, "Support"],
]

template_support_upgrades = [
    [9, 7, "UP"],
    [8, 8, "UP"],
    [14, 6, "UP"],
    [3, 11, "UP"],
    [2, 11, "UP"],
    [10, 6, "UP"],
]

core_layout = (
    template_core_walls
    + template_core_turrets[0:3]
    + template_wall_upgrades
    + template_turret_upgrades[0:3]
)


def upgrade(unit, game_state):
    skew = defense_skew(game_state)
    if skew != -1 and abs(unit[0] - skew) > 10:
        # if there is a skew and our unit is > 10 away from it, dont upgrade
        return

    unit_type = game_state.game_map[unit[0], unit[1]][0].unit_type
    if unit_type == "FF":
        wall_upgrade(unit, game_state)
    else:
        game_state.attempt_upgrade((unit[0], unit[1]))


def wall_upgrade(unit, game_state):
    if unit[1] == 13 and len(game_state.game_map[unit[0], 14]) != 0:
        mirror = game_state.game_map[unit[0], 14]
        if (
            (mirror[0].unit_type == "FF" and mirror[1].unit_type == "UP")
            or (mirror[0].unit_type == "EF")
            or (mirror[0].unit_type == "DF")
        ):
            return  # we dont want to upgrade it is mirroring an upgraded wall, turret, or support

    game_state.attempt_upgrade((unit[0], unit[1]))


unit_map = {
    "FF": 0,  # wall
    "EF": 0,  # support
    "DF": 0,  # turret
    "UP-FF": 0,  # upgraded Wall
    "UP-EF": 0,  # upgraded Support
    "UP-DF": 0,  # upgraded Turret
    "PI": 0,  # scout
    "EI": 0,  # demolisher
    "SI": 0,  # interceptor
}


def unitMap(gameMap):
    my_units = unit_map.copy()
    en_units = unit_map.copy()

    matrix = gameMap._GameMap__map
    n = len(matrix)

    for y in range(n):
        for x in range(n):
            if gameMap.in_arena_bounds((x, y)):
                unit = matrix[x][y]
                if len(unit) == 1 and y > 13:
                    en_units[unit[0].unit_type] += 1
                elif len(unit) == 2 and y > 13:
                    ut = unit[0].unit_type
                    en_units["UP-" + ut] += 1
                elif len(unit) == 1:
                    my_units[unit[0].unit_type] += 1
                elif len(unit) == 2:
                    ut = unit[0].unit_type
                    my_units["UP-" + ut] += 1
    return my_units, en_units


def gridRatings(game_state):
    en_defense = [0] * 28
    my_defense = [0] * 28

    matrix = game_state.game_map._GameMap__map
    n = len(matrix)

    for x in range(n):
        for y in range(n):
            if game_state.game_map.in_arena_bounds((x, y)):
                unit = matrix[x][y]
                if y > 13:
                    en_defense[x] += calc_defense_score(unit)
                else:
                    my_defense[x] += calc_defense_score(unit)

    return en_defense, my_defense


def calc_defense_score(unit):
    if len(unit) == 1:
        if unit[0] == "FF":
            return 0.25
        elif unit[0] == "EF":
            return 0.2
        elif unit[0] == "DF":
            return 5
    elif len(unit) == 2:
        if unit[0] == "FF":
            return 1
        elif unit[0] == "EF":
            return 0.7
        elif unit[0] == "DF":
            return 10
    return 0


def defense_skew(game_state):
    """
    determines if there is a skew in the defense setup:
    -1 : no skew
    0 : far left skew
    7 : center left skew
    14 : center right skew
    21 : far right skew
    """

    en_defense, my_defense = gridRatings(game_state)
    med_en = numpy.median(en_defense)
    med_my = numpy.median(my_defense)
    idx_en = en_defense.index(med_en)
    idx_my = my_defense.index(med_my)

    # get enemy weight average location, and my weight average location

    if abs(idx_en - idx_my) > 7:
        return idx_en
    return -1


def build_template(game_state):
    """
    Fancy for loop to build out the template defense
    """

    for unit in core_layout:
        if game_state.get_resource(0) < 1:
            break
        elif unit[2] == "UP":
            upgrade(unit, game_state)
        elif unit[2] == "Wall":
            game_state.attempt_spawn("FF", (unit[0], unit[1]))
        elif unit[2] == "Turret":
            game_state.attempt_spawn("DF", (unit[0], unit[1]))
        else:
            game_state.attempt_spawn("EF", (unit[0], unit[1]))
