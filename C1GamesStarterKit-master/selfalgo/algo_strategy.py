from __future__ import annotations

import gamelib
import random
import math
import warnings
from sys import maxsize
import json
import numpy as np
# import dill
import pickle
from collections import defaultdict
from functools import partial

"""
Most of the algo code you write will be in this file unless you create new
modules yourself. Start by modifying the 'on_turn' function.

Advanced strategy tips: 

  - You can analyze action frames by modifying on_action_frame function

  - The GameState.map object can be manually manipulated to create hypothetical 
  board states. Though, we recommended making a copy of the map to preserve 
  the actual current map state.
"""

train = True

class AlgoStrategy(gamelib.AlgoCore):
    def __init__(self):
        super().__init__()
        seed = random.randrange(maxsize)
        random.seed(seed)
        gamelib.debug_write('Random seed: {}'.format(seed))

    def on_game_start(self, config):
        """ 
        Read in config and perform any initial setup here 
        """
        gamelib.debug_write('Configuring your custom algo strategy...')
        self.config = config
        global WALL, SUPPORT, TURRET, SCOUT, DEMOLISHER, INTERCEPTOR, MP, SP
        WALL = config["unitInformation"][0]["shorthand"]
        SUPPORT = config["unitInformation"][1]["shorthand"]
        TURRET = config["unitInformation"][2]["shorthand"]
        SCOUT = config["unitInformation"][3]["shorthand"]
        DEMOLISHER = config["unitInformation"][4]["shorthand"]
        INTERCEPTOR = config["unitInformation"][5]["shorthand"]
        MP = 1
        SP = 0
        # This is a good place to do initial setup
        self.scored_on_locations = []

        self.prev_scored_on = 0
        self.prev_en_health = 30
        self.prev_action = None

        self.action_to_func = {
            0: self.build_wallset,
            1: self.build_turr,
            2: self.build_supp,
            3: self.deploy_demo,
            4: self.deploy_inter,
        }

        try:
            self.agent = load_object("./selfalgo/agent3.pki")
        except FileNotFoundError:
            self.agent = Agent(len(self.action_to_func))
        self.agent.update_epsilon()
        # self.agent.update_lr(0.1*20000/len(self.agent.q_values))
        self.agent.update_defd(partial(np.array, [0.225, .225, 0.1, 0.225, 0.225]))
        # try:
        #     self.agent = load_object("agent.pki")
        #     print("Success upload of agent")
        # except:
        #     self.agent = Agent(len(self.action_to_func))
        #     print("Starting new agent")

        self.UNIT_TYPE_TO_INDEX = {}
        WALL = "FF"
        self.UNIT_TYPE_TO_INDEX[WALL] = 0
        SUPPORT = "EF"
        self.UNIT_TYPE_TO_INDEX[SUPPORT] = 1
        TURRET = "DF"
        self.UNIT_TYPE_TO_INDEX[TURRET] = 2
        SCOUT = "PI"
        self.UNIT_TYPE_TO_INDEX[SCOUT] = 3
        DEMOLISHER = "EI"
        self.UNIT_TYPE_TO_INDEX[DEMOLISHER] = 4
        INTERCEPTOR = "SI"
        self.UNIT_TYPE_TO_INDEX[INTERCEPTOR] = 5
        REMOVE = "RM"
        self.UNIT_TYPE_TO_INDEX[REMOVE] = 6
        UPGRADE = "UP"
        self.UNIT_TYPE_TO_INDEX[UPGRADE] = 7

    def on_turn(self, turn_state):
        """
        This function is called every turn with the game state wrapper as
        an argument. The wrapper stores the state of the arena and has methods
        for querying its state, allocating your current resources as planned
        unit deployments, and transmitting your intended deployments to the
        game engine.
        """
        game_state = gamelib.GameState(self.config, turn_state)
        # game_state.attempt_spawn(DEMOLISHER, [24, 10], 3)
        gamelib.debug_write('Performing turn {} of your custom algo strategy'.format(game_state.turn_number))
        game_state.suppress_warnings(True)  #Comment or remove this line to enable warnings.

        self.starter_strategy(game_state)

        game_state.submit_turn()


    """
    NOTE: All the methods after this point are part of the sample starter-algo
    strategy and can safely be replaced for your custom algo.
    """

    def starter_strategy(self, game_state):
        """
        For defense we will use a spread out layout and some interceptors early on.
        We will place turrets near locations the opponent managed to score on.
        For offense we will use long range demolishers if they place stationary units near the enemy's front.
        If there are no stationary units to attack in the front, we will send Scouts to try and score quickly.
        """
        # self.agent.update(game_state)
        # for i in range(int(game_state.get_resource(0)//5)):
        #     action = self.agent.get_action()
        #     func = self.action_to_func[action]
        #     if action == 0: # wall
        #         func(game_state=game_state, loc=[random.randint(2, 24), 13])
        #     elif action == 1: # turret
        #         func(game_state=game_state, loc=[random.randint(1, 8)*3, 12])
        #     elif action == 2:
        #         func(game_state=game_state, loc=[0, 0])
        #     elif action == 3:
        #         func(game_state=game_state, loc=[[24, 10]])
        #     else:
        #         func(game_state=game_state, loc=[[24, 10]])
        obs = self.rework(game_state.game_map._GameMap__map) # TODO: need to continually update map

        # determine rewards
        rewards = {
            0: 0.5 + (len(self.scored_on_locations) - self.prev_scored_on),
            1: 0.5 + (len(self.scored_on_locations) - self.prev_scored_on),
            2: 0.1,
            3: 0.5 + self.prev_en_health - game_state.enemy_health,
            4: 0.5 + self.prev_en_health - game_state.enemy_health,
        }

        # do specific stuff with certain actions
        action_activator = {
            0: lambda act_f: act_f(game_state=game_state, loc=[random.randint(2, 24), 13]), # wall
            1: lambda act_f: act_f(game_state=game_state, loc=[random.randint(1, 8)*3, 12]), # turret
            2: lambda act_f: act_f(game_state=game_state, loc=[0, 0]),
            3: lambda act_f: act_f(game_state=game_state, loc=[[24, 10]]),
            4: lambda act_f: act_f(game_state=game_state, loc=[[24, 10]]),
        }

        for i in range(int(game_state.get_resource(0)//5)):
            if game_state.turn_number > 1:
                self.agent.update(self.prev_obs, self.prev_action, rewards[self.prev_action], False, obs) # fix updates so is in batches
            action = self.agent.get_action(obs)
            gamelib.debug_write(action)
            func = self.action_to_func[action] # determine which function to use
            action_activator[action](func)
            # func(game_state=game_state, loc=[random.randint(2, 24), 13])

            self.prev_action = action
        self.prev_obs = obs

        self.prev_scored_on = len(self.scored_on_locations)
        self.prev_en_health = game_state.enemy_health

        # TODO: need to DECAY somehow
        # # First, place basic defenses
        # self.build_defences(game_state)
        # # Now build reactive defenses based on where the enemy scored
        # self.build_reactive_defense(game_state)

        # # If the turn is less than 5, stall with interceptors and wait to see enemy's base
        # if game_state.turn_number < 5:
        #     self.stall_with_interceptors(game_state)
        # else:
        #     # Now let's analyze the enemy base to see where their defenses are concentrated.
        #     # If they have many units in the front we can build a line for our demolishers to attack them at long range.
        #     if self.detect_enemy_unit(game_state, unit_type=None, valid_x=None, valid_y=[14, 15]) > 10:
        #         self.demolisher_line_strategy(game_state)
        #     else:
        #         # They don't have many units in the front so lets figure out their least defended area and send Scouts there.

        #         # Only spawn Scouts every other turn
        #         # Sending more at once is better since attacks can only hit a single scout at a time
        #         if game_state.turn_number % 2 == 1:
        #             # To simplify we will just check sending them from back left and right
        #             scout_spawn_location_options = [[13, 0], [14, 0]]
        #             best_location = self.least_damage_spawn_location(game_state, scout_spawn_location_options)
        #             game_state.attempt_spawn(SCOUT, best_location, 1000)

        #         # Lastly, if we have spare SP, let's build some supports
        #         support_locations = [[11, 11], [12, 11], [13, 11], [14, 11]]
        #         game_state.attempt_spawn(SUPPORT, support_locations)

    def build_wallset(self, game_state, loc, size=5):
        """
        Build walls
        x, and y are center; builds horizontally
        """
        wall_locations = []
        for i in range(-size//2, size//2):
            wall_locations.append([loc[0]+i, loc[1]])

        game_state.attempt_spawn(WALL, wall_locations)
        game_state.attempt_upgrade(wall_locations)

    def build_turr(self, game_state, loc):
        game_state.attempt_spawn(TURRET, [[loc[0]-1,loc[1]], [loc[0]+1,loc[1]]])
        # game_state.attempt_upgrade(loc)

    def build_supp(self, game_state, loc):
        support_locations = [[11, 11], [12, 11], [13, 11], [14, 11]]
        game_state.attempt_spawn(SUPPORT, support_locations)

    def deploy_inter(self, game_state, loc):
        game_state.attempt_spawn(INTERCEPTOR, loc, 1000)

    def deploy_demo(self, game_state, loc):
        game_state.attempt_spawn(DEMOLISHER, loc, 1000)


    def demolisher_line_strategy(self, game_state):
        """
        Build a line of the cheapest stationary unit so our demolisher can attack from long range.
        """
        # First let's figure out the cheapest unit
        # We could just check the game rules, but this demonstrates how to use the GameUnit class
        stationary_units = [WALL, TURRET, SUPPORT]
        cheapest_unit = WALL
        for unit in stationary_units:
            unit_class = gamelib.GameUnit(unit, game_state.config)
            if unit_class.cost[game_state.MP] < gamelib.GameUnit(cheapest_unit, game_state.config).cost[game_state.MP]:
                cheapest_unit = unit

        # Now let's build out a line of stationary units. This will prevent our demolisher from running into the enemy base.
        # Instead they will stay at the perfect distance to attack the front two rows of the enemy base.
        for x in range(27, 5, -1):
            game_state.attempt_spawn(cheapest_unit, [x, 11])

        # Now spawn demolishers next to the line
        # By asking attempt_spawn to spawn 1000 units, it will essentially spawn as many as we have resources for
        game_state.attempt_spawn(DEMOLISHER, [24, 10], 1000)

    def least_damage_spawn_location(self, game_state, location_options):
        """
        This function will help us guess which location is the safest to spawn moving units from.
        It gets the path the unit will take then checks locations on that path to 
        estimate the path's damage risk.
        """
        damages = []
        # Get the damage estimate each path will take
        for location in location_options:
            path = game_state.find_path_to_edge(location)
            damage = 0
            for path_location in path:
                # Get number of enemy turrets that can attack each location and multiply by turret damage
                damage += len(game_state.get_attackers(path_location, 0)) * gamelib.GameUnit(TURRET, game_state.config).damage_i
            damages.append(damage)
        
        # Now just return the location that takes the least damage
        return location_options[damages.index(min(damages))]

    def detect_enemy_unit(self, game_state, unit_type=None, valid_x = None, valid_y = None):
        total_units = 0
        for location in game_state.game_map:
            if game_state.contains_stationary_unit(location):
                for unit in game_state.game_map[location]:
                    if unit.player_index == 1 and (unit_type is None or unit.unit_type == unit_type) and (valid_x is None or location[0] in valid_x) and (valid_y is None or location[1] in valid_y):
                        total_units += 1
        return total_units
        
    def filter_blocked_locations(self, locations, game_state):
        filtered = []
        for location in locations:
            if not game_state.contains_stationary_unit(location):
                filtered.append(location)
        return filtered

    def on_action_frame(self, turn_string):
        """
        This is the action frame of the game. This function could be called 
        hundreds of times per turn and could slow the algo down so avoid putting slow code here.
        Processing the action frames is complicated so we only suggest it if you have time and experience.
        Full doc on format of a game frame at in json-docs.html in the root of the Starterkit.
        """
        # Let's record at what position we get scored on
        state = json.loads(turn_string)
        events = state["events"]
        breaches = events["breach"]
        for breach in breaches:
            location = breach[0]
            unit_owner_self = True if breach[4] == 1 else False
            # When parsing the frame data directly, 
            # 1 is integer for yourself, 2 is opponent (StarterKit code uses 0, 1 as player_index instead)
            if not unit_owner_self:
                gamelib.debug_write("Got scored on at: {}".format(location))
                self.scored_on_locations.append(location)
                gamelib.debug_write("All locations: {}".format(self.scored_on_locations))

    def get_act_to_func(self):
        return self.action_to_func

    def rework(self, matrix):
        full_lst = []
        for x in range(len(matrix[0])):
            curr = []
            for y in range(len(matrix)):
                if not matrix[x][y]:
                    curr.append(-1)
                else:
                    curr.append(len(matrix[x][y])*10+self.UNIT_TYPE_TO_INDEX[matrix[x][y][0].unit_type])
            full_lst.append(tuple(curr))
        return tuple(full_lst)

class Agent:

    def __init__(self, act_len=1) -> None:
        self.act_len = act_len

        # self.q_values = defaultdict(lambda: np.zeros(act_len))
        self.q_values = defaultdict(partial(np.array, [0.225, .225, 0.1, 0.225, 0.225]))

        learning_rate = 0.05
        n_episodes = 1_000
        initial_epsilon = 1.0 # TODO: determine good reduction to decide when to use own strats
        epsilon_decay = initial_epsilon / (n_episodes / 2)  # reduce the exploration over time
        final_epsilon = 0.05
        discount_factor = 0.95

        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def get_action(self):
        elements = range(self.act_len)
        probabilities = [.3, .2, .05, .2, .25]
        np.random.choice(elements, 1, p=probabilities)
        return random.randint(0, self.act_len-1)

    def get_action(self, matrix) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return np.random.randint(self.act_len)

        # with probability (1 - epsilon) act greedily (exploit)
        else:
            return int(np.argmax(self.q_values[matrix]))

    def update(
        self,
        obs,
        action: int,
        reward: float,
        terminated: bool,
        next_obs,
    ):
        """Updates the Q-value of an action."""
        # print(next_obs)
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        )

        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def update_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay*len(self.q_values)/50000)
        # self.epsilon = 0.5

    def update_lr(self, val):
        self.lr = val

    def update_defd(self, part):
        self.q_values.setdefault(part)


def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

# def save_object_d(obj, filename):
#     with open(filename, 'wb') as outp:  # Overwrites any existing file.
#         dill.dump(obj, outp, dill.HIGHEST_PROTOCOL)

# save_object(agent, "agent.pkl")

def load_object(filename):
    with open(filename,"rb") as f:
        a = pickle.load(f)
    return a

# def load_object_d(filename):
#     with open(filename,"rb") as f:
#         a = dill.load(f)
#     return a

def find_maxes(lst):
    try:
        counter = [0]*len(lst[0])
        for l in lst:
            counter[np.argmax(l)]+=1
        return counter
    except:
        print(l)

if __name__ == "__main__":
    algo = AlgoStrategy()
    algo.start()

    if train:
        save_object(algo.agent, "./selfalgo/agent3.pki")

"""
a = load_object("./selfalgo/agent3.pki")
find_maxes(list(a.q_values.values())[1:])
a.epsilon
len(a.q_values)

a = load_object("./selfalgo/agent3.pki")
a.q_values = {key:val for key, val in a.q_values.items() if type(val) != type(None)}
save_object(a, "./selfalgo/agent3.pki")
"""