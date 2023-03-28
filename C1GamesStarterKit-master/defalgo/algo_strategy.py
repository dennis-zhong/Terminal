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

train = False

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
        self.prev_actions = []
        self.prev_obs = []

        self.action_to_func = dict.fromkeys(list(range(12)), self.build_wallset)
        self.action_to_func.update(dict.fromkeys(np.array(range(8))+12, self.build_turr))

        try:
            self.agent = load_object("./defalgo/agent.pki")
        except FileNotFoundError:
            self.agent = Agent(len(self.action_to_func))
        self.agent.update_epsilon()
        # self.agent.update_lr(0.1*20000/len(self.agent.q_values))
        # self.agent.update_defdic(partial(np.zeros, 20, float))
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
        # gamelib.debug_write('Performing turn {} of your custom algo strategy'.format(game_state.turn_number))
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
        obs = self.rework(game_state.game_map) # TODO: need to continually update map

        # determine rewards
        rewards = {
            0: 0.5 + (len(self.scored_on_locations) - self.prev_scored_on),
            1: 0.5 + (len(self.scored_on_locations) - self.prev_scored_on),
            2: 0.1,
            3: 0.5 + self.prev_en_health - game_state.enemy_health,
            4: 0.5 + self.prev_en_health - game_state.enemy_health,
        }

        # do specific stuff with certain actions
        action_activator = { # 12 wall possibilities (2,4,6,8,10,12,14,18,20,22,24), 8 turret possibilities (3,6,9,12,15,18,21,24)
            0: lambda act_f: act_f(game_state=game_state, loc=[2, 13]), # wall
            1: lambda act_f: act_f(game_state=game_state, loc=[4, 13]), # turret
            2: lambda act_f: act_f(game_state=game_state, loc=[6, 13]),
            3: lambda act_f: act_f(game_state=game_state, loc=[8, 13]),
            4: lambda act_f: act_f(game_state=game_state, loc=[10, 13]),
            5: lambda act_f: act_f(game_state=game_state, loc=[12, 13]),
            6: lambda act_f: act_f(game_state=game_state, loc=[14, 13]),
            7: lambda act_f: act_f(game_state=game_state, loc=[16, 13]),
            8: lambda act_f: act_f(game_state=game_state, loc=[18, 13]),
            9: lambda act_f: act_f(game_state=game_state, loc=[20, 13]),
            10: lambda act_f: act_f(game_state=game_state, loc=[22, 13]),
            11: lambda act_f: act_f(game_state=game_state, loc=[24, 13]),
            12: lambda act_f: act_f(game_state=game_state, loc=[3, 12]),
            13: lambda act_f: act_f(game_state=game_state, loc=[6, 12]),
            14: lambda act_f: act_f(game_state=game_state, loc=[9, 12]),
            15: lambda act_f: act_f(game_state=game_state, loc=[12, 12]),
            16: lambda act_f: act_f(game_state=game_state, loc=[15, 12]),
            17: lambda act_f: act_f(game_state=game_state, loc=[18, 12]),
            18: lambda act_f: act_f(game_state=game_state, loc=[21, 12]),
            19: lambda act_f: act_f(game_state=game_state, loc=[24, 12]),
        }

        if game_state.turn_number > 1: # update prev actions
            for past_obv, past_act in zip(self.prev_obs, self.prev_actions):
                # self.agent.update(self.prev_obs, past_act, rewards[past_act], False, obs) # fix updates so is in batches
                up = self.agent.update(past_obv, past_act, rewards[0], False, obs)
                pass
                # gamelib.debug_write("updated acts:", up)
            self.prev_actions = []
            self.prev_obs = []

        for i in range(int(game_state.get_resource(0)//5)): # TODO need better mechanic
            action = self.agent.get_action(obs)
            func = self.action_to_func[action] # determine which function to use
            action_activator[action](func)
            # func(game_state=game_state, loc=[random.randint(2, 24), 13])
            self.prev_actions.append(action)
            self.prev_obs.append(obs)
            obs = self.rework(game_state.game_map) # Should continually update
        # gamelib.debug_write("prev acts:", self.prev_actions)

        self.prev_scored_on = len(self.scored_on_locations)
        self.prev_en_health = game_state.enemy_health


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
        # return game_state.attempt_spawn(TURRET, [[loc[0]-1,loc[1]], [loc[0]+1,loc[1]]])
        succ = game_state.attempt_spawn(TURRET, loc)
        game_state.attempt_upgrade(loc)
        return succ

    def build_supp(self, game_state, loc):
        support_locations = [[11, 11], [12, 11], [13, 11], [14, 11]]
        return game_state.attempt_spawn(SUPPORT, support_locations)

    def deploy_inter(self, game_state, loc):
        return game_state.attempt_spawn(INTERCEPTOR, loc, 1000)

    def deploy_demo(self, game_state, loc):
        return game_state.attempt_spawn(DEMOLISHER, loc, 1000)

    def build_turr_wall(self, game_state, loc):
        if game_state.get_resource(0) < game_state.type_cost(WALL)*3+game_state.type_cost(TURRET):
            return 0
        wall_succ = game_state.attempt_spawn(WALL, [[loc[0]-1, loc[1]], [loc[0], loc[1]], [loc[0]+1, loc[1]]])
        turr_succ = game_state.attempt_spawn(TURRET, [loc[0], loc[1]-1])
        return wall_succ+turr_succ == 4

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
                # gamelib.debug_write("Got scored on at: {}".format(location))
                self.scored_on_locations.append(location)
                # gamelib.debug_write("All locations: {}".format(self.scored_on_locations))

    def get_act_to_func(self):
        return self.action_to_func

    def rework(self, gameMap):
        matrix = gameMap._GameMap__map
        full_lst = []
        for y in range(len(matrix)):
            for x in range(len(matrix[0])):
                if gameMap.in_arena_bounds((x, y)):
                    if not matrix[x][y]:
                        full_lst.append(-1)
                    else:
                        full_lst.append(len(matrix[x][y])*10+self.UNIT_TYPE_TO_INDEX[matrix[x][y][0].unit_type])
        return tuple(full_lst)

class Agent:

    def __init__(self, act_len=1) -> None:
        self.act_len = act_len

        # self.q_values = defaultdict(lambda: np.zeros(act_len))
        self.q_values = defaultdict(partial(np.zeros, 20, float))

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
            return int(self.get_rand_max(self.q_values[matrix]))

    def get_rand_max(self, vals):
        highest = max(vals)
        all_maxes = []
        for i in range(len(vals)):
            if vals[i] == highest:
                all_maxes.append(i)
        return random.choice(all_maxes)

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

        self.q_values[obs][action] += float(self.lr * temporal_difference)
        # self.q_values[obs][action] = 0.05
        # val = self.q_values[obs][action]
        self.training_error.append(temporal_difference)
        return (action, self.q_values[obs][action],  self.lr*temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def update_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay*len(self.q_values)/50000) # TODO: always check constant is right
        # self.epsilon = 0.75

    def update_lr(self, val):
        self.lr = val

    def update_defdic(self, part):
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
        counter = [0]*len(lst[1])
        for l in lst:
            counter[np.argmax(l)]+=1
        return counter
    except:
        print(l)

if __name__ == "__main__":
    algo = AlgoStrategy()
    algo.start()

    if train:
        save_object(algo.agent, "./defalgo/agent.pki")

"""
a = load_object("./defalgo/agent.pki")
find_maxes(list(a.q_values.values())[1:])
a.epsilon
len(a.q_values)

a = load_object("./defalgo/agent3.pki")
a.q_values = {key:val for key, val in a.q_values.items() if type(val) != type(None)}
save_object(a, "./defalgo/agent3.pki")
"""