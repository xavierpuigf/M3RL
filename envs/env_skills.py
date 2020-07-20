# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import numpy as np
import random
import copy
import pickle
from queue import PriorityQueue
import ipdb
from collections import deque, namedtuple
from termcolor import colored

from utils.utils import *


TERM_COLORS = {
    0: 'red',
    1: 'green',
    2: 'cyan',
    3: 'magenta',
    4: 'yellow',
    5: 'blue'
}
ARROWS = ['^', '<', 'v', '>']
DX = [-1,  0, 1, 0]
DY = [ 0, -1, 0, 1]


class Skills_v0:
    """Base implementation, 1 skill (S2 setting)"""
    def __init__(self, 
                 nb_agent_types=4, nb_resource_types=4, nb_pay_types=2, 
                 include_type=False, include_desire=False, obstacle=False):
        """
        nb_agent_types: the number of types of agents
        nb_resource_types: the number of types of resources (or the number of goals/tasks)
        nb_pay_types: payment levels (from 1 to nb_pay_types)
        include_type: whether to show ground-truth agent type in observation
        include_desire: whether to show ground-truth preferred resource of agents 
                        in observation
        obstacle: whether to cinlude obstacle in the environment
        """
        self.action_space = ['move', 'turn left', 'turn right', 'dig', 'stop']
        self.action_size = len(self.action_space)
        if not obstacle:
            self.init_map = [
            "***********",
            "*.........*",
            "*.........*",
            "*.........*",
            "*.........*",
            "*.........*",
            "*.........*",
            "*.........*",
            "*.........*",
            "*.........*",
            "***********"
            ]
        else:
            self.init_map = [
            "***********",
            "*...*.....*",
            "*...*.....*",
            "....***...*",
            "*.....*...*",
            "*.....*...*",
            "*.........*",
            "***.......*",
            "*.........*",
            "*.........*",
            "***********"
            ]
        
        self.init_map = [list(row) for row in self.init_map]
        self.map_dim = (len(self.init_map), len(self.init_map[0]))
        
        self.nb_resource_types = nb_resource_types
        self.resource_syms = ['=', '+', '&', '#'] 
        self.nb_agent_types = nb_agent_types
        # speeds indicate skills for different tasks. 0 means "can not perform for the task"
        self.init_speeds = [[10, 0, 0, 0],
                            [0, 10, 0, 0],
                            [0, 0, 10, 0],
                            [0, 0, 0, 10]]
        self.goal_types = list(range(self.nb_resource_types))
        self.nb_goal_types = len(self.goal_types)
        self.nb_pay_types = nb_pay_types
        self.include_type = include_type
        self.include_desire = include_desire
        if include_type:
            if include_desire:
                self.obs_dim = (1 + 1 + self.nb_resource_types + self.map_dim[0] + self.map_dim[1] + 4 \
                                      + self.nb_goal_types + self.nb_resource_types,) + self.map_dim 
            else:
                self.obs_dim = (1 + 1 + self.nb_resource_types + self.map_dim[0] + self.map_dim[1] + 4 \
                                      + self.nb_goal_types,) + self.map_dim        
        else:
            self.obs_dim = (1 + 1 + self.nb_resource_types + self.map_dim[0] + self.map_dim[1] + 4,) + self.map_dim

        self.reward_weight = [min(nb_pay_types + 1, 5)] * nb_resource_types
        self.cost_weight = list(range(1, nb_pay_types + 1)) 

        random.seed(1)


    def generate_population(self, population_size=100):
        """generate a population of agents"""
        random.seed(1)
        self.full_population = [None] * population_size
        for identity in range(population_size):
            if population_size <= 4:
                agent_type = identity
            else:
                agent_type = random.randint(0, self.nb_agent_types - 1)
            desire = random.randint(0, self.nb_resource_types - 1)
            self.full_population[identity] = {'agent_type': agent_type, 'speed': copy.deepcopy(self.init_speeds[agent_type]), 'desire': desire, 'identity': identity}
        self.size_full = population_size
        ave_speeds = [0] * self.nb_goal_types
        for agent in self.full_population:
            for goal in range(self.nb_goal_types):
                ave_speeds[goal] += int(agent['speed'][goal] > 0)
        print('ability distribution:', [ave_speed / self.size_full for ave_speed in ave_speeds])


    def update_population(self, nb_agents_list, goal_type_list, inc_list):
        """update the agent population
        Args
            nb_agents_list: number of agents to be updated for each goal in goal_type_list
            goal_type_list: a list of goals to be updated
            inc_list:  whether to add or remove skills
        """
        for nb_agents, goal_type, inc in zip(nb_agents_list, goal_type_list, inc_list):
            if inc: # add a new ability to nb_agents
                indices_no_skill = [agent_id for agent_id in range(self.size_full) if self.full_population[agent_id]['speed'][goal_type] < 10]
                if len(indices_no_skill) < nb_agents:
                    indices = random.sample(range(self.size_full), nb_agents)
                else:
                    indices = random.sample(indices_no_skill, nb_agents)
                for agent_id in indices:
                    self.full_population[agent_id]['speed'][goal_type] = 10
            else: # remove an existing ability from nb_agents
                indices_skill = [agent_id for agent_id in range(self.size_full) if self.full_population[agent_id]['speed'][goal_type] > 0]
                if len(indices_skill) < nb_agents:
                    indices = random.sample(range(self.size_full), nb_agents)
                else:
                    indices = random.sample(indices_skill, nb_agents)
                for agent_id in indices:
                    self.full_population[agent_id]['speed'][goal_type] = 0
        ave_speeds = [0] * self.nb_goal_types
        for agent in self.full_population:
            for goal in range(self.nb_goal_types):
                ave_speeds[goal] += int(agent['speed'][goal] > 0)
        print('ability distribution:', [ave_speed / self.size_full for ave_speed in ave_speeds])


    def save_population(self, path):
        """save the population"""
        pickle.dump(self.full_population, open(path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


    def load_population(self, path):
        """load population"""
        self.full_population = pickle.load(open(path, 'rb'))
        self.size_full = len(self.full_population)
        ave_speeds = [0] * self.nb_goal_types
        for agent in self.full_population:
            for goal in range(self.nb_goal_types):
                ave_speeds[goal] += int(agent['speed'][goal] > 0)
        print('ability distribution:', [ave_speed / self.size_full for ave_speed in ave_speeds])


    def _is_in_bound(self, pos):
        """check if a pos is in the map boundary"""
        return pos[0] > 0 and pos[0] < self.map_dim[0] - 1 and \
               pos[1] > 0 and pos[1] < self.map_dim[1] - 1

    
    def _is_reachable(self, agent_id, pos, other_agents, ignore_resource=False):
        """check if an agent can reach pos"""
        if not self._is_in_bound(pos) or \
           self.map[pos[0]][pos[1]] == '*' or (self.resources[pos] is not None and not ignore_resource):
            return False
        for agent in other_agents:
            if agent['pos'] == pos: return False
        return True


    def _is_collectable(self, agent_id, pos, resources):
        """check if a pos has collectable items
        two requirements: exists and is not being occupied by other agents
        """
        return self._is_in_bound(pos) and resources[pos] is not None and \
               resources[pos]['collector'] in [None, agent_id]


    def get_reward(self, goal):
        """ret reward"""
        reward = 0
        agents_reached_goal = []
        for res in self.collected_res:
            if res['type'] == goal:
                reward += 1
                agents_reached_goal.append(res['collector'])
        return reward, agents_reached_goal



    
    def get_action_names(self, actions):
        """retrieve action names based on action indices"""
        return [self.action_space[action] for action in actions]


    def _status_after_action(self, agent_indices, actions, resources=None, agents=None, ignore_resource=False):
        """Tentatively taking an action and return the expected new status
        considering pos occupation after previous agents' moves to avoid conflicts
        """
        if resources is None:
            cur_resources = copy.deepcopy(self.resources)
        else:
            cur_resources = copy.deepcopy(resources)
        cur_agents = copy.deepcopy(self.agents)
        if agents is not None:
            for indi_it, indi in enumerate(agent_indices):
                cur_agents[indi] = copy.deepcopy(agents[indi_it])

        # *......X..*
        indices = list(range(len(actions)))
        shuffled_agent_indices = [agent_indices[index] for index in indices]
        shuffled_actions = [actions[index] for index in indices]
        collected_material = False
        for agent_id, action in zip(shuffled_agent_indices, shuffled_actions):
            agent = cur_agents[agent_id]
            cur_pos = agent['pos']
            cur_dir = agent['dir']
            if action != 'dig':
                cur_agents[agent_id]['digged'] = 0
            if action == 'move':
                cur_pos = (cur_pos[0] + DX[cur_dir], cur_pos[1] + DY[cur_dir])
            elif action == 'turn left':
                cur_dir = (cur_dir + 1) % 4
            elif action == 'turn right':
                cur_dir = (cur_dir + 3) % 4
            elif action == 'dig':
                item_pos = (cur_pos[0] + DX[cur_dir], cur_pos[1] + DY[cur_dir])
                if self._is_collectable(agent_id, item_pos, cur_resources):
                    cur_resources[item_pos]['collector'] = agent_id
                    u = random.random() * 10
                    effect = 10 if u < agent['speed'][cur_resources[item_pos]['type']] else 0
                    digged_amount = min(cur_resources[item_pos]['hp'], effect)
                    cur_resources[item_pos]['hp'] -= digged_amount
                    cur_agents[agent_id]['digged'] += digged_amount
            else:
                continue
            cur_agents[agent_id]['dir'] = cur_dir

            other_agents = [cur_agents[i] for i in range(agent_id)] \
                         + [self.agents[i] for i in range(agent_id + 1, self.nb_agents)]
            # print(agent_id, cur_pos, ignore_resource)
            if self._is_reachable(agent_id, cur_pos, other_agents, ignore_resource):
                if ignore_resource and cur_resources[cur_pos] is not None and cur_resources[cur_pos]['type'] < self.nb_resource_types:
                    # Crossing the palce will get you the element
                    resource_type = cur_resources[cur_pos]['type']
                    # ipdb.set_trace()
                    cur_agents[agent_id]['inventory'][resource_type] += 1
                    # cur_resources[cur_pos] = None
                    # cur_resources[cur_pos]['collector'] = agent_id
                    # cur_resources[cur_pos]['hp'] = 0
                    cur_resources[cur_pos] = None
                    # print("COLLECTED")
                elif ignore_resource and cur_resources[cur_pos] is not None:
                    for resource_type in range(self.nb_resource_types):
                        cur_agents[agent_id]['inventory'][resource_type] = 0
                    cur_resources[cur_pos] = None

                    collected_material = True
                cur_agents[agent_id]['pos'] = cur_pos
        # if collected_material:
        #     ipdb.set_trace()
        return cur_resources, cur_agents


    def send_action(self, agent_indices, actions):
        """send actions for a set of agents"""
        print("o ", self.agents)
        cur_resources, cur_agents = self._status_after_action(agent_indices, actions, ignore_resource=True)
        self.resources = copy.deepcopy(cur_resources)
        self.agents = copy.deepcopy(cur_agents)
        print("! ", self.agents)

    def setup(self, nb_agents=1, nb_resources=1, episode_id=None): 
        """set up a new game"""
        self.nb_agents = nb_agents
        self.nb_resources = nb_resources
        self.resources = dict()
        for row_id in range(self.map_dim[0]):
            for col_id in range(self.map_dim[1]):
                self.resources[(row_id, col_id)] = None
        self.remaining_resources = []#[None] * nb_resources
        self.dest = []
        self.resource_count = [0] * self.nb_resource_types
        self.agents = [None] * nb_agents
        self.map = copy.deepcopy(self.init_map)
        if episode_id is not None:
            random.seed(123 + episode_id * 1000)

        for res_id in range(nb_resources):
            res_type = random.randint(0, self.nb_resource_types - 1)
            if self.resource_count[res_type] == 0:
                while True:
                    res_pos = (random.randint(1, self.map_dim[0] - 2), random.randint(1, self.map_dim[1] - 2))
                    if self.resources[res_pos] is None: break
            else:
                res_pos = None
                indices = list(range(len(self.remaining_resources)))
                random.shuffle(indices)
                for res_id in indices:
                    prev_res_pos = self.remaining_resources[res_id]
                    if self.resources[prev_res_pos]['type'] == res_type:
                        dir_indices = list(range(4))
                        random.shuffle(dir_indices)
                        for dir_index in dir_indices:
                            dx, dy = DX[dir_index], DY[dir_index]
                            cur_pos = (prev_res_pos[0] + dx, prev_res_pos[1] + dy)
                            if self._is_in_bound(cur_pos) and self.map[cur_pos[0]][cur_pos[1]] != '*' and \
                               self.resources[cur_pos] is None:
                               res_pos = cur_pos
                               break
                    if res_pos is not None:
                        break
                if res_pos is None:
                    while True:
                        res_pos = (random.randint(1, self.map_dim[0] - 2), random.randint(1, self.map_dim[1] - 2))
                        if self.resources[res_pos] is None: break
            self.resources[res_pos] = {'type': res_type, 'sym': self.resource_syms[res_type], 
                                       'pos': res_pos, 'hp': 10, 'collector': None}
            self.remaining_resources.append(res_pos)
            self.resource_count[res_type] += 1

        self.resource_weights = [None] * nb_agents


        dest_count = [0] * self.nb_resource_types
        print(dest_count)
        # remaining_dest = []
        for dest in range(nb_resources):
            res_type = random.randint(0, self.nb_resource_types - 1)
            if dest_count[res_type] == 0:
                while True:
                    res_pos = (random.randint(1, self.map_dim[0] - 2), random.randint(1, self.map_dim[1] - 2))
                    if self.resources[res_pos] is None: break
            else:
                res_pos = None
                indices = list(range(len(self.dest)))
                random.shuffle(indices)
                for res_id in indices:
                    prev_res_pos = self.dest[res_id]
                    if self.resources[prev_res_pos]['type'] == res_type + self.nb_resource_types:
                        dir_indices = list(range(4))
                        random.shuffle(dir_indices)
                        for dir_index in dir_indices:
                            dx, dy = DX[dir_index], DY[dir_index]
                            cur_pos = (prev_res_pos[0] + dx, prev_res_pos[1] + dy)
                            if self._is_in_bound(cur_pos) and self.map[cur_pos[0]][cur_pos[1]] != '*' and \
                               self.resources[cur_pos] is None:
                               res_pos = cur_pos
                               break
                    if res_pos is not None:
                        break
                if res_pos is None:
                    while True:
                        res_pos = (random.randint(1, self.map_dim[0] - 2), random.randint(1, self.map_dim[1] - 2))
                        if self.resources[res_pos] is None: break
            self.resources[res_pos] = {'type': res_type + self.nb_resource_types, 'sym': 'O',
                                       'pos': res_pos, 'hp': 10, 'collector': None}
            self.dest.append(res_pos)
            dest_count[res_type] += 1

        for agent_id in range(nb_agents):
            while True:
                pos = (random.randint(1, self.map_dim[0] - 2), random.randint(1, self.map_dim[1] - 2))
                if self._is_reachable(agent_id, pos, self.agents[:agent_id]): break
            agent_dir = random.randint(0, 3)
            while True:
                if nb_agents == self.size_full:
                    agent = self.full_population[agent_id]
                    agent_type = agent['agent_type']
                    agent_desire = agent['desire']
                    agent_identity = agent['identity']
                    agent_speed = agent['speed']
                    break
                else:
                    agent = random.choice(self.full_population)
                    agent_type = agent['agent_type']
                    agent_desire = agent['desire']
                    agent_identity = agent['identity']
                    agent_speed = agent['speed']
                    if agent_id > 0:
                        found = False
                        for prev_agent_id in range(agent_id):
                            if self.agents[prev_agent_id]['identity'] == agent_identity:
                                found = True
                                break
                        if found: continue
                    if agent_id == 0 or self.agents[agent_id - 1]['type'] != agent_type or self.nb_agent_types < 3:
                        break
            agent_inventory = [0] * self.nb_resource_types
            self.agents[agent_id] = {'type': agent_type, 'identity': agent_identity, 'speed': agent_speed, 'desire': agent_desire,
                                     'pos': pos, 'dir': agent_dir, 'digged': 0, 'inventory': agent_inventory}
            self.resource_weights[agent_id] = [0] * self.nb_resource_types
            self.resource_weights[agent_id][agent_desire] = 1


        self.steps = 0
        self.running = True
        self.achieved = [False] * self.nb_goal_types
        # ipdb.set_trace()


    def search_path(self, time_limit, goal=0, actionable_agents=None, return_actions=True, verbose=0):

        nb_actionable_agents = len(actionable_agents)
        agent_id = actionable_agents[0]
        other_agents = [agent for other_agent_id, agent in enumerate(self.agents) \
                            if other_agent_id != agent_id]
        init_pos = self.agents[agent_id]['pos']
        init_dir = self.agents[agent_id]['dir']
        dig_pos = (init_pos[0] + DX[init_dir], init_pos[1] + DY[init_dir])
        if dig_pos in self.remaining_resources and self.resources[dig_pos]['type'] == goal and \
            (self.resources[dig_pos]['collector'] in [None, agent_id]):
            return 0, [{0: 3}]
        avail = False
        for res_pos in self.remaining_resources:
            if self.resources[res_pos]['type'] == goal and self.resources[res_pos]['collector'] is None:
                avail = True
                break
        if not avail:
            if return_actions:
                return -1, [{0: self.action_size - 1}]
            else:
                return -1

        q = deque()
        q.append((init_pos, init_dir, 0))
        pre = dict()
        pre[(init_pos, init_dir)] = None
        pre_action = dict()
        pre_action[(init_pos, init_dir)] = None
        found = False
        while q and not found:
            cur_pos, cur_dir, t = q.popleft()
            for action in range(self.action_size - 2):
                if action == 0:
                    nxt_pos = (cur_pos[0] + DX[cur_dir], cur_pos[1] + DY[cur_dir])
                    if not self._is_reachable(agent_id, nxt_pos, other_agents):
                        continue
                    nxt_dir = cur_dir
                else:
                    nxt_pos = cur_pos
                    nxt_dir = (cur_dir + (1 if action == 1 else 3)) % 4
                if (nxt_pos, nxt_dir) not in pre:
                    q.append((nxt_pos, nxt_dir, t + 1))
                    pre[(nxt_pos, nxt_dir)] = (cur_pos, cur_dir)
                    pre_action[(nxt_pos, nxt_dir)] = action
                dig_pos = (nxt_pos[0] + DX[nxt_dir], nxt_pos[1] + DY[nxt_dir])
                if self._is_collectable(agent_id, dig_pos, self.resources):
                    if self.resources[dig_pos]['type'] == goal:
                        found = True
                        break
        T = -1
        if found:
            T = 0
            cur_pos = nxt_pos
            cur_dir = nxt_dir
            actions = []
            while pre_action[(cur_pos, cur_dir)] is not None:
                T += 1
                actions.insert(0, {0: pre_action[(cur_pos, cur_dir)]})
                cur_pos, cur_dir = pre[(cur_pos, cur_dir)]
        else:
            actions = [{0: self.action_size - 1}]
        return T, actions


    def step(self):
        """update one step"""
        self.steps += 1
        self.collected_res = []
        remaining_res_indices = []
        for res_id, res_pos in enumerate(self.remaining_resources):
            res = self.resources[res_pos]
            if res is None or res['hp'] <= 0:
                self.resources[res_pos] = None
                if res is not None:
                    self.collected_res.append(copy.deepcopy(res))
                    self.resource_count[res['type']] -= 1
                else:
                    pass
                    # ipdb.set_trace()
            else:
                remaining_res_indices.append(res_id)
        self.remaining_resources = [self.remaining_resources[res_id] for res_id in remaining_res_indices]
        self.running = len(self.remaining_resources) > 0

        gt_rewards, gt_costs = dict(), dict()
        agents_reached_goal = dict()

        for goal in range(self.nb_goal_types):
            reward, agent_list = self.get_reward(goal)

            gt_rewards[goal] = reward
            agents_reached_goal[goal] = agent_list
            
        return gt_rewards, gt_costs, agents_reached_goal, not self.running


    def get_world_state(self, actionable_agents=None):
        """get the world state (non-actionable agents combined in a channel)"""
        if actionable_agents is not None:
            state = np.zeros((1 + self.nb_resource_types + 2,) + self.map_dim)
        else:
            state = np.zeros((1 + self.nb_resource_types + 1,) + self.map_dim)
        for row_id in range(self.map_dim[0]):
            for col_id in range(self.map_dim[1]):
                sym = self.map[row_id][col_id]
                if sym == '*':
                    state[1, row_id, col_id] = 1
                else:
                    res = self.resources[(row_id, col_id)]
                    if res is not None:
                        state[2 + res['type'], row_id, col_id] = 1
                    else:
                        state[0, row_id, col_id] = 1
                    
        if actionable_agents is not None:
            for agent_id, agent in enumerate(self.agents):
                if agent_id not in actionable_agents:
                    state[2 + self.nb_resource_types, agent['pos'][0], agent['pos'][1]] = 1
        return state

    def state_to_hash(self, state, count_char_resource=False):
        resources, agents = state
        resource_hash = -1 * np.ones((self.map_dim[0], self.map_dim[1]))
        for position, content in resources.items():
            if content is not None:
                resource_hash[position[0], position[1]] = content['type']
        agent_hash = tuple([(agent['pos'], agent['dir']) for agent in agents])
        if count_char_resource:

            agent_resource_hash = tuple([tuple(agent['inventory']) for agent in agents])
            print("INVENT", agent_resource_hash)
            hash_code = (resource_hash.tostring(), agent_hash, agent_resource_hash)
        else:
            hash_code = (resource_hash.tostring(), agent_hash)
        return hash_code

    def state_from_hash(self, hash):
        resources = {}
        resource = np.fromstring(hash[0]).reshape(self.map_dim[0], self.map_dim[1])
        for i in range(resource.shape[0]):
            for k in range(resource.shape[1]):

                res_pos = (i, k)
                res_type = int(resource[i][k])
                if res_type >= 0:
                    if res_type < self.nb_resource_types:
                        resources[res_pos] = {'type': res_type, 'sym': self.resource_syms[res_type],
                                              'pos': res_pos, 'hp': 10, 'collector': None}
                    else:
                        resources[res_pos] = {'type': res_type, 'sym': 'O',
                                              'pos': res_pos, 'hp': 0, 'collector': None}

                else:
                    resources[res_pos] = None
        agents = copy.deepcopy(self.agents)
        for agent_id, agent in enumerate(agents):
            agent['pos'] = hash[1][agent_id][0]
            agent['dir'] = hash[1][agent_id][1]
            if len(list(hash)) > 2:
                agent['inventory'] = list(hash[2][agent_id])

        state = resources, agents
        return state

    def set_state(self, current_state):
        resources, agents = current_state
        self.agents = agents
        self.resources = resources

    def get_state(self):
        return self.resources, self.agents

    def generate_img(self):
        """display the state"""
        cur_map = copy.deepcopy(self.map)
        colors = copy.deepcopy(self.map)
        for res_pos in self.dest:
            cur_map[res_pos[0]][res_pos[1]] = 'O'

        for agent_id, agent in enumerate(self.agents):
            cur_map[agent['pos'][0]][agent['pos'][1]] = ARROWS[agent['dir']]
            colors[agent['pos'][0]][agent['pos'][1]] = TERM_COLORS[agent_id]
        for res_pos in self.remaining_resources:
            cur_map[res_pos[0]][res_pos[1]] = self.resources[res_pos]['sym']

        res = 8
        image_out = np.zeros((h*res, w*res, 3))
        sign2col = {
            '=': [255, 0, 0],
            '+': [0, 255, 0],
            '&': [0, 0, 255],
            '#': [255, 0, 255],
            '.': [255, 255, 0],
            '*': [0, 0, 0]
        }


    def print_state(self):
        """display the state"""
        cur_map = copy.deepcopy(self.map)
        colors = copy.deepcopy(self.map)
        for res_pos in self.dest:
            cur_map[res_pos[0]][res_pos[1]] = 'O'

        for agent_id, agent in enumerate(self.agents):
            cur_map[agent['pos'][0]][agent['pos'][1]] = ARROWS[agent['dir']]
            colors[agent['pos'][0]][agent['pos'][1]] = TERM_COLORS[agent_id]
        for res_pos in self.remaining_resources:
            cur_map[res_pos[0]][res_pos[1]] = self.resources[res_pos]['sym']



        print('map:')
        for row_id in range(self.map_dim[0]):
            for col_id in range(self.map_dim[1]):
                if cur_map[row_id][col_id] not in ARROWS:
                    print(colored(cur_map[row_id][col_id], 'white'), end='')
                else:
                    print(colored(cur_map[row_id][col_id], colors[row_id][col_id]), end='')
            print('')
        print('agent_type:')
        for agent_id, agent in enumerate(self.agents):
            print(colored(' '.join([self.resource_syms[res_id] + ':' + '%02d' % speed 
                    for res_id, speed in enumerate(agent['speed'])]), TERM_COLORS[agent_id]))
            print(self.resource_weights[agent_id])

if __name__ == "__main__":
    env = Skills_v0()
    pdb.set_trace()
