import copy
from . import search
import ipdb
import numpy as np

class LowLevelPlanner:
    def __init__(self, env):
        self.env = copy.deepcopy(env)
        self.cache = {}

    def _get_walls_only(self, state):
        # state is unused
        return ','.join([''.join(r) for r in self.env.map])


    def query_cache(self, state, pos_dir_init, pos_dir_end):
        return None
        new_state = self._get_walls_only(state)
        hash = (new_state, pos_dir_init, pos_dir_end)
        if hash in self.cache:
            # ipdb.set_trace()
            return self.cache[hash]
        else:
            return None

    def save_cache(self, state, pos_init, pos_end, plan, cost):
        return None
        new_state = self._get_walls_only(state)
        hash = (new_state, pos_init, pos_end)
        # ipdb.set_trace()
        self.cache[hash] = (plan, cost, state)

    def get_plan(self, current_state, agent_id, goal_pos):
        # self.env.set_state(current_state)
        curr_res, curr_agents = current_state
        pos_dir_init = curr_agents[agent_id]
        pos_init = pos_dir_init[0]
        pos_end = goal_pos

        cache_val = self.query_cache(current_state, pos_dir_init, pos_end)
        if cache_val is not None:
            return cache_val


        def goal_fn(state_goal):
            curr_res, curr_agents = state_goal
            # print("IS GOAL:", curr_agents[agent_id][0], goal_pos)
            return curr_agents[agent_id][0] == goal_pos

        def heuristic_fn(curr_state):
            # ipdb.set_trace()
            curr_res, curr_agents = curr_state
            # ipdb.set_trace()
            curr_pos = curr_agents[agent_id][0]
            return abs(goal_pos[0] - curr_pos[0]) + abs(goal_pos[1] - curr_pos[1])

        def expand_fn(curr_state):
            # return list of action, successor, cost
            # print(curr_state[1])
            curr_state = self.env.state_from_hash(curr_state)
            curr_res, curr_agents = curr_state
            action_names = ['move', 'turn left', 'turn right'] # self.env.action_space
            list_expansion = []
            for action_name in action_names:
                # ipdb.set_trace()
                new_state = self.env._status_after_action([agent_id], [action_name], curr_res, curr_agents, ignore_resource=True)
                # print(action_name, curr_state[1][0], new_state[1][0], agent_id)s
                new_state = self.env.state_to_hash(new_state)
                cost = 1

                list_expansion.append((action_name, new_state, cost))
            return list_expansion

        search_tree = search.SearchTree(current_state, goal_fn, expand_fn, heuristic_fn)
        path, cost, state = search_tree.A_star_graph_search()
        path_actions = [p[0] for p in path]
        self.save_cache(current_state, pos_dir_init, pos_end, path_actions, cost)
        return path_actions, cost, state


class MidLevelPlanner:
    def __init__(self, env):
        self.env = copy.deepcopy(env)
        self.ll_planner = LowLevelPlanner(env)


    def get_actions(self, current_state):

        current_state = self.env.state_from_hash(current_state)
        curr_agent = current_state[1][0]
        # print("Expanding", curr_agent)
        if sum(curr_agent['inventory']) == 0:
            # Resource collection
            positions = [x for x, y in current_state[0].items() if y is not None and y['type'] < self.env.nb_resource_types]
            # print("COLLECT")
        else:
            # print("PLACE")
            # Resource placing
            # print(current_state[0])
            positions = [x for x, y in current_state[0].items() if y is not None and y['type'] >= self.env.nb_resource_types]

        list_paths = []
        agent_id = 0
        for position in positions:
            current_state_hash = self.env.state_to_hash(current_state)
            path_actions, cost, _ = self.ll_planner.get_plan(current_state_hash, 0, position)
            curr_res, curr_agents = current_state
            # ipdb.set_trace()
            for i in range(1, len(path_actions)):
                action_name = path_actions[i]
                # ipdb.set_trace()
                final_state = self.env._status_after_action([agent_id], [action_name], curr_res, curr_agents, ignore_resource=True)
                # print(action_name, final_state[1][0]['pos'])
                curr_res, curr_agents = final_state
            # print(final_state[1], final_state[0])
            list_paths.append((path_actions, self.env.state_to_hash(final_state, count_char_resource=True), cost))
        # ipdb.set_trace()
        if len(positions) == 0:
            pass
            # ipdb.set_trace()
        return list_paths

    def get_plan(self, current_state, agent_id):
        def goal_fn(state_goal):
            # ipdb.set_trace()
            curr_state = self.env.state_from_hash(state_goal)
            return len([x for x in curr_state[0].values() if x is not None]) == 0

        def heuristic_fn(state_goal):
            # print(state_goal)
            curr_state = self.env.state_from_hash(state_goal)
            return len([x for x in curr_state[0].values() if x is None])

        # print(current_state)
        search_tree = search.SearchTree(current_state, goal_fn, self.get_actions, heuristic_fn)
        path, cost, state = search_tree.A_star_graph_search()
        # ipdb.set_trace()
        return path, cost, state
