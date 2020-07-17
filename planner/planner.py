import copy
from . import search
import ipdb

class LowLevelPlanner:
    def __init__(self, env):
        self.env = copy.deepcopy(env)

    def get_plan(self, current_state, agent_id, goal_pos):
        # self.env.set_state(current_state)

        def goal_fn(state_goal):
            curr_res, curr_agents = state_goal
            print("IS GOAL:", curr_agents[agent_id][0], goal_pos)
            return curr_agents[agent_id][0] == goal_pos

        def heuristic_fn(curr_state):
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
                new_state = self.env._status_after_action([agent_id], [action_name], curr_res, curr_agents)
                # print(action_name, curr_state[1][0], new_state[1][0], agent_id)
                new_state = self.env.state_to_hash(new_state)
                cost = 1

                list_expansion.append((action_name, new_state, cost))
            return list_expansion

        search_tree = search.SearchTree(current_state, goal_fn, expand_fn, heuristic_fn)
        path, cost = search_tree.A_star_graph_search()
        return path, cost


