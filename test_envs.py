# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import argparse
import torch

from agents.M3RL import M3RL
from envs import *
import ipdb
from planner import *

if __name__ == '__main__':
    # env = env_skills.Skills_v0()
    # env.generate_population()
    # env.setup()
    #
    # planner = planner.LowLevelPlanner(env)
    # finished = False
    # while not finished:
    #     env.print_state()
    #     state = env.state_to_hash(env.get_state())
    #     path, cost = planner.get_plan(state, 0, env.remaining_resources[0])
    #     # path, cost = planner.get_plan(state, 0, (1,2))
    #     # print(chr(27) + "[2J")
    #
    #     # print(path[1][0])
    #     if len(path) < 2:
    #         finished = True
    #     else:
    #         env.send_action([0], [path[1][0]])
    #         env.step()
    # env.print_state()

    env = env_skills.Skills_v0()
    env.generate_population()
    env.setup()

    planner = planner.MidLevelPlanner(env)
    finished = False
    while not finished:
        env.print_state()
        state = env.state_to_hash(env.get_state(), count_char_resource=True)

        path, cost, st = planner.get_plan(state, 0)
        # path, cost = planner.get_plan(state, 0, (1,2))
        # print(chr(27) + "[2J")

        # print(path[1][0])
        if len(path) < 2:
            finished = True
        else:
            env.send_action([0], [path[1][0][1]])
            env.step()
    env.print_state()