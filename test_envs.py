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
    env = env_skills.Skills_v0()
    env.generate_population()
    env.setup()
    while True:
        env.print_state()
        planner = planner.LowLevelPlanner(env)
        state = env.state_to_hash(env.get_state())
        # planner.get_plan(state, 0, env.remaining_resources[0])
        path, cost = planner.get_plan(state, 0, (1,2))
        print(path[1][0])
        ipdb.set_trace()