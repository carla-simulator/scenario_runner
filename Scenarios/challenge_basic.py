#!/usr/bin/env python

#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Control Loss Vehicle scenario:

The scenario realizes that the vehicle looses control due to
bad road conditions, etc. and checks to see if the vehicle
regains control and corrects it's course.
"""
import importlib
import random
import os
import py_trees

from ScenarioManager.atomic_scenario_behavior import *
from ScenarioManager.atomic_scenario_criteria import *
from ScenarioManager.timer import TimeOut
from Scenarios.basic_scenario import *


CHALLENGE_BASIC_SCENARIOS = [
    "ChallengeBasic"
]


class ChallengeBasic(BasicScenario):

    """
    Implementation of "Control Loss Vehicle" (Traffic Scenario 01)
    """

    category = "ChallengeBasic"

    timeout = 100000            # Timeout of scenario in seconds

    # ego vehicle parameters
    _no_of_jitter_actions = 20
    _noise_mean = 0      # Mean value of steering noise
    _noise_std = 0.02    # Std. deviation of steerning noise
    _dynamic_mean = 0.05
    _abort_distance_to_intersection = 20
    _start_distance = 20
    _end_distance = 80

    _agent_path = "/home/grossanc/Projects/scenario_runner/Challenge/agents/MyAgent.py"

    def __init__(self, world, ego_vehicle, other_actors, town, randomize=False, debug_mode=False):
        """
        Setup all relevant parameters and create scenario
        """
        super(ChallengeBasic, self).__init__("ChallengeBasic", ego_vehicle, other_actors, town, world, debug_mode)

        self.agent = None

    def _create_behavior(self):
        """
        """

        # first we instantiate the Agent
        module_name = os.path.basename(self._agent_path).split('.')[0]
        module_spec = importlib.util.spec_from_file_location(module_name, self._agent_path)
        foo = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(foo)
        self.agent = getattr(foo, foo.__name__)()

        # Build behavior tree
        sequence = py_trees.composites.Sequence("Sequence Behavior")

        import pdb; pdb.set_trace()
        return sequence

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = []

        collision_criterion = CollisionTest(self.ego_vehicle)
        criteria.append(collision_criterion)

        return criteria
