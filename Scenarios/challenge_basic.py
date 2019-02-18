#!/usr/bin/env python

#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Basic CARLA Autonomous Driving training scenario
"""

import importlib
import random
import os
import py_trees

from ScenarioManager.atomic_scenario_behavior import *
from ScenarioManager.atomic_scenario_criteria import *
from ScenarioManager.timer import TimeOut
from Scenarios.basic_scenario import *


CHALLENGE_BASIC_SCENARIOS = ["ChallengeBasic"]


class ChallengeBasic(BasicScenario):

    """
    Implementation of a dummy scenario
    """

    category = "ChallengeBasic"
    radius = 10.0           # meters
    timeout = 1000            # Timeout of scenario in seconds

    def __init__(self, world, ego_vehicle, other_actors, town, randomize=False, debug_mode=False, config=None):
        """
        Setup all relevant parameters and create scenario
        """
        self.config = config
        self.target = self.config.target

        super(ChallengeBasic, self).__init__("ChallengeBasic", ego_vehicle, other_actors, town, world, debug_mode, True)

    def _create_behavior(self):
        """
        """
        # Build behavior tree
        sequence = py_trees.composites.Sequence("Sequence Behavior")
        idle_behavior = Idle()
        sequence.add_child(idle_behavior)

        return sequence

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = []

        collision_criterion = CollisionTest(self.ego_vehicle)
        target_criterion = ReachedRegionTest(self.ego_vehicle,
                                             min_x=self.target.transform.location.x - self.radius,
                                             max_x=self.target.transform.location.x + self.radius,
                                             min_y=self.target.transform.location.y - self.radius,
                                             max_y=self.target.transform.location.y + self.radius)

        criteria.append(collision_criterion)
        criteria.append(target_criterion)

        return criteria
