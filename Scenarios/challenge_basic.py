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

    timeout = 2 * 60            # Timeout of scenario in seconds
    _end_distance = 800

    def __init__(self, world, ego_vehicle, other_actors, town, randomize=False, debug_mode=False):
        """
        Setup all relevant parameters and create scenario
        """
        super(ChallengeBasic, self).__init__("ChallengeBasic", ego_vehicle, other_actors, town, world, debug_mode)

    def _create_behavior(self):
        """
        """

        # Build behavior tree
        sequence = py_trees.composites.Sequence("Sequence Behavior")

        # endcondition: Check if vehicle reached waypoint _end_distance from here:
        location, _ = get_location_in_distance(self.ego_vehicle, self._end_distance)
        end_condition = InTriggerDistanceToLocation(self.ego_vehicle, location, 2.0)

        # Build behavior tree
        sequence.add_child(end_condition)

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
