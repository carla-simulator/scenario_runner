#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Vehicle Manuevering In Opposite Direction:

Vehicle is passing another vehicle in a rural area, in daylight, under clear
weather conditions, at a non-junction with a posted speed limit of 55 mph or more;
and encroaches into another vehicle traveling in the opposite direcntotion.
"""

import random

import py_trees

from srunner.scenariomanager.atomic_scenario_behavior import *
from srunner.scenariomanager.atomic_scenario_criteria import *
from srunner.scenariomanager.timer import TimeOut
from srunner.scenarios.basic_scenario import *


MANEUVER_OPPOSITE_DIRECTION = [
    "ManeuverOppositeDirection"
]


class ManeuverOppositeDirection(BasicScenario):

    """
    Implementation class for Traffic Scenario 06,
    "Vehicle Manuevering In Opposite Direction".
    """

    category = "ManeuverOppositeDirection"
    timeout = 120            # Timeout of scenario in seconds

    def __init__(self, world, ego_vehicle, other_actors, town, randomize=False, debug_mode=False, config=None):
        """
        Setup all relevant parameters and create scenario
        """
        super(ManeuverOppositeDirection, self).__init__(
            "FollowVehicle",
            ego_vehicle,
            other_actors,
            town,
            world,
            debug_mode)

    def _create_behavior(self):
        """
        The behavior tree returned by this method is as follows:
        The ego vehicle is trying to pass a leading vehicle in the same lane
        by moving onto the oncoming lane while another vehicle is moving in the
        opposite direction in the oncoming lane.
        """

        # Build behavior tree
        sequence = py_trees.composites.Sequence("Sequence Behavior")

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

    def __del__(self):
        """
        Remove all actors upon deletion
        """
        self.remove_all_actors()
