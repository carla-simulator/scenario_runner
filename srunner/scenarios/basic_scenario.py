#!/usr/bin/env python

# Copyright (c) 2018-2019 Intel Labs.
# authors: Fabian Oboril (fabian.oboril@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provide the basic class for all user-defined scenarios.
"""

from __future__ import print_function

import py_trees

from srunner.scenariomanager.scenario_manager import Scenario


def get_location_in_distance(actor, distance):
    """
    Obtain a location in a given distance from the current actor's location.
    Note: Search is stopped on first intersection.

    @return obtained location and the traveled distance
    """
    waypoint = actor.get_world().get_map().get_waypoint(actor.get_location())
    traveled_distance = 0
    while not waypoint.is_intersection and traveled_distance < distance:
        waypoint_new = waypoint.next(1.0)[-1]
        traveled_distance += waypoint_new.transform.location.distance(waypoint.transform.location)
        waypoint = waypoint_new

    return waypoint.transform.location, traveled_distance


class BasicScenario(object):

    """
    Base class for user-defined scenario
    """

    _town = None            # Name of the map that is used
    category = None         # Scenario category, e.g. control_loss, follow_leading_vehicle, ...
    name = None             # Name of the scenario
    criteria_list = []      # List of evaluation criteria
    timeout = 60            # Timeout of scenario in seconds
    scenario = None

    ego_vehicle = None
    other_actors = []

    def __init__(self, name, ego_vehicle, other_actors, town, world, debug_mode=False, terminate_on_failure=False):
        """
        Setup all relevant parameters and create scenario
        and instantiate scenario manager
        """

        # Check if the CARLA server uses the correct map
        self._town = town
        self._check_town(world)

        self.ego_vehicle = ego_vehicle
        self.other_actors = other_actors
        self.name = name
        self.terminate_on_failure = terminate_on_failure

        # Setup scenario
        if debug_mode:
            py_trees.logging.level = py_trees.logging.Level.DEBUG

        behavior = self._create_behavior()
        criteria = self._create_test_criteria()
        self.scenario = Scenario(behavior, criteria, self.name, self.timeout, self.terminate_on_failure)

    def _create_behavior(self):
        """
        Pure virtual function to setup user-defined scenario behavior
        """
        raise NotImplementedError(
            "This function is re-implemented by all scenarios"
            "If this error becomes visible the class hierarchy is somehow broken")

    def _create_test_criteria(self):
        """
        Pure virtual function to setup user-defined evaluation criteria for the
        scenario
        """
        raise NotImplementedError(
            "This function is re-implemented by all scenarios"
            "If this error becomes visible the class hierarchy is somehow broken")

    def _check_town(self, world):
        if world.map_name != self._town:
            print("The CARLA server uses the wrong map!")
            print("This scenario requires to use map {}".format(self._town))
            raise Exception("The CARLA server uses the wrong map!")
