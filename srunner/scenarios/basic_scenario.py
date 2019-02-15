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
import math

import numpy as np
import py_trees
import carla

from srunner.scenariomanager.scenario_manager import Scenario
from agents.tools.misc import vector

def get_intersection(ego_actor, other_actor):
    """
    Obtain a intersection point between two actor's location

    @return point of intersection
    """
    waypoint = ego_actor.get_world().get_map().get_waypoint(ego_actor.get_location())
    waypoint_other = other_actor.get_world().get_map().get_waypoint(other_actor.get_location())
    flag = float("inf")
    while True:
        current_location = waypoint.transform.location
        waypoint_choice = waypoint.next(1)

        #   Select the straighter path at intersection
        if len(waypoint_choice) > 1:
            loc_projection = current_location + carla.Location(
                x=math.cos(math.radians(waypoint.transform.rotation.yaw)),
                y=math.sin(math.radians(waypoint.transform.rotation.yaw)))
            v_current = vector(current_location, loc_projection)
            max_dot = -1*float('inf')
            for wp_select in waypoint_choice:
                v_select = vector(current_location, wp_select.transform.location)
                dot_select = np.dot(v_current, v_select)
                if dot_select > max_dot:
                    max_dot = dot_select
                    waypoint = wp_select
        else:
            waypoint = waypoint_choice[0]

        distance = current_location.distance(waypoint_other.transform.location)
        if distance > flag:
            break
        flag = distance
    return current_location

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
        if world.get_map().name != self._town:
            print("The CARLA server uses the wrong map!")
            print("This scenario requires to use map {}".format(self._town))
            raise Exception("The CARLA server uses the wrong map!")
