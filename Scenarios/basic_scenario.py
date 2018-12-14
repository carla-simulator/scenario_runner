#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: Fabian Oboril (fabian.oboril@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provide the basic class for all user-defined scenarios.
"""

from __future__ import print_function
import random
import sys

import py_trees

from ScenarioManager.scenario_manager import Scenario


def setup_vehicle(world, model, spawn_point, hero=False):
    """
    Function to setup the most relevant vehicle parameters,
    incl. spawn point and vehicle model.
    """
    blueprint_library = world.get_blueprint_library()

    # Get vehicle by model
    blueprint = random.choice(blueprint_library.filter(model))
    if hero:
        blueprint.set_attribute('role_name', 'hero')
    else:
        blueprint.set_attribute('role_name', 'scenario')

    vehicle = world.try_spawn_actor(blueprint, spawn_point)

    if vehicle is None:
        sys.exit(
            "Error: Unable to spawn vehicle {} at {}".format(model, spawn_point))

    # Let's put the vehicle to drive around
    vehicle.set_autopilot(False)

    return vehicle


class BasicScenario(object):

    """
    Base class for user-defined scenario
    """

    _town = None            # Name of the map that is used
    name = None             # Name of the scenario
    criteria_list = []      # List of evaluation criteria
    timeout = 60            # Timeout of scenario in seconds
    scenario = None

    ego_vehicle = None
    other_vehicles = []

    def __init__(self, name, town, world, debug_mode=False):
        """
        Setup all relevant parameters and create scenario
        and instantiate scenario manager
        """

        # Check if the CARLA server uses the correct map
        self._town = town
        self._check_town(world)

        self.name = name

        # Setup scenario
        if debug_mode:
            py_trees.logging.level = py_trees.logging.Level.DEBUG

        behavior = self._create_behavior()
        criteria = self._create_test_criteria()
        self.scenario = Scenario(
            behavior, criteria, self.name, self.timeout)

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
            sys.exit(-1)

    def __del__(self):
        """
        Cleanup.
        - Removal of the vehicles
        """
        actors = [self.ego_vehicle] + self.other_vehicles
        for actor in actors:
            if actor is not None:
                actor.destroy()
                actor = None
