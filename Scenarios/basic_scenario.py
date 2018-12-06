#!/usr/bin/env python

#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provide the basic class for all user-defined scenarios.
"""

import random
import sys

import py_trees
import carla

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

    name = None
    criteria_list = []      # List of evaluation criteria
    timeout = 60            # Timeout of scenario in seconds
    scenario = None

    ego_vehicle = None
    other_vehicles = []

    def __init__(self, name, debug_mode=False):
        """
        Setup all relevant parameters and create scenario
        and instantiate scenario manager
        """

        self.name = name

        # Setup scenario
        if debug_mode:
            py_trees.logging.level = py_trees.logging.Level.DEBUG

        behavior = self.create_behavior()
        criteria = self.create_test_criteria()
        self.scenario = Scenario(
            behavior, criteria, self.name, self.timeout)

    def create_behavior(self):
        """
        Pure virtual function to setup user-defined scenario behavior
        """

        raise NotImplementedError(
            "This function is re-implemented by all scenarios"
            "If this error becomes visible the class hierarchy is somehow broken")

    def create_test_criteria(self):
        """
        Pure virtual function to setup user-defined evaluation criteria for the
        scenario
        """

        raise NotImplementedError(
            "This function is re-implemented by all scenarios"
            "If this error becomes visible the class hierarchy is somehow broken")

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
