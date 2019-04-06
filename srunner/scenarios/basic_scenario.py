#!/usr/bin/env python

# Copyright (c) 2018-2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provide the basic class for all user-defined scenarios.
"""

from __future__ import print_function

import py_trees
import math
import carla
import numpy as np

from srunner.scenariomanager.atomic_scenario_behavior import InTimeToArrivalToLocation
from srunner.scenariomanager.carla_data_provider import CarlaActorPool, CarlaDataProvider
from srunner.scenariomanager.scenario_manager import Scenario


class BasicScenario(object):

    """
    Base class for user-defined scenario
    """

    def __init__(self, name, ego_vehicle, config, world, debug_mode=False, terminate_on_failure=False, criteria_enable=False):
        """
        Setup all relevant parameters and create scenario
        and instantiate scenario manager
        """
        self.other_actors = []
        if not self.timeout:
            self.timeout = 60    # If no timeout was provided, set it to 60 seconds

        self.category = None     # Scenario category, e.g. control_loss, follow_leading_vehicle, ...
        self.criteria_list = []  # List of evaluation criteria
        self.scenario = None
        # Check if the CARLA server uses the correct map
        self._town = config.town
        self._check_town(world)

        self.ego_vehicle = ego_vehicle
        self.name = name
        self.terminate_on_failure = terminate_on_failure

        # Initializing adversarial actors
        self._initialize_actors(config)
        if not CarlaDataProvider.is_sync_mode():
            world.wait_for_tick()
        else:
            world.tick()

        # Setup scenario
        if debug_mode:
            py_trees.logging.level = py_trees.logging.Level.DEBUG

        behavior = self._create_behavior()

        criteria = None
        if criteria_enable:
            criteria = self._create_test_criteria()

        # Add a trigger condition for the behavior to ensure the behavior is only activated, when it is relevant

        start_location = None
        if config.trigger_point:
            start_location = config.trigger_point.location     # start location of the scenario

        time_to_start_location = 2.0                               # seconds
        behavior_seq = py_trees.composites.Sequence()
        if start_location:
            behavior_seq.add_child(InTimeToArrivalToLocation(self.ego_vehicle, time_to_start_location, start_location))
        behavior_seq.add_child(behavior)

        self.scenario = Scenario(behavior_seq, criteria, self.name, self.timeout, self.terminate_on_failure)

    def _initialize_actors(self, config):
        """
        Default initialization of other actors.
        Override this method in child class to provide custom initialization.
        """
        for actor in config.other_actors:
            new_actor = CarlaActorPool.request_new_actor(actor.model,
                                                         actor.transform,
                                                         hero=False,
                                                         autopilot=actor.autopilot,
                                                         random_location=actor.random_location)
            if new_actor is None:
                raise Exception("Error: Unable to add actor {} at {}".format(actor.model, actor.transform))

            self.other_actors.append(new_actor)

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
        if CarlaDataProvider.get_map().name != self._town:
            print("The CARLA server uses the wrong map!")
            print("This scenario requires to use map {}".format(self._town))
            raise Exception("The CARLA server uses the wrong map!")

    def remove_all_actors(self):
        """
        Remove all actors
        """
        for i, _ in enumerate(self.other_actors):
            if self.other_actors[i] is not None:
                CarlaActorPool.remove_actor_by_id(self.other_actors[i].id)
                self.other_actors[i] = None
        self.other_actors = []
