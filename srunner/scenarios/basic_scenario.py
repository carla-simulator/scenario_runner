#!/usr/bin/env python

# Copyright (c) 2018-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provide the basic class for all user-defined scenarios.
"""

from __future__ import print_function

import py_trees

import srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions as conditions
from srunner.scenariomanager.carla_data_provider import CarlaActorPool, CarlaDataProvider
from srunner.scenariomanager.scenario_manager import Scenario


class BasicScenario(object):

    """
    Base class for user-defined scenario
    """

    def __init__(self, name, ego_vehicles, config, world,
                 debug_mode=False, terminate_on_failure=False, criteria_enable=False):
        """
        Setup all relevant parameters and create scenario
        and instantiate scenario manager
        """
        self.other_actors = []
        if not self.timeout:     # pylint: disable=access-member-before-definition
            self.timeout = 60    # If no timeout was provided, set it to 60 seconds

        self.criteria_list = []  # List of evaluation criteria
        self.scenario = None
        # Check if the CARLA server uses the correct map
        self._town = config.town
        self._check_town(world)

        self.ego_vehicles = ego_vehicles
        self.name = name
        self.terminate_on_failure = terminate_on_failure

        # Initializing adversarial actors
        self._initialize_actors(config)
        if world.get_settings().synchronous_mode:
            CarlaDataProvider.perform_carla_tick()
        else:
            world.wait_for_tick()

        # Setup scenario
        if debug_mode:
            py_trees.logging.level = py_trees.logging.Level.DEBUG

        behavior = self._create_behavior()

        criteria = None
        if criteria_enable:
            criteria = self._create_test_criteria()

        # Add a trigger condition for the behavior to ensure the behavior is only activated, when it is relevant
        behavior_seq = py_trees.composites.Sequence()
        trigger_behavior = self._setup_scenario_trigger(config)
        if trigger_behavior:
            behavior_seq.add_child(trigger_behavior)

        if behavior is not None:
            behavior_seq.add_child(behavior)
            behavior_seq.name = behavior.name

        self.scenario = Scenario(behavior_seq, criteria, self.name, self.timeout, self.terminate_on_failure)

    def _initialize_actors(self, config):
        """
        Default initialization of other actors.
        Override this method in child class to provide custom initialization.
        """

        new_actors = CarlaActorPool.request_new_actors(config.other_actors)
        if new_actors is None:
            raise Exception("Error: Unable to add actors")

        for new_actor in new_actors:
            self.other_actors.append(new_actor)

    def _setup_scenario_trigger(self, config):
        """
        This function creates a trigger maneuver, that has to be finished before the real scenario starts.
        This implementation focuses on the first available ego vehicle.

        The function can be overloaded by a user implementation inside the user-defined scenario class.
        """
        start_location = None
        if config.trigger_points and config.trigger_points[0]:
            start_location = config.trigger_points[0].location     # start location of the scenario

        ego_vehicle_route = CarlaDataProvider.get_ego_vehicle_route()

        if start_location:
            if ego_vehicle_route:
                return conditions.InTriggerDistanceToLocationAlongRoute(self.ego_vehicles[0],
                                                                        ego_vehicle_route,
                                                                        start_location,
                                                                        5)
            return conditions.InTimeToArrivalToLocation(self.ego_vehicles[0],
                                                        2.0,
                                                        start_location)

        return None

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

    def change_control(self, control):  # pylint: disable=no-self-use
        """
        This is a function that changes the control based on the scenario determination
        :param control: a carla vehicle control
        :return: a control to be changed by the scenario.

        Note: This method should be overriden by the user-defined scenario behavior
        """
        return control

    def remove_all_actors(self):
        """
        Remove all actors
        """
        for i, _ in enumerate(self.other_actors):
            if self.other_actors[i] is not None:
                if CarlaActorPool.actor_id_exists(self.other_actors[i].id):
                    CarlaActorPool.remove_actor_by_id(self.other_actors[i].id)
                self.other_actors[i] = None
        self.other_actors = []
