#!/usr/bin/env python

# Copyright (c) 2018-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Scenario in which the ego has to yield its lane to emergency vehicle.
"""

from __future__ import print_function

import py_trees
import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import ActorTransformSetter, ActorDestroy, Idle, AdaptiveConstantVelocityAgentBehavior
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest, YieldToEmergencyVehicleTest
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.tools.background_manager import RemoveRoadLane, ReAddEgoRoadLane


class YieldToEmergencyVehicle(BasicScenario):
    """
    This class holds everything required for a scenario in which the ego has to yield its lane to emergency vehicle.
    The background activity will be removed from the lane the emergency vehicle will pass through, 
    and will be recreated once the scenario is over.

    Should be on the highway which is long enough and has no junctions.
    There should be at least two lanes on the highway.
    """

    def __init__(self, world, ego_vehicles, config, debug_mode=False, criteria_enable=True,
                 timeout=180):
        """
        Setup all relevant parameters and create scenario
        and instantiate scenario manager
        """
        self._world = world
        self._map = CarlaDataProvider.get_map()
        self.timeout = timeout
        self._ev_drive_time = 12  # seconds

        # m/s. How much the EV is expected to be faster than the EGO
        self._speed_increment = 15

        if 'distance' in config.other_parameters:
            self._distance = float(
                config.other_parameters['distance']['value'])
        else:
            self._distance = 15  # m

        self._trigger_location = config.trigger_points[0].location
        self._reference_waypoint = self._map.get_waypoint(
            self._trigger_location)
        self._ev_start_transform = None

        super().__init__("YieldToEmergencyVehicle",
                         ego_vehicles,
                         config,
                         world,
                         debug_mode,
                         criteria_enable=criteria_enable)

    def _initialize_actors(self, config):
        """
        Custom initialization
        """
        # Spawn emergency vehicle
        ev_points = self._reference_waypoint.previous(
            self._distance)
        if ev_points:
            self._ev_start_transform = ev_points[0].transform
        else:
            raise Exception(
                "Couldn't find viable position for the emergency vehicle")

        actor = CarlaDataProvider.request_new_actor(
            "vehicle.*.*", self._ev_start_transform, rolename='scenario', attribute_filter={'special_type': 'emergency'})
        if actor is None:
            raise Exception(
                "Couldn't spawn the emergency vehicle")
        # Remove its physics so that it doesn't fall
        actor.set_simulate_physics(False)
        # Move the actor underground
        new_location = actor.get_location()
        new_location.z -= 500
        actor.set_location(new_location)
        # Turn on special lights
        actor.set_light_state(carla.VehicleLightState(
            carla.VehicleLightState.Special1 | carla.VehicleLightState.Special2))
        self.other_actors.append(actor)

    def _create_behavior(self):
        """
        - Remove BA from current lane
        - Teleport Emergency Vehicle(EV) behind the ego
        - [Parallel SUCCESS_ON_ONE]
            - Idle(20 seconds)
            - AdaptiveConstantVelocityAgentBehavior
        - Destroy EV
        - [Parallel SUCCESS_ON_ONE]
            - DriveDistance(ego, 30)
        - Recover BA
        """

        sequence = py_trees.composites.Sequence()

        if self.route_mode:
            sequence.add_child(RemoveRoadLane(self._reference_waypoint))

        # Teleport EV behind the ego
        sequence.add_child(ActorTransformSetter(
            self.other_actors[0], self._ev_start_transform))

        # Emergency Vehicle runs for self._ev_drive_time seconds
        ev_end_condition = py_trees.composites.Parallel("Waiting for emergency vehicle driving for a certein distance",
                                                        policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)

        ev_end_condition.add_child(Idle(self._ev_drive_time))

        ev_end_condition.add_child(AdaptiveConstantVelocityAgentBehavior(
            self.other_actors[0], self.ego_vehicles[0], speed_increment=self._speed_increment))

        sequence.add_child(ev_end_condition)

        sequence.add_child(ActorDestroy(self.other_actors[0]))

        if self.route_mode:
            sequence.add_child(ReAddEgoRoadLane())

        return sequence

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criterias = []
        criterias.append(YieldToEmergencyVehicleTest(
            self.ego_vehicles[0], self.other_actors[0]))
        if not self.route_mode:
            criterias.append(CollisionTest(self.ego_vehicles[0]))

        return criterias

    def __del__(self):
        """
        Remove all actors and traffic lights upon deletion
        """
        self.remove_all_actors()
