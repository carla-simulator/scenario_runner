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

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import ActorTransformSetter, ActorDestroy, WaypointFollower, SwitchOutsideRouteLanesTest
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest, YieldToEmergencyVehicleTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import DriveDistance, WaitForTime
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.tools.background_manager import RemoveLane, SwitchRouteSources

import random


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
        self._actor_types = ["vehicle.carlamotors.firetruck", "vehicle.ford.ambulance",
                             "vehicle.dodge.charger_police", "vehicle.dodge.charger_police_2020"]
        self._ev_drive_time = 20  # seconds
        self._ev_speed = 80

        if 'emergency_vehicle_distance' in config.other_parameters:
            self._emergency_vehicle_distance = float(
                config.other_parameters['emergency_vehicle_distance']['value'])
        else:
            self._emergency_vehicle_distance = 5  # m

        self._trigger_location = config.trigger_points[0].location
        self._reference_waypoint = self._map.get_waypoint(
            self._trigger_location)

        super(YieldToEmergencyVehicle, self).__init__("YieldToEmergencyVehicle",
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

        spawn_transform = self._reference_waypoint.transform
        spawn_transform.location.z -= 500

        actor = CarlaDataProvider.request_new_actor(
            random.choice(self._actor_types), spawn_transform, rolename='scenario')
        if actor is None:
            raise Exception(
                "Couldn't spawn the emergency vehicle")
        self.other_actors.append(actor)

    def _create_behavior(self):
        """
        - Remove BA from current lane
        - Teleport Emergency Vehicle(EV) behind the ego
        - Deactivate OutsideRouteLanesTest
        - [Parallel SUCCESS_ON_ONE]
            - WaitForTime(20 seconds)
            - WaypointFollower(EV, speed=80, avoid_collision=True)
        - Destroy EV
        - [Parallel SUCCESS_ON_ONE]
            - DriveDistance(ego, 30)
        - Activate OutsideRouteLanesTest
        - Recover BA
        """

        sequence = py_trees.composites.Sequence()

        sequence.add_child(RemoveLane(self._reference_waypoint.lane_id))
        # sequence.add_child(SwitchLane(self._reference_waypoint.lane_id, False))

        # Teleport EV behind the ego
        ev_points = self._map.get_waypoint(self.ego_vehicles[0].get_location()).previous(
            self._emergency_vehicle_distance)
        if ev_points:
            ev_start_transform = ev_points[0].transform
        else:
            raise Exception(
                "Couldn't find viable position for the emergency vehicle")
        sequence.add_child(ActorTransformSetter(
            self.other_actors[0], ev_start_transform))

        sequence.add_child(SwitchOutsideRouteLanesTest(False))

        ev_end_condition = py_trees.composites.Parallel("Waiting for emergency vehicle driving for a certein distance",
                                                        policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)

        ev_end_condition.add_child(WaitForTime(self._ev_drive_time))

        ev_end_condition.add_child(WaypointFollower(
            self.other_actors[0], self._ev_speed, avoid_collision=True))
        sequence.add_child(ev_end_condition)

        sequence.add_child(ActorDestroy(self.other_actors[0]))

        end_condition = py_trees.composites.Parallel("Waiting for ego driving for a certein distance",
                                                     policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        end_condition.add_child(DriveDistance(
            self.ego_vehicles[0], 30))

        sequence.add_child(end_condition)

        sequence.add_child(SwitchOutsideRouteLanesTest(True))

        sequence.add_child(SwitchRouteSources(True))
        # sequence.add_child(SwitchLane(self._reference_waypoint.lane_id, True))

        return sequence

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        if self.route_mode:
            return[YieldToEmergencyVehicleTest(self.ego_vehicles[0], self.other_actors[0])]

        return [CollisionTest(self.ego_vehicles[0]), YieldToEmergencyVehicleTest(self.ego_vehicles[0], self.other_actors[0])]

    def __del__(self):
        """
        Remove all actors and traffic lights upon deletion
        """
        self.remove_all_actors()
