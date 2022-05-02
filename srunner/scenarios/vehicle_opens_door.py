#!/usr/bin/env python

# Copyright (c) 2018-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Scenarios in which another (opposite) vehicle 'illegally' takes
priority, e.g. by running a red traffic light.
"""

from __future__ import print_function

import py_trees
import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest

from srunner.scenariomanager.scenarioatomics.atomic_behaviors import OpenVehicleDoor, HandBrakeVehicle
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (InTriggerDistanceToLocation,
                                                                               InTimeToArrivalToLocation,
                                                                               DriveDistance)
from srunner.scenarios.basic_scenario import BasicScenario

from srunner.tools.background_manager import LeaveSpaceInFront



class VehicleOpensDoor(BasicScenario):
    """
    This class holds everything required for a scenario in which another vehicle runs a red light
    in front of the ego, forcing it to react. This vehicles are 'special' ones such as police cars,
    ambulances or firetrucks.
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=180):
        """
        Setup all relevant parameters and create scenario
        and instantiate scenario manager
        """
        self._world = world
        self._map = CarlaDataProvider.get_map()

        self.timeout = timeout
        self._wait_duration = 15
        self._end_distance = 40
        self._min_trigger_dist = 5
        self._reaction_time = 2.0

        if 'distance' in config.other_parameters:
            self._parked_distance = config.other_parameters['distance']['value']
        else:
            self._parked_distance = 50

        if 'direction' in config.other_parameters:
            self._direction = config.other_parameters['direction']['value']
        else:
            self._direction = 'right'
        if self._direction not in ('left', 'right'):
            raise ValueError(f"'direction' must be either 'right' or 'left' but {self._direction} was given")

        super().__init__("VehicleOpensDoor", ego_vehicles, config, world, debug_mode, criteria_enable=criteria_enable)

    def _initialize_actors(self, config):
        """
        Creates a parked vehicle on the side of the road
        """
        trigger_location = config.trigger_points[0].location
        starting_wp = self._map.get_waypoint(trigger_location)
        front_wps = starting_wp.next(self._parked_distance)
        if len(front_wps) == 0:
            raise ValueError("Couldn't find a spot to place the adversary vehicle")
        elif len(front_wps) > 1:
            print("WARNING: Found a diverging lane. Choosing one at random")
        self._front_wp = front_wps[0]

        if self._direction == 'left':
            parked_wp = self._front_wp.get_left_lane()
        else:
            parked_wp = self._front_wp.get_right_lane()

        if parked_wp is None:
            raise ValueError("Couldn't find a spot to place the adversary vehicle")

        self.parked_actor = CarlaDataProvider.request_new_actor(
            "*vehicle.*", parked_wp.transform, attribute_filter={'has_dynamic_doors': True, 'base_type': 'car'})
        if not self.parked_actor:
            raise ValueError("Couldn't spawn the parked vehicle")
        self.other_actors.append(self.parked_actor)

    def _create_behavior(self):
        """
        Hero vehicle is entering a junction in an urban area, at a signalized intersection,
        while another actor runs a red lift, forcing the ego to break.
        """
        sequence = py_trees.composites.Sequence(name="CrossingActor")
        sequence.add_child(HandBrakeVehicle(self.parked_actor, True))

        if self.route_mode:
            sequence.add_child(LeaveSpaceInFront(self._parked_distance))

        collision_location = self._front_wp.transform.location

        # Wait until ego is close to the adversary
        trigger_adversary = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE, name="TriggerAdversaryStart")
        trigger_adversary.add_child(InTimeToArrivalToLocation(
            self.ego_vehicles[0], self._reaction_time, collision_location))
        trigger_adversary.add_child(InTriggerDistanceToLocation(
            self.ego_vehicles[0], collision_location, self._min_trigger_dist))
        sequence.add_child(trigger_adversary)

        door = carla.VehicleDoor.FR if self._direction == 'left' else carla.VehicleDoor.FL

        sequence.add_child(OpenVehicleDoor(self.parked_actor, door, self._wait_duration))
        sequence.add_child(DriveDistance(self.ego_vehicles[0], self._end_distance))

        return sequence

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        if self.route_mode:
            return []
        return [CollisionTest(self.ego_vehicles[0])]

    def __del__(self):
        """
        Remove all actors and traffic lights upon deletion
        """
        self.remove_all_actors()
