#!/usr/bin/env python

# Copyright (c) 2018-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Scenario in which the ego is parked between two vehicles and has to maneuver to start the route.
"""

from __future__ import print_function

import py_trees
import carla


from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import SwitchOutsideRouteLanesTest, ActorSource, ActorTransformSetter
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import InTriggerDistanceToLocation
from srunner.scenarios.basic_scenario import BasicScenario

import math
import operator


def convert_dict_to_location(actor_dict):
    """
    Convert a JSON string to a Carla.Location
    """
    location = carla.Location(
        x=float(actor_dict['x']),
        y=float(actor_dict['y']),
        z=float(actor_dict['z'])
    )
    return location


def get_same_dir_transform_by_distance(transform, distance: float):
    """
    Get a transform in the same direction as the given one.
    With {distance} (m) offset.
    Works for arbitrary point on the map.
    """
    new_t = carla.Transform()
    yaw = transform.rotation.yaw
    new_t.location.x = transform.location.x + \
        distance*math.cos(math.radians(yaw))
    new_t.location.y = transform.location.y + \
        distance*math.sin(math.radians(yaw))
    new_t.location.z = transform.location.z
    new_t.rotation = transform.rotation
    return new_t


class ParkingExit(BasicScenario):
    """
    This class holds everything required for a scenario in which the ego would be teleported to the parking lane.
    Once the scenario is triggered, the OutsideRouteLanesTest will be deactivated since the ego is out of the driving lane.
    Then blocking vehicles will be generated in front of and behind the parking point.
    The ego need to exit from the parking lane and then merge into the driving lane.
    After the ego is {end_distance} meters away from the parking point, the OutsideRouteLanesTest will be activated and the scenario ends.
    
    Note 1: For route mode, this shall be the first scenario of the route. The trigger point shall be the first point of the route waypoints.
    
    Note 2: Make sure there are enough space for spawning blocking vehicles.
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
        self._parking_point = convert_dict_to_location(
            config.other_parameters['parking_point'])

        if 'front_vehicle_distance' in config.other_parameters:
            self._front_vehicle_distance = float(
                config.other_parameters['front_vehicle_distance']['value'])
        else:
            self._front_vehicle_distance = 20  # m

        if 'behind_vehicle_distance' in config.other_parameters:
            self._behind_vehicle_distance = float(
                config.other_parameters['behind_vehicle_distance']['value'])
        else:
            self._behind_vehicle_distance = 5  # m

        if 'end_distance' in config.other_parameters:
            self._end_distance = float(
                config.other_parameters['end_distance']['value'])
        else:
            self._end_distance = 25  # m

        super(ParkingExit, self).__init__("ParkingExit",
                                          ego_vehicles,
                                          config,
                                          world,
                                          debug_mode,
                                          criteria_enable=criteria_enable)

    def _create_behavior(self):
        """
        Deactivate OutsideRouteLanesTest, then move ego to the parking point, generate blocking vehicles in front of and behind the ego.
        After ego drives away, actviate OutsideRouteLanesTest, end scenario.
        """

        sequence = py_trees.composites.Sequence()

        parking_transform = front_transform = carla.Transform()
        parking_transform.location = self._parking_point
        # To get rotation like this because the return value of the
        # get_waypoint(self, location, project_to_road=True, lane_type=carla.LaneType.Parking)
        # would be weird in some maps. So using get_waypoint(self._parking_point) to get a waypoint in driving lane
        # which has the same rotation with the parking point is a safe choice.
        parking_transform.rotation = self._map.get_waypoint(
            self._parking_point).transform.rotation

        # Put blocking vehicles
        front_transform = get_same_dir_transform_by_distance(
            parking_transform, self._front_vehicle_distance)
        front_transform.location.z += 0.5
        behind_transform = get_same_dir_transform_by_distance(
            parking_transform, -1*self._behind_vehicle_distance)
        behind_transform.location.z += 0.5

        actor_source_front = ActorSource(
            ['vehicle.audi.tt', 'vehicle.tesla.model3', 'vehicle.nissan.micra'],
            front_transform, 1, "ParkingExit_front_transform_queue", 1)

        actor_source_behind = ActorSource(
            ['vehicle.audi.tt', 'vehicle.tesla.model3', 'vehicle.nissan.micra'],
            behind_transform, 1, "ParkingExit_front_transform_queue", 1)

        # Deactivate OutsideRouteLanesTest
        sequence.add_child(SwitchOutsideRouteLanesTest(False))

        # Teleport ego to the parking point
        sequence.add_child(ActorTransformSetter(
            self.ego_vehicles[0], parking_transform))

        root = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        root.add_child(actor_source_front)
        root.add_child(actor_source_behind)

        end_condition = py_trees.composites.Sequence()
        end_condition.add_child(InTriggerDistanceToLocation(
            self.ego_vehicles[0], self._parking_point, self._end_distance, operator.gt, name="EndTrigger"))
        root.add_child(end_condition)
        sequence.add_child(root)

        sequence.add_child(SwitchOutsideRouteLanesTest(True))

        return sequence

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        return [CollisionTest(self.ego_vehicles[0])]

    def __del__(self):
        """
        Remove all actors and traffic lights upon deletion
        """
        print("del scenario")
        self.remove_all_actors()
