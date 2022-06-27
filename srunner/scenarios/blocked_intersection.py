#!/usr/bin/env python

# Copyright (c) 2018-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Scenario with low visibility, the ego performs a turn only to find out that the end is blocked by another vehicle.
"""

from __future__ import print_function

import carla
import py_trees
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (
    ActorDestroy, Idle)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import \
    CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import \
    InTriggerDistanceToVehicle
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.tools.background_manager import HandleJunctionScenario


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


class BlockedIntersection(BasicScenario):
    """
    This class holds everything required for a scenario in which with low visibility, 
    the ego performs a turn only to find out that the end is blocked by another vehicle.
    The ego is expected to not see the blockage until far into the junction, resulting in a hard brake.

    User needs to specify the location of the blocker.
    This scenario is expected to spawn obstacles on the sidewalk.
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

        self._trigger_location = config.trigger_points[0].location
        self._reference_waypoint = self._map.get_waypoint(
            self._trigger_location)

        self._blocker_location = convert_dict_to_location(
            config.other_parameters['blocker_point'])
        self._blocker_waypoint = self._map.get_waypoint(self._blocker_location)
        self._block_distance = 10  # m. Will stop blocking when ego is within this distance
        self._block_time = 5  # s

        if 'obstacle_model' in config.other_parameters:
            self._obstacle_model = config.other_parameters['obstacle_model']['value']
        else:
            self._obstacle_model = "static.prop.trampoline"

        if 'obstacle_gap' in config.other_parameters:
            self._obstacle_gap = int(
                config.other_parameters['obstacle_gap']['value'])
        else:
            self._obstacle_gap = 2

        # Extra obstacles are not included. One obstacle by default.
        self._obstacle_amount = 1

        # The amount of obstacles that invade the road
        if 'extra_obstacle' in config.other_parameters:
            self._extra_obstacle = int(
                config.other_parameters['extra_obstacle']['value'])
        else:
            self._extra_obstacle = 2

        super(BlockedIntersection, self).__init__("BlockedIntersection",
                                                  ego_vehicles,
                                                  config,
                                                  world,
                                                  debug_mode,
                                                  criteria_enable=criteria_enable)

    def _initialize_actors(self, config):
        """
        Custom initialization
        """
        # Spawn obstacles to block the view

        # Move to the right sidewalk
        sidewalk_waypoint = self._reference_waypoint

        while sidewalk_waypoint.lane_type != carla.LaneType.Sidewalk:
            right_wp = sidewalk_waypoint.get_right_lane()
            if right_wp is None:
                raise RuntimeError("Can't find sidewalk to spawn obstacles")
            sidewalk_waypoint = right_wp

        sidewalk_points = sidewalk_waypoint.next_until_lane_end(
            self._obstacle_gap)
        sidewalk_transforms = [wp.transform for wp in sidewalk_points]

        # Add some obstacles to invade the road
        for _ in range(self._extra_obstacle):
            end_transform_1 = sidewalk_transforms[-1]
            end_transform_2 = sidewalk_transforms[-2]
            delta_location = carla.Location(x=end_transform_1.location.x-end_transform_2.location.x,
                                            y=end_transform_1.location.y-end_transform_2.location.y,
                                            z=end_transform_1.location.z-end_transform_2.location.z)
            extra_location = end_transform_1.location + delta_location
            extra_transform = carla.Transform(extra_location, carla.Rotation())
            sidewalk_transforms.append(extra_transform)

        obs_transforms = sidewalk_transforms[-1 *
                                             min(len(sidewalk_transforms), self._obstacle_amount + self._extra_obstacle):]

        # Spawn obstacles
        actors = CarlaDataProvider.request_new_batch_actors(
            self._obstacle_model, len(obs_transforms), obs_transforms, rolename='scenario')
        self.other_actors += actors

        # Spawn the blocker vehicle
        actor = CarlaDataProvider.request_new_actor(
            "vehicle.*.*", self._blocker_waypoint.transform, rolename='scenario')
        if actor is None:
            raise Exception(
                "Couldn't spawn the blocker vehicle")
        self.other_actors.append(actor)

    def _create_behavior(self):
        """
        When ego arrives behind the blocker, idel for a while, then clear the blocker.
        """

        sequence = py_trees.composites.Sequence()

        if self.route_mode:
            sequence.add_child(HandleJunctionScenario(
                clear_junction=True,
                clear_ego_entry=True,
                remove_entries=[],
                remove_exits=[],
                stop_entries=True,
                extend_road_exit=0
            ))
        # Ego go behind the blocker
        sequence.add_child(InTriggerDistanceToVehicle(
            self.other_actors[-1], self.ego_vehicles[0],  self._block_distance))
        sequence.add_child(Idle(self._block_time))
        sequence.add_child(ActorDestroy(self.other_actors[-1]))

        return sequence

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        if self.route_mode:
            return[]

        return [CollisionTest(self.ego_vehicles[0])]

    def __del__(self):
        """
        Remove all actors and traffic lights upon deletion
        """
        self.remove_all_actors()
