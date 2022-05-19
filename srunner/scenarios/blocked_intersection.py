#!/usr/bin/env python

# Copyright (c) 2018-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Scenario in which the ego TODO
"""

from __future__ import print_function

import py_trees
import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import ActorTransformSetter, ActorDestroy, Idle
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import DriveDistance, InTriggerDistanceToVehicle
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.tools.background_manager import SwitchLane, RemoveJunctionEntry


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
    This class holds everything required for a scenario in which the ego TODO
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
        self._block_distance = 10 # m. Will stop blocking when ego is within this distance
        self._block_time = 5 # s

        self._obstacle_horizontal_gap = 3  # m
        self._obstacle_vertical_gap = 2  # m
        self._obstacle_model = "static.prop.trampoline"

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
                break  # No more right lanes
            sidewalk_waypoint = right_wp

        obs_points = sidewalk_waypoint.next_until_lane_end(
            self._obstacle_horizontal_gap)
        # Only need some obstacles near junction
        obs_points = obs_points[-1*min(len(obs_points), 4):]

        for obs_point in obs_points:
            actor = CarlaDataProvider.request_new_actor(
                self._obstacle_model, obs_point.transform, rolename='scenario')
            if actor is None:
                raise Exception(
                    "Couldn't spawn obstacles")
            self.other_actors.append(actor)

        # Spawn the blocker vehicle



        # ev_points = self._reference_waypoint.previous(
        #     self._emergency_vehicle_distance)
        # if ev_points:
        #     self._actor_start_transform = ev_points[0].transform
        # else:
        #     raise Exception(
        #         "Couldn't find viable position for the emergency vehicle")

        actor = CarlaDataProvider.request_new_actor(
            "vehicle.*.*", self._blocker_waypoint.transform, rolename='scenario')
        if actor is None:
            raise Exception(
                "Couldn't spawn the blocker vehicle")
        self.other_actors.append(actor)

    def _create_behavior(self):
        """
        TODO
        """

        sequence = py_trees.composites.Sequence()

        sequence.add_child(RemoveJunctionEntry(self._reference_waypoint))
        sequence.add_child(SwitchLane(self._blocker_waypoint.lane_id, False))

        # Ego go behind the blocker
        blocker_wait = py_trees.composites.Parallel("Wait for ego to come close",
                                              policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        blocker_wait.add_child(InTriggerDistanceToVehicle(self.other_actors[-1], self.ego_vehicles[0],  self._block_distance))
        sequence.add_child(blocker_wait)
        sequence.add_child(Idle(self._block_time))
        sequence.add_child(ActorDestroy(self.other_actors[-1]))
        
        # End
        sequence.add_child(SwitchLane(self._blocker_waypoint.lane_id, True))
        # sequence.add_child(Idle(20))

        # sequence.add_child(SwitchLane(self._reference_waypoint.lane_id, False))

        # # Teleport EV behind the ego
        # sequence.add_child(ActorTransformSetter(
        #     self.other_actors[0], self._actor_start_transform))
        # sequence.add_child(SetInitSpeed(self.other_actors[0], 20))

        # # Emergency Vehicle runs for self._ev_drive_time seconds
        # ev_end_condition = py_trees.composites.Parallel("Waiting for emergency vehicle driving for a certein distance",
        #                                                 policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)

        # ev_end_condition.add_child(Idle(self._ev_drive_time))

        # target_locations = self._reference_waypoint.next(2000)
        # target_location = target_locations[0].transform.location
        # ev_end_condition.add_child(FasterBasicAgentBehavior(
        #     self.other_actors[0], self.ego_vehicles[0], target_location))

        # sequence.add_child(ev_end_condition)

        # sequence.add_child(ActorDestroy(self.other_actors[0]))

        # # End condition
        # end_condition = py_trees.composites.Parallel("Waiting for ego driving for a certein distance",
        #                                              policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        # end_condition.add_child(DriveDistance(
        #     self.ego_vehicles[0], 30))

        # sequence.add_child(end_condition)

        # sequence.add_child(SwitchLane(self._reference_waypoint.lane_id, True))

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
