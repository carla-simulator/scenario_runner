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
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import ActorDestroy,  WaypointFollower, LaneChange, AccelerateToVelocity, SetInitSpeed
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import DriveDistance, WaitUntilInFront
from srunner.scenarios.basic_scenario import BasicScenario

class HighwayEntryCutIn(BasicScenario):
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
        self._drive_distance = 100
        self._other_init_speed = 100 / 3.6
        self._other_end_speed = 70 / 3.6
        self._factor = 0

        super(HighwayEntryCutIn, self).__init__("HighwayEntryCutIn",
                                                ego_vehicles,
                                                config,
                                                world,
                                                debug_mode,
                                                criteria_enable=criteria_enable)

    def _initialize_actors(self, config):
        """
        Custom initialization
        """
        starting_location = config.other_actors[0].transform.location
        starting_waypoint = self._map.get_waypoint(starting_location)

        other_vehicle = CarlaDataProvider.request_new_actor(config.other_actors[0].model, starting_waypoint.transform)
        self.other_actors.append(other_vehicle)

    def _create_behavior(self):
        """
        Hero vehicle is entering a junction in an urban area, at a signalized intersection,
        while another actor runs a red lift, forcing the ego to break.
        """

        behaviour = py_trees.composites.Sequence("ExitFreewayCutIn")

        # Make the vehicle "sync" with the ego_vehicle
        sync_arrival = py_trees.composites.Parallel(
            "Sync Arrival", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        sync_arrival.add_child(WaitUntilInFront(self.other_actors[0], self.ego_vehicles[0], factor=self._factor))

        sync_behavior = py_trees.composites.Sequence()
        sync_behavior.add_child(SetInitSpeed(self.other_actors[0], self._other_init_speed))
        sync_behavior.add_child(WaypointFollower(self.other_actors[0], self._other_init_speed * 1000))
        sync_arrival.add_child(sync_behavior)

        # Force a lane change
        lane_change = LaneChange(
            self.other_actors[0],
            speed=self._other_end_speed,
            direction='left',
            distance_same_lane=1,
            distance_other_lane=30,
            distance_lane_change=60)

        # End of the scenario
        end_condition = py_trees.composites.Parallel(
            "End condition", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)

        end_condition.add_child(DriveDistance(self.ego_vehicles[0], self._drive_distance))
        end_condition.add_child(WaypointFollower(self.other_actors[0], self._other_end_speed))

        # behaviour.add_child(init_speed)
        behaviour.add_child(sync_arrival)
        behaviour.add_child(lane_change)
        behaviour.add_child(end_condition)
        behaviour.add_child(ActorDestroy(self.other_actors[0]))

        return behaviour

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
        self.remove_all_actors()
