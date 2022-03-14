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
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import ActorDestroy, BasicAgentBehavior
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import DriveDistance, InTimeToArrivalToVehicle
from srunner.scenarios.basic_scenario import BasicScenario


class BycicleGroup(BasicScenario):
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
        self._bycicle_speed = float(config.other_parameters['bycicle_speed']['value'])
        self._offset = float(config.other_parameters['bycicle_offset']['value'])
        self.timeout = timeout
        self._drive_distance = 200
        self._arrival_time = 7

        super(BycicleGroup, self).__init__("BycicleGroup",
                                           ego_vehicles,
                                           config,
                                           world,
                                           debug_mode,
                                           criteria_enable=criteria_enable)

    def _initialize_actors(self, config):
        """
        Custom initialization
        """
        bycicle_transform = config.other_actors[0].transform
        bycicle_wp = self._map.get_waypoint(bycicle_transform.location)

        # Displace the wp to the side
        self._displacement = self._offset * bycicle_wp.lane_width / 2
        r_vec = bycicle_wp.transform.get_right_vector()
        w_loc = bycicle_wp.transform.location
        w_loc += carla.Location(x=self._displacement * r_vec.x, y=self._displacement * r_vec.y)
        bycicle_transform = carla.Transform(w_loc, bycicle_wp.transform.rotation)

        bycicle = CarlaDataProvider.request_new_actor(config.other_actors[0].model, bycicle_transform)
        self.other_actors.append(bycicle)

    def _create_behavior(self):
        """
        Hero vehicle is entering a junction in an urban area, at a signalized intersection,
        while another actor runs a red lift, forcing the ego to break.
        """
        sequence = py_trees.composites.Sequence()
        behavior = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        sequence.add_child(InTimeToArrivalToVehicle(self.ego_vehicles[0], self.other_actors[0], self._arrival_time))
        behavior.add_child(BasicAgentBehavior(
            self.other_actors[0], target_speed=self._bycicle_speed, opt_dict={'offset': self._displacement}))
        behavior.add_child(DriveDistance(self.ego_vehicles[0], self._drive_distance))
        sequence.add_child(behavior)
        sequence.add_child(ActorDestroy(self.other_actors[0]))
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
        self.remove_all_actors()
