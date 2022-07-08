#!/usr/bin/env python

# Copyright (c) 2018-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Scenario in which the ego is about to turn right 
when a vehicle coming from the opposite lane invades the ego's lane, forcing the ego to move right to avoid a possible collision.
"""

from __future__ import print_function

import py_trees
import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (ActorTransformSetter,
                                                                      ActorDestroy,
                                                                      BasicAgentBehavior,
                                                                      ScenarioTimeout)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest, ScenarioTimeoutTest
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.tools.background_manager import LeaveSpaceInFront


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


class InvadingTurn(BasicScenario):
    """
    This class holds everything required for a scenario in which the ego is about to turn right 
    when a vehicle coming from the opposite lane invades the ego's lane, 
    forcing the ego to move right to avoid a possible collision.

    This scenario is expected to take place on a road that has only one lane in each direction.
    """

    def __init__(self, world, ego_vehicles, config, debug_mode=False, criteria_enable=True,
                 timeout=180):
        """
        Setup all relevant parameters and create scenario
        and instantiate scenario manager
        """
        self._map = CarlaDataProvider.get_map()
        self.timeout = timeout

        self._trigger_location = config.trigger_points[0].location
        self._reference_waypoint = self._map.get_waypoint(
            self._trigger_location)

        # Distance between the trigger point and the start location of the adversary
        if 'distance' in config.other_parameters:
            self._distance = float(
                config.other_parameters['distance']['value'])
        else:
            self._distance = 120  # m

        self._adversary_end = self._reference_waypoint.get_left_lane().transform.location

        # The width (m) of the vehicle invading the opposite lane.
        if 'offset' in config.other_parameters:
            self._offset = float(
                config.other_parameters['offset']['value'])
        else:
            self._offset = 0.5

        if 'timeout' in config.other_parameters:
            self._scenario_timeout = float(config.other_parameters['flow_distance']['value'])
        else:
            self._scenario_timeout = 180

        super(InvadingTurn, self).__init__("InvadingTurn",
                                           ego_vehicles,
                                           config,
                                           world,
                                           debug_mode,
                                           criteria_enable=criteria_enable)

    def _initialize_actors(self, config):
        """
        Custom initialization
        """
        # Spawn adversary actor
        self._adversary_start_waypoint = self._reference_waypoint.next(self._distance)[
            0]
        self._adversary_start_waypoint = self._adversary_start_waypoint.get_left_lane()
        if self._adversary_start_waypoint:
            self._adversary_start_transform = self._adversary_start_waypoint.transform
        else:
            raise Exception(
                "Couldn't find viable position for the adversary vehicle")
        spawn_transform = carla.Transform(
            self._adversary_start_transform.location, self._adversary_start_transform.rotation)

        spawn_transform.location.z += 5  # Avoid colliding with BA actors
        actor = CarlaDataProvider.request_new_actor(
            "vehicle.*", spawn_transform, rolename='scenario', attribute_filter={'base_type': 'car', 'has_lights': True})
        if actor is None:
            raise Exception(
                "Couldn't spawn the adversary vehicle")

        # Remove its physics so that it doesn't fall
        actor.set_simulate_physics(False)
        # Move the actor underground
        new_location = actor.get_location()
        new_location.z -= 500
        actor.set_location(new_location)

        self.other_actors.append(actor)

        # Calculate the real offset
        lane_width = self._adversary_start_waypoint.lane_width
        car_width = actor.bounding_box.extent.y*2
        self._offset = -1*(self._offset + 0.5*lane_width - 0.5*car_width)

    def _create_behavior(self):
        """
        The adversary vehicle will go to the target place while invading another lane.
        """

        sequence = py_trees.composites.Sequence()

        if self.route_mode:
            sequence.add_child(LeaveSpaceInFront(self._distance))

        # Teleport adversary
        sequence.add_child(ActorTransformSetter(
            self.other_actors[0], self._adversary_start_transform))

        behavior = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        behavior.add_child(BasicAgentBehavior(
            self.other_actors[0], self._adversary_end, opt_dict={'offset': self._offset}))
        behavior.add_child(DriveDistance(self.ego_vehicles[0], self._distance))
        behavior.add_child(ScenarioTimeout(self._scenario_timeout, self.config.name))

        sequence.add_child(behavior)
        sequence.add_child(ActorDestroy(self.other_actors[0]))

        return sequence

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = [ScenarioTimeoutTest(self.ego_vehicles[0], self.config.name)]
        if not self.route_mode:
            criteria.append(CollisionTest(self.ego_vehicles[0]))
        return criteria

    def __del__(self):
        """
        Remove all actors and traffic lights upon deletion
        """
        self.remove_all_actors()
