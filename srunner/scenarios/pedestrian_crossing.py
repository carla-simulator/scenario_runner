#!/usr/bin/env python
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Object crash without prior vehicle action scenario:
The scenario realizes the user controlled ego vehicle
moving along the road and encountering a cyclist ahead.
"""

from __future__ import print_function
from ast import walk

import math
import py_trees
import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (ActorDestroy,
                                                                      KeepVelocity,
                                                                      Idle,
                                                                      WalkerFlow)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (InTriggerDistanceToLocation,
                                                                               InTimeToArrivalToLocation,
                                                                               DriveDistance)
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.tools.scenario_helper import get_location_in_distance_from_wp

from srunner.tools.background_manager import LeaveSpaceInFront, RemoveJunctionEntry


def convert_dict_to_transform(actor_dict):
    """
    Convert a JSON string to a Carla.Transform
    """
    location = carla.Location(
        x=float(actor_dict['x']),
        y=float(actor_dict['y']),
        z=float(actor_dict['z'])
    )
    transform = carla.Transform(
        location, carla.Rotation(yaw=float(actor_dict['yaw'])))
    return transform


class PedestrianCrossing(BasicScenario):

    """
    This class holds everything required for a group of natual pedestrians crossing the road.
    The ego vehicle is passing through a road,
    And encounters a group of pedestrians crossing the road.

    This is a single ego vehicle scenario.

    Notice that the pavement is expected to be self._start_distance (30m) ahead of the trigger point.
    """

    def __init__(self, world, ego_vehicles, config,
                 adversary_type='walker.*',
                 randomize=False, debug_mode=False, criteria_enable=True, timeout=60):
        """
        Setup all relevant parameters and create scenario
        """
        self._wmap = CarlaDataProvider.get_map()
        self._trigger_location = config.trigger_points[0].location
        self._reference_waypoint = self._wmap.get_waypoint(
            self._trigger_location)
        self._num_lane_changes = 0

        self._start_distance = 30
        self._retry_dist = 0.4

        self._adversary_type = adversary_type  # blueprint filter of the adversary
        self._adversary_transform = None
        self._collision_wp = None

        self._source_wp = None
        self._start_walker_flow = convert_dict_to_transform(
            config.other_parameters['start_walker_flow'])
        self._sink_transforms = []
        self._sink_transforms_prob = []
        self._spawn_time_interval = 3

        end_walker_flow_list = [
            v for k, v in config.other_parameters.items() if 'end_walker_flow' in k]

        for item in end_walker_flow_list:
            self._sink_transforms.append(convert_dict_to_transform(item))
            self._sink_transforms_prob.append(float(item['p']))

        if 'source_dist_interval' in config.other_parameters:
            self._source_dist_interval = [
                float(config.other_parameters['source_dist_interval']['from']),
                float(config.other_parameters['source_dist_interval']['to'])
            ]
        else:
            self._source_dist_interval = [2, 6]  # m

        # Min distance to the collision location that triggers the adversary [m]
        self._ego_end_distance = 40
        self.timeout = timeout

        super(PedestrianCrossing, self).__init__("PedestrianCrossing",
                                                 ego_vehicles,
                                                 config,
                                                 world,
                                                 debug_mode,
                                                 criteria_enable=criteria_enable)

    def _create_behavior(self):
        """
        After invoking this scenario, pedestrians will wait for the ego
        vehicle to enter trigger distance region,
        the pedestrians starts crossing the road once the condition meets.
        Scenario will end when all pedestrians reach its 
        timeout (collision_duration) or max distance (collision_distance).
        """
        sequence = py_trees.composites.Sequence(name="CrossingActor")
        if self.route_mode:
            sequence.add_child(RemoveJunctionEntry(self._reference_waypoint))

        # Move the adversary
        walker_root = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE, name="AdversaryMove")

        walker_root.add_child(WalkerFlow(
            self._start_walker_flow, self._sink_transforms, self._sink_transforms_prob, self._source_dist_interval))
        walker_root.add_child(Idle(40))

        sequence.add_child(walker_root)

        # Remove everything
        sequence.add_child(DriveDistance(
            self.ego_vehicles[0], self._ego_end_distance, name="EndCondition"))

        return sequence

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        if self.route_mode:
            return []
        return [CollisionTest(self.ego_vehicles[0])]
