#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Vehicle Manuevering In Opposite Direction:

Vehicle is passing another vehicle in a rural area, in daylight, under clear
weather conditions, at a non-junction and encroaches into another
vehicle traveling in the opposite direcntotion.
"""

import random

import py_trees

from srunner.scenariomanager.atomic_scenario_behavior import *
from srunner.scenariomanager.atomic_scenario_criteria import *
from srunner.scenariomanager.timer import TimeOut
from srunner.scenarios.basic_scenario import *
from srunner.scenarios.scenario_helper import get_location_in_distance
from srunner.scenarios.config_parser import ActorConfigurationData


MANEUVER_OPPOSITE_DIRECTION = [
    "ManeuverOppositeDirection"
]


class ManeuverOppositeDirection(BasicScenario):

    """
    Implementation class for Traffic Scenario 06,
    "Vehicle Manuevering In Opposite Direction".
    """

    category = "ManeuverOppositeDirection"
    timeout = 120

    def __init__(self, world, ego_vehicle, other_actors, town, randomize=False, debug_mode=False, config=None):
        """
        Setup all relevant parameters and create scenario
        """

        other_vehicle1_location, _ = get_location_in_distance(ego_vehicle, 100)
        other_vehicle2_location, _ = get_location_in_distance(ego_vehicle, 200)
        other_vehicle1_waypoint = world.get_map().get_waypoint(other_vehicle1_location)
        other_vehicle2_waypoint = world.get_map().get_waypoint(other_vehicle2_location)
        other_vehicle2_waypoint = other_vehicle2_waypoint.get_left_lane()

        parameter_list = []
        parameter_list.append(ActorConfigurationData('vehicle.tesla.model3', other_vehicle1_waypoint.transform))
        parameter_list.append(ActorConfigurationData('vehicle.tesla.model3', other_vehicle2_waypoint.transform))

        super(ManeuverOppositeDirection, self).__init__(
            "FollowVehicle",
            ego_vehicle,
            parameter_list,
            town,
            world,
            debug_mode)

    def _create_behavior(self):
        """
        The behavior tree returned by this method is as follows:
        The ego vehicle is trying to pass a leading vehicle in the same lane
        by moving onto the oncoming lane while another vehicle is moving in the
        opposite direction in the oncoming lane.
        """

        # Leaf nodes
        ego_drive_distance = DriveDistance(self.ego_vehicle, 300)
        start_trigger = InTriggerDistanceToVehicle(self.other_actors[0], self.ego_vehicle, 60)
        waypoint_follower_1 = WaypointFollower(self.other_actors[0], 20)
        waypoint_follower_2 = WaypointFollower(self.other_actors[1], 30)

        # Non-leaf nodes
        root = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        sequence = py_trees.composites.Sequence()
        waypoint_follow_node = py_trees.composites.Parallel()

        # Building tree
        root.add_child(ego_drive_distance)
        root.add_child(sequence)
        sequence.add_child(start_trigger)
        sequence.add_child(waypoint_follow_node)
        waypoint_follow_node.add_child(waypoint_follower_1)
        waypoint_follow_node.add_child(waypoint_follower_2)

        return root

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = []

        collision_criterion = CollisionTest(self.ego_vehicle)
        criteria.append(collision_criterion)

        return criteria

    def __del__(self):
        """
        Remove all actors upon deletion
        """
        self.remove_all_actors()
