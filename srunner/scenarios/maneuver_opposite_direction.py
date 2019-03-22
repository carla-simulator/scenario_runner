#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Vehicle Manuevering In Opposite Direction:

Vehicle is passing another vehicle in a rural area, in daylight, under clear
weather conditions, at a non-junction and encroaches into another
vehicle traveling in the opposite direction.
"""

import py_trees

from srunner.scenariomanager.atomic_scenario_behavior import *
from srunner.scenariomanager.atomic_scenario_criteria import *
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenarios.basic_scenario import *
from srunner.scenarios.scenario_helper import get_waypoint_in_distance


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

    def __init__(self, world, ego_vehicle, config, randomize=False, debug_mode=False, criteria_enable=True):
        """
        Setup all relevant parameters and create scenario
        """

        self._map = CarlaDataProvider.get_map()
        self._first_vehicle_location = 50
        self._second_vehicle_location = self._first_vehicle_location + 40
        self._ego_vehicle_drive_distance = self._second_vehicle_location * 2
        self._start_distance = self._first_vehicle_location * 0.9
        self._opposite_speed = 30  # km/h
        self._reference_waypoint = self._map.get_waypoint(config.trigger_point.location)

        super(ManeuverOppositeDirection, self).__init__(
            "ManeuverOppositeDirection",
            ego_vehicle,
            config,
            world,
            debug_mode,
            criteria_enable=criteria_enable)

    def _initialize_actors(self, config):
        """
        Custom initialization
        """
        first_vehicle_waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, self._first_vehicle_location)
        second_vehicle_waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, self._second_vehicle_location)
        second_vehicle_waypoint = second_vehicle_waypoint.get_left_lane()

        first_vehicle = CarlaActorPool.request_new_actor('vehicle.nissan.patrol', first_vehicle_waypoint.transform)
        second_vehicle = CarlaActorPool.request_new_actor('vehicle.audi.tt', second_vehicle_waypoint.transform)

        self.other_actors.append(first_vehicle)
        self.other_actors.append(second_vehicle)

    def _create_behavior(self):
        """
        The behavior tree returned by this method is as follows:
        The ego vehicle is trying to pass a leading vehicle in the same lane
        by moving onto the oncoming lane while another vehicle is moving in the
        opposite direction in the oncoming lane.
        """

        # Leaf nodes
        ego_drive_distance = DriveDistance(self.ego_vehicle, self._ego_vehicle_drive_distance)
        waypoint_follower = WaypointFollower(self.other_actors[1], self._opposite_speed)
        opposite_start_trigger = InTriggerDistanceToVehicle(
            self.other_actors[0], self.ego_vehicle, self._start_distance)

        # Non-leaf nodes
        root = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        sequence = py_trees.composites.Sequence()

        # Building tree
        root.add_child(ego_drive_distance)
        root.add_child(sequence)
        sequence.add_child(opposite_start_trigger)
        sequence.add_child(waypoint_follower)

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
