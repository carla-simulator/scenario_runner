#!/usr/bin/env python

#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import sys

import py_trees

import carla

from srunner.scenariomanager.atomic_scenario_behavior import *
from srunner.scenariomanager.atomic_scenario_criteria import *
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenarios.basic_scenario import *
from srunner.scenarios.scenario_helper import *

TURN_LEFT_SIGNALIZED_JUNCTION_SCENARIOS = [
    "SignalizedJunctionLeftTurn"
]


class SignalizedJunctionLeftTurn(BasicScenario):

    """
    Implementation class for Hero
    Vehicle turning left at signalized junction scenario,
    Traffic Scenario 08.
    """

    def __init__(self, world, ego_vehicle, config, randomize=False, debug_mode=False, criteria_enable=True):
        """
        Setup all relevant parameters and create scenario
        """
        self.category = "SignalizedJunctionLeftTurn"
        self.timeout = 80  # Timeout of scenario in seconds
        self._target_vel = 35
        self._brake_value = 0.02
        self._dist_to_intersection = 4
        self._drive_distance = 50
        self._ego_distance = 20
        self._traffic_light = None

        super(SignalizedJunctionLeftTurn, self).__init__("SignalizedJunctionLeftTurn",
                                                         ego_vehicle,
                                                         config,
                                                         world,
                                                         debug_mode,
                                                         criteria_enable=criteria_enable)

        self._traffic_light = CarlaDataProvider.get_next_traffic_light(self.ego_vehicle, False)
        if self._traffic_light is None:
            print("No traffic light for the given location of the ego vehicle found")
            sys.exit(-1)
        self._traffic_light.set_state(carla.TrafficLightState.Green)
        self._traffic_light.set_green_time(self.timeout)
        # other vehicle's traffic light
        traffic_light_other = CarlaDataProvider.get_next_traffic_light(self.other_actors[0], False)
        if traffic_light_other is None:
            print("No traffic light for the given location of the other vehicle found")
            sys.exit(-1)
        traffic_light_other.set_state(carla.TrafficLightState.Green)
        traffic_light_other.set_green_time(self.timeout)

    def _create_behavior(self):
        """
        Hero vehicle is turning left in an urban area,
        at a signalized intersection and cuts across the path of another vehicle
        coming straight crossing from an opposite direction,
        After 80 seconds, a timeout stops the scenario.
        """
        _location = get_crossing_point(self.ego_vehicle)
        start_other_trigger = InTriggerDistanceToLocation(self.ego_vehicle, _location, self._dist_to_intersection)
        # Selecting straight path at intersection
        target_waypoint = generate_target_waypoint(
            CarlaDataProvider.get_map().get_waypoint(self.other_actors[0].get_location()), 0)
        # Generating waypoint list till next intersection
        plan = []
        wp_choice = target_waypoint.next(5.0)
        while len(wp_choice) == 1:
            target_waypoint = wp_choice[0]
            plan.append((target_waypoint, RoadOption.LANEFOLLOW))
            wp_choice = target_waypoint.next(5.0)

        move_actor = WaypointFollower(self.other_actors[0], self._target_vel, plan=plan)
        drive_actor = DriveDistance(self.other_actors[0], self._drive_distance)
        stop_other = StopVehicle(self.other_actors[0], self._brake_value)
        end_condition = DriveDistance(self.ego_vehicle, self._ego_distance)
        move_actor_parallel = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        move_actor_parallel.add_child(move_actor)
        move_actor_parallel.add_child(drive_actor)

        sequence = py_trees.composites.Sequence()
        sequence.add_child(start_other_trigger)
        sequence.add_child(move_actor_parallel)
        sequence.add_child(stop_other)
        sequence.add_child(end_condition)

        return sequence

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = []

        collison_criteria = CollisionTest(self.ego_vehicle)
        criteria.append(collison_criteria)

        return criteria

    def __del__(self):
        """
        Remove all actors upon deletion
        """
        self.remove_all_actors()
