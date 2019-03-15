#!/usr/bin/env python

#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import sys

import py_trees

import carla

from srunner.scenariomanager.atomic_scenario_behavior import *
from srunner.scenariomanager.atomic_scenario_criteria import *
from srunner.scenarios.basic_scenario import *
from srunner.scenarios.scenario_helper import *

TURNING_RIGHT_SIGNALIZED_JUNCTION_SCENARIOS = [
    "SignalizedJunctionRightTurn"
]


class SignalizedJunctionRightTurn(BasicScenario):

    """
    Implementation class for Hero
    Vehicle turning right at signalized junction scenario,
    Traffic Scenario 09.
    """

    def __init__(self, world, ego_vehicle, config, randomize=False, debug_mode=False):
        """
        Setup all relevant parameters and create scenario
        """
        self.category = "SignalizedJunctionLeftTurn"
        self.timeout = 80  # Timeout of scenario in seconds
        self._target_vel = 45
        self._drive_distance = 50
        self._trigger_dist_loc = 9
        self._brake_value = 0.5
        self._ego_distance = 10
        self._traffic_light = None

        super(SignalizedJunctionRightTurn, self).__init__("HeroActorTurningRightAtSignalizedJunction",
                                                          ego_vehicle,
                                                          config,
                                                          world,
                                                          debug_mode)

        self._traffic_light = CarlaDataProvider.get_next_traffic_light(self.ego_vehicle, False)
        if self._traffic_light is None:
            print("No traffic light for the given location of the ego vehicle found")
            sys.exit(-1)
        self._traffic_light.set_state(carla.TrafficLightState.Red)
        self._traffic_light.set_red_time(self.timeout)
        # other vehicle's traffic light
        traffic_light_other = CarlaDataProvider.get_next_traffic_light(self.other_actors[0], False)
        if traffic_light_other is None:
            print("No traffic light for the given location of the other vehicle found")
            sys.exit(-1)
        traffic_light_other.set_state(carla.TrafficLightState.Green)
        traffic_light_other.set_green_time(self.timeout)

    def _create_behavior(self):
        """
        Hero vehicle is turning right in an urban area,
        at a signalized intersection, while other actor coming straight
        from left.The hero actor may turn right either before other actor
        passes intersection or later, without any collision.
        After 80 seconds, a timeout stops the scenario.
        """
        start_trigger_loc, _ = get_location_in_distance(self.ego_vehicle, 10)
        start_other_trigger = InTriggerDistanceToLocation(self.ego_vehicle, start_trigger_loc, self._trigger_dist_loc)
        # Selecting straight path at intersection
        target_waypoint = generate_target_waypoint(
            self.other_actors[0].get_world().get_map().get_waypoint(
                self.other_actors[0].get_location()), 0)
        # Generating waypoint list till next intersection
        plan = []
        wp_choice = target_waypoint.next(5.0)
        while len(wp_choice) == 1:
            target_waypoint = wp_choice[0]
            plan.append((target_waypoint, RoadOption.LANEFOLLOW))
            wp_choice = target_waypoint.next(5.0)

        move_actor = WaypointFollower(self.other_actors[0], self._target_vel, plan=plan)
        drive_actor = DriveDistance(self.other_actors[0], self._drive_distance)
        move_actor_parallel = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        move_actor_parallel.add_child(move_actor)
        move_actor_parallel.add_child(drive_actor)
        # stop other actor
        stop_other = StopVehicle(self.other_actors[0], self._brake_value)
        # end condition
        end_condition = DriveDistance(self.ego_vehicle, self._ego_distance)

        # Behavior tree
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
        self._traffic_light = None
        self.remove_all_actors()
