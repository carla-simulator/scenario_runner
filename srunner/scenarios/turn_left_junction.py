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

TURN_LEFT_JUNCTION_SCENARIOS = [
    "VehicleTurnLeftAtJunction"
]

class VehicleTurnLeftAtJunction(BasicScenario):

    """
    Implementation class for
    'Vehicle turning left at signalized junction scenario,
    Traffic Scenario 08.
    """
    def __init__(self, world, ego_vehicle, other_actors, town, randomize=False, debug_mode=False):
        """
        Setup all relevant parameters and create scenario
        """
        self.category = "VehicleTurnLeftAtJunction"
        self.timeout = 60     #Timeout of scenario in seconds
        self._target_vel = 25
        self._drive_distance = 30
        self._trigger_distance = 1
        self._trigger_dist_loc = 8
        self._brake_value = 0.02
        self._ego_distance = 20
        self._traffic_light = None
        super(VehicleTurnLeftAtJunction, self).__init__("VehicleTurnLeftAtJunction",
                                                        ego_vehicle,
                                                        other_actors,
                                                        town,
                                                        world,
                                                        debug_mode)

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
        """
        start_trigger_loc, _ = get_location_in_distance(self.ego_vehicle, 10)
        start_other_trigger = InTriggerDistanceToLocation(self.ego_vehicle, start_trigger_loc, self._trigger_dist_loc)
        move_other_actor = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        trigger_other_actor = WaypointFollower(self.other_actors[0], self._target_vel)
        stop_other = InTriggerDistanceToNextIntersection(self.other_actors[0], self._trigger_distance)
        move_other_actor.add_child(trigger_other_actor)
        move_other_actor.add_child(stop_other)
        stop_parallel = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        stop_other_actor = KeepVelocity(self.other_actors[0], self._target_vel)
        drive_further = DriveDistance(self.other_actors[0], self._drive_distance)
        stop_parallel.add_child(stop_other_actor)
        stop_parallel.add_child(drive_further)
        brake_other_actor = StopVehicle(self.other_actors[0], self._brake_value)
        end_condition = DriveDistance(self.ego_vehicle, self._ego_distance)
        sequence = py_trees.composites.Sequence()
        sequence.add_child(start_other_trigger)
        sequence.add_child(move_other_actor)
        sequence.add_child(stop_parallel)
        sequence.add_child(brake_other_actor)
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

        for vehicle in self.other_actors:
            collison_criteria = CollisionTest(vehicle)

        return criteria
