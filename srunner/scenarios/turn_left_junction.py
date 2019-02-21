#!/usr/bin/env python

#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import carla
import sys

import py_trees

from srunner.scenariomanager.atomic_scenario_behavior import *
from srunner.scenariomanager.atomic_scenario_criteria import *
from srunner.scenarios.basic_scenario import *

TURN_LEFT_JUNCTION_SCENARIOS = [
    "VehicleTurnLeftAtJunction"
]

class VehicleTurnLeftAtJunction(BasicScenario):

    """
    Implementation class for
    'Vehicle turning left at signalized junction scenario,
    Traffic Scenario 08.
    """
    category = "VehicleTurnLeftAtJunction"

    timeout = 60     #Timeout of scenario in seconds

    def __init__(self, world, ego_vehicle, other_actors, town, randomize=False, debug_mode=False):
        """
        Setup all relevant parameters and create scenario
        """
        super(VehicleTurnLeftAtJunction, self).__init__("VehicleTurnLeftAtJunction",
                                                        ego_vehicle,
                                                        other_actors,
                                                        town,
                                                        world,
                                                        debug_mode)
        set_traffic_lights_state(ego_vehicle, carla.TrafficLightState.Green)

    def _create_behavior(self):
        """
        """
        start_trigger_location, _ = get_location_in_distance(self.ego_vehicle, 10)
        start_other_trigger = InTriggerDistanceToLocation(self.ego_vehicle, start_trigger_location, 8)
        move_other_actor = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        trigger_other_actor = KeepVelocity(self.other_actors[0], 25)
        stop_other = InTriggerDistanceToNextIntersection(self.other_actors[0], 1)
        move_other_actor.add_child(trigger_other_actor)
        move_other_actor.add_child(stop_other)
        stop_parallel = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        stop_other_actor = KeepVelocity(self.other_actors[0], 20)
        drive_further = DriveDistance(self.other_actors[0], 25)
        stop_parallel.add_child(stop_other_actor)
        stop_parallel.add_child(drive_further)
        brake_other_actor = StopVehicle(self.other_actors[0], 0.02)
        sequence = py_trees.composites.Sequence()
        sequence.add_child(start_other_trigger)
        sequence.add_child(move_other_actor)
        sequence.add_child(stop_parallel)
        sequence.add_child(brake_other_actor)

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
