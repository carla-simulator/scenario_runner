#!/usr/bin/env python

#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
"""
Object crash with prior vehicle action scenario:
The scenario realizes the user controlled ego vehicle
moving along the road and encounters a cyclist ahead after taking a right and a left turn.
"""

import py_trees
import carla

from ScenarioManager.atomic_scenario_behavior import *
from ScenarioManager.atomic_scenario_criteria import *
from ScenarioManager.timer import TimeOut
from Scenarios.basic_scenario import *


VEHICLE_TURNING_SCENARIOS = [
    "VehicleTurningRight",
    "VehicleTurningLeft"
]


class VehicleTurningRight(BasicScenario):

    """
    This class holds everything required for a simple object crash
    with prior vehicle action involving a vehicle and a cyclist.
    The ego vehicle is passing through a road and encounters
    a cyclist after taking a right turn.
    """

    category = "VehicleTurning"

    timeout = 90

    # ego vehicle parameters
    _ego_vehicle_velocity_allowed = 30
    _ego_driven_distance = 55
    _ego_acceptable_distance = 35

    # other vehicle parameters
    _other_actor_target_velocity = 10
    _trigger_distance_from_ego = 14
    _other_actor_max_throttle = 1.0
    _other_actor_max_brake = 1.0

    _location_of_collision = carla.Location(x=93.1, y=44.8, z=39)

    def __init__(self, world, ego_vehicle, other_actors, town, randomize=False, debug_mode=False):
        """
        Setup all relevant parameters and create scenario
        """

        super(VehicleTurningRight, self).__init__("VehicleTurningRight",
                                                  ego_vehicle,
                                                  other_actors,
                                                  town,
                                                  world,
                                                  debug_mode)

    def _create_behavior(self):
        """
        After invoking this scenario, cyclist will wait for the user
        controlled vehicle to enter the in the trigger distance region,
        the cyclist starts crossing the road once the condition meets,
        ego vehicle has to avoid the crash after a right turn, but
        continue driving after the road is clear.If this does not happen
        within 90 seconds, a timeout stops the scenario.
        """
        # leaf nodes
        trigger_distance = InTriggerDistanceToVehicle(
            self.other_actors[0],
            self.ego_vehicle,
            self._trigger_distance_from_ego)
        stop_other_actor = StopVehicle(
            self.other_actors[0],
            self._other_actor_max_brake)
        timeout_other_actor = TimeOut(5)
        start_other = KeepVelocity(
            self.other_actors[0],
            self._other_actor_target_velocity)
        trigger_other = InTriggerRegion(
            self.other_actors[0],
            85.5, 86.5,
            41, 43)
        stop_other = StopVehicle(
            self.other_actors[0],
            self._other_actor_max_brake)
        timeout_other = TimeOut(3)
        sync_arrival = SyncArrival(
            self.other_actors[0], self.ego_vehicle, self._location_of_collision)
        sync_arrival_stop = InTriggerDistanceToVehicle(self.other_actors[0],
                                                       self.ego_vehicle,
                                                       6)

        # non leaf nodes
        root = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        scenario_sequence = py_trees.composites.Sequence()
        keep_velocity_other = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        sync_arrival_parallel = py_trees.composites.Parallel(
            "Synchronize arrival times",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)

        # building the tress
        root.add_child(scenario_sequence)
        scenario_sequence.add_child(trigger_distance)
        scenario_sequence.add_child(sync_arrival_parallel)
        scenario_sequence.add_child(stop_other_actor)
        scenario_sequence.add_child(timeout_other_actor)
        scenario_sequence.add_child(keep_velocity_other)
        scenario_sequence.add_child(stop_other)
        scenario_sequence.add_child(timeout_other)
        sync_arrival_parallel.add_child(sync_arrival)
        sync_arrival_parallel.add_child(sync_arrival_stop)
        keep_velocity_other.add_child(start_other)
        keep_velocity_other.add_child(trigger_other)

        return root

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = []

        max_velocity_criterion = MaxVelocityTest(
            self.ego_vehicle,
            self._ego_vehicle_velocity_allowed)
        collision_criterion = CollisionTest(self.ego_vehicle)
        driven_distance_criterion = DrivenDistanceTest(
            self.ego_vehicle,
            self._ego_driven_distance,
            distance_acceptable=self._ego_acceptable_distance)

        criteria.append(max_velocity_criterion)
        criteria.append(collision_criterion)
        criteria.append(driven_distance_criterion)

        return criteria


class VehicleTurningLeft(BasicScenario):

    """
    This class holds everything required for a simple object crash
    with prior vehicle action involving a vehicle and a cyclist.
    The ego vehicle is passing through a road and encounters
    a cyclist after taking a left turn.
    """

    category = "VehicleTurning"

    timeout = 90

    # ego vehicle parameters
    _ego_vehicle_velocity_allowed = 30
    _ego_driven_distance = 60
    _ego_acceptable_distance = 40

    # other vehicle parameters
    _other_actor_target_velocity = 10
    _trigger_distance_from_ego = 23
    _other_actor_max_throttle = 1.0
    _other_actor_max_brake = 1.0

    _location_of_collision = carla.Location(x=88.6, y=75.8, z=38)

    def __init__(self, world, ego_vehicle, other_actors, town, randomize=False, debug_mode=False):
        """
        Setup all relevant parameters and create scenario
        """

        super(VehicleTurningLeft, self).__init__("VehicleTurningLeft",
                                                 ego_vehicle,
                                                 other_actors,
                                                 town,
                                                 world,
                                                 debug_mode)

    def _create_behavior(self):
        """
        After invoking this scenario, cyclist will wait for the user
        controlled vehicle to enter the in the trigger distance region,
        the cyclist starts crossing the road once the condition meets,
        ego vehicle has to avoid the crash after a left turn, but
        continue driving after the road is clear.If this does not happen
        within 90 seconds, a timeout stops the scenario.
        """
        # leaf nodes
        trigger_distance = InTriggerDistanceToVehicle(
            self.other_actors[0],
            self.ego_vehicle,
            self._trigger_distance_from_ego)
        stop_other_actor = StopVehicle(
            self.other_actors[0],
            self._other_actor_max_brake)
        timeout_other_actor = TimeOut(5)
        start_other = KeepVelocity(
            self.other_actors[0],
            self._other_actor_target_velocity)
        trigger_other = InTriggerRegion(
            self.other_actors[0],
            95, 96,
            78, 79)
        stop_other = StopVehicle(
            self.other_actors[0],
            self._other_actor_max_brake)
        timeout_other = TimeOut(3)

        sync_arrival = SyncArrival(
            self.other_actors[0], self.ego_vehicle, self._location_of_collision)
        sync_arrival_stop = InTriggerDistanceToVehicle(self.other_actors[0],
                                                       self.ego_vehicle,
                                                       6)

        # non leaf nodes
        root = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        scenario_sequence = py_trees.composites.Sequence()
        keep_velocity_other = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)

        sync_arrival_parallel = py_trees.composites.Parallel(
            "Synchronize arrival times",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)

        # building the tress
        root.add_child(scenario_sequence)
        scenario_sequence.add_child(trigger_distance)
        scenario_sequence.add_child(sync_arrival_parallel)
        scenario_sequence.add_child(stop_other_actor)
        scenario_sequence.add_child(timeout_other_actor)
        scenario_sequence.add_child(keep_velocity_other)
        scenario_sequence.add_child(stop_other)
        scenario_sequence.add_child(timeout_other)
        sync_arrival_parallel.add_child(sync_arrival)
        sync_arrival_parallel.add_child(sync_arrival_stop)
        keep_velocity_other.add_child(start_other)
        keep_velocity_other.add_child(trigger_other)

        return root

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = []

        max_velocity_criterion = MaxVelocityTest(
            self.ego_vehicle,
            self._ego_vehicle_velocity_allowed)
        collision_criterion = CollisionTest(self.ego_vehicle)
        driven_distance_criterion = DrivenDistanceTest(
            self.ego_vehicle,
            self._ego_driven_distance,
            distance_acceptable=self._ego_acceptable_distance)

        criteria.append(max_velocity_criterion)
        criteria.append(collision_criterion)
        criteria.append(driven_distance_criterion)

        return criteria
