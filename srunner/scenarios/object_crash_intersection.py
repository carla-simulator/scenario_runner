#!/usr/bin/env python
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
"""
Object crash with prior vehicle action scenario:
The scenario realizes the user controlled ego vehicle
moving along the road and encounters a cyclist ahead after taking a right and a left turn.
"""

import py_trees
import carla

from srunner.scenariomanager.atomic_scenario_behavior import *
from srunner.scenariomanager.atomic_scenario_criteria import *
from srunner.scenariomanager.timer import TimeOut
from srunner.scenarios.basic_scenario import *


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

    timeout = 60

    # ego vehicle parameters
    _ego_vehicle_velocity_allowed = 30
    _ego_vehicle_distance_driven = 55

    # other vehicle parameters
    _other_actor_target_velocity = 15
    _other_actor_max_brake = 1.0

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
        trigger_distance = InTriggerDistanceToNextIntersection(self.ego_vehicle, 8)
        start_other_actor = AccelerateToVelocity(self.other_actors[0], 1.0, self._other_actor_target_velocity)
        trigger_other = DriveDistance(self.other_actors[0], 3)
        stop_other_actor = StopVehicle(self.other_actors[0], self._other_actor_max_brake)
        timeout_other = TimeOut(5)
        start_actor = AccelerateToVelocity(self.other_actors[0], 1.0, self._other_actor_target_velocity)
        trigger_other_actor = DriveDistance(self.other_actors[0], 6)
        stop_actor = StopVehicle(self.other_actors[0], self._other_actor_max_brake)
        timeout_other_actor = TimeOut(5)

        # non leaf nodes
        root = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        scenario_sequence = py_trees.composites.Sequence()
        keep_velocity_other_parallel = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        keep_velocity_other = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)

        # building tree
        root.add_child(scenario_sequence)
        scenario_sequence.add_child(trigger_distance)
        scenario_sequence.add_child(keep_velocity_other_parallel)
        scenario_sequence.add_child(stop_other_actor)
        scenario_sequence.add_child(timeout_other)
        scenario_sequence.add_child(keep_velocity_other)
        scenario_sequence.add_child(stop_actor)
        scenario_sequence.add_child(timeout_other_actor)
        keep_velocity_other_parallel.add_child(start_other_actor)
        keep_velocity_other_parallel.add_child(trigger_other)
        keep_velocity_other.add_child(start_actor)
        keep_velocity_other.add_child(trigger_other_actor)

        return root

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = []

        max_velocity_criterion = MaxVelocityTest(self.ego_vehicle,
                                                 self._ego_vehicle_velocity_allowed,
                                                 optional=True)
        collision_criterion = CollisionTest(self.ego_vehicle)
        keep_lane_criterion = KeepLaneTest(self.ego_vehicle, optional=True)
        driven_distance_criterion = DrivenDistanceTest(self.ego_vehicle,
                                                       self._ego_vehicle_distance_driven,
                                                       distance_acceptable=35,
                                                       optional=True)

        criteria.append(max_velocity_criterion)
        criteria.append(collision_criterion)
        criteria.append(keep_lane_criterion)
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

    timeout = 60

    # ego vehicle parameters
    _ego_vehicle_velocity_allowed = 30
    _ego_vehicle_distance_driven = 60

    # other vehicle parameters
    _other_actor_target_velocity = 15
    _other_actor_max_brake = 1.0

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
        trigger_distance = InTriggerDistanceToNextIntersection(self.ego_vehicle, 8)
        start_other_actor = AccelerateToVelocity(self.other_actors[0], 1.0, self._other_actor_target_velocity)
        trigger_other = DriveDistance(self.other_actors[0], 3)
        stop_other_actor = StopVehicle(self.other_actors[0], self._other_actor_max_brake)
        timeout_other = TimeOut(5)
        start_actor = AccelerateToVelocity(self.other_actors[0], 1.0, self._other_actor_target_velocity)
        trigger_other_actor = DriveDistance(self.other_actors[0], 6)
        stop_actor = StopVehicle(self.other_actors[0], self._other_actor_max_brake)
        timeout_other_actor = TimeOut(5)

        # non leaf nodes
        root = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        scenario_sequence = py_trees.composites.Sequence()
        keep_velocity_other_parallel = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        keep_velocity_other = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)

        # building tree
        root.add_child(scenario_sequence)
        scenario_sequence.add_child(trigger_distance)
        scenario_sequence.add_child(keep_velocity_other_parallel)
        scenario_sequence.add_child(stop_other_actor)
        scenario_sequence.add_child(timeout_other)
        scenario_sequence.add_child(keep_velocity_other)
        scenario_sequence.add_child(stop_actor)
        scenario_sequence.add_child(timeout_other_actor)
        keep_velocity_other_parallel.add_child(start_other_actor)
        keep_velocity_other_parallel.add_child(trigger_other)
        keep_velocity_other.add_child(start_actor)
        keep_velocity_other.add_child(trigger_other_actor)

        return root

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = []

        max_velocity_criterion = MaxVelocityTest(self.ego_vehicle,
                                                 self._ego_vehicle_velocity_allowed,
                                                 optional=True)
        collision_criterion = CollisionTest(self.ego_vehicle)
        keep_lane_criterion = KeepLaneTest(self.ego_vehicle, optional=True)
        driven_distance_criterion = DrivenDistanceTest(self.ego_vehicle,
                                                       self._ego_vehicle_distance_driven,
                                                       distance_acceptable=35,
                                                       optional=True)

        criteria.append(max_velocity_criterion)
        criteria.append(collision_criterion)
        criteria.append(keep_lane_criterion)
        criteria.append(driven_distance_criterion)

        return criteria
    