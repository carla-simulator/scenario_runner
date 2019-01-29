#!/usr/bin/env python

#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Object crash without prior vehicle action scenario:
The scenario realizes the user controlled ego vehicle
moving along the road and encountering a cyclist ahead.
"""

import py_trees

from ScenarioManager.atomic_scenario_behavior import *
from ScenarioManager.atomic_scenario_criteria import *
from ScenarioManager.timer import TimeOut
from Scenarios.basic_scenario import *


OBJECT_CROSSING_SCENARIOS = [
    "StationaryObjectCrossing",
    "DynamicObjectCrossing"
]


class StationaryObjectCrossing(BasicScenario):

    """
    This class holds everything required for a simple object crash
    without prior vehicle action involving a vehicle and a cyclist.
    The ego vehicle is passing through a road and encounters
    a stationary cyclist.
    """

    category = "ObjectCrossing"

    timeout = 60

    # ego vehicle parameters
    _ego_vehicle_velocity_allowed = 20
    _ego_vehicle_distance_to_other = 35

    def __init__(self, world, ego_vehicle, other_actors, town, randomize=False, debug_mode=False):
        """
        Setup all relevant parameters and create scenario
        """

        super(StationaryObjectCrossing, self).__init__("Stationaryobjectcrossing",
                                                       ego_vehicle,
                                                       other_actors,
                                                       town,
                                                       world,
                                                       debug_mode)

    def _create_behavior(self):
        """
        Only behavior here is to wait
        """
        redundant = TimeOut(self.timeout - 5)
        return redundant

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created
        that is later used in parallel behavior tree.
        """
        criteria = []

        max_velocity_criterion = MaxVelocityTest(
            self.ego_vehicle,
            self._ego_vehicle_velocity_allowed,
            optional=True)
        collision_criterion = CollisionTest(self.ego_vehicle)
        keep_lane_criterion = KeepLaneTest(self.ego_vehicle, optional=True)
        driven_distance_criterion = DrivenDistanceTest(
            self.ego_vehicle,
            self._ego_vehicle_distance_to_other)

        criteria.append(max_velocity_criterion)
        criteria.append(collision_criterion)
        criteria.append(keep_lane_criterion)
        criteria.append(driven_distance_criterion)

        return criteria


class DynamicObjectCrossing(BasicScenario):

    """
    This class holds everything required for a simple object crash
    without prior vehicle action involving a vehicle and a cyclist,
    The ego vehicle is passing through a road,
    And encounters a cyclist crossing the road.
    """

    category = "ObjectCrossing"

    timeout = 60

    # ego vehicle parameters
    _ego_vehicle_velocity_allowed = 10
    _ego_vehicle_distance_driven = 50

    # other vehicle parameters
    _other_actor_target_velocity = 10
    _trigger_distance_from_ego = 35
    _other_actor_max_throttle = 1.0
    _other_actor_max_brake = 1.0

    def __init__(self, world, ego_vehicle, other_actors, town, randomize=False, debug_mode=False):
        """
        Setup all relevant parameters and create scenario
        """

        super(DynamicObjectCrossing, self).__init__("Dynamicobjectcrossing",
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
        then after 60 seconds, a timeout stops the scenario
        """
        # leaf nodes
        trigger_dist = InTriggerDistanceToVehicle(
            self.other_actors[0],
            self.ego_vehicle,
            self._trigger_distance_from_ego)
        start_other_actor = KeepVelocity(
            self.other_actors[0],
            self._other_actor_target_velocity)
        trigger_other = InTriggerRegion(
            self.other_actors[0],
            46, 50,
            128, 129.5)
        stop_other_actor = StopVehicle(
            self.other_actors[0],
            self._other_actor_max_brake)
        timeout_other = TimeOut(10)
        start_vehicle = KeepVelocity(
            self.other_actors[0],
            self._other_actor_target_velocity)
        trigger_other_actor = InTriggerRegion(
            self.other_actors[0],
            46, 50,
            137, 139)
        stop_vehicle = StopVehicle(
            self.other_actors[0],
            self._other_actor_max_brake)
        timeout_other_actor = TimeOut(3)

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
        scenario_sequence.add_child(trigger_dist)
        scenario_sequence.add_child(keep_velocity_other_parallel)
        scenario_sequence.add_child(stop_other_actor)
        scenario_sequence.add_child(timeout_other)
        scenario_sequence.add_child(keep_velocity_other)
        scenario_sequence.add_child(stop_vehicle)
        scenario_sequence.add_child(timeout_other_actor)
        keep_velocity_other_parallel.add_child(start_other_actor)
        keep_velocity_other_parallel.add_child(trigger_other)
        keep_velocity_other.add_child(start_vehicle)
        keep_velocity_other.add_child(trigger_other_actor)

        return root

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = []

        max_velocity_criterion = MaxVelocityTest(
            self.ego_vehicle,
            self._ego_vehicle_velocity_allowed,
            optional=True)
        collision_criterion = CollisionTest(self.ego_vehicle)
        keep_lane_criterion = KeepLaneTest(self.ego_vehicle, optional=True)
        driven_distance_criterion = DrivenDistanceTest(
            self.ego_vehicle, self._ego_vehicle_distance_driven)

        criteria.append(max_velocity_criterion)
        criteria.append(collision_criterion)
        criteria.append(keep_lane_criterion)
        criteria.append(driven_distance_criterion)

        return criteria
