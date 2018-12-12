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
import carla

from ScenarioManager.atomic_scenario_behavior import *
from ScenarioManager.atomic_scenario_criteria import *
from ScenarioManager.timer import TimeOut
from Scenarios.basic_scenario import *


class StationaryObjectCrossing(BasicScenario):

    """
    This class holds everything required for a simple object crash
    without prior vehicle action involving a vehicle and a cyclist.
    The ego vehicle is passing through a road and encounters
    a stationary cyclist.
    """

    timeout = 60

    # ego vehicle parameters
    ego_vehicle_model = 'vehicle.carlamotors.carlacola'
    ego_vehicle_start_x = 110
    ego_vehicle_start = carla.Transform(
        carla.Location(x=ego_vehicle_start_x, y=129, z=1),
        carla.Rotation(yaw=180))
    ego_vehicle_velocity_allowed = 20
    ego_vehicle_distance_to_other = 30

    # other vehicle parameters
    other_vehicle_model = 'vehicle.diamondback.century'
    other_vehicle_start_x = 70
    other_vehicle_start = carla.Transform(
        carla.Location(x=other_vehicle_start_x, y=129, z=0),
        carla.Rotation(yaw=270))

    def __init__(self, world, debug_mode=False):
        """
        Setup all relevant parameters and create scenario
        and instantiate scenario manager
        """
        self.other_vehicles = [setup_vehicle(world,
                                             self.other_vehicle_model,
                                             self.other_vehicle_start)]
        self.ego_vehicle = setup_vehicle(world,
                                         self.ego_vehicle_model,
                                         self.ego_vehicle_start,
                                         hero=True)

        super(StationaryObjectCrossing, self).__init__(
            name="stationaryobjectcrossing",
            debug_mode=debug_mode)

    def _create_behavior(self):
        """
        Example of a user defined scenario behavior. This function should be
        adapted by the user for other scenarios.
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
            self.ego_vehicle_velocity_allowed)
        collision_criterion = CollisionTest(self.ego_vehicle)
        keep_lane_criterion = KeepLaneTest(self.ego_vehicle)
        driven_distance_criterion = DrivenDistanceTest(
            self.ego_vehicle,
            self.ego_vehicle_distance_to_other)

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

    timeout = 60

    # ego vehicle parameters
    ego_vehicle_model = 'vehicle.carlamotors.carlacola'
    ego_vehicle_start_x = 90
    ego_vehicle_start = carla.Transform(
        carla.Location(x=ego_vehicle_start_x, y=129, z=1),
        carla.Rotation(yaw=180))
    ego_vehicle_velocity_allowed = 10
    ego_vehicle_distance_driven = 30

    # other vehicle parameters
    other_vehicles = []
    other_vehicle_model = 'vehicle.diamondback.century'
    other_vehicle_start_x = 47.5
    other_vehicle_start = carla.Transform(
        carla.Location(x=other_vehicle_start_x, y=124, z=1),
        carla.Rotation(yaw=90))
    other_vehicle_target_velocity = 10
    trigger_distance_from_ego_vehicle = 35
    other_vehicle_max_throttle = 1.0
    other_vehicle_max_brake = 1.0

    def __init__(self, world, debug_mode=False):
        """
        Setup all relevant parameters and create scenario
        and instantiate scenario manager
        """
        self.other_vehicles = [setup_vehicle(world,
                                             self.other_vehicle_model,
                                             self.other_vehicle_start)]
        self.ego_vehicle = setup_vehicle(world,
                                         self.ego_vehicle_model,
                                         self.ego_vehicle_start,
                                         hero=True)

        super(DynamicObjectCrossing, self).__init__(
            name="dynamicobjectcrossing",
            debug_mode=debug_mode)

    def _create_behavior(self):
        """
        After invoking this scenario, cyclist will wait for the user
        controlled vehicle to enter the in the trigger distance region,
        the cyclist starts crossing the road once the condition meets,
        then after 60 seconds, a timeout stops the scenario

        """
        # leaf nodes
        trigger_dist = InTriggerDistanceToVehicle(
            self.other_vehicles[0],
            self.ego_vehicle,
            self.trigger_distance_from_ego_vehicle)
        start_other_vehicle = KeepVelocity(
            self.other_vehicles[0],
            self.other_vehicle_target_velocity)
        trigger_other = InTriggerRegion(
            self.other_vehicles[0],
            46, 50,
            128, 129.5)
        stop_other_vehicle = StopVehicle(
            self.other_vehicles[0],
            self.other_vehicle_max_brake)
        timeout_other = TimeOut(10)
        start_vehicle = KeepVelocity(
            self.other_vehicles[0],
            self.other_vehicle_target_velocity)
        trigger_other_vehicle = InTriggerRegion(
            self.other_vehicles[0],
            46, 50,
            137, 139)
        stop_vehicle = StopVehicle(
            self.other_vehicles[0],
            self.other_vehicle_max_brake)
        timeout_other_vehicle = TimeOut(5)
        root_timeout = TimeOut(self.timeout)

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
        root.add_child(root_timeout)
        scenario_sequence.add_child(trigger_dist)
        scenario_sequence.add_child(keep_velocity_other_parallel)
        scenario_sequence.add_child(stop_other_vehicle)
        scenario_sequence.add_child(timeout_other)
        scenario_sequence.add_child(keep_velocity_other)
        scenario_sequence.add_child(stop_vehicle)
        scenario_sequence.add_child(timeout_other_vehicle)
        keep_velocity_other_parallel.add_child(start_other_vehicle)
        keep_velocity_other_parallel.add_child(trigger_other)
        keep_velocity_other.add_child(start_vehicle)
        keep_velocity_other.add_child(trigger_other_vehicle)

        return root

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = []

        max_velocity_criterion = MaxVelocityTest(
            self.ego_vehicle,
            self.ego_vehicle_velocity_allowed)
        collision_criterion = CollisionTest(self.ego_vehicle)
        keep_lane_criterion = KeepLaneTest(self.ego_vehicle)
        driven_distance_criterion = DrivenDistanceTest(
            self.ego_vehicle, self.ego_vehicle_distance_driven)

        criteria.append(max_velocity_criterion)
        criteria.append(collision_criterion)
        criteria.append(keep_lane_criterion)
        criteria.append(driven_distance_criterion)

        return criteria
