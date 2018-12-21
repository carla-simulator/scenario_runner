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


class VehicleTurningRight(BasicScenario):
    """
    This class holds everything required for a simple object crash
    with prior vehicle action involving a vehicle and a cyclist.
    The ego vehicle is passing through a road and encounters
    a cyclist after taking a right turn.

    Location: Town01
    """

    timeout = 90

    # ego vehicle parameters
    _ego_vehicle_model = 'vehicle.nissan.micra'
    _ego_vehicle_start = carla.Transform(
        carla.Location(x=130, y=55, z=38.5),
        carla.Rotation(yaw=180))
    _ego_vehicle_velocity_allowed = 30
    _ego_driven_distance = 55
    _ego_acceptable_distance = 35

    # other vehicle parameters
    _other_vehicle_model = 'vehicle.diamondback.century'
    _other_vehicle_start = carla.Transform(
        carla.Location(x=95.5, y=42, z=38.5),
        carla.Rotation(yaw=180))
    _other_vehicle_target_velocity = 10
    _trigger_distance_from_ego = 14
    _other_vehicle_max_throttle = 1.0
    _other_vehicle_max_brake = 1.0

    def __init__(self, world, debug_mode=False):
        """
        Setup all relevant parameters and create scenario
        and instantiate scenario manager
        """
        self.other_vehicles = [setup_vehicle(world,
                                             self._other_vehicle_model,
                                             self._other_vehicle_start)]
        self.ego_vehicle = setup_vehicle(world,
                                         self._ego_vehicle_model,
                                         self._ego_vehicle_start,
                                         hero=True)

        super(VehicleTurningRight, self).__init__(
            name="vehicleturningright",
            town="Town01",
            world=world,
            debug_mode=debug_mode)

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
            self.other_vehicles[0],
            self.ego_vehicle,
            self._trigger_distance_from_ego)
        start_other_vehicle = KeepVelocity(
            self.other_vehicles[0],
            self._other_vehicle_target_velocity)
        trigger_other_vehicle = InTriggerRegion(
            self.other_vehicles[0],
            91, 93,
            41, 43)
        stop_other_vehicle = StopVehicle(
            self.other_vehicles[0],
            self._other_vehicle_max_brake)
        timeout_other_vehicle = TimeOut(5)
        start_other = KeepVelocity(
            self.other_vehicles[0],
            self._other_vehicle_target_velocity)
        trigger_other = InTriggerRegion(
            self.other_vehicles[0],
            85.5, 86.5,
            41, 43)
        stop_other = StopVehicle(
            self.other_vehicles[0],
            self._other_vehicle_max_brake)
        timeout_other = TimeOut(20)
        root_timeout = TimeOut(self.timeout)

        # non leaf nodes
        root = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        scenario_sequence = py_trees.composites.Sequence()
        keep_velocity_other_vehicle = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        keep_velocity_other = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)

        # building the tress
        root.add_child(scenario_sequence)
        root.add_child(root_timeout)
        scenario_sequence.add_child(trigger_distance)
        scenario_sequence.add_child(keep_velocity_other_vehicle)
        scenario_sequence.add_child(stop_other_vehicle)
        scenario_sequence.add_child(timeout_other_vehicle)
        scenario_sequence.add_child(keep_velocity_other)
        scenario_sequence.add_child(stop_other)
        scenario_sequence.add_child(timeout_other)
        keep_velocity_other_vehicle.add_child(start_other_vehicle)
        keep_velocity_other_vehicle.add_child(trigger_other_vehicle)
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

    Location: Town01
    """

    timeout = 90

    # ego vehicle parameters
    _ego_vehicle_model = 'vehicle.nissan.micra'
    _ego_vehicle_start = carla.Transform(
        carla.Location(x=130, y=55, z=38.5),
        carla.Rotation(yaw=180))
    _ego_vehicle_velocity_allowed = 30
    _ego_driven_distance = 60
    _ego_acceptable_distance = 40

    # other vehicle parameters
    _other_vehicle_model = 'vehicle.diamondback.century'
    _other_vehicle_start = carla.Transform(
        carla.Location(x=85, y=78.8, z=38.5),
        carla.Rotation(yaw=0))
    _other_vehicle_target_velocity = 10
    _trigger_distance_from_ego = 18
    _other_vehicle_max_throttle = 1.0
    _other_vehicle_max_brake = 1.0

    def __init__(self, world, debug_mode=False):
        """
        Setup all relevant parameters and create scenario
        and instantiate scenario manager
        """
        self.other_vehicles = [setup_vehicle(world,
                                             self._other_vehicle_model,
                                             self._other_vehicle_start)]
        self.ego_vehicle = setup_vehicle(world,
                                         self._ego_vehicle_model,
                                         self._ego_vehicle_start,
                                         hero=True)

        super(VehicleTurningLeft, self).__init__(
            name="vehicleturningleft",
            town="Town01",
            world=world,
            debug_mode=debug_mode)

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
            self.other_vehicles[0],
            self.ego_vehicle,
            self._trigger_distance_from_ego)
        start_other_vehicle = KeepVelocity(
            self.other_vehicles[0],
            self._other_vehicle_target_velocity)
        trigger_other_vehicle = InTriggerRegion(
            self.other_vehicles[0],
            87.5, 89,
            78, 79)
        stop_other_vehicle = StopVehicle(
            self.other_vehicles[0],
            self._other_vehicle_max_brake)
        timeout_other_vehicle = TimeOut(5)
        start_other = KeepVelocity(
            self.other_vehicles[0],
            self._other_vehicle_target_velocity)
        trigger_other = InTriggerRegion(
            self.other_vehicles[0],
            95, 96,
            78, 79)
        stop_other = StopVehicle(
            self.other_vehicles[0],
            self._other_vehicle_max_brake)
        timeout_other = TimeOut(20)
        root_timeout = TimeOut(self.timeout)

        # non leaf nodes
        root = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        scenario_sequence = py_trees.composites.Sequence()
        keep_velocity_other_vehicle = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        keep_velocity_other = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)

        # building the tress
        root.add_child(scenario_sequence)
        root.add_child(root_timeout)
        scenario_sequence.add_child(trigger_distance)
        scenario_sequence.add_child(keep_velocity_other_vehicle)
        scenario_sequence.add_child(stop_other_vehicle)
        scenario_sequence.add_child(timeout_other_vehicle)
        scenario_sequence.add_child(keep_velocity_other)
        scenario_sequence.add_child(stop_other)
        scenario_sequence.add_child(timeout_other)
        keep_velocity_other_vehicle.add_child(start_other_vehicle)
        keep_velocity_other_vehicle.add_child(trigger_other_vehicle)
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
