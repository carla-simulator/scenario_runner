#!/usr/bin/env python

#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Follow leading vehicle scenario:

The scenario realizes a common driving behavior, in which the
user-controlled ego vehicle follows a leading car driving down
a given road. At some point the leading car has to slow down and
finally stop. The ego vehicle has to react accordingly to avoid
a collision. The scenario ends either via a timeout, or if the ego
vehicle stopped close enough to the leading vehicle
"""

import random

import py_trees
import carla

from ScenarioManager.atomic_scenario_behavior import *
from ScenarioManager.atomic_scenario_criteria import *
from Scenarios.basic_scenario import *


class FollowLeadingVehicle(BasicScenario):

    """
    This class holds everything required for a simple "Follow a leading vehicle"
    scenario involving two vehicles.
    """

    timeout = 60            # Timeout of scenario in seconds

    # ego vehicle parameters
    ego_vehicle_model = 'vehicle.carlamotors.carlacola'
    ego_vehicle_start_x = 107
    ego_vehicle_start = carla.Transform(
        carla.Location(x=ego_vehicle_start_x, y=133, z=39), carla.Rotation(yaw=0))
    ego_max_velocity_allowed = 20   # Maximum allowed velocity [m/s]
    ego_avg_velocity_expected = 4   # Average expected velocity [m/s]
    ego_distance_to_other = 50      # Min. driven distance of ego vehicle [m]

    # other vehicle
    other_vehicle_model = 'vehicle.*'
    other_vehicle_start_x = ego_vehicle_start_x + ego_distance_to_other
    other_vehicle_start = carla.Transform(
        carla.Location(x=other_vehicle_start_x, y=133.5, z=39), carla.Rotation(yaw=0))
    other_vehicle_target_velocity = 15                  # Target velocity of other vehicle
    trigger_distance_from_ego = 15              # Starting point of other vehicle maneuver
    other_vehicle_max_throttle = 1.0                    # Maximum throttle of other vehicle
    other_vehicle_max_brake = 1.0                       # Maximum brake of other vehicle
    other_vehicle_distance = 50 + \
        random.randint(0, 50)  # Distance the other vehicle should drive

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

        super(FollowLeadingVehicle, self).__init__(name="FollowVehicle",
                                                   debug_mode=debug_mode)

    def _create_behavior(self):
        """
        Example of a user defined scenario behavior. This function should be
        adapted by the user for other scenarios.

        The scenario defined after is a "follow leading vehicle" scenario. After
        invoking this scenario, it will wait for the user controlled vehicle to
        enter the start region, then make a traffic participant to accelerate
        until reaching a certain speed, then keep this speed for 2 seconds,
        before initiating a stopping maneuver. Finally, the user-controlled
        vehicle has to reach a target region.
        If this does not happen within 60 seconds, a timeout stops the scenario
        """

        # start condition
        startcondition = InTimeToArrivalToLocation(
            self.ego_vehicle,
            4,
            self.other_vehicles[0].get_location(),
            name="Waiting for start position")

        # get to velocity and keep it for certain distance
        keep_velocity_for_distance = py_trees.composites.Parallel(
            "Keep velocity for distance",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        keep_velocity = KeepVelocity(
            self.other_vehicles[0],
            self.other_vehicle_target_velocity)
        keep_velocity_distance = DriveDistance(
            self.other_vehicles[0],
            self.other_vehicle_distance,
            name="Distance")
        keep_velocity_for_distance.add_child(keep_velocity)
        keep_velocity_for_distance.add_child(keep_velocity_distance)

        # stop vehicle
        stop = StopVehicle(
            self.other_vehicles[0],
            self.other_vehicle_max_brake)

        # end condition
        endcondition = py_trees.composites.Parallel(
            "Waiting for end position",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL)
        endcondition_part1 = InTriggerDistanceToVehicle(
            self.other_vehicles[0],
            self.ego_vehicle,
            distance=10,
            name="FinalDistance")
        endcondition_part2 = TriggerVelocity(
            self.ego_vehicle, target_velocity=0, name="FinalSpeed")
        endcondition.add_child(endcondition_part1)
        endcondition.add_child(endcondition_part2)

        # Build behavior tree
        sequence = py_trees.composites.Sequence("Sequence Behavior")
        sequence.add_child(startcondition)
        sequence.add_child(keep_velocity_for_distance)
        sequence.add_child(stop)
        sequence.add_child(endcondition)

        return sequence

    def _create_test_criteria(self):
        """
        Example of a user defined test catalogue.
        This function should be adapted by the user.

        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = []

        max_velocity_criterion = MaxVelocityTest(
            self.ego_vehicle,
            self.ego_max_velocity_allowed)
        collision_criterion = CollisionTest(self.ego_vehicle)
        keep_lane_criterion = KeepLaneTest(self.ego_vehicle)
        driven_distance_criterion = DrivenDistanceTest(
            self.ego_vehicle,
            self.ego_distance_to_other + self.other_vehicle_distance - 10)
        avg_velocity_criterion = AverageVelocityTest(
            self.ego_vehicle, self.ego_avg_velocity_expected)

        criteria.append(max_velocity_criterion)
        criteria.append(collision_criterion)
        criteria.append(keep_lane_criterion)
        criteria.append(driven_distance_criterion)
        criteria.append(avg_velocity_criterion)

        # Add the collision and lane checks for all vehicles as well
        for vehicle in self.other_vehicles:
            collision_criterion = CollisionTest(vehicle)
            keep_lane_criterion = KeepLaneTest(vehicle)
            criteria.append(collision_criterion)
            criteria.append(keep_lane_criterion)

        return criteria


class FollowLeadingVehicleWithObstacle(BasicScenario):

    """
    This class holds a scenario similar to FollowLeadingVehicle
    but there is a (hidden) obstacle in front of the leading vehicle
    """

    timeout = 60            # Timeout of scenario in seconds

    # ego vehicle parameters
    ego_vehicle_model = 'vehicle.carlamotors.carlacola'
    ego_vehicle_start_x = 107
    ego_vehicle_start = carla.Transform(
        carla.Location(x=ego_vehicle_start_x, y=133, z=39), carla.Rotation(yaw=0))
    ego_max_velocity_allowed = 20   # Maximum allowed velocity [m/s]
    ego_avg_velocity_expected = 4   # Average expected velocity [m/s]
    ego_distance_to_other = 50      # Min. driven distance of ego vehicle [m]

    # other vehicle
    other_vehicles = []
    other_vehicle_model = 'vehicle.volkswagen.t2'
    other_vehicle_start_x = ego_vehicle_start_x + ego_distance_to_other
    other_vehicle_start = carla.Transform(
        carla.Location(x=other_vehicle_start_x, y=133.5, z=39), carla.Rotation(yaw=0))
    other_vehicle_target_velocity = 15      # Target velocity of other vehicle
    trigger_distance_from_ego = 15          # Starting point of other vehicle maneuver
    other_vehicle_max_throttle = 1.0        # Maximum throttle of other vehicle
    other_vehicle_max_brake = 1.0           # Maximum brake of other vehicle
    other_vehicle_distance = 40             # Distance the other vehicle should drive

    other_vehicle_model_no2 = 'vehicle.gazelle.omafiets'
    other_vehicle_start_x_no2 = other_vehicle_start_x + \
        10 + random.randint(other_vehicle_distance, 80)
    other_vehicle_start_no2 = carla.Transform(
        carla.Location(x=other_vehicle_start_x_no2, y=133.5, z=39), carla.Rotation(yaw=0))

    def __init__(self, world, debug_mode=False):
        """
        Setup all relevant parameters and create scenario
        and instantiate scenario manager
        """
        self.other_vehicles = [setup_vehicle(world,
                                             self.other_vehicle_model,
                                             self.other_vehicle_start),
                               setup_vehicle(world,
                                             self.other_vehicle_model_no2,
                                             self.other_vehicle_start_no2)]
        self.ego_vehicle = setup_vehicle(world,
                                         self.ego_vehicle_model,
                                         self.ego_vehicle_start,
                                         hero=True)

        super(FollowLeadingVehicleWithObstacle, self).__init__(
            name="FollowLeadingVehicleWithObstacle",
            debug_mode=debug_mode)

    def _create_behavior(self):
        """
        Example of a user defined scenario behavior. This function should be
        adapted by the user for other scenarios.

        The scenario defined after is a "follow leading vehicle" scenario. After
        invoking this scenario, it will wait for the user controlled vehicle to
        enter the start region, then make a traffic participant to accelerate
        until reaching a certain speed, then keep this speed for 2 seconds,
        before initiating a stopping maneuver. Finally, the user-controlled
        vehicle has to reach a target region.
        If this does not happen within 60 seconds, a timeout stops the scenario
        """

        # start condition
        startcondition = InTimeToArrivalToLocation(
            self.ego_vehicle,
            4,
            self.other_vehicles[0].get_location(),
            name="Waiting for start position")

        # get to velocity and keep it for certain distance
        keep_velocity_for_distance = py_trees.composites.Parallel(
            "Keep velocity for duration",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        keep_velocity = KeepVelocity(
            self.other_vehicles[0],
            self.other_vehicle_target_velocity)
        keep_velocity_distance = DriveDistance(
            self.other_vehicles[0],
            self.other_vehicle_distance,
            name="Distance")
        keep_velocity_for_distance.add_child(keep_velocity)
        keep_velocity_for_distance.add_child(keep_velocity_distance)

        # use autopilot
        use_autopilot = py_trees.composites.Parallel(
            "Use autopilot for distance",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        autopilot = UseAutoPilot(self.other_vehicles[0])
        endcondition = InTriggerDistanceToVehicle(
            self.other_vehicles[1],
            self.other_vehicles[0],
            distance=8,
            name="AutoPilotEnd")
        use_autopilot.add_child(autopilot)
        use_autopilot.add_child(endcondition)

        # end condition
        endcondition = py_trees.composites.Parallel(
            "Waiting for end position",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL)
        endcondition_part1 = InTriggerDistanceToVehicle(
            self.other_vehicles[0],
            self.ego_vehicle,
            distance=10,
            name="FinalDistance")
        endcondition_part2 = TriggerVelocity(
            self.ego_vehicle, target_velocity=0, name="FinalSpeed")
        endcondition.add_child(endcondition_part1)
        endcondition.add_child(endcondition_part2)

        # Build behavior tree
        sequence = py_trees.composites.Sequence("Sequence Behavior")
        sequence.add_child(startcondition)
        sequence.add_child(keep_velocity_for_distance)
        sequence.add_child(use_autopilot)
        sequence.add_child(endcondition)

        return sequence

    def _create_test_criteria(self):
        """
        Example of a user defined test catalogue.
        This function should be adapted by the user.

        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = []

        max_velocity_criterion = MaxVelocityTest(
            self.ego_vehicle,
            self.ego_max_velocity_allowed)
        collision_criterion = CollisionTest(self.ego_vehicle)
        keep_lane_criterion = KeepLaneTest(self.ego_vehicle)
        driven_distance_criterion = DrivenDistanceTest(
            self.ego_vehicle,
            self.ego_distance_to_other + self.other_vehicle_distance - 10)
        avg_velocity_criterion = AverageVelocityTest(
            self.ego_vehicle, self.ego_avg_velocity_expected)

        criteria.append(max_velocity_criterion)
        criteria.append(collision_criterion)
        criteria.append(keep_lane_criterion)
        criteria.append(driven_distance_criterion)
        criteria.append(avg_velocity_criterion)

        # Add the collision and lane checks for all vehicles as well
        for vehicle in self.other_vehicles:
            collision_criterion = CollisionTest(vehicle)
            keep_lane_criterion = KeepLaneTest(vehicle)
            criteria.append(collision_criterion)
            criteria.append(keep_lane_criterion)

        return criteria
