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
a collision.
"""

import random
import sys

import py_trees
import carla

from ScenarioManager.atomic_scenario_behavior import *
from ScenarioManager.atomic_scenario_criteria import *
from ScenarioManager.scenario_manager import Scenario
from ScenarioManager.timer import TimeOut


class FollowLeadingVehicle(object):

    """
    This class holds everything required for a simple "Follow a leading vehicle"
    scenario involving two vehicles.
    """

    name = "FollowVehicle"
    criteria_list = []      # List of evaluation criteria
    timeout = 60            # Timeout of scenario in seconds
    scenario = None

    # ego vehicle parameters
    ego_vehicle = None
    ego_vehicle_model = 'vehicle.carlamotors.carlacola'
    ego_vehicle_start = carla.Transform(
        carla.Location(x=312, y=129, z=39), carla.Rotation(yaw=180))
    ego_vehicle_max_velocity_allowed = 20   # Maximum allowed velocity [m/s]
    ego_vehicle_avg_velocity_expected = 4   # Average expected velocity [m/s]
    ego_vehicle_driven_distance = 110       # Min. driven distance of ego vehicle [m]

    # other vehicle
    other_vehicles = []
    other_vehicle_model = 'vehicle.*'
    other_vehicle_start = carla.Transform(
        carla.Location(x=263, y=129, z=39), carla.Rotation(yaw=180))
    other_vehicle_target_velocity = 15      # Target velocity of other vehicle
    trigger_distance_from_ego_vehicle = 15  # Starting point of other vehicle maneuver
    other_vehicle_max_throttle = 1.0        # Maximum throttle of other vehicle
    other_vehicle_max_brake = 1.0           # Maximum brake of other vehicle

    def __init__(self, world, debug_mode=False):
        """
        Setup all relevant parameters and create scenario
        and instantiate scenario manager
        """
        self.other_vehicles = [self.setup_vehicle(world,
                                                  self.other_vehicle_model,
                                                  self.other_vehicle_start)]
        self.ego_vehicle = self.setup_vehicle(world,
                                              self.ego_vehicle_model,
                                              self.ego_vehicle_start)

        # Setup scenario

        if debug_mode:
            py_trees.logging.level = py_trees.logging.Level.DEBUG

        behavior = self.create_behavior()
        criteria = self.create_test_criteria()
        self.scenario = Scenario(
            behavior, criteria, self.name, self.timeout)

    def setup_vehicle(self, world, model, spawn_point):
        """
        Function to setup the most relevant vehicle parameters,
        incl. spawn point and vehicle model.
        """
        blueprint_library = world.get_blueprint_library()

        # Get vehicle by model
        blueprint = random.choice(blueprint_library.filter(model))
        vehicle = world.try_spawn_actor(blueprint, spawn_point)

        if vehicle is None:
            sys.exit(
                "Error: Unable to spawn vehicle {} at {}".format(model, spawn_point))

        # Let's put the vehicle to drive around.
        vehicle.set_autopilot(False)

        return vehicle

    def create_behavior(self):
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

        # accelerate to target_velocity but at most for 5 seconds
        accelerate = py_trees.composites.Parallel(
            "Accelerate with timeout",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        accelerate_behavior = AccelerateToVelocity(
            self.other_vehicles[0],
            self.other_vehicle_max_throttle,
            self.other_vehicle_target_velocity)
        accelerate_timeout = TimeOut(timeout=5, name="Duration")
        accelerate.add_child(accelerate_behavior)
        accelerate.add_child(accelerate_timeout)

        # keep velocity for 2 seconds
        keep_velocity_for_duration = py_trees.composites.Parallel(
            "Keep velocity for duration",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        keep_velocity = KeepVelocity(
            self.other_vehicles[0],
            self.other_vehicle_target_velocity)
        keep_velocity_duration = TimeOut(timeout=2, name="Duration")
        keep_velocity_for_duration.add_child(keep_velocity)
        keep_velocity_for_duration.add_child(keep_velocity_duration)

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
        sequence.add_child(accelerate)
        sequence.add_child(keep_velocity_for_duration)
        sequence.add_child(stop)
        sequence.add_child(endcondition)

        return sequence

    def create_test_criteria(self):
        """
        Example of a user defined test catalogue.
        This function should be adapted by the user.

        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = []

        max_velocity_criterion = MaxVelocityTest(
            self.ego_vehicle,
            self.ego_vehicle_max_velocity_allowed)
        collision_criterion = CollisionTest(
            self.ego_vehicle)
        keep_lane_criterion = KeepLaneTest(
            self.ego_vehicle)
        driven_distance_criterion = DrivenDistanceTest(
            self.ego_vehicle, self.ego_vehicle_driven_distance)
        avg_velocity_criterion = AverageVelocityTest(
            self.ego_vehicle, self.ego_vehicle_avg_velocity_expected)

        criteria.append(max_velocity_criterion)
        criteria.append(collision_criterion)
        criteria.append(keep_lane_criterion)
        criteria.append(driven_distance_criterion)
        criteria.append(avg_velocity_criterion)

        return criteria

    def __del__(self):
        """
        Cleanup.
        - Removal of the vehicles
        """
        actors = [self.ego_vehicle] + self.other_vehicles
        for actor in actors:
            actor.destroy()
            actor = None
