#!/usr/bin/env python

# Copyright (c) 2018-2019 Intel Labs.
# authors: Fabian Oboril (fabian.oboril@intel.com)
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
import carla

import random

import py_trees

from srunner.scenariomanager.atomic_scenario_behavior import *
from srunner.scenariomanager.atomic_scenario_criteria import *
from srunner.scenariomanager.timer import TimeOut
from srunner.scenarios.basic_scenario import *
from srunner.scenarios.scenario_helper import get_location_in_distance
from srunner.scenarios.config_parser import ActorConfigurationData


FOLLOW_LEADING_VEHICLE_SCENARIOS = [
    "FollowLeadingVehicle",
    "FollowLeadingVehicleWithObstacle"
]


class FollowLeadingVehicle(BasicScenario):

    """
    This class holds everything required for a simple "Follow a leading vehicle"
    scenario involving two vehicles.
    """

    category = "FollowLeadingVehicle"

    timeout = 120            # Timeout of scenario in seconds

    # ego vehicle parameters
    _ego_max_velocity_allowed = 20        # Maximum allowed velocity [m/s]
    _ego_avg_velocity_expected = 4        # Average expected velocity [m/s]
    _ego_other_distance_start = 4         # time to arrival that triggers scenario starts

    # other vehicle
    _other_actor_max_brake = 1.0                  # Maximum brake of other actor
    _other_actor_stop_in_front_intersection = 30  # Stop ~30m in front of intersection

    def __init__(self, world, ego_vehicle, config, randomize=False, debug_mode=False):
        """
        Setup all relevant parameters and create scenario

        If randomize is True, the scenario parameters are randomized
        """

        self.category = "FollowLeadingVehicle"

        self.timeout = 120            # Timeout of scenario in seconds

        # ego vehicle parameters
        self._ego_other_distance_start = 4         # time to arrival that triggers scenario starts

        # other vehicle
        self._other_actor_max_brake = 1.0                  # Maximum brake of other actor
        self._other_actor_stop_in_front_intersection = 30  # Stop ~30m in front of intersection

        parameter_list = []

        #   Other vehicle 1
        model = 'vehicle.tesla.model3'
        spawn_location, _ = get_location_in_distance(ego_vehicle, 20)
        spawn_location.z += 10
        spawn_waypoint = ego_vehicle.get_world().get_map().get_waypoint(spawn_location)
        spawn_transform = carla.Transform(spawn_location, spawn_waypoint.transform.rotation)
        parameter_list.append(ActorConfigurationData(model, spawn_transform))

        super(FollowLeadingVehicle, self).__init__("FollowVehicle",
                                                   ego_vehicle,
                                                   config,
                                                   world,
                                                   debug_mode)

        if randomize:
            self._ego_other_distance_start = random.randint(4, 8)

            # Example code how to randomize start location
            # distance = random.randint(20, 80)
            # new_location, _ = get_location_in_distance(self.ego_vehicle, distance)
            # waypoint = world.get_map().get_waypoint(new_location)
            # waypoint.transform.location.z += 39
            # self.other_actors[0].set_transform(waypoint.transform)

    @staticmethod
    def initialize_actors(ego_vehicle):
        """
        This method returns the list of participant actors and their initial positions for the scenario
        """
        parameter_list = []

        #   Other vehicle 1
        model = 'vehicle.tesla.model3'
        spawn_location, _ = get_location_in_distance(ego_vehicle, 50)
        spawn_location.z = 40
        print spawn_location
        spawn_transform = carla.Transform(
            spawn_location,
            ego_vehicle.get_transform().rotation)
        parameter_list.append((model, spawn_transform))

        return parameter_list

    def _create_behavior(self):
        """
        The scenario defined after is a "follow leading vehicle" scenario. After
        invoking this scenario, it will wait for the user controlled vehicle to
        enter the start region, then make the other actor to drive until reaching
        the next intersection. Finally, the user-controlled vehicle has to be close
        enough to the other actor to end the scenario.
        If this does not happen within 60 seconds, a timeout stops the scenario
        """

        hand_brake_apply = HandBrakeVehicle(self.other_actors[0], True, name="HandBraking")

        # start condition
        startcondition = py_trees.composites.Parallel(
            "Waiting for start position",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)

        startcondition.add_child(InTimeToArrivalToLocation(self.ego_vehicle,
                                                           self._ego_other_distance_start,
                                                           self.other_actors[0].get_location()))
        startcondition.add_child(InTriggerDistanceToVehicle(self.ego_vehicle,
                                                            self.other_actors[0],
                                                            15))

        hand_brake_release = HandBrakeVehicle(self.other_actors[0], False, name="ReleasingHandBrake")

        # let the other actor drive until next intersection
        # @todo: We should add some feedback mechanism to respond to ego_vehicle behavior
        driving_to_next_intersection = py_trees.composites.Parallel(
            "DrivingTowardsIntersection",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)

        driving_to_next_intersection.add_child(WaypointFollower(self.other_actors[0], 55))
        driving_to_next_intersection.add_child(InTriggerDistanceToNextIntersection(
            self.other_actors[0], self._other_actor_stop_in_front_intersection))

        # stop vehicle
        stop = StopVehicle(self.other_actors[0], self._other_actor_max_brake)

        # end condition
        endcondition = py_trees.composites.Parallel("Waiting for end position",
                                                    policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL)
        endcondition_part1 = InTriggerDistanceToVehicle(self.other_actors[0],
                                                        self.ego_vehicle,
                                                        distance=20,
                                                        name="FinalDistance")
        endcondition_part2 = StandStill(self.ego_vehicle, name="StandStill")
        endcondition.add_child(endcondition_part1)
        endcondition.add_child(endcondition_part2)

        # Build behavior tree
        sequence = py_trees.composites.Sequence("Sequence Behavior")
        sequence.add_child(hand_brake_apply)
        sequence.add_child(startcondition)
        sequence.add_child(hand_brake_release)
        sequence.add_child(driving_to_next_intersection)
        sequence.add_child(stop)
        sequence.add_child(endcondition)

        return sequence

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = []

        collision_criterion = CollisionTest(self.ego_vehicle)
        keep_lane_criterion = KeepLaneTest(self.ego_vehicle)

        criteria.append(collision_criterion)
        criteria.append(keep_lane_criterion)

        return criteria

    def __del__(self):
        """
        Remove all actors upon deletion
        """
        self.remove_all_actors()


class FollowLeadingVehicleWithObstacle(BasicScenario):

    """
    This class holds a scenario similar to FollowLeadingVehicle
    but there is an obstacle in front of the leading vehicle
    """

    category = "FollowLeadingVehicle"

    timeout = 120            # Timeout of scenario in seconds

    # ego vehicle parameters
    _ego_max_velocity_allowed = 20   # Maximum allowed velocity [m/s]
    _ego_avg_velocity_expected = 4   # Average expected velocity [m/s]
    _ego_other_distance_start = 4    # time to arrival that triggers scenario starts

    # other vehicle
    _other_actor_max_brake = 1.0                  # Maximum brake of other vehicle
    _other_actor_stop_in_front_intersection = 30  # Stop ~30m in front of intersection

    def __init__(self, world, ego_vehicle, config, randomize=False, debug_mode=False):
        """
        Setup all relevant parameters and create scenario
        """
        self.category = "FollowLeadingVehicle"

        self.timeout = 120            # Timeout of scenario in seconds

        # ego vehicle parameters
        self._ego_other_distance_start = 4    # time to arrival that triggers scenario starts

        # other vehicle
        self._other_actor_max_brake = 1.0                  # Maximum brake of other vehicle

        parameter_list = []

        #   Other vehicle 1
        model = 'vehicle.tesla.model3'
        model_1 = 'vehicle.diamondback.century'

        spawn_location, _ = get_location_in_distance(ego_vehicle, 30)
        spawn_waypoint = ego_vehicle.get_world().get_map().get_waypoint(spawn_location)
        spawn_transform = carla.Transform(spawn_location, spawn_waypoint.transform.rotation)

        spawn_location_1, _ = get_location_in_distance(ego_vehicle, 100)
        spawn_waypoint_1 = ego_vehicle.get_world().get_map().get_waypoint(spawn_location_1)
        yaw_1 = spawn_waypoint_1.transform.rotation.yaw + 90
        spawn_transform_1 = carla.Transform(spawn_location_1, carla.Rotation(
                                            pitch=spawn_waypoint_1.transform.rotation.pitch,
                                            yaw=yaw_1, roll=spawn_waypoint_1.transform.rotation.roll))

        parameter_list.append(ActorConfigurationData(model, spawn_transform))
        parameter_list.append(ActorConfigurationData(model_1, spawn_transform_1))

        super(FollowLeadingVehicleWithObstacle, self).__init__("FollowLeadingVehicleWithObstacle",
                                                               ego_vehicle,
                                                               config,
                                                               world,
                                                               debug_mode)
        if randomize:
            self._ego_other_distance_start = random.randint(4, 8)

    def _create_behavior(self):
        """
        The scenario defined after is a "follow leading vehicle" scenario. After
        invoking this scenario, it will wait for the user controlled vehicle to
        enter the start region, then make the other actor to drive towards obstacle.
        Once obstacle clears the road, make the other actor to drive towards the
        next intersection. Finally, the user-controlled vehicle has to be close
        enough to the other actor to end the scenario.
        If this does not happen within 60 seconds, a timeout stops the scenario
        """

        hand_brake_apply = HandBrakeVehicle(self.other_actors[0], True, name="HandBraking")

        # start condition
        startcondition = InTimeToArrivalToLocation(self.ego_vehicle,
                                                   self._ego_other_distance_start,
                                                   self.other_actors[0].get_location(),
                                                   name="Waiting for start position")

        hand_brake_release = HandBrakeVehicle(self.other_actors[0], False, name="ReleasingHandBrake")

        # let the other actor drive until next intersection
        driving_to_next_intersection = py_trees.composites.Parallel(
            "Driving towards Intersection",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)

        obstacle_clear_road = py_trees.composites.Parallel("Obstalce clearing road",
                                                           policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        obstacle_clear_road.add_child(DriveDistance(self.other_actors[1], 4))
        obstacle_clear_road.add_child(KeepVelocity(self.other_actors[1], 5))

        stop_near_intersection = py_trees.composites.Parallel(
            "Waiting for end position near Intersection",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        stop_near_intersection.add_child(WaypointFollower(self.other_actors[0], 35))
        stop_near_intersection.add_child(InTriggerDistanceToNextIntersection(self.other_actors[0], 35))

        driving_to_next_intersection.add_child(WaypointFollower(self.other_actors[0], 35))
        driving_to_next_intersection.add_child(InTriggerDistanceToVehicle(self.other_actors[1],
                                                                          self.other_actors[0], 15))

        # end condition
        endcondition = py_trees.composites.Parallel("Waiting for end position",
                                                    policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL)
        endcondition_part1 = InTriggerDistanceToVehicle(self.other_actors[0],
                                                        self.ego_vehicle,
                                                        distance=20,
                                                        name="FinalDistance")
        endcondition_part2 = StandStill(self.ego_vehicle, name="FinalSpeed")
        endcondition.add_child(endcondition_part1)
        endcondition.add_child(endcondition_part2)

        # Build behavior tree
        sequence = py_trees.composites.Sequence("Sequence Behavior")
        sequence.add_child(hand_brake_apply)
        sequence.add_child(startcondition)
        sequence.add_child(hand_brake_release)
        sequence.add_child(driving_to_next_intersection)
        sequence.add_child(StopVehicle(self.other_actors[0], self._other_actor_max_brake))
        sequence.add_child(TimeOut(3))
        sequence.add_child(obstacle_clear_road)
        sequence.add_child(stop_near_intersection)
        sequence.add_child(StopVehicle(self.other_actors[0], self._other_actor_max_brake))
        sequence.add_child(endcondition)

        return sequence

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = []

        collision_criterion = CollisionTest(self.ego_vehicle)
        keep_lane_criterion = KeepLaneTest(self.ego_vehicle)

        criteria.append(collision_criterion)
        criteria.append(keep_lane_criterion)

        return criteria

    def __del__(self):
        """
        Remove all actors upon deletion
        """
        self.remove_all_actors()