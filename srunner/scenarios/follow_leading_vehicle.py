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

import random

import py_trees

from srunner.scenariomanager.atomic_scenario_behavior import *
from srunner.scenariomanager.atomic_scenario_criteria import *
from srunner.scenariomanager.timer import TimeOut
from srunner.scenarios.basic_scenario import *


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

    def __init__(self, world, ego_vehicle, other_actors, town, randomize=False, debug_mode=False):
        """
        Setup all relevant parameters and create scenario

        If randomize is True, the scenario parameters are randomized
        """
        super(FollowLeadingVehicle, self).__init__("FollowVehicle",
                                                   ego_vehicle,
                                                   other_actors,
                                                   town,
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

    def _create_behavior(self):
        """
        The scenario defined after is a "follow leading vehicle" scenario. After
        invoking this scenario, it will wait for the user controlled vehicle to
        enter the start region, then make the other actor to drive until reaching
        the next intersection. Finally, the user-controlled vehicle has to be close
        enough to the other actor to end the scenario.
        If this does not happen within 60 seconds, a timeout stops the scenario
        """

        # start condition
        startcondition = py_trees.composites.Parallel(
            "Waiting for start position",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)

        startcondition.add_child(InTimeToArrivalToLocation(self.ego_vehicle,
                                                           self._ego_other_distance_start,
                                                           self.other_actors[0].get_location()))
        startcondition.add_child(InTriggerDistanceToVehicle(self.ego_vehicle,
                                                            self.other_actors[0],
                                                            10))

        # let the other actor drive until next intersection
        # @todo: We should add some feedback mechanism to respond to ego_vehicle behavior
        driving_to_next_intersection = py_trees.composites.Parallel(
            "Waiting for end position",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        driving_to_next_intersection.add_child(UseAutoPilot(self.other_actors[0]))
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
        sequence.add_child(startcondition)
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

        max_velocity_criterion = MaxVelocityTest(self.ego_vehicle,
                                                 self._ego_max_velocity_allowed,
                                                 optional=True)
        collision_criterion = CollisionTest(self.ego_vehicle)
        keep_lane_criterion = KeepLaneTest(self.ego_vehicle)
        avg_velocity_criterion = AverageVelocityTest(self.ego_vehicle, self._ego_avg_velocity_expected, optional=True)

        criteria.append(max_velocity_criterion)
        criteria.append(collision_criterion)
        criteria.append(keep_lane_criterion)
        criteria.append(avg_velocity_criterion)

        # Add the collision and lane checks for all vehicles as well
        for vehicle in self.other_actors:
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

    category = "FollowLeadingVehicle"

    timeout = 120            # Timeout of scenario in seconds

    # ego vehicle parameters
    _ego_max_velocity_allowed = 20   # Maximum allowed velocity [m/s]
    _ego_avg_velocity_expected = 4   # Average expected velocity [m/s]
    _ego_other_distance_start = 4    # time to arrival that triggers scenario starts

    # other vehicle
    _other_actor_max_brake = 1.0                  # Maximum brake of other vehicle
    _other_actor_stop_in_front_intersection = 30  # Stop ~30m in front of intersection

    def __init__(self, world, ego_vehicle, other_actors, town, randomize=False, debug_mode=False):
        """
        Setup all relevant parameters and create scenario
        """

        super(FollowLeadingVehicleWithObstacle, self).__init__("FollowLeadingVehicleWithObstacle",
                                                               ego_vehicle,
                                                               other_actors,
                                                               town,
                                                               world,
                                                               debug_mode)

        if randomize:
            self._ego_other_distance_start = random.randint(2, 8)

    def _create_behavior(self):
        """
        The scenario defined after is a "follow leading vehicle" scenario. After
        invoking this scenario, it will wait for the user controlled vehicle to
        enter the start region, then make the other actor to drive until reaching
        the next intersection. Finally, the user-controlled vehicle has to be close
        enough to the other actor to end the scenario.
        If this does not happen within 60 seconds, a timeout stops the scenario
        """

        # start condition
        startcondition = InTimeToArrivalToLocation(self.ego_vehicle,
                                                   self._ego_other_distance_start,
                                                   self.other_actors[0].get_location(),
                                                   name="Waiting for start position")

        # let the other actor drive until next intersection
        # @todo: We should add some feedback mechanism to respond to ego_vehicle behavior
        driving_to_next_intersection = py_trees.composites.Parallel(
            "Waiting for end position",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)

        driving_considering_bike = py_trees.composites.Parallel(
            "Drive with AutoPilot",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL)
        driving_considering_bike.add_child(UseAutoPilot(self.other_actors[0]))
        obstacle_sequence = py_trees.composites.Sequence("Obstacle sequence behavior")
        obstacle_sequence.add_child(InTriggerDistanceToVehicle(self.other_actors[0],
                                                               self.other_actors[1],
                                                               10))
        obstacle_sequence.add_child(TimeOut(5))
        obstacle_clear_road = py_trees.composites.Parallel("Obstalce clearing road",
                                                           policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        obstacle_clear_road.add_child(DriveDistance(self.other_actors[1], 4))
        obstacle_clear_road.add_child(KeepVelocity(self.other_actors[1], 5))

        obstacle_sequence.add_child(obstacle_clear_road)
        obstacle_sequence.add_child(StopVehicle(self.other_actors[1], self._other_actor_max_brake))
        driving_considering_bike.add_child(obstacle_sequence)

        driving_to_next_intersection.add_child(InTriggerDistanceToNextIntersection(
            self.other_actors[0], self._other_actor_stop_in_front_intersection))
        driving_to_next_intersection.add_child(driving_considering_bike)

        # stop vehicle
        stop = StopVehicle(self.other_actors[0], self._other_actor_max_brake)

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
        sequence.add_child(startcondition)
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

        max_velocity_criterion = MaxVelocityTest(self.ego_vehicle,
                                                 self._ego_max_velocity_allowed,
                                                 optional=True)
        collision_criterion = CollisionTest(self.ego_vehicle)
        keep_lane_criterion = KeepLaneTest(self.ego_vehicle)
        avg_velocity_criterion = AverageVelocityTest(self.ego_vehicle, self._ego_avg_velocity_expected, optional=True)

        criteria.append(max_velocity_criterion)
        criteria.append(collision_criterion)
        criteria.append(keep_lane_criterion)
        criteria.append(avg_velocity_criterion)

        # Add the collision and lane checks for all vehicles as well
        for vehicle in self.other_actors:
            collision_criterion = CollisionTest(vehicle)
            keep_lane_criterion = KeepLaneTest(vehicle)
            criteria.append(collision_criterion)
            criteria.append(keep_lane_criterion)

        return criteria
