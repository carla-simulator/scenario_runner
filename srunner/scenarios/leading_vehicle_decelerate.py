#!/usr/bin/env python

#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Leading vehicle decelerate scenario:

The scenario realizes a common driving behavior, in which the
user-controlled ego vehicle follows a leading car driving down
a given road. At some point the leading car has to decelerate.
The ego vehicle has to react accordingly by changing lane to avoid a
collision and follow the leading car in other lane. The scenario ends
either via a timeout, or if the ego vehicle drives some distance.
"""

from __future__ import print_function
import sys

import py_trees
import carla

from srunner.scenariomanager.atomic_scenario_behavior import *
from srunner.scenariomanager.atomic_scenario_criteria import *
from srunner.scenarios.basic_scenario import *
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenarios.config_parser import ActorConfigurationData
from srunner.scenarios.scenario_helper import get_location_in_distance


LEADING_VEHICLE_DECELERATE_SCENARIOS = [
    "LeadingVehicleDecelerate"
]


class LeadingVehicleDecelerate(BasicScenario):

    """
    This class holds everything required for a simple "Leading vehicle decelerate"
    scenario involving a user controlled vehicle and two other actors.
    """

    def __init__(self, world, ego_vehicle, config, randomize=False, debug_mode=False):
        """
        Setup all relevant parameters and create scenario
        """

        self.category = "LeadingVehicleDecelerate"

        self.timeout = 90        # Timeout of scenario in seconds

        # ego vehicle parameters
        self._ego_max_vel = 100        # Maximum allowed velocity [m/s]

        # other vehicle parameters
        self._other_target_vel = 55      # Target velocity of other vehicle

        self._traffic_light = None

        parameter_list = []

        model_1 = 'vehicle.volkswagen.t2'
        model_2 = 'vehicle.nissan.patrol'

        spawn_location_1, _ = get_location_in_distance(ego_vehicle, 50)
        spawn_location_1.z += 1
        spawn_waypoint_1 = ego_vehicle.get_world().get_map().get_waypoint(spawn_location_1)

        if spawn_waypoint_1.lane_change & carla.LaneChange.Left:
            spawn_waypoint_2 = spawn_waypoint_1.get_left_lane()
        elif spawn_waypoint_1.lane_change & carla.LaneChange.Right:
            spawn_waypoint_2 = spawn_waypoint_1.get_right_lane()

        spawn_transform_1 = carla.Transform(spawn_location_1, spawn_waypoint_1.transform.rotation)
        spawn_transform_2 = carla.Transform(
            spawn_waypoint_2.transform.location, spawn_waypoint_2.transform.rotation)
        parameter_list.append(ActorConfigurationData(model_1, spawn_transform_1))
        parameter_list.append(ActorConfigurationData(model_2, spawn_transform_2))
        config.other_actors = parameter_list

        super(LeadingVehicleDecelerate, self).__init__("LeadingVehicleDeceleratingInMultiLaneSetUp",
                                                       ego_vehicle,
                                                       config,
                                                       world,
                                                       debug_mode)

    def _create_behavior(self):
        """
        The scenario defined after is a "leading vehicle decelerate" scenario. After
        invoking this scenario, the user controlled vehicle has to drive towards the
        moving other actors, then make the leading actor to decelerate when user controlled
        vehicle is at some close distance. Finally, the user-controlled vehicle has to change
        lane to avoid collision and follow other leading actor in other lane to end the scenario.
        If this does not happen within 60 seconds, a timeout stops the scenario or the ego vehicle
        drives certain distance and stops the scenario.
        """
        # traffic light
        self._traffic_light = CarlaDataProvider.get_next_traffic_light(self.other_actors[0], False)
        if self._traffic_light is None:
            print("No traffic light for the given location found")
            sys.exit(-1)
        self._traffic_light.set_state(carla.TrafficLightState.Green)
        self._traffic_light.set_green_time(self.timeout)

        # hand brakes
        brake = py_trees.composites.Parallel(
            "Apply and release hand brakes",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        brake_1 = py_trees.composites.Parallel(
            "Hand brake for other actor",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL)
        brake_2 = py_trees.composites.Parallel(
            "Hand brake for leading actor",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL)
        brake_1.add_child(HandBrakeVehicle(self.other_actors[1], True))
        brake_1.add_child(HandBrakeVehicle(self.other_actors[1], False))
        brake_2.add_child(HandBrakeVehicle(self.other_actors[0], True))
        brake_2.add_child(HandBrakeVehicle(self.other_actors[0], False))
        brake.add_child(brake_1)
        brake.add_child(brake_2)

        # both actors moving in same direction
        driving_in_same_direction = py_trees.composites.Parallel(
            "Both actors driving in same direction",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)

        keep_velocity = py_trees.composites.Parallel(
            "Trigger condition for deceleration",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        keep_velocity.add_child(WaypointFollower(self.other_actors[0], self._other_target_vel))
        keep_velocity.add_child(InTriggerDistanceToVehicle(self.other_actors[0], self.ego_vehicle, 30))

        # deceleration of leading actor
        deceleration = py_trees.composites.Parallel(
            "Deceleration of leading actor",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        decelerate = self._other_target_vel / 3.2
        deceleration.add_child(WaypointFollower(self.other_actors[0], decelerate))
        deceleration.add_child(DriveDistance(self.other_actors[0], 100))

        leading_actor_sequence_behavior = py_trees.composites.Sequence("Decelerating actor sequence behavior")
        leading_actor_sequence_behavior.add_child(keep_velocity)
        leading_actor_sequence_behavior.add_child(deceleration)
        leading_actor_sequence_behavior.add_child(WaypointFollower(self.other_actors[0], self._other_target_vel))

        # end condition
        endcondition = DriveDistance(self.ego_vehicle, 350)

        # Build behavior tree
        sequence = py_trees.composites.Sequence("Sequence behavior")
        sequence.add_child(brake)
        sequence.add_child(driving_in_same_direction)
        driving_in_same_direction.add_child(leading_actor_sequence_behavior)
        driving_in_same_direction.add_child(WaypointFollower(self.other_actors[1], self._other_target_vel))
        driving_in_same_direction.add_child(endcondition)

        return sequence

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = []

        collision_criterion = CollisionTest(self.ego_vehicle)
        criteria.append(collision_criterion)

        return criteria

    def __del__(self):
        self._traffic_light = None
        self.remove_all_actors()
