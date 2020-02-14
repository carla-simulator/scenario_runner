#!/usr/bin/env python

#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
"""
Object crash with prior vehicle action scenario:
The scenario realizes the user controlled ego vehicle
moving along the road and encounters a cyclist ahead after taking a right or left turn.
"""

from __future__ import print_function

import math
import py_trees

import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider, CarlaActorPool
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (ActorTransformSetter,
                                                                      ActorDestroy,
                                                                      KeepVelocity)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (InTriggerDistanceToLocationAlongRoute,
                                                                               InTriggerDistanceToVehicle,
                                                                               DriveDistance)
from srunner.scenariomanager.timer import TimeOut
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.tools.scenario_helper import generate_target_waypoint


def get_opponent_transform(_start_distance, waypoint, trigger_location, last_waypoint_lane):
    """
    Calculate the transform of the adversary
    """

    offset = {"orientation": 270, "position": 90, "z": 0.25, "k": 1.0}
    _wp = waypoint.next(_start_distance)
    if _wp:
        _wp = _wp[-1]
    else:
        raise RuntimeError("Cannot get next waypoint !")

    if last_waypoint_lane == carla.LaneType.Shoulder:
        lane_width = 2.5
    elif last_waypoint_lane == carla.LaneType.Sidewalk:
        lane_width = 2.5
    else:
        lane_width = 4.0

    location = _wp.transform.location
    orientation_yaw = _wp.transform.rotation.yaw + offset["orientation"]
    position_yaw = _wp.transform.rotation.yaw + offset["position"]

    offset_location = carla.Location(
        offset['k'] * lane_width * math.cos(math.radians(position_yaw)),
        offset['k'] * lane_width * math.sin(math.radians(position_yaw)))
    location += offset_location
    location.z = trigger_location.z + offset["z"]
    transform = carla.Transform(location, carla.Rotation(yaw=orientation_yaw))

    return transform


class VehicleTurningRight(BasicScenario):

    """
    This class holds everything required for a simple object crash
    with prior vehicle action involving a vehicle and a cyclist.
    The ego vehicle is passing through a road and encounters
    a cyclist after taking a right turn.
    (Traffic Scenario 4)

    This is a single ego vehicle scenario
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=60):
        """
        Setup all relevant parameters and create scenario
        """
        # other vehicle parameters
        self._other_actor_target_velocity = 10
        self._wmap = CarlaDataProvider.get_map()
        self._reference_waypoint = self._wmap.get_waypoint(config.trigger_points[0].location)
        self._trigger_location = config.trigger_points[0].location
        self._other_actor_transform = None
        self._num_lane_changes = 0
        # Timeout of scenario in seconds
        self.timeout = timeout
        # Total Number of attempts to relocate a vehicle before spawning
        self._number_of_attempts = 6
        # Number of attempts made so far
        self._spawn_attempted = 0

        self._ego_route = CarlaDataProvider.get_ego_vehicle_route()

        super(VehicleTurningRight, self).__init__("VehicleTurningRight",
                                                  ego_vehicles,
                                                  config,
                                                  world,
                                                  debug_mode,
                                                  criteria_enable=criteria_enable)

    def _initialize_actors(self, config):
        """
        Custom initialization
        """

        waypoint = self._reference_waypoint
        waypoint = generate_target_waypoint(waypoint, 1)
        _start_distance = 8
        while True:
            wp_next = waypoint.get_right_lane()
            self._num_lane_changes += 1
            if wp_next is not None:
                _start_distance += 1
                waypoint = wp_next
                if waypoint.lane_type == carla.LaneType.Shoulder or waypoint.lane_type == carla.LaneType.Sidewalk:
                    last_waypoint_lane = waypoint.lane_type
                    break

            else:
                last_waypoint_lane = waypoint.lane_type
                break

        while True:
            try:
                self._other_actor_transform = get_opponent_transform(_start_distance, waypoint,
                                                                     self._trigger_location, last_waypoint_lane)
                first_vehicle = CarlaActorPool.request_new_actor('vehicle.diamondback.century',
                                                                 self._other_actor_transform)
                first_vehicle.set_simulate_physics(enabled=False)

                break
            except RuntimeError as r:
                # In the case there is an object just move a little bit and retry
                print("Base transform is blocking objects ", self._other_actor_transform)
                _start_distance += 0.2
                self._spawn_attempted += 1
                if self._spawn_attempted >= self._number_of_attempts:
                    raise r
        # Set the transform to -500 z after we are able to spawn it
        actor_transform = carla.Transform(
            carla.Location(self._other_actor_transform.location.x,
                           self._other_actor_transform.location.y,
                           self._other_actor_transform.location.z - 500),
            self._other_actor_transform.rotation)
        first_vehicle.set_transform(actor_transform)
        first_vehicle.set_simulate_physics(enabled=False)
        self.other_actors.append(first_vehicle)

    def _create_behavior(self):
        """
        After invoking this scenario, cyclist will wait for the user
        controlled vehicle to enter the in the trigger distance region,
        the cyclist starts crossing the road once the condition meets,
        ego vehicle has to avoid the crash after a right turn, but
        continue driving after the road is clear.If this does not happen
        within 90 seconds, a timeout stops the scenario.
        """

        root = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE, name="IntersectionRightTurn")

        lane_width = self._reference_waypoint.lane_width
        lane_width = lane_width + (1.10 * lane_width * self._num_lane_changes)

        if self._ego_route is not None:
            trigger_distance = InTriggerDistanceToLocationAlongRoute(self.ego_vehicles[0], self._ego_route,
                                                                     self._other_actor_transform.location, 20)
        else:
            trigger_distance = InTriggerDistanceToVehicle(self.other_actors[0], self.ego_vehicles[0], 20)

        actor_velocity = KeepVelocity(self.other_actors[0], self._other_actor_target_velocity)
        actor_traverse = DriveDistance(self.other_actors[0], 0.30 * lane_width)
        post_timer_velocity_actor = KeepVelocity(self.other_actors[0], self._other_actor_target_velocity)
        post_timer_traverse_actor = DriveDistance(self.other_actors[0], 0.70 * lane_width)
        end_condition = TimeOut(5)

        # non leaf nodes
        scenario_sequence = py_trees.composites.Sequence()
        actor_ego_sync = py_trees.composites.Parallel(
            "Synchronization of actor and ego vehicle",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        after_timer_actor = py_trees.composites.Parallel(
            "After timout actor will cross the remaining lane_width",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)

        # building the tree
        root.add_child(scenario_sequence)
        scenario_sequence.add_child(ActorTransformSetter(self.other_actors[0], self._other_actor_transform,
                                                         name='TransformSetterTS4'))
        scenario_sequence.add_child(trigger_distance)
        scenario_sequence.add_child(actor_ego_sync)
        scenario_sequence.add_child(after_timer_actor)
        scenario_sequence.add_child(end_condition)
        scenario_sequence.add_child(ActorDestroy(self.other_actors[0]))
        actor_ego_sync.add_child(actor_velocity)
        actor_ego_sync.add_child(actor_traverse)

        after_timer_actor.add_child(post_timer_velocity_actor)
        after_timer_actor.add_child(post_timer_traverse_actor)
        return root

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = []
        collision_criterion = CollisionTest(self.ego_vehicles[0])
        criteria.append(collision_criterion)
        return criteria

    def __del__(self):
        """
        Remove all actors upon deletion
        """
        self.remove_all_actors()


class VehicleTurningLeft(BasicScenario):

    """
    This class holds everything required for a simple object crash
    with prior vehicle action involving a vehicle and a cyclist.
    The ego vehicle is passing through a road and encounters
    a cyclist after taking a left turn. Scenario 4

    This is a single ego vehicle scenario
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=60):
        """
        Setup all relevant parameters and create scenario
        """
        self._other_actor_target_velocity = 10
        self._wmap = CarlaDataProvider.get_map()
        self._reference_waypoint = self._wmap.get_waypoint(config.trigger_points[0].location)
        self._trigger_location = config.trigger_points[0].location
        self._other_actor_transform = None
        self._num_lane_changes = 0
        # Timeout of scenario in seconds
        self.timeout = timeout
        # Total Number of attempts to relocate a vehicle before spawning
        self._number_of_attempts = 6
        # Number of attempts made so far
        self._spawn_attempted = 0

        self._ego_route = CarlaDataProvider.get_ego_vehicle_route()

        super(VehicleTurningLeft, self).__init__("VehicleTurningLeft",
                                                 ego_vehicles,
                                                 config,
                                                 world,
                                                 debug_mode,
                                                 criteria_enable=criteria_enable)

    def _initialize_actors(self, config):
        """
        Custom initialization
        """
        waypoint = self._reference_waypoint
        waypoint = generate_target_waypoint(waypoint, -1)
        _start_distance = 8
        while True:
            wp_next = waypoint.get_right_lane()
            self._num_lane_changes += 1
            if wp_next is not None:
                _start_distance += 1
                waypoint = wp_next
                if waypoint.lane_type == carla.LaneType.Shoulder or waypoint.lane_type == carla.LaneType.Sidewalk:
                    last_waypoint_lane = waypoint.lane_type
                    break

            else:
                last_waypoint_lane = waypoint.lane_type
                break
        while True:
            try:
                self._other_actor_transform = get_opponent_transform(_start_distance, waypoint,
                                                                     self._trigger_location, last_waypoint_lane)
                first_vehicle = CarlaActorPool.request_new_actor('vehicle.diamondback.century',
                                                                 self._other_actor_transform)
                first_vehicle.set_simulate_physics(enabled=False)

                break
            except RuntimeError as r:
                # In the case there is an object just move a little bit and retry
                print(" Base transform is blocking objects ", self._other_actor_transform)
                _start_distance += 0.2
                self._spawn_attempted += 1
                if self._spawn_attempted >= self._number_of_attempts:
                    raise r
            # Set the transform to -500 z after we are able to spawn it
        actor_transform = carla.Transform(
            carla.Location(self._other_actor_transform.location.x,
                           self._other_actor_transform.location.y,
                           self._other_actor_transform.location.z - 500),
            self._other_actor_transform.rotation)
        first_vehicle.set_transform(actor_transform)
        first_vehicle.set_simulate_physics(enabled=False)
        self.other_actors.append(first_vehicle)

    def _create_behavior(self):
        """
        After invoking this scenario, cyclist will wait for the user
        controlled vehicle to enter the in the trigger distance region,
        the cyclist starts crossing the road once the condition meets,
        ego vehicle has to avoid the crash after a left turn, but
        continue driving after the road is clear.If this does not happen
        within 90 seconds, a timeout stops the scenario.
        """
        root = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE, name="IntersectionLeftTurn")

        lane_width = self._reference_waypoint.lane_width
        lane_width = lane_width + (1.10 * lane_width * self._num_lane_changes)
        if self._ego_route is not None:
            trigger_distance = InTriggerDistanceToLocationAlongRoute(self.ego_vehicles[0], self._ego_route,
                                                                     self._other_actor_transform.location, 20)
        else:
            trigger_distance = InTriggerDistanceToVehicle(self.other_actors[0], self.ego_vehicles[0], 25)

        actor_velocity = KeepVelocity(self.other_actors[0], self._other_actor_target_velocity)
        actor_traverse = DriveDistance(self.other_actors[0], 0.30 * lane_width)
        post_timer_velocity_actor = KeepVelocity(self.other_actors[0], self._other_actor_target_velocity)
        post_timer_traverse_actor = DriveDistance(self.other_actors[0], 0.70 * lane_width)
        end_condition = TimeOut(5)
        # non leaf nodes
        scenario_sequence = py_trees.composites.Sequence()

        actor_ego_sync = py_trees.composites.Parallel(
            "Synchronization of actor and ego vehicle",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        after_timer_actor = py_trees.composites.Parallel(
            "After timout actor will cross the remaining lane_width",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)

        # building the tree
        root.add_child(scenario_sequence)
        scenario_sequence.add_child(ActorTransformSetter(self.other_actors[0], self._other_actor_transform,
                                                         name='TransformSetterTS4'))
        scenario_sequence.add_child(trigger_distance)
        scenario_sequence.add_child(actor_ego_sync)
        scenario_sequence.add_child(after_timer_actor)
        scenario_sequence.add_child(end_condition)
        scenario_sequence.add_child(ActorDestroy(self.other_actors[0]))

        actor_ego_sync.add_child(actor_velocity)
        actor_ego_sync.add_child(actor_traverse)

        after_timer_actor.add_child(post_timer_velocity_actor)
        after_timer_actor.add_child(post_timer_traverse_actor)
        return root

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = []
        collision_criterion = CollisionTest(self.ego_vehicles[0])

        criteria.append(collision_criterion)
        return criteria

    def __del__(self):
        """
        Remove all actors upon deletion
        """
        self.remove_all_actors()


class VehicleTurningRoute(BasicScenario):

    """
    This class holds everything required for a simple object crash
    with prior vehicle action involving a vehicle and a cyclist.
    The ego vehicle is passing through a road and encounters
    a cyclist after taking a right turn.
    (Traffic Scenario 4)

    This is a single ego vehicle scenario
    """

    SUBTYPE_INDEX_TRANSLATION = {
        "S4left": -1,
        "S4right": 1
    }

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=60):
        """
        Setup all relevant parameters and create scenario
        """
        # other vehicle parameters
        self._other_actor_target_velocity = 10
        self._wmap = CarlaDataProvider.get_map()
        self._reference_waypoint = self._wmap.get_waypoint(config.trigger_points[0].location)
        self._trigger_location = config.trigger_points[0].location
        self._other_actor_transform = None
        self._num_lane_changes = 0
        # Timeout of scenario in seconds
        self.timeout = timeout
        # Total Number of attempts to relocate a vehicle before spawning
        self._number_of_attempts = 6
        # Number of attempts made so far
        self._spawn_attempted = 0

        self._ego_route = CarlaDataProvider.get_ego_vehicle_route()

        super(VehicleTurningRoute, self).__init__("VehicleTurningRoute",
                                                  ego_vehicles,
                                                  config,
                                                  world,
                                                  debug_mode,
                                                  criteria_enable=criteria_enable)

    def _initialize_actors(self, config):
        """
        Custom initialization
        """

        waypoint = self._reference_waypoint
        direction = self.SUBTYPE_INDEX_TRANSLATION[config.subtype]
        waypoint = generate_target_waypoint(waypoint, direction)
        _start_distance = 8
        while True:
            wp_next = waypoint.get_right_lane()
            self._num_lane_changes += 1
            if wp_next is not None:
                _start_distance += 1
                waypoint = wp_next
                if waypoint.lane_type == carla.LaneType.Shoulder or waypoint.lane_type == carla.LaneType.Sidewalk:
                    last_waypoint_lane = waypoint.lane_type
                    break

            else:
                last_waypoint_lane = waypoint.lane_type
                break

        while True:
            try:
                self._other_actor_transform = get_opponent_transform(_start_distance, waypoint,
                                                                     self._trigger_location, last_waypoint_lane)
                first_vehicle = CarlaActorPool.request_new_actor('vehicle.diamondback.century',
                                                                 self._other_actor_transform)
                first_vehicle.set_simulate_physics(enabled=False)

                break
            except RuntimeError as r:
                # In the case there is an object just move a little bit and retry
                print("Base transform is blocking objects ", self._other_actor_transform)
                _start_distance += 0.2
                self._spawn_attempted += 1
                if self._spawn_attempted >= self._number_of_attempts:
                    raise r
        # Set the transform to -500 z after we are able to spawn it
        actor_transform = carla.Transform(
            carla.Location(self._other_actor_transform.location.x,
                           self._other_actor_transform.location.y,
                           self._other_actor_transform.location.z - 500),
            self._other_actor_transform.rotation)
        first_vehicle.set_transform(actor_transform)
        first_vehicle.set_simulate_physics(enabled=False)
        self.other_actors.append(first_vehicle)

    def _create_behavior(self):
        """
        After invoking this scenario, cyclist will wait for the user
        controlled vehicle to enter the in the trigger distance region,
        the cyclist starts crossing the road once the condition meets,
        ego vehicle has to avoid the crash after a right turn, but
        continue driving after the road is clear.If this does not happen
        within 90 seconds, a timeout stops the scenario.
        """

        root = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE, name="IntersectionRightTurn")

        lane_width = self._reference_waypoint.lane_width
        lane_width = lane_width + (1.10 * lane_width * self._num_lane_changes)

        if self._ego_route is not None:
            trigger_distance = InTriggerDistanceToLocationAlongRoute(self.ego_vehicles[0], self._ego_route,
                                                                     self._other_actor_transform.location, 20)
        else:
            trigger_distance = InTriggerDistanceToVehicle(self.other_actors[0], self.ego_vehicles[0], 20)

        actor_velocity = KeepVelocity(self.other_actors[0], self._other_actor_target_velocity)
        actor_traverse = DriveDistance(self.other_actors[0], 0.30 * lane_width)
        post_timer_velocity_actor = KeepVelocity(self.other_actors[0], self._other_actor_target_velocity)
        post_timer_traverse_actor = DriveDistance(self.other_actors[0], 0.70 * lane_width)
        end_condition = TimeOut(5)

        # non leaf nodes
        scenario_sequence = py_trees.composites.Sequence()
        actor_ego_sync = py_trees.composites.Parallel(
            "Synchronization of actor and ego vehicle",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        after_timer_actor = py_trees.composites.Parallel(
            "After timout actor will cross the remaining lane_width",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)

        # building the tree
        root.add_child(scenario_sequence)
        scenario_sequence.add_child(ActorTransformSetter(self.other_actors[0], self._other_actor_transform,
                                                         name='TransformSetterTS4'))
        scenario_sequence.add_child(trigger_distance)
        scenario_sequence.add_child(actor_ego_sync)
        scenario_sequence.add_child(after_timer_actor)
        scenario_sequence.add_child(end_condition)
        scenario_sequence.add_child(ActorDestroy(self.other_actors[0]))
        actor_ego_sync.add_child(actor_velocity)
        actor_ego_sync.add_child(actor_traverse)

        after_timer_actor.add_child(post_timer_velocity_actor)
        after_timer_actor.add_child(post_timer_traverse_actor)
        return root

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = []
        collision_criterion = CollisionTest(self.ego_vehicles[0])
        criteria.append(collision_criterion)
        return criteria

    def __del__(self):
        """
        Remove all actors upon deletion
        """
        self.remove_all_actors()
