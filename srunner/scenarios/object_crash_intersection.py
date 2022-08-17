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

import py_trees

import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (ActorDestroy,
                                                                      HandBrakeVehicle,
                                                                      KeepVelocity,
                                                                      ActorTransformSetter)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (InTriggerDistanceToLocation,
                                                                               InTimeToArrivalToLocation,
                                                                               DriveDistance)
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.tools.scenario_helper import (generate_target_waypoint,
                                           generate_target_waypoint_in_route,
                                           get_same_dir_lanes,
                                           get_opposite_dir_lanes)

from srunner.tools.background_manager import LeaveCrossingSpace


def get_sidewalk_transform(waypoint, offset):
    """
    Processes the waypoint transform to find a suitable spawning one at the sidewalk.
    It first rotates the transform so that it is pointing towards the road and then moves a
    bit to the side waypoint that aren't part of sidewalks, as they might be invading the road
    """

    new_rotation = waypoint.transform.rotation
    new_rotation.yaw += offset['yaw']

    if waypoint.lane_type == carla.LaneType.Sidewalk:
        new_location = waypoint.transform.location
    else:
        right_vector = waypoint.transform.get_right_vector()
        offset_location = carla.Location(offset["k"] * right_vector.x, offset["k"] * right_vector.y)
        new_location = waypoint.transform.location + offset_location
    new_location.z += offset['z']

    return carla.Transform(new_location, new_rotation)


class BaseVehicleTurning(BasicScenario):

    """
    This class holds everything required for a simple object crash
    with prior vehicle action involving a vehicle and a cyclist.
    The ego vehicle is passing through a road and encounters
    a cyclist after taking a turn.

    This is a single ego vehicle scenario
    """
    _subtype = None

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=60, name="BaseVehicleTurning"):
        """
        Setup all relevant parameters and create scenario
        """

        self._wmap = CarlaDataProvider.get_map()
        self._trigger_location = config.trigger_points[0].location
        self._reference_waypoint = self._wmap.get_waypoint(self._trigger_location)
        self._ego_route = config.route

        self._start_distance = 11
        self._spawn_dist = self._start_distance
        self._number_of_attempts = 6
        self._retry_dist = 0.4

        self._adversary_transform = None

        self._collision_wp = None
        self._adversary_speed = 1.8  # Speed of the adversary [m/s]
        self._reaction_time = 1.8  # Time the agent has to react to avoid the collision [s]
        self._min_trigger_dist = 6.0  # Min distance to the collision location that triggers the adversary [m]
        self._ego_end_distance = 40

        self._offset = {"yaw": 270, "z": 0.2, "k": 1.5}

        self.timeout = timeout
        super(BaseVehicleTurning, self).__init__(
            name, ego_vehicles, config, world, debug_mode, criteria_enable=criteria_enable)

    def _get_target_waypoint(self):
        """
        Gets the first waypoint after the junction.
        This method depends on the subtype of VehicleTurning scenario
        """
        if self._subtype == 'right':
            return generate_target_waypoint(self._reference_waypoint, 1)
        elif self._subtype == 'left':
            return generate_target_waypoint(self._reference_waypoint, -1)
        elif self._subtype == 'route':
            return generate_target_waypoint_in_route(self._reference_waypoint, self._ego_route)
        else:
            raise ValueError("Trying to run a VehicleTurning scenario with a wrong subtype")

    def _initialize_actors(self, config):
        """
        Custom initialization
        """
        # Get the waypoint right after the junction
        waypoint = self._get_target_waypoint()
        move_dist = self._start_distance
        while self._number_of_attempts > 0:
            # Move to the front
            waypoint = waypoint.next(move_dist)[0]
            self._collision_wp = waypoint

            # Move to the right
            sidewalk_waypoint = waypoint
            while sidewalk_waypoint.lane_type != carla.LaneType.Sidewalk:
                right_wp = sidewalk_waypoint.get_right_lane()
                if right_wp is None:
                    break  # No more right lanes
                sidewalk_waypoint = right_wp

            # Get the adversary transform and spawn it
            self._adversary_transform = get_sidewalk_transform(sidewalk_waypoint, self._offset)
            adversary = CarlaDataProvider.request_new_actor('vehicle.diamondback.century', self._adversary_transform)
            if adversary is None:
                self._number_of_attempts -= 1
                move_dist = self._retry_dist
                self._spawn_dist += self._retry_dist
                continue

            # Both actors where summoned, end
            break

        if self._number_of_attempts == 0:
            raise ValueError("Couldn't find viable position for the adversary")

        if isinstance(adversary, carla.Vehicle):
            adversary.apply_control(carla.VehicleControl(hand_brake=True))
        self.other_actors.append(adversary)

    def _create_behavior(self):
        """
        After invoking this scenario, cyclist will wait for the user
        controlled vehicle to enter the in the trigger distance region,
        the cyclist starts crossing the road once the condition meets,
        ego vehicle has to avoid the crash after a turn, but
        continue driving after the road is clear.If this does not happen
        within 90 seconds, a timeout stops the scenario.
        """
        sequence = py_trees.composites.Sequence(name="CrossingActorIntersection")
        collision_location = self._collision_wp.transform.location
        collision_distance = collision_location.distance(self._adversary_transform.location)
        collision_duration = collision_distance / self._adversary_speed

        # Adversary trigger behavior
        trigger_adversary = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE, name="TriggerAdversaryStart")
        trigger_adversary.add_child(InTimeToArrivalToLocation(
            self.ego_vehicles[0], self._reaction_time, collision_location))
        trigger_adversary.add_child(InTriggerDistanceToLocation(
            self.ego_vehicles[0], collision_location, self._min_trigger_dist))

        sequence.add_child(trigger_adversary)
        sequence.add_child(HandBrakeVehicle(self.other_actors[0], False))

        # Move the adversary.
        speed_duration = 2.0 * collision_duration
        speed_distance = 2.0 * collision_distance
        if self.route_mode:
            sequence.add_child(LeaveCrossingSpace(self._collision_wp))
        sequence.add_child(KeepVelocity(
            self.other_actors[0], self._adversary_speed, True,
            speed_duration, speed_distance, name="AdversaryCrossing")
        )

        # Remove everything
        sequence.add_child(ActorDestroy(self.other_actors[0], name="DestroyAdversary"))
        sequence.add_child(DriveDistance(self.ego_vehicles[0], self._ego_end_distance, name="EndCondition"))

        return sequence

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        return [CollisionTest(self.ego_vehicles[0])]

    def __del__(self):
        """
        Remove all actors upon deletion
        """
        self.remove_all_actors()


class VehicleTurningRight(BaseVehicleTurning):
    """
    Version of the VehicleTurning scenario where
    the adversary is placed at the right side after the junction
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=60):
        """
        Setup all relevant parameters and create scenario
        """
        self._subtype = 'right'
        super(VehicleTurningRight, self).__init__(
            world, ego_vehicles, config, randomize, debug_mode, criteria_enable, timeout, "VehicleTurningRight")


class VehicleTurningLeft(BaseVehicleTurning):
    """
    Version of the VehicleTurning scenario where
    the adversary is placed at the left side after the junction
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=60):
        """
        Setup all relevant parameters and create scenario
        """
        self._subtype = 'left'
        super(VehicleTurningLeft, self).__init__(
            world, ego_vehicles, config, randomize, debug_mode, criteria_enable, timeout, "VehicleTurningLeft")


class VehicleTurningRoute(BaseVehicleTurning):
    """
    Version of the VehicleTurning scenario where
    the adversary is placed using the route path
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=60):
        """
        Setup all relevant parameters and create scenario
        """
        self._subtype = 'route'
        super(VehicleTurningRoute, self).__init__(
            world, ego_vehicles, config, randomize, debug_mode, criteria_enable, timeout, "VehicleTurningRoute")

    def _create_test_criteria(self):
        """
        Empty, the route already has a collision criteria
        """
        return []


class VehicleTurningRoutePedestrian(BasicScenario):

    """
    This class holds everything required for a simple object crash
    with prior vehicle action involving a vehicle and a cyclist.
    The ego vehicle is passing through a road and encounters
    a cyclist after taking a turn.

    This is a single ego vehicle scenario
    """
    _subtype = None

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=60, name="VehicleTurningRoutePedestrian"):
        """
        Setup all relevant parameters and create scenario
        """

        self._wmap = CarlaDataProvider.get_map()
        self._trigger_location = config.trigger_points[0].location
        self._reference_waypoint = self._wmap.get_waypoint(self._trigger_location)
        self._ego_route = config.route

        self._collision_wp = None
        self._adversary_speed = 1.8  # Speed of the adversary [m/s]
        self._reaction_time = 1.8  # Time the agent has to react to avoid the collision [s]
        self._min_trigger_dist = 6.0  # Min distance to the collision location that triggers the adversary [m]
        self._ego_end_distance = 40

        self._offset = {"yaw": 270, "z": 0.2, "k": 1.5}

        self.timeout = timeout
        super().__init__(name, ego_vehicles, config, world, debug_mode, criteria_enable=criteria_enable)

    def _initialize_actors(self, config):
        """
        Custom initialization
        """
        # Get the waypoint right after the junction
        waypoint = generate_target_waypoint_in_route(self._reference_waypoint, self._ego_route)
        self._collision_wp = waypoint.next(0.5)[0]  # Some wps are still part of the junction

        # Get the right waypoint at the sidewalk
        same_dir_wps = get_same_dir_lanes(self._collision_wp)
        right_wp = same_dir_wps[0]
        while right_wp.lane_type != carla.LaneType.Sidewalk:
            side_wp = right_wp.get_right_lane()
            if side_wp is None:
                break
            right_wp = side_wp

        # Get the left waypoint at the sidewalk
        other_dir_wps = get_opposite_dir_lanes(self._collision_wp)
        if other_dir_wps:
            # With opposite lane
            left_wp = other_dir_wps[-1]
            while left_wp.lane_type != carla.LaneType.Sidewalk:
                side_wp = left_wp.get_right_lane()
                if side_wp is None:
                    break
                left_wp = side_wp
        else:
            # Without opposite lane
            self._offset['yaw'] = 90
            left_wp = same_dir_wps[-1]
            while left_wp.lane_type != carla.LaneType.Sidewalk:
                side_wp = left_wp.get_left_lane()
                if side_wp is None:
                    break
                left_wp = side_wp

        self._adversary_distance = right_wp.transform.location.distance(left_wp.transform.location)

        entry_vec = self._reference_waypoint.transform.get_forward_vector()
        exit_vec = waypoint.transform.get_forward_vector()
        cross_prod = entry_vec.cross(exit_vec)
        spawn_wp = right_wp if cross_prod.z < 0 else left_wp

        # Get the adversary transform and spawn it
        self._spawn_transform = get_sidewalk_transform(spawn_wp, self._offset)
        adversary = CarlaDataProvider.request_new_actor('walker.*', self._spawn_transform)
        if adversary is None:
            raise ValueError("Couldn't spawn adversary")

        self.other_actors.append(adversary)
        adversary.set_location(self._spawn_transform.location + carla.Location(z=-200))

    def _create_behavior(self):
        """
        """
        sequence = py_trees.composites.Sequence(name="VehicleTurningRoutePedestrian")
        sequence.add_child(ActorTransformSetter(self.other_actors[0], self._spawn_transform, True))

        collision_location = self._collision_wp.transform.location

        # Adversary trigger behavior
        trigger_adversary = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE, name="TriggerAdversaryStart")
        trigger_adversary.add_child(InTimeToArrivalToLocation(
            self.ego_vehicles[0], self._reaction_time, collision_location))
        trigger_adversary.add_child(InTriggerDistanceToLocation(
            self.ego_vehicles[0], collision_location, self._min_trigger_dist))

        sequence.add_child(trigger_adversary)
        if self.route_mode:
            sequence.add_child(LeaveCrossingSpace(self._collision_wp))

        # Move the adversary.
        speed_distance = self._adversary_distance
        speed_duration = self._adversary_distance / self._adversary_speed
        sequence.add_child(KeepVelocity(
            self.other_actors[0], self._adversary_speed, True,
            speed_duration, speed_distance, name="AdversaryCrossing")
        )

        # Remove everything
        sequence.add_child(ActorDestroy(self.other_actors[0], name="DestroyAdversary"))
        sequence.add_child(DriveDistance(self.ego_vehicles[0], self._ego_end_distance, name="EndCondition"))

        return sequence

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        return [CollisionTest(self.ego_vehicles[0])]

    def __del__(self):
        """
        Remove all actors upon deletion
        """
        self.remove_all_actors()