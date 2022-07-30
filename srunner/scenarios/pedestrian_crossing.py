#!/usr/bin/env python
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Pedestrians crossing through the middle of the lane.
"""

from __future__ import print_function

import py_trees
import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import ActorDestroy, KeepVelocity
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (InTriggerDistanceToLocation,
                                                                               InTimeToArrivalToLocation,
                                                                               DriveDistance)
from srunner.scenarios.basic_scenario import BasicScenario

from srunner.tools.background_manager import HandleJunctionScenario

from srunner.tools.scenario_helper import get_same_dir_lanes, get_opposite_dir_lanes


def convert_dict_to_location(actor_dict):
    """
    Convert a JSON string to a Carla.Location
    """
    location = carla.Location(
        x=float(actor_dict['x']),
        y=float(actor_dict['y']),
        z=float(actor_dict['z'])
    )
    return location


class PedestrianCrossing(BasicScenario):

    """
    This class holds everything required for a group of natual pedestrians crossing the road.
    The ego vehicle is passing through a road,
    And encounters a group of pedestrians crossing the road.

    This is a single ego vehicle scenario.

    Notice that the initial pedestrian will walk from the start of the junction ahead to end_walker_flow_1.
    """

    def __init__(self, world, ego_vehicles, config, debug_mode=False, criteria_enable=True, timeout=60):
        """
        Setup all relevant parameters and create scenario
        """
        self._wmap = CarlaDataProvider.get_map()
        self._trigger_location = config.trigger_points[0].location
        self._reference_waypoint = self._wmap.get_waypoint(self._trigger_location)

        self._adversary_speed = 1.3  # Speed of the adversary [m/s]
        self._reaction_time = 4.0  # Time the agent has to react to avoid the collision [s]
        self._min_trigger_dist = 12.0  # Min distance to the collision location that triggers the adversary [m]
        self._ego_end_distance = 40
        self.timeout = timeout

        self._crosswalk_dist = 1
        self._crosswalk_right_dist = 1

        super().__init__("PedestrianCrossing",
                          ego_vehicles,
                          config,
                          world,
                          debug_mode,
                          criteria_enable=criteria_enable)

    def _initialize_actors(self, config):

        # Get the start point of the initial pedestrian
        collision_wp = self._reference_waypoint
        while True:
            next_wps = collision_wp.next(1)
            if not next_wps:
                raise ValueError("Couldn't find a waypoint to spawn the pedestrians")
            if next_wps[0].is_junction:
                break
            collision_wp = next_wps[0]

        self._collision_wp = collision_wp

        # Get the crosswalk start point
        start_wp = collision_wp
        while start_wp.lane_type != carla.LaneType.Sidewalk:
            wp = start_wp.get_right_lane()
            if wp is None:
                raise ValueError("Couldn't find a waypoint to start the flow")
            start_wp = wp

        # Displace it to the crosswalk. Move forwards towards the crosswalk
        start_vec = start_wp.transform.get_forward_vector()
        start_right_vec = start_wp.transform.get_right_vector()

        spawn_loc = start_wp.transform.location + carla.Location(
            self._crosswalk_dist * start_vec.x + self._crosswalk_right_dist * start_right_vec.x,
            self._crosswalk_dist * start_vec.y + self._crosswalk_right_dist * start_right_vec.y,
            self._crosswalk_dist * start_vec.z + self._crosswalk_right_dist * start_right_vec.z + 1.0
        )

        spawn_rotation = start_wp.transform.rotation
        spawn_rotation.yaw += 270
        spawn_transform = carla.Transform(spawn_loc, spawn_rotation)

        adversary = CarlaDataProvider.request_new_actor('walker.*', spawn_transform)
        if adversary is None:
            raise ValueError("Failed to spawn an adversary")

        self._collision_dist = spawn_transform.location.distance(self._collision_wp.transform.location)

        self.other_actors.append(adversary)

    def _create_behavior(self):
        """
        After invoking this scenario, cyclist will wait for the user
        controlled vehicle to enter trigger distance region,
        the cyclist starts crossing the road once the condition meets,
        then after 60 seconds, a timeout stops the scenario
        """
        sequence = py_trees.composites.Sequence(name="CrossingActor")
        if self.route_mode:
            sequence.add_child(HandleJunctionScenario(
                clear_junction=False,
                clear_ego_entry=True,
                remove_entries=[],
                remove_exits=[],
                stop_entries=False,
                extend_road_exit=0
            ))

        collision_location = self._collision_wp.transform.location

        # Wait until ego is close to the adversary
        trigger_adversary = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE, name="TriggerAdversaryStart")
        trigger_adversary.add_child(InTimeToArrivalToLocation(
            self.ego_vehicles[0], self._reaction_time, collision_location))
        trigger_adversary.add_child(InTriggerDistanceToLocation(
            self.ego_vehicles[0], collision_location, self._min_trigger_dist))
        sequence.add_child(trigger_adversary)

        # Move the adversary
        move_distance = 2 * self._collision_dist  # Cross the whole road (supposing symetry in both directions)
        move_duration = move_distance / self._adversary_speed
        sequence.add_child(KeepVelocity(
            self.other_actors[0], self._adversary_speed,
            duration=move_duration, distance=move_distance, name="AdversaryCrossing"))

        # Remove everything
        sequence.add_child(ActorDestroy(self.other_actors[0], name="DestroyAdversary"))
        sequence.add_child(DriveDistance(self.ego_vehicles[0], self._ego_end_distance, name="EndCondition"))

        return sequence

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        if self.route_mode:
            return []
        return [CollisionTest(self.ego_vehicles[0])]