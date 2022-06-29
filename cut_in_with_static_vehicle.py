#!/usr/bin/env python
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Parking cut in scenario synchronizes a vehicle that is parked at a side lane
to cut in in front of the ego vehicle, forcing it to break
"""

from __future__ import print_function

import py_trees
import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (ActorDestroy,
                                                    WaypointFollower,
                                                    BasicAgentBehavior
                                                    )
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (InTriggerDistanceToLocation,
                                                                               InTimeToArrivalToLocation,
                                                                               DriveDistance)
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.tools.background_manager import SwitchLane

class ParkingCutIn(BasicScenario):

    """
    Parking cut in scenario synchronizes a vehicle that is parked at a side lane
    to cut in in front of the ego vehicle, forcing it to break
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True, timeout=60):
        """
        Setup all relevant parameters and create scenario
        """
        self._wmap = CarlaDataProvider.get_map()
        self._trigger_location = config.trigger_points[0].location
        self._reference_waypoint = self._wmap.get_waypoint(self._trigger_location)
        self._cut_in_distance =  25
        self._blocker_distance = 16
        self._front_distance = 30
        self._adversary_speed = 10.0  # Speed of the adversary [m/s]
        self._reaction_time = 2.5  # Time the agent has to react to avoid the collision [s]
        self._min_trigger_dist = 18.0  # Min distance to the collision location that triggers the adversary [m]
        self._end_distance = 10 # 40
        self.timeout = timeout
        if 'direction' in config.other_parameters:
            self._direction = config.other_parameters['direction']['value']
        else:
            self._direction = "right"
        super().__init__("ParkingCutIn",
                         ego_vehicles,
                         config,
                         world,
                         debug_mode,
                         criteria_enable=criteria_enable)

    def _initialize_actors(self, config):
        """
        Custom initialization
        """
        # Spawn the blocker vehicle
        blocker_wps = self._reference_waypoint.next(self._blocker_distance)
        if not blocker_wps:
            raise ValueError("Couldn't find a proper position for the cut in vehicle")
        self._blocker_wp = blocker_wps[0]
        if self._direction == 'left':
            parking_wp = self._blocker_wp.get_left_lane()
        else:
            parking_wp = self._blocker_wp.get_right_lane()

        self._blocker_actor = CarlaDataProvider.request_new_actor(
            'vehicle.*', parking_wp.transform, 'scenario', attribute_filter={'base_type': 'car', 'has_lights':True})
        if not self._blocker_actor:
            raise ValueError("Couldn't spawn the parked actor")
        self._blocker_actor.apply_control(carla.VehicleControl(hand_brake=True))
        self.other_actors.append(self._blocker_actor)

        side_transform = self._get_displaced_transform(self._blocker_actor, parking_wp)
        self._blocker_actor.set_location(side_transform)

        collision_wps = self._reference_waypoint.next(self._cut_in_distance)
        if not collision_wps:
            raise ValueError("Couldn't find a proper position for the cut in vehicle")
        self._collision_wp = collision_wps[0]

        # Get the parking direction
        if self._direction == 'left':
            parking_wp = self._collision_wp.get_left_lane()
        else:
            parking_wp = self._collision_wp.get_right_lane()

        self._parked_actor = CarlaDataProvider.request_new_actor(
            'vehicle.*', parking_wp.transform, 'scenario', attribute_filter={'base_type': 'car', 'has_lights':True})
        if not self._parked_actor:
            raise ValueError("Couldn't spawn the parked actor")
        self.other_actors.append(self._parked_actor)

        side_transform = self._get_displaced_transform(self._parked_actor, parking_wp)
        self._parked_actor.set_location(side_transform)

        self._front_wps = self._collision_wp.next(self._front_distance)
        self._front_wp = self._front_wps[0].transform.location

    def _get_displaced_transform(self, actor, wp):
        """
        Calculates the transforming such that the actor is at the sidemost part of the lane
        """
        # Move the actor to the edge of the lane near the sidewalk
        displacement = (wp.lane_width - actor.bounding_box.extent.y) / 4
        displacement_vector = wp.transform.get_right_vector()
        if self._direction == 'left':
            displacement_vector *= -1

        new_location = wp.transform.location + carla.Location(x=displacement*displacement_vector.x,
                                                              y=displacement*displacement_vector.y,
                                                              z=displacement*displacement_vector.z)
        new_location.z += 0.05  # Just in case, avoid collisions with the ground
        return new_location

    def _create_behavior(self):
        """
        After invoking this scenario, a parked vehicle will wait for the ego to
        be close-by, merging into its lane, forcing it to break.
        """
        sequence = py_trees.composites.Sequence(name="CrossingActor")
        if self.route_mode:
            other_car = self._wmap.get_waypoint(self.other_actors[1].get_location())
            sequence.add_child(SwitchLane(other_car.lane_id, False))

        collision_location = self._collision_wp.transform.location

        # Wait until ego is close to the adversary
        trigger_adversary = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE, name="TriggerAdversaryStart")
        trigger_adversary.add_child(InTimeToArrivalToLocation(
            self.ego_vehicles[0], self._reaction_time, collision_location))  #<reaction_time :Success
        trigger_adversary.add_child(InTriggerDistanceToLocation(
            self.ego_vehicles[0], collision_location, self._min_trigger_dist)) # A < B :Success

        sequence.add_child(trigger_adversary)

        # The adversary change the lane
        sequence.add_child(BasicAgentBehavior(self.other_actors[1],target_location=self._front_wp))

        # Move the adversary
        cut_in = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE, name="Cut in behavior")
        other_car = self._wmap.get_waypoint(self.other_actors[1].get_location())
        sequence.add_child(SwitchLane(other_car.lane_id, True))
        cut_in.add_child(WaypointFollower(self.other_actors[1], self._adversary_speed))

        sequence.add_child(cut_in)

        # Remove everything
        sequence.add_child(ActorDestroy(self.other_actors[0], name="DestroyAdversary"))
        sequence.add_child(ActorDestroy(self.other_actors[1], name="DestroyBlocker"))

        return sequence

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        if self.route_mode:
            return []
        return [CollisionTest(self.ego_vehicles[0])]

    def __del__(self):
        """
        Remove all actors upon deletion
        """
        self.remove_all_actors()