#!/usr/bin/env python

# Copyright (c) 2019 Intel Corporation
# Copyright (c) 2019-2022 Intel Corporation

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

from __future__ import print_function

import py_trees
import carla

from agents.navigation.local_planner import RoadOption

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (ActorDestroy,
                                                                      WaypointFollower,
                                                                      BasicAgentBehavior)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (InTriggerDistanceToLocation,
                                                                               InTimeToArrivalToLocation)
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.tools.background_manager import RemoveRoadLane

class CutInWithStaticVehicle(BasicScenario):

    """
    Cut in(with static vehicle) scenario synchronizes a vehicle that is parked at a side lane
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
        self.timeout = timeout
        if 'direction' in config.other_parameters:
            self._direction = config.other_parameters['direction']['value']
        else:
            self._direction = "right"
        super().__init__("CutInWithStaticVehicle",
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
            sideblocker_wp = self._blocker_wp.get_left_lane()
        else:
            sideblocker_wp = self._blocker_wp.get_right_lane()

        self._blocker_actor = CarlaDataProvider.request_new_actor(
            'vehicle.*', sideblocker_wp.transform, 'scenario', attribute_filter={'base_type': 'car', 'has_lights':True})
        if not self._blocker_actor:
            raise ValueError("Couldn't spawn the parked actor")
        self._blocker_actor.apply_control(carla.VehicleControl(hand_brake=True))
        self.other_actors.append(self._blocker_actor)

        collision_wps = self._reference_waypoint.next(self._cut_in_distance)
        if not collision_wps:
            raise ValueError("Couldn't find a proper position for the cut in vehicle")
        self._collision_wp = collision_wps[0]

        # GGet the parking direction of the car that will be cut in
        if self._direction == 'left':
            cutin_wp = self._collision_wp.get_left_lane()
        else:
            cutin_wp = self._collision_wp.get_right_lane()

        self._parked_actor = CarlaDataProvider.request_new_actor(
            'vehicle.*', cutin_wp.transform, 'scenario', attribute_filter={'base_type': 'car', 'has_lights':True})
        if not self._parked_actor:
            raise ValueError("Couldn't spawn the parked actor")
        self.other_actors.append(self._parked_actor)

        self._front_wps = self._collision_wp.next(self._front_distance)
        self._front_wp = self._front_wps[0]

        self._plan = [[self._collision_wp, RoadOption.STRAIGHT],
                      [self._front_wp, RoadOption.STRAIGHT] ]

    def _create_behavior(self):
        """
        After invoking this scenario, a parked vehicle will wait for the ego to
        be close-by, merging into its lane, forcing it to break.
        """
        sequence = py_trees.composites.Sequence(name="CrossingActor")
        if self.route_mode:
            other_car = self._wmap.get_waypoint(self.other_actors[1].get_location())
            sequence.add_child(RemoveRoadLane(other_car))

        collision_location = self._collision_wp.transform.location

        # Wait until ego is close to the adversary
        trigger_adversary = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE, name="TriggerAdversaryStart")
        trigger_adversary.add_child(InTimeToArrivalToLocation(
            self.ego_vehicles[0], self._reaction_time, collision_location))  
        trigger_adversary.add_child(InTriggerDistanceToLocation(
            self.ego_vehicles[0], collision_location, self._min_trigger_dist)) 

        sequence.add_child(trigger_adversary)

        # The adversary change the lane
        sequence.add_child(BasicAgentBehavior(self.other_actors[1], plan=self._plan))

        # Move the adversary
        cut_in = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE, name="Cut in behavior")
        other_car = self._wmap.get_waypoint(self.other_actors[1].get_location())
        sequence.add_child(RemoveRoadLane(other_car))
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

        
