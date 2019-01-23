#!/usr/bin/env python

# Copyright (c) 2018-2019 Intel Labs.
# authors: Fabian Oboril (fabian.oboril@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Scenarios in which another (opposite) vehicle 'illegally' takes
priority, e.g. by running a red traffic light.
"""

from __future__ import print_function
import sys

import py_trees
import carla

from ScenarioManager.atomic_scenario_behavior import *
from ScenarioManager.atomic_scenario_criteria import *
from Scenarios.basic_scenario import *


RUNNING_RED_LIGHT_SCENARIOS = [
    "OppositeVehicleRunningRedLight"
]


class OppositeVehicleRunningRedLight(BasicScenario):

    """
    This class holds everything required for a scenario,
    in which an other vehicle takes priority from the ego
    vehicle, by running a red traffic light (while the ego
    vehicle has green)
    """

    timeout = 180            # Timeout of scenario in seconds

    # ego vehicle parameters
    _ego_max_velocity_allowed = 20       # Maximum allowed velocity [m/s]
    _ego_avg_velocity_expected = 4       # Average expected velocity [m/s]
    _ego_expected_driven_distance = 88   # Expected driven distance [m]
    _ego_distance_to_traffic_light = 15  # Trigger distance to traffic light [m]
    _ego_end_position = carla.Location(x=-3, y=-90, z=0)   # End position
    _ego_distance_to_end_position = 5    # Allowed distance to end position [m]
    _intersection_location = carla.Location(x=-3, y=-150, z=0)

    # other vehicle
    _other_actor_target_velocity = 15      # Target velocity of other vehicle
    _other_actor_max_brake = 1.0           # Maximum brake of other vehicle
    _other_actor_distance = 30             # Distance the other vehicle should drive

    _traffic_light_location = carla.Location(x=-11.5, y=-125.0, z=0.15)
    _traffic_light = None
    _location_of_collision = carla.Location(x=0, y=-135, z=1)

    def __init__(self, world, ego_vehicle, other_actors, town, debug_mode=False):
        """
        Setup all relevant parameters and create scenario
        and instantiate scenario manager
        """

        for actor in world.get_actors().filter('traffic.traffic_light'):
            if actor.get_location().distance(self._traffic_light_location) < 1.0:
                self._traffic_light = actor

        if self._traffic_light is None:
            print("No traffic light for the given location found")
            sys.exit(-1)

        super(OppositeVehicleRunningRedLight, self).__init__("OppositeVehicleRunningRedLight",
                                                             ego_vehicle,
                                                             other_actors,
                                                             town,
                                                             world,
                                                             debug_mode)

    def _create_behavior(self):
        """
        Scenario behavior:
        The other vehicle waits until the ego vehicle is close enough to the
        intersection and that its own traffic light is red. Then, it will start
        driving and 'illegally' cross the intersection. After a short distance
        it should stop again, outside of the intersection. The ego vehicle has
        to avoid the crash, but continue driving after the intersection is clear.

        If this does not happen within 120 seconds, a timeout stops the scenario
        """

        # start condition
        startcondition = InTriggerDistanceToLocation(
            self.ego_vehicle,
            self._intersection_location,
            self._ego_distance_to_traffic_light,
            name="Waiting for start position")

        # wait until traffic light for ego vehicle is green
        wait_for_green = WaitForTrafficLightState(self._traffic_light, "Green")

        sync_arrival_parallel = py_trees.composites.Parallel(
            "Synchronize arrival times",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        sync_arrival = SyncArrival(
            self.other_actors[0], self.ego_vehicle, self._location_of_collision)
        sync_arrival_stop = InTriggerDistanceToVehicle(self.other_actors[0],
                                                       self.ego_vehicle,
                                                       15)
        sync_arrival_parallel.add_child(sync_arrival)
        sync_arrival_parallel.add_child(sync_arrival_stop)

        keep_velocity_for_distance = py_trees.composites.Parallel(
            "Keep velocity for distance",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        keep_velocity = KeepVelocity(
            self.other_actors[0],
            self._other_actor_target_velocity)
        keep_velocity_distance = DriveDistance(
            self.other_actors[0],
            self._other_actor_distance,
            name="Distance")
        keep_velocity_for_distance.add_child(keep_velocity)
        keep_velocity_for_distance.add_child(keep_velocity_distance)

        # finally wait that ego vehicle reached target position
        wait = InTriggerDistanceToLocation(
            self.ego_vehicle,
            self._ego_end_position,
            self._ego_distance_to_end_position,
            name="Waiting for end position")

        # Build behavior tree
        sequence = py_trees.composites.Sequence("Sequence Behavior")
        sequence.add_child(startcondition)
        sequence.add_child(wait_for_green)
        sequence.add_child(sync_arrival_parallel)
        sequence.add_child(keep_velocity_for_distance)
        sequence.add_child(wait)

        return sequence

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = []

        max_velocity_criterion = MaxVelocityTest(
            self.ego_vehicle,
            self._ego_max_velocity_allowed,
            optional=True)
        collision_criterion = CollisionTest(self.ego_vehicle)
        driven_distance_criterion = DrivenDistanceTest(
            self.ego_vehicle,
            self._ego_expected_driven_distance)

        criteria.append(max_velocity_criterion)
        criteria.append(collision_criterion)
        criteria.append(driven_distance_criterion)

        # Add the collision and lane checks for all vehicles as well
        for vehicle in self.other_actors:
            collision_criterion = CollisionTest(vehicle)
            criteria.append(collision_criterion)

        return criteria

    def __del__(self):
        self._traffic_light = None
