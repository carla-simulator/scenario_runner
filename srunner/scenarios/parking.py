#!/usr/bin/env python
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Parking scenario:
The scenario realizes the ego vehicle in a parking lot
and encountering a pedestrian and other static obstacles.
"""

from __future__ import print_function

import math
import py_trees
import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (ActorTransformSetter,
                                                                      ActorDestroy,
                                                                      AccelerateToVelocity,
                                                                      HandBrakeVehicle,
                                                                      KeepVelocity,
                                                                      StopVehicle)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (InTriggerDistanceToLocationAlongRoute,
                                                                               InTimeToArrivalToVehicle,
                                                                               DriveDistance)
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import Idle
from srunner.scenariomanager.timer import TimeOut
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.tools.scenario_helper import get_location_in_distance_from_wp


class ParkingScenario(BasicScenario):

    """
    This class holds everything required for a parking lot scenario
    The ego vehicle is passing through the parking lot and encounters
    a pedestrian and other static obstalces.

    This is a single ego vehicle scenario
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=60):
        """
        Setup all relevant parameters and create scenario
        """
        self._wmap = CarlaDataProvider.get_map()
        self._reference_waypoint = self._wmap.get_waypoint(config.trigger_points[0].location)
        # ego vehicle parameters
        self._ego_vehicle_distance_driven = 40

        # other vehicle parameters
        self._other_actor_target_velocity = 10
        # Timeout of scenario in seconds
        self.timeout = timeout

        super(ParkingScenario, self).__init__("ParkingScenario",
                                                       ego_vehicles,
                                                       config,
                                                       world,
                                                       debug_mode,
                                                       criteria_enable=criteria_enable)

    def _initialize_actors(self, config):
        """
        Custom initialization
        """
        _start_distance = 40
        location, _ = get_location_in_distance_from_wp(
                        self._reference_waypoint, _start_distance
        )
        waypoint = self._wmap.get_waypoint(location)

        self.spawn_adversary(waypoint.transform)

        blockers = {'':, }

        for blocker_name, blocker in blockers.items():
            self.spawn_blocker(blocker_name, blocker_transform, waypoint.transform)

    def spawn_adversary(self, start_transform):

        pass

    def spawn_blocker(self, prop_name, prop_transform, start_transform):
        pass

        self.world.get_blueprint_library().filter('static.prop.container')[0]


    def _create_behavior(self):
        """
        Only behavior here is to wait
        """
        actor_stand = Idle(15)

        end_condition = DriveDistance(
            self.ego_vehicles[0],
            self._ego_vehicle_distance_driven)

        # non leaf nodes
        scenario_sequence = py_trees.composites.Sequence()

        # building tree
        scenario_sequence.add_child(actor_stand)

        for i, _ in enumerate(self.other_actors):
            scenario_sequence.add_child(ActorDestroy(self.other_actors[i]))

        scenario_sequence.add_child(end_condition)

        return scenario_sequence

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
