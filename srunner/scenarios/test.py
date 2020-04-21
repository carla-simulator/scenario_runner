#!/usr/bin/env python
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
In this scenario, the ego vehicle is located at the
rightmost lane of a freeway but a vehicle that is on its
left lane wants to take the exit and cuts in from the left
"""

import py_trees
import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider, CarlaActorPool
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import SetRelativeOSCVelocity
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import DriveDistance
from srunner.scenarios.basic_scenario import BasicScenario


class Test(BasicScenario):

    """
    In this scenario, the ego vehicle is located at the rightmost lane of a freeway but
    a vehicle that is on its left lane wants to take the exit and cuts in from the left
    """

    timeout = 80

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=80):

        self.timeout = timeout
        self._map = CarlaDataProvider.get_map()
        self._reference_waypoint = self._map.get_waypoint(config.trigger_points[0].location)

        super(Test, self).__init__("Test",
                                    ego_vehicles,
                                    config,
                                    world,
                                    debug_mode,
                                    criteria_enable=criteria_enable)

    def _initialize_actors(self, config):

        # Spawn the other actor
        other_actor_transform = config.other_actors[0].transform
        other_vehicle = CarlaActorPool.request_new_actor("vehicle.lincoln.mkz2017", other_actor_transform)
        self.other_actors.append(other_vehicle)

    def _create_behavior(self):
        """
        """

        behaviour = py_trees.composites.Sequence("Test")

        # Make the vehicle "sync" with the ego_vehicle
        behaviour.add_child(SetRelativeOSCVelocity(self.other_actors[0],
                                                   self.ego_vehicles[0],
                                                   0,
                                                   'delta',
                                                   True,
                                                   distance = 70,
                                                   name="SetRelativeVelocity"))

        return behaviour

    def _create_test_criteria(self):
        """
        A list of all test criteria is created, which is later used in the parallel behavior tree.
        """
        criteria = []

        collision_criterion = CollisionTest(self.ego_vehicles[0])

        criteria.append(collision_criterion)

        return criteria

    def __del__(self):
        """
        Remove all actors after deletion.
        """
        self.remove_all_actors()
