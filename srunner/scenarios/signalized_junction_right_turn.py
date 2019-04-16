#!/usr/bin/env python

#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Collection of traffic scenarios where the ego vehicle (hero)
is making a right turn
"""

from __future__ import print_function
from six.moves.queue import Queue

import carla
import py_trees
from agents.navigation.local_planner import RoadOption

from srunner.scenariomanager.atomic_scenario_behavior import *
from srunner.scenariomanager.atomic_scenario_criteria import *
from srunner.scenarios.basic_scenario import *
from srunner.tools.scenario_helper import *


TURNING_RIGHT_SIGNALIZED_JUNCTION_SCENARIOS = [
    "SignalizedJunctionRightTurn"
]


class SignalizedJunctionRightTurn(BasicScenario):

    """
    Implementation class for Hero
    Vehicle turning right at signalized junction scenario,
    Traffic Scenario 09.
    """
    category = "SignalizedJunctionRightTurn"

    timeout = 90  # Timeout of scenario in seconds

    def __init__(self, world, ego_vehicle, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=80):
        """
        Setup all relevant parameters and create scenario
        """
        self._world = world
        self._map = CarlaDataProvider.get_map()
        self._target_vel = 25
        self._brake_value = 0.5
        self._ego_distance = 110
        self._traffic_light = None
        self._other_actor_transform = None
        self._blackboard_queue_name = 'SignalizedJunctionRightTurn/actor_flow_queue'
        self._queue = Blackboard().set(self._blackboard_queue_name, Queue())
        # Timeout of scenario in seconds
        self.timeout = timeout
        self._initialized = True
        super(SignalizedJunctionRightTurn, self).__init__("HeroActorTurningRightAtSignalizedJunction",
                                                          ego_vehicle,
                                                          config,
                                                          world,
                                                          debug_mode,
                                                          criteria_enable=criteria_enable)

        self._traffic_light = CarlaDataProvider.get_next_traffic_light(self.ego_vehicle, False)
        traffic_light_other = CarlaDataProvider.get_next_traffic_light(self.other_actors[0], False)
        if self._traffic_light is None or traffic_light_other is None:
            print("No traffic light for the given location found")
            self._initialized = False
            return
        self._traffic_light.set_state(carla.TrafficLightState.Green)
        self._traffic_light.set_green_time(self.timeout)
        # other vehicle's traffic light
        traffic_light_other = CarlaDataProvider.get_next_traffic_light(self.other_actors[0], False)
        traffic_light_other.set_state(carla.TrafficLightState.Green)
        traffic_light_other.set_green_time(self.timeout)

    def _initialize_actors(self, config):
        """
        Custom initialization
        """
        if self._initialized:
            self._other_actor_transform = config.other_actors[0].transform
            first_vehicle_transform = carla.Transform(
                carla.Location(config.other_actors[0].transform.location.x,
                               config.other_actors[0].transform.location.y,
                               config.other_actors[0].transform.location.z - 500),
                config.other_actors[0].transform.rotation)
            first_vehicle = CarlaActorPool.request_new_actor(config.other_actors[0].model, first_vehicle_transform)
            self.other_actors.append(first_vehicle)

    def _create_behavior(self):
        """
        Hero vehicle is turning right in an urban area,
        at a signalized intersection, while other actor coming straight
        from left.The hero actor may turn right either before other actor
        passes intersection or later, without any collision.
        After 80 seconds, a timeout stops the scenario.
        """
        sequence = py_trees.composites.Sequence("Sequence Behavior")

        if self._initialized:
            # Selecting straight path at intersection
            target_waypoint = generate_target_waypoint(
                CarlaDataProvider.get_map().get_waypoint(self.other_actors[0].get_location()), 0)
            # Generating waypoint list till next intersection
            plan = []
            wp_choice = target_waypoint.next(1.0)
            while not wp_choice[0].is_intersection:
                target_waypoint = wp_choice[0]
                plan.append((target_waypoint, RoadOption.LANEFOLLOW))
                wp_choice = target_waypoint.next(1.0)
            # adding flow of actors
            actor_source = ActorSource(
                self._world, ['vehicle.*', 'vehicle.nissan.patrol', 'vehicle.nissan.micra'],
                self._other_actor_transform, 15, self._blackboard_queue_name)
            # destroying flow of actors
            actor_sink = ActorSink(self._world, plan[-1][0].transform.location, 10)
            # follow waypoints untill next intersection
            move_actor = WaypointFollower(self.other_actors[0], self._target_vel, plan=plan,
                                          blackboard_queue_name=self._blackboard_queue_name, avoid_collision=True)
            # wait
            wait = DriveDistance(self.ego_vehicle, self._ego_distance)

            # Behavior tree
            root = py_trees.composites.Parallel(
                policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
            root.add_child(wait)
            root.add_child(actor_source)
            root.add_child(actor_sink)
            root.add_child(move_actor)

            sequence.add_child(ActorTransformSetter(self.other_actors[0], self._other_actor_transform))
            sequence.add_child(root)
            sequence.add_child(ActorDestroy(self.other_actors[0]))

        return sequence

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = []

        collison_criteria = CollisionTest(self.ego_vehicle)
        criteria.append(collison_criteria)

        return criteria

    def __del__(self):
        self._traffic_light = None
        self.remove_all_actors()
