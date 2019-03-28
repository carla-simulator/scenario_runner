#!/usr/bin/env python

#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import sys

import py_trees

import carla
from agents.navigation.local_planner import RoadOption

from srunner.scenariomanager.atomic_scenario_behavior import *
from srunner.scenariomanager.atomic_scenario_criteria import *
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenarios.basic_scenario import *
from srunner.scenarios.scenario_helper import *

from queue import Queue

TURN_LEFT_SIGNALIZED_JUNCTION_SCENARIOS = [
    "SignalizedJunctionLeftTurn"
]


class SignalizedJunctionLeftTurn(BasicScenario):

    """
    Implementation class for Hero
    Vehicle turning left at signalized junction scenario,
    Traffic Scenario 08.
    """

    def __init__(self, world, ego_vehicle, config, randomize=False, debug_mode=False, criteria_enable=True):
        """
        Setup all relevant parameters and create scenario
        """

        self._world = world
        self.category = "SignalizedJunctionLeftTurn"
        self.timeout = 60  # Timeout of scenario in seconds
        self._target_vel = 35
        self._drive_distance = 200
        self._traffic_light = None
        self._other_actor_transform = None
        self._sink_location = None
        self._blackboard_queue_name = 'SignalizedJunctionLeftTurn/actor_flow_queue'
        self._queue = Blackboard().set(self._blackboard_queue_name, Queue())

        super(SignalizedJunctionLeftTurn, self).__init__("SignalizedJunctionLeftTurn",
                                                         ego_vehicle,
                                                         config,
                                                         world,
                                                         debug_mode,
                                                         criteria_enable=criteria_enable)

        self._traffic_light = CarlaDataProvider.get_next_traffic_light(self.ego_vehicle, False)
        if self._traffic_light is None:
            print("No traffic light for the given location of the ego vehicle found")
            sys.exit(-1)
        self._traffic_light.set_state(carla.TrafficLightState.Green)
        self._traffic_light.set_green_time(self.timeout)
        # other vehicle's traffic light
        traffic_light_other = CarlaDataProvider.get_next_traffic_light(self.other_actors[0], False)
        if traffic_light_other is None:
            print("No traffic light for the given location of the other vehicle found")
            sys.exit(-1)
        traffic_light_other.set_state(carla.TrafficLightState.Green)
        traffic_light_other.set_green_time(self.timeout)

    def _initialize_actors(self, config):
        """
        Custom initialization
        """
        self._other_actor_transform = config.other_actors[0].transform
        first_vehicle_transform = carla.Transform(
            carla.Location(config.other_actors[0].transform.location.x,
                           config.other_actors[0].transform.location.y,
                           config.other_actors[0].transform.location.z - 5),
            config.other_actors[0].transform.rotation)
        first_vehicle = CarlaActorPool.request_new_actor(config.other_actors[0].model, first_vehicle_transform)
        self.other_actors.append(first_vehicle)

    def _create_behavior(self):
        """
        Hero vehicle is turning left in an urban area,
        at a signalized intersection and cuts across the path of another vehicle
        coming straight crossing from an opposite direction,
        After 80 seconds, a timeout stops the scenario.
        """

        root = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)

        actor_source = ActorSource(
            self._world, ['vehicle.audi.tt', 'vehicle.tesla.model3', 'vehicle.nissan.micra'],
            self._other_actor_transform, 20, self._blackboard_queue_name)

        # Selecting straight path at intersection
        target_waypoint = generate_target_waypoint(
            CarlaDataProvider.get_map().get_waypoint(self.other_actors[0].get_location()), 0)

        # Generating waypoint list till next intersection
        plan = []
        wp_choice = target_waypoint.next(1.0)
        while len(wp_choice) == 1:
            target_waypoint = wp_choice[0]
            plan.append((target_waypoint, RoadOption.LANEFOLLOW))
            wp_choice = target_waypoint.next(1.0)

        waypoint_follower = WaypointFollower(
            self.other_actors[0], self._target_vel, plan=plan, blackboard_queue_name=self._blackboard_queue_name, avoid_collision=False)

        sink_waypoint = target_waypoint.next(1)[0]
        while not sink_waypoint.is_intersection:
            sink_waypoint = sink_waypoint.next(1)[0]
        self._sink_location = sink_waypoint.transform.location

        actor_sink = ActorSink(self._world, self._sink_location, 10)

        drive_distance = DriveDistance(self.ego_vehicle, self._drive_distance)

        root.add_child(actor_source)
        root.add_child(waypoint_follower)
        root.add_child(actor_sink)
        root.add_child(drive_distance)

        return root

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
        """
        Remove all actors upon deletion
        """
        self.remove_all_actors()
