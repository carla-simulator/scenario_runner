#!/usr/bin/env python

#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Collection of traffic scenarios where the ego vehicle (hero)
is making a left turn
"""

import carla
import py_trees
import numpy.random as random

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import ActorSourceSinkPair, TrafficLightFreezer
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import WaitEndIntersection
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.tools.scenario_helper import (generate_target_waypoint,
                                           get_junction_topology, 
                                           filter_junction_wp_direction,
                                           get_traffic_light_in_lane)

from leaderboard.utils.background_manager import Scenario8Manager


class SignalizedJunctionLeftTurn(BasicScenario):

    """
    Implementation class for Hero
    Vehicle turning left at signalized junction scenario,
    Traffic Scenario 08.

    This is a single ego vehicle scenario
    """

    timeout = 80  # Timeout of scenario in seconds

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=80):
        """
        Setup all relevant parameters and create scenario
        """
        self._world = world
        self._map = CarlaDataProvider.get_map()
        self._source_dist = 20
        self._source_dist_interval = [15, 25]
        self._sink_dist = 20
        self._rng = random.RandomState(2000)
        self._direction = 'opposite'
        self.timeout = timeout
        super(SignalizedJunctionLeftTurn, self).__init__("SignalizedJunctionLeftTurn",
                                                         ego_vehicles,
                                                         config,
                                                         world,
                                                         debug_mode,
                                                         criteria_enable=criteria_enable)

    def _initialize_actors(self, config):
        """
        Default initialization of other actors.
        Override this method in child class to provide custom initialization.
        """
        ego_location = config.trigger_points[0].location
        ego_wp = CarlaDataProvider.get_map().get_waypoint(ego_location)

        # Get the junction
        starting_wp = ego_wp
        while not starting_wp.is_junction:
            starting_wps = starting_wp.next(1.0)
            if len(starting_wps) == 0:
                raise ValueError("Failed to find junction as a waypoint with no next was detected")
            starting_wp = starting_wps[0]
        junction = starting_wp.get_junction()

        # Get the opposite entry lane wp
        entry_wps, _ = get_junction_topology(junction)
        source_entry_wps = filter_junction_wp_direction(starting_wp, entry_wps, self._direction)
        if not source_entry_wps:
            raise ValueError("Trying to find a lane in the {} direction but none was found".format(self._direction))

        # Get the source transform
        source_entry_wp = self._rng.choice(source_entry_wps)
        source_wps = source_entry_wp.previous(self._source_dist)
        if len(source_wps) == 0:
            raise ValueError("Failed to find a source location as a waypoint with no previous was detected")
        self._source_transform = source_wps[0].transform

        # Get the sink location
        sink_exit_wp = generate_target_waypoint(self._map.get_waypoint(self._source_transform.location), 0)
        sink_wps = sink_exit_wp.next(self._sink_dist)
        if len(sink_wps) == 0:
            raise ValueError("Failed to find a sink location as a waypoint with no next was detected")
        self._sink_location = sink_wps[0].transform.location

        # get traffic lights
        tls = self._world.get_traffic_lights_in_junction(junction.id)
        ego_tl = get_traffic_light_in_lane(ego_wp, tls)
        source_tl = get_traffic_light_in_lane(source_wps[0], tls)
        self._tl_dict = {}
        for tl in tls:
            if tl == ego_tl or tl == source_tl:
                self._tl_dict[tl] = carla.TrafficLightState.Green
            else:
                self._tl_dict[tl] = carla.TrafficLightState.Red

    def _create_behavior(self):
        """
        Hero vehicle is turning left in an urban area at a signalized intersection,
        where, a flow of actors coming straight is present.
        """

        root = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        root.add_child(WaitEndIntersection(self.ego_vehicles[0]))
        root.add_child(ActorSourceSinkPair(
            self._source_transform, self._sink_location, self._source_dist_interval, 2))
        root.add_child(TrafficLightFreezer(self._tl_dict))

        sequence = py_trees.composites.Sequence("Sequence Behavior")
        if CarlaDataProvider.get_ego_vehicle_route():
            sequence.add_child(Scenario8Manager(self._direction))
        sequence.add_child(root)

        return sequence

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        return [CollisionTest(self.ego_vehicles[0])]

    def __del__(self):
        self._traffic_light = None
        self.remove_all_actors()

