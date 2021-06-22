#!/usr/bin/env python

# Copyright (c) 2018-2020 Intel Corporation
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
import numpy.random as random
from agents.navigation.local_planner import RoadOption

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (ActorTransformSetter,
                                                                      ActorDestroy,
                                                                      Idle,
                                                                      SyncArrival,
                                                                      BasicAgentBehavior,
                                                                      TrafficLightFreezer)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (WaitEndIntersection,
                                                                               InTriggerDistanceToLocation)
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.tools.scenario_helper import (get_geometric_linear_intersection,
                                           generate_target_waypoint,
                                           get_junction_topology,
                                           filter_junction_wp_direction,
                                           get_traffic_light_in_lane)

from leaderboard.utils.background_manager import Scenario7Manager

class OppositeVehicleRunningRedLight(BasicScenario):
    """
    This class holds everything required for a scenario in which another vehicle runs a red light
    in front of the ego, forcing it to react. This vehicles are 'special' ones such as police cars,
    ambulances or firetrucks.
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=180):
        """
        Setup all relevant parameters and create scenario
        and instantiate scenario manager
        """
        self._world = world
        self._map = CarlaDataProvider.get_map()
        self._source_dist = 20
        self._exit_speed = 30
        self._sink_dist = 20
        self._sync_stop_dist = 10
        self._direction = self._get_other_actor_direction(config)
        self._opposite_bp_wildcards = ['*firetruck*', '*ambulance*', '*police*']  #Wildcard patterns of the blueprints
        self.timeout = timeout
        self._rng = random.RandomState(2000)

        super(OppositeVehicleRunningRedLight, self).__init__("OppositeVehicleRunningRedLight",
                                                             ego_vehicles,
                                                             config,
                                                             world,
                                                             debug_mode,
                                                             criteria_enable=criteria_enable)

    def _get_other_actor_direction(self, config):
        """Returns the direction the other actor will come through"""
        if not CarlaDataProvider.get_ego_vehicle_route():
            return None
        subtype = config.subtype.lower()
        if 'left' in subtype:
            return 'left'
        if 'right' in subtype:
            return 'right'
        if 'opposite' in subtype:
            return 'opposite'
        raise ValueError("Unknown scenario subtype")

    def _initialize_actors(self, config):
        """
        Custom initialization
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
        if self._direction:
            # At routes, use the given subtype
            entry_wps, _ = get_junction_topology(junction)
            source_entry_wps = filter_junction_wp_direction(starting_wp, entry_wps, self._direction)
            if not source_entry_wps:
                raise ValueError("Couldn't find a lane in the {} direction".format(self._direction))
        else:
            # Outside routes, test different directions
            for direction in ['right', 'left']:
                entry_wps, _ = get_junction_topology(junction)
                source_entry_wps = filter_junction_wp_direction(starting_wp, entry_wps, direction)
                if source_entry_wps:
                    self._direction = direction
                    break
            if not self._direction:
                raise ValueError("Trying to find a lane to spawn the opposite actor but none was found")

        # Get the source transform
        self._entry_plan = []
        source_wp = source_entry_wps[0]
        added_dist = self._source_dist
        while added_dist > 0:
            source_wps = source_wp.previous(1.0)
            if len(source_wps) == 0:
                raise ValueError("Failed to find a source location as a waypoint with no previous was detected")
            source_wp = source_wps[0]
            self._entry_plan.insert(0, ([source_wp, RoadOption.LANEFOLLOW]))
            added_dist -=1

        source_transform = source_wp.transform
        self._spawn_location = carla.Transform(
            source_transform.location + carla.Location(z=0.1),
            source_transform.rotation
        )
        opposite_bp_wildcard = self._rng.choice(self._opposite_bp_wildcards)
        opposite_actor = CarlaDataProvider.request_new_actor(opposite_bp_wildcard, self._spawn_location)
        if not opposite_actor:
            raise Exception("Couldn't spawn the actor")
        self.other_actors.append(opposite_actor)

        opposite_transform = carla.Transform(
            source_transform.location - carla.Location(z=500),
            source_transform.rotation
        )
        opposite_actor.set_transform(opposite_transform)
        opposite_actor.set_simulate_physics(enabled=False)

        # Get the out of junction plan
        sink_exit_wp = generate_target_waypoint(self._map.get_waypoint(source_transform.location), 0)
        self._exit_plan = []
        next_wp = sink_exit_wp
        added_dist = self._sink_dist
        while added_dist > 0:
            next_wps = next_wp.next(1.0)
            if len(next_wps) == 0:
                break
            next_wp = next_wps[0]
            self._exit_plan.append([next_wp, RoadOption.LANEFOLLOW])
            added_dist -= 1

        self._collision_location = get_geometric_linear_intersection(
            starting_wp.transform.location, source_entry_wp.transform.location)
        collision_waypoint = self._map.get_waypoint(self._collision_location)
        self._entry_plan.append([collision_waypoint, RoadOption.LANEFOLLOW])

        # Get the relevant traffic lights
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
        Hero vehicle is entering a junction in an urban area, at a signalized intersection,
        while another actor runs a red lift, forcing the ego to break.
        """
        # First part of the behavior, synchronize the actors
        sync_arrival = py_trees.composites.Parallel("Synchronize actors",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        sync_arrival.add_child(
            SyncArrival(self.other_actors[0], self.ego_vehicles[0], self._collision_location))
        sync_arrival.add_child(
            InTriggerDistanceToLocation(self.other_actors[0], self._collision_location, self._sync_stop_dist))

        # Second part, move the other actor out of the way
        move_actor_exit = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        move_actor_exit.add_child(
            BasicAgentBehavior(self.other_actors[0], plan=self._exit_plan, target_speed=self._exit_speed))
        move_actor_exit.add_child(
            InTriggerDistanceToLocation(self.other_actors[0], self._exit_plan[-1][0].transform.location, 10))
        move_actor_exit.add_child(Idle(15))  # In case both actors crash and get stuck

        # Behavior tree
        sequence = py_trees.composites.Sequence()
        sync_arrival.add_child(
            SyncArrival(self.other_actors[0], self.ego_vehicles[0], self._collision_location, self._entry_plan))
        sequence.add_child(sync_arrival)
        sequence.add_child(move_actor_exit)

        main_behavior = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        main_behavior.add_child(TrafficLightFreezer(self._tl_dict))
        main_behavior.add_child(sequence)

        root = py_trees.composites.Sequence()
        if CarlaDataProvider.get_ego_vehicle_route():
            root.add_child(Scenario7Manager(self._direction))
        root.add_child(ActorTransformSetter(self.other_actors[0], self._spawn_location))
        root.add_child(main_behavior)
        root.add_child(ActorDestroy(self.other_actors[0]))
        root.add_child(WaitEndIntersection(self.ego_vehicles[0]))

        return root

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        return [CollisionTest(self.ego_vehicles[0])]

    def __del__(self):
        """
        Remove all actors and traffic lights upon deletion
        """
        self._traffic_light = None
        self.remove_all_actors()
