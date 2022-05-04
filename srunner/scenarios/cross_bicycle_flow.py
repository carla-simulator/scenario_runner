#!/usr/bin/env python

# Copyright (c) 2018-2022 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Scenarios in which the ego has to cross a flow of bycicles
"""

from __future__ import print_function

import py_trees
import carla

from agents.navigation.global_route_planner import GlobalRoutePlanner

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import BicycleFlow, TrafficLightFreezer
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import WaitEndIntersection
from srunner.scenarios.basic_scenario import BasicScenario

from srunner.tools.scenario_helper import get_closest_traffic_light
from srunner.tools.background_manager import ClearJunction
from agents.navigation.local_planner import RoadOption


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


class CrossingBycicleFlow(BasicScenario):
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
        self.timeout = timeout

        self._start_flow = convert_dict_to_location(config.other_parameters['start_actor_flow'])
        self._end_dist_flow = 40  # m
        self._sink_distance = 2

        self._end_distance = 40

        if 'flow_speed' in config.other_parameters:
            self._flow_speed = float(config.other_parameters['flow_speed']['value'])
        else:
            self._flow_speed = 10 # m/s

        if 'source_dist_interval' in config.other_parameters:
            self._source_dist_interval = [
                float(config.other_parameters['source_dist_interval']['from']),
                float(config.other_parameters['source_dist_interval']['to'])
            ]
        else:
            self._source_dist_interval = [5, 7] # m

        if 'green_light_delay' in config.other_parameters:
            self._green_light_delay = float(config.other_parameters['green_light_delay']['value'])
        else:
            self._green_light_delay = 3 # s

        self._reference_waypoint = self._map.get_waypoint(config.trigger_points[0].location)

        super().__init__("CrossingBycicleFlow",
                         ego_vehicles,
                         config,
                         world,
                         debug_mode,
                         criteria_enable=criteria_enable)

    def _create_behavior(self):
        """
        Hero vehicle is entering a junction in an urban area, at a signalized intersection,
        while another actor runs a red lift, forcing the ego to break.
        """
        source_wp = self._map.get_waypoint(self._start_flow, lane_type=carla.LaneType.Biking)
        if not source_wp:
            raise ValueError("Couldn't find a biking lane")
        elif source_wp.transform.location.distance(self._start_flow) > 10:
            raise ValueError("Couldn't find a biking lane at the specified location")

        # Get the plan
        plan = []
        junction_id = -1
        plan_step = 0
        wp = source_wp
        while True:
            next_wps = wp.next(1)
            if not next_wps:
                raise ValueError("Couldn't find a proper plan for the bicycle flow")
            next_wp = next_wps
            wp = next_wp[0]
            plan.append([next_wp[0], RoadOption.LANEFOLLOW])

            if plan_step == 0 and wp.is_junction:
                junction_id = wp.get_junction().id
                plan_step += 1
            elif plan_step == 1 and not wp.is_junction:
                plan_step += 1
                exit_loc = wp.transform.location
            elif plan_step == 2 and exit_loc.distance(wp.transform.location) > self._end_dist_flow:
                break

        # Get the relevant traffic lights
        tls = self._world.get_traffic_lights_in_junction(junction_id)
        ego_tl = get_closest_traffic_light(self._reference_waypoint, tls)
        self._flow_tl_dict = {}
        self._init_tl_dict = {}
        for tl in tls:
            if tl == ego_tl:
                self._flow_tl_dict[tl] = carla.TrafficLightState.Green
                self._init_tl_dict[tl] = carla.TrafficLightState.Red
            else:
                self._flow_tl_dict[tl] = carla.TrafficLightState.Red
                self._init_tl_dict[tl] = carla.TrafficLightState.Red

        root = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        root.add_child(BicycleFlow(plan, self._source_dist_interval, self._sink_distance, self._flow_speed))

        # End condition, when the ego exits the junction
        end_condition = py_trees.composites.Sequence()
        end_condition.add_child(WaitEndIntersection(self.ego_vehicles[0]))
        root.add_child(end_condition)

        # Freeze the traffic lights to allow the flow to populate the junction
        tl_freezer_sequence = py_trees.composites.Sequence("Traffic Light Behavior")
        tl_freezer_sequence.add_child(TrafficLightFreezer(self._init_tl_dict, duration=self._green_light_delay))
        tl_freezer_sequence.add_child(TrafficLightFreezer(self._flow_tl_dict))
        root.add_child(tl_freezer_sequence)

        # Add the BackgroundActivity behaviors
        if not self.route_mode:
            return root

        sequence = py_trees.composites.Sequence()
        sequence.add_child(ClearJunction())
        sequence.add_child(root)
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
        Remove all actors and traffic lights upon deletion
        """
        self.remove_all_actors()
