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

import py_trees
import carla

from agents.navigation.global_route_planner import GlobalRoutePlanner

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import ActorFlow
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import InTriggerDistanceToLocation, WaitEndIntersection
from srunner.scenarios.basic_scenario import BasicScenario

from srunner.tools.background_manager import ChangeRoadBehavior, JunctionScenarioManager, ExtentExitRoadSpace, StopEntries
from srunner.tools.scenario_helper import get_same_dir_lanes

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

class EnterActorFlow(BasicScenario):
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

        self._start_actor_flow = convert_dict_to_location(config.other_parameters['start_actor_flow'])
        self._end_actor_flow = convert_dict_to_location(config.other_parameters['end_actor_flow'])
        self._sink_distance = 2

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

        super(EnterActorFlow, self).__init__("EnterActorFlow",
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
        source_wp = self._map.get_waypoint(self._start_actor_flow)
        sink_wp = self._map.get_waypoint(self._end_actor_flow)

        # Get all lanes
        source_wps = get_same_dir_lanes(source_wp)
        sink_wps = get_same_dir_lanes(sink_wp)
        num_flows = min(len(source_wps), len(sink_wps))

        root = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        for i in range(num_flows):
            root.add_child(ActorFlow(
                source_wps[i], sink_wps[i], self._source_dist_interval, self._sink_distance, self._flow_speed))
            root.add_child(InTriggerDistanceToLocation(self.ego_vehicles[0], sink_wps[i].transform.location, self._sink_distance))

        sequence = py_trees.composites.Sequence()
        if CarlaDataProvider.get_ego_vehicle_route():
            sequence.add_child(JunctionScenarioManager('left', True, False, True))

            grp = GlobalRoutePlanner(CarlaDataProvider.get_map(), 2.0)
            route = grp.trace_route(source_wp.transform.location, sink_wp.transform.location)
            extra_space = 0
            for i in range(-2, -len(route)-1, -1):
                current_wp = route[i][0]
                extra_space += current_wp.transform.location.distance(route[i+1][0].transform.location)
                if current_wp.is_junction:
                    sequence.add_child(ExtentExitRoadSpace(distance=extra_space, direction='left'))
                    break

            sequence.add_child(ChangeRoadBehavior(switch_source=False))
        sequence.add_child(root)
        if CarlaDataProvider.get_ego_vehicle_route():
            sequence.add_child(ChangeRoadBehavior(switch_source=True))

        return sequence

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
        self.remove_all_actors()


class CrossActorFlow(BasicScenario):
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

        self._start_actor_flow = convert_dict_to_location(config.other_parameters['start_actor_flow'])
        self._end_actor_flow = convert_dict_to_location(config.other_parameters['end_actor_flow'])
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

        super(CrossActorFlow, self).__init__("CrossActorFlow",
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
        source_wp = self._map.get_waypoint(self._start_actor_flow)
        sink_wp = self._map.get_waypoint(self._end_actor_flow)

        self.world.debug.draw_point(source_wp.transform.location + carla.Location(z=3))
        self.world.debug.draw_point(sink_wp.transform.location + carla.Location(z=3))

        grp = GlobalRoutePlanner(CarlaDataProvider.get_map(), 2.0)
        route = grp.trace_route(source_wp.transform.location, sink_wp.transform.location)
        junction_id = None
        for wp, _ in route:
            if wp.is_junction:
                junction_id = wp.get_junction().id
                break

        root = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        root.add_child(ActorFlow(
            source_wp, sink_wp, self._source_dist_interval, self._sink_distance, self._flow_speed))
        end_condition = py_trees.composites.Sequence()
        end_condition.add_child(WaitEndIntersection(self.ego_vehicles[0], junction_id))
        
        root.add_child(end_condition)

        sequence = py_trees.composites.Sequence()
        if CarlaDataProvider.get_ego_vehicle_route():
            sequence.add_child(JunctionScenarioManager('opposite', True, True, True))
            sequence.add_child(StopEntries())
        sequence.add_child(root)

        return sequence

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
        self.remove_all_actors()
