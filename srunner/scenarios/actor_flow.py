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

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import ActorFlow
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import InTriggerDistanceToLocation, WaitEndIntersection
from srunner.scenarios.basic_scenario import BasicScenario

from srunner.tools.background_manager import (SwitchRouteSources,
                                              ChangeOppositeBehavior,
                                              HandleJunctionScenario,
                                              RemoveRoadLane)
from srunner.tools.scenario_helper import get_same_dir_lanes, generate_target_waypoint_in_route

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

        ego_location = config.trigger_points[0].location
        self._reference_waypoint = CarlaDataProvider.get_map().get_waypoint(ego_location)

        super().__init__("EnterActorFlow",
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

        root = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)

        for source_wp, sink_wp in zip(source_wps, sink_wps):
            root.add_child(InTriggerDistanceToLocation(self.ego_vehicles[0], sink_wp.transform.location, self._sink_distance))
            root.add_child(ActorFlow(
                source_wp, sink_wp, self._source_dist_interval, self._sink_distance, self._flow_speed, add_initial_actors=True))

        sequence = py_trees.composites.Sequence()
        if self.route_mode:
            grp = CarlaDataProvider.get_global_route_planner()
            route = grp.trace_route(source_wp.transform.location, sink_wp.transform.location)
            extra_space = 0
            for i in range(-2, -len(route)-1, -1):
                current_wp = route[i][0]
                extra_space += current_wp.transform.location.distance(route[i+1][0].transform.location)
                if current_wp.is_junction:
                    break

            sequence.add_child(HandleJunctionScenario(
                clear_junction=False,
                clear_ego_entry=True,
                remove_entries=source_wps,
                remove_exits=[],
                stop_entries=False,
                extend_road_exit=extra_space
            ))
            sequence.add_child(SwitchRouteSources(False))
        sequence.add_child(root)
        if self.route_mode:
            sequence.add_child(SwitchRouteSources(True))

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


class EnterActorFlowV2(EnterActorFlow):
    """
    Variation of EnterActorFlow for special highway entry exits with dedicated lanes
    """
    def _create_behavior(self):
        """
        Hero vehicle is entering a junction in an urban area, at a signalized intersection,
        while another actor runs a red lift, forcing the ego to break.
        """
        source_wp = self._map.get_waypoint(self._start_actor_flow)
        sink_wp = self._map.get_waypoint(self._end_actor_flow)

        # Get all lanes
        sink_wps = get_same_dir_lanes(sink_wp)

        root = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        root.add_child(ActorFlow(
                source_wp, sink_wp, self._source_dist_interval, self._sink_distance, self._flow_speed, add_initial_actors=True))
        for sink_wp in sink_wps:
            root.add_child(InTriggerDistanceToLocation(self.ego_vehicles[0], sink_wp.transform.location, self._sink_distance))

        exit_wp = generate_target_waypoint_in_route(self._reference_waypoint, self.config.route)
        exit_wp = exit_wp.next(10)[0]  # just in case the junction maneuvers don't match

        sequence = py_trees.composites.Sequence()
        if self.route_mode:
            sequence.add_child(HandleJunctionScenario(
                clear_junction=False,
                clear_ego_entry=True,
                remove_entries=[source_wp],
                remove_exits= get_same_dir_lanes(exit_wp),
                stop_entries=False,
                extend_road_exit=0
            ))
            sequence.add_child(SwitchRouteSources(False))

        sequence.add_child(root)
        if self.route_mode:
            sequence.add_child(SwitchRouteSources(True))

        return sequence


class HighwayExit(BasicScenario):
    """
    This scenario is similar to CrossActorFlow
    It will remove the BackgroundActivity from the lane where ActorFlow starts.
    Then vehicles (cars) will start driving from start_actor_flow location to end_actor_flow location
    in a relatively high speed, forcing the ego to accelerate to cut in the actor flow 
    then exit from the highway.
    This scenario works when Background Activity is running in route mode. And there should be no junctions in front of the ego.
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

        super().__init__("HighwayExit",
                         ego_vehicles,
                         config,
                         world,
                         debug_mode,
                         criteria_enable=criteria_enable)

    def _create_behavior(self):
        """
        Vehicles run from the start to the end continuously.
        """
        source_wp = self._map.get_waypoint(self._start_actor_flow)
        sink_wp = self._map.get_waypoint(self._end_actor_flow)

        grp = CarlaDataProvider.get_global_route_planner()
        route = grp.trace_route(source_wp.transform.location, sink_wp.transform.location)
        junction_id = None
        for wp, _ in route:
            if wp.is_junction:
                junction_id = wp.get_junction().id
                break

        root = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        root.add_child(ActorFlow(
            source_wp, sink_wp, self._source_dist_interval, self._sink_distance, self._flow_speed, add_initial_actors=True))
        end_condition = py_trees.composites.Sequence()
        end_condition.add_child(WaitEndIntersection(self.ego_vehicles[0], junction_id))

        root.add_child(end_condition)

        sequence = py_trees.composites.Sequence()

        if self.route_mode:
            sequence.add_child(RemoveRoadLane(source_wp))
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


class MergerIntoSlowTraffic(BasicScenario):
    """
    This scenario is similar to EnterActorFlow
    It will remove the BackgroundActivity from the lane where ActorFlow starts.
    Then vehicles (cars) will start driving from start_actor_flow location to end_actor_flow location
    in a relatively low speed, ego car must merger into this slow traffic flow.
    This scenario works when Background Activity is running in route mode. And applies to a confluence
    area at a highway intersection.
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
        self._trigger_point=config.trigger_points[0].location

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

        ego_location = config.trigger_points[0].location
        self._reference_waypoint = CarlaDataProvider.get_map().get_waypoint(ego_location)

        super().__init__("MergerIntoSlowTraffic",
                         ego_vehicles,
                         config,
                         world,
                         debug_mode,
                         criteria_enable=criteria_enable)

    def _create_behavior(self):
        """
        the ego vehicle mergers into a slow traffic flow from the freeway entrance.
        """
        source_wp = self._map.get_waypoint(self._start_actor_flow)
        sink_wp = self._map.get_waypoint(self._end_actor_flow)

        # Get all lanes
        source_wps = get_same_dir_lanes(source_wp)
        sink_wps = get_same_dir_lanes(sink_wp)

        root = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        for source_wp, sink_wp in zip(source_wps, sink_wps):
            root.add_child(InTriggerDistanceToLocation(self.ego_vehicles[0], sink_wp.transform.location, self._sink_distance))
        root.add_child(ActorFlow(
            source_wp, sink_wp, self._source_dist_interval, self._sink_distance, self._flow_speed, add_initial_actors=True))

        sequence = py_trees.composites.Sequence()
        if self.route_mode:

            grp = CarlaDataProvider.get_global_route_planner()
            route = grp.trace_route(source_wp.transform.location, sink_wp.transform.location)
            extra_space = 0
            for i in range(-2, -len(route)-1, -1):
                current_wp = route[i][0]
                extra_space += current_wp.transform.location.distance(route[i+1][0].transform.location)
                if current_wp.is_junction:
                    break

            sequence.add_child(HandleJunctionScenario(
                clear_junction=False,
                clear_ego_entry=True,
                remove_entries=[source_wp],
                remove_exits=[],
                stop_entries=False,
                extend_road_exit=extra_space
            ))
            sequence.add_child(SwitchRouteSources(False))
        sequence.add_child(root)
        if self.route_mode:
            sequence.add_child(SwitchRouteSources(True))

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


class MergerIntoSlowTrafficV2(MergerIntoSlowTraffic):
    """
    Variation of MergerIntoSlowTraffic 
    """

    def _create_behavior(self):
        """
        the ego vehicle mergers into a slow traffic flow from the freeway entrance.
        """
        source_wp = self._map.get_waypoint(self._start_actor_flow)
        sink_wp = self._map.get_waypoint(self._end_actor_flow)

        sink_wps = get_same_dir_lanes(sink_wp)

        root = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        root.add_child(ActorFlow(
            source_wp, sink_wp, self._source_dist_interval, self._sink_distance, self._flow_speed, add_initial_actors=True))
        for sink_wp in sink_wps:
            root.add_child(InTriggerDistanceToLocation(self.ego_vehicles[0], sink_wp.transform.location, self._sink_distance))

        exit_wp = generate_target_waypoint_in_route(self._reference_waypoint, self.config.route)
        exit_wp = exit_wp.next(10)[0]  # just in case the junction maneuvers don't match

        sequence = py_trees.composites.Sequence()
        if self.route_mode:
            sequence.add_child(HandleJunctionScenario(
                clear_junction=False,
                clear_ego_entry=True,
                remove_entries=[source_wp],
                remove_exits=get_same_dir_lanes(exit_wp),
                stop_entries=False,
                extend_road_exit=0
            ))
            sequence.add_child(SwitchRouteSources(False))
        sequence.add_child(root)
        if self.route_mode:
            sequence.add_child(SwitchRouteSources(True))

        return sequence


class InterurbanActorFlow(BasicScenario):
    """
    Scenario specifically made for the interurban intersections,
    where the ego leaves the interurban road by turning left, crossing an actor flow.
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

        self._reference_wp = self._map.get_waypoint(config.trigger_points[0].location)
        exit_wp = generate_target_waypoint_in_route(self._reference_wp, config.route)
        exit_wp = exit_wp.next(10)[0]  # just in case the junction maneuvers don't match
        self._other_entry_wp = exit_wp.get_left_lane()
        if not self._other_entry_wp or self._other_entry_wp.lane_type != carla.LaneType.Driving:
            raise ValueError("Couldn't find an end position")

        super().__init__("InterurbanActorFlow",
                         ego_vehicles,
                         config,
                         world,
                         debug_mode,
                         criteria_enable=criteria_enable)

    def _create_behavior(self):
        """
        Create an actor flow at the opposite lane which the ego has to cross
        """
        source_wp = self._map.get_waypoint(self._start_actor_flow)
        sink_wp = self._map.get_waypoint(self._end_actor_flow)


        root = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        root.add_child(ActorFlow(
            source_wp, sink_wp, self._source_dist_interval, self._sink_distance, self._flow_speed))
        end_condition = py_trees.composites.Sequence()
        end_condition.add_child(WaitEndIntersection(self.ego_vehicles[0]))

        root.add_child(end_condition)

        sequence = py_trees.composites.Sequence()

        if self.route_mode:
            sequence.add_child(HandleJunctionScenario(
                clear_junction=False,
                clear_ego_entry=False,
                remove_entries=[source_wp, self._other_entry_wp],
                remove_exits=[],
                stop_entries=False,
                extend_road_exit=0
            ))
            sequence.add_child(ChangeOppositeBehavior(active=False))
        sequence.add_child(root)
        if self.route_mode:
            sequence.add_child(ChangeOppositeBehavior(active=True))

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


class InterurbanAdvancedActorFlow(BasicScenario):
    """
    Scenario specifically made for the interurban intersections,
    where the ego incorportates into the interurban road by turning left,
    first crossing an actor flow, and then merging into another one.
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

        self._start_actor_flow_1 = convert_dict_to_location(config.other_parameters['start_actor_flow'])
        self._end_actor_flow_1 = convert_dict_to_location(config.other_parameters['end_actor_flow'])

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

        self._reference_wp = self._map.get_waypoint(config.trigger_points[0].location)
        self._exit_wp = generate_target_waypoint_in_route(self._reference_wp, config.route)

        super().__init__("InterurbanAdvancedActorFlow",
                         ego_vehicles,
                         config,
                         world,
                         debug_mode,
                         criteria_enable=criteria_enable)

    def _create_behavior(self):
        """
        the ego vehicle mergers into a slow traffic flow from the freeway entrance.
        """
        source_wp_1 = self._map.get_waypoint(self._start_actor_flow_1)
        sink_wp_1 = self._map.get_waypoint(self._end_actor_flow_1)

        source_wp_2 = sink_wp_1.get_left_lane()
        if not source_wp_2 or source_wp_2.lane_type != carla.LaneType.Driving:
            raise ValueError("Couldn't find a position for the actor flow")
        sink_wp_2 = source_wp_1.get_left_lane()
        if not sink_wp_2 or sink_wp_2.lane_type != carla.LaneType.Driving:
            raise ValueError("Couldn't find a position for the actor flow")

        root = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        root.add_child(InTriggerDistanceToLocation(self.ego_vehicles[0], sink_wp_2.transform.location, self._sink_distance))
        root.add_child(ActorFlow(
            source_wp_1, sink_wp_1, self._source_dist_interval, self._sink_distance, self._flow_speed))
        root.add_child(ActorFlow(
            source_wp_2, sink_wp_2, self._source_dist_interval, self._sink_distance, self._flow_speed))

        sequence = py_trees.composites.Sequence()
        if self.route_mode:

            grp = CarlaDataProvider.get_global_route_planner()
            route = grp.trace_route(source_wp_2.transform.location, sink_wp_2.transform.location)
            extra_space = 0
            for i in range(-2, -len(route)-1, -1):
                current_wp = route[i][0]
                extra_space += current_wp.transform.location.distance(route[i+1][0].transform.location)
                if current_wp.is_junction:
                    break

            sequence.add_child(HandleJunctionScenario(
                clear_junction=True,
                clear_ego_entry=True,
                remove_entries=[source_wp_1, source_wp_2],
                remove_exits=[self._exit_wp],
                stop_entries=False,
                extend_road_exit=extra_space
            ))
            sequence.add_child(SwitchRouteSources(False))
            sequence.add_child(ChangeOppositeBehavior(active=False))

        sequence.add_child(root)

        if self.route_mode:
            sequence.add_child(SwitchRouteSources(True))
            sequence.add_child(ChangeOppositeBehavior(active=True))

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
