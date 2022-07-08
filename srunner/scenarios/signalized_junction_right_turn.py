#!/usr/bin/env python

#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Collection of traffic scenarios where the ego vehicle (hero)
is making a right turn
"""

from __future__ import print_function

import py_trees

import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import ActorFlow, TrafficLightFreezer, ScenarioTimeout
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest, ScenarioTimeoutTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import WaitEndIntersection
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.tools.scenario_helper import (generate_target_waypoint,
                                           get_junction_topology,
                                           filter_junction_wp_direction,
                                           get_same_dir_lanes)

from srunner.tools.background_manager import HandleJunctionScenario


class SignalizedJunctionRightTurn(BasicScenario):

    """
    Scenario where the vehicle is turning right at an intersection an has to avoid
    colliding with a vehicle coming from its left
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=80):
        """
        Setup all relevant parameters and create scenario
        """
        self._world = world
        self._map = CarlaDataProvider.get_map()
        self.timeout = timeout

        if 'flow_speed' in config.other_parameters:
            self._flow_speed = float(config.other_parameters['flow_speed']['value'])
        else:
            self._flow_speed = 20 # m/s

        if 'source_dist_interval' in config.other_parameters:
            self._source_dist_interval = [
                float(config.other_parameters['source_dist_interval']['from']),
                float(config.other_parameters['source_dist_interval']['to'])
            ]
        else:
            self._source_dist_interval = [25, 50] # m

        self._direction = 'left'

        # The faster the flow, the further they are spawned, leaving time to react to them
        self._source_dist = 5 * self._flow_speed
        self._sink_dist = 3 * self._flow_speed

        self._green_light_delay = 5  # Wait before the ego's lane traffic light turns green
        self._flow_tl_dict = {}
        self._init_tl_dict = {}

        if 'timeout' in config.other_parameters:
            self._scenario_timeout = float(config.other_parameters['flow_distance']['value'])
        else:
            self._scenario_timeout = 180

        super().__init__("SignalizedJunctionRightTurn",
                         ego_vehicles,
                         config,
                         world,
                         debug_mode,
                         criteria_enable=criteria_enable)

    def _initialize_actors(self, config):
        """
        Custom initialization
        """
        ego_location = config.trigger_points[0].location
        self._ego_wp = CarlaDataProvider.get_map().get_waypoint(ego_location)

        # Get the junction
        starting_wp = self._ego_wp
        ego_junction_dist = 0
        while not starting_wp.is_junction:
            starting_wps = starting_wp.next(1.0)
            if len(starting_wps) == 0:
                raise ValueError("Failed to find a junction")
            starting_wp = starting_wps[0]
            ego_junction_dist += 1
        junction = starting_wp.get_junction()

        # Get the source entry lane wp
        entry_wps, _ = get_junction_topology(junction)
        source_entry_wps = filter_junction_wp_direction(starting_wp, entry_wps, self._direction)
        if not source_entry_wps:
            raise ValueError("Trying to find a lane in the {} direction but none was found".format(self._direction))

        # Get the rightmost lane
        source_entry_wp = source_entry_wps[0]
        while True:
            right_wp = source_entry_wp.get_right_lane()
            if not right_wp or right_wp.lane_type != carla.LaneType.Driving:
                break
            source_entry_wp = right_wp

        # Get the source transform
        source_wp = source_entry_wp
        source_junction_dist = 0
        while source_junction_dist < self._source_dist:
            source_wps = source_wp.previous(5)
            if len(source_wps) == 0:
                raise ValueError("Failed to find a source location")
            if source_wps[0].is_junction:
                break
            source_wp = source_wps[0]
            source_junction_dist += 5

        self._source_wp = source_wp
        source_transform = self._source_wp.transform

        # Get the sink location
        sink_exit_wp = generate_target_waypoint(self._map.get_waypoint(source_transform.location), 0)
        sink_wps = sink_exit_wp.next(self._sink_dist)
        if len(sink_wps) == 0:
            raise ValueError("Failed to find a sink location")
        self._sink_wp = sink_wps[0]

        self._get_traffic_lights(junction, ego_junction_dist, source_junction_dist)

    def _get_traffic_lights(self, junction, ego_dist, source_dist):
        """Get the traffic light of the junction, mapping their states"""
        tls = self._world.get_traffic_lights_in_junction(junction.id)
        if not tls:
            raise ValueError("No traffic lights found, use the NonSignalized version instead")

        ego_landmark = self._ego_wp.get_landmarks_of_type(ego_dist + 2, "1000001")[0]
        ego_tl = self._world.get_traffic_light(ego_landmark)
        source_landmark = self._source_wp.get_landmarks_of_type(source_dist + 2, "1000001")[0]
        source_tl = self._world.get_traffic_light(source_landmark)
        for tl in tls:
            if tl.id == ego_tl.id:
                self._flow_tl_dict[tl] = carla.TrafficLightState.Green
                self._init_tl_dict[tl] = carla.TrafficLightState.Red
            elif tl.id == source_tl.id:
                self._flow_tl_dict[tl] = carla.TrafficLightState.Green
                self._init_tl_dict[tl] = carla.TrafficLightState.Green
            else:
                self._flow_tl_dict[tl] = carla.TrafficLightState.Red
                self._init_tl_dict[tl] = carla.TrafficLightState.Red

    def _create_behavior(self):
        """
        Hero vehicle is turning right in an urban area, at a signalized intersection,
        while other actor coming straight from the left. The ego has to avoid colliding with it
        """

        sequence = py_trees.composites.Sequence(name="JunctionRightTurn")
        if self.route_mode:
            sequence.add_child(HandleJunctionScenario(
                clear_junction=True,
                clear_ego_entry=True,
                remove_entries=get_same_dir_lanes(self._source_wp),
                remove_exits=[],
                stop_entries=False,
                extend_road_exit=self._sink_dist
            ))

        root = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        root.add_child(WaitEndIntersection(self.ego_vehicles[0]))
        root.add_child(ActorFlow(
            self._source_wp, self._sink_wp, self._source_dist_interval, 2, self._flow_speed))
        root.add_child(ScenarioTimeout(self._scenario_timeout, self.config.name))

        tl_freezer_sequence = py_trees.composites.Sequence("Traffic Light Behavior")
        tl_freezer_sequence.add_child(TrafficLightFreezer(self._init_tl_dict, duration=self._green_light_delay))
        tl_freezer_sequence.add_child(TrafficLightFreezer(self._flow_tl_dict))
        root.add_child(tl_freezer_sequence)

        sequence.add_child(root)
        return sequence

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = [ScenarioTimeoutTest(self.ego_vehicles[0], self.config.name)]
        if not self.route_mode:
            criteria.append(CollisionTest(self.ego_vehicles[0]))
        return criteria

    def __del__(self):
        """
        Remove all actors upon deletion
        """
        self.remove_all_actors()


class NonSignalizedJunctionRightTurn(BasicScenario):

    """
    Scenario where the vehicle is turning right at an intersection an has to avoid
    colliding with a vehicle coming from its left
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=80):
        """
        Setup all relevant parameters and create scenario
        """
        self._world = world
        self._map = CarlaDataProvider.get_map()
        self.timeout = timeout

        if 'flow_speed' in config.other_parameters:
            self._flow_speed = float(config.other_parameters['flow_speed']['value'])
        else:
            self._flow_speed = 20 # m/s

        if 'source_dist_interval' in config.other_parameters:
            self._source_dist_interval = [
                float(config.other_parameters['source_dist_interval']['from']),
                float(config.other_parameters['source_dist_interval']['to'])
            ]
        else:
            self._source_dist_interval = [25, 50] # m

        self._direction = 'left'

        # The faster the flow, the further they are spawned, leaving time to react to them
        self._source_dist = 5 * self._flow_speed
        self._sink_dist = 3 * self._flow_speed

        if 'timeout' in config.other_parameters:
            self._scenario_timeout = float(config.other_parameters['flow_distance']['value'])
        else:
            self._scenario_timeout = 180

        super().__init__("NonSignalizedJunctionRightTurn",
                         ego_vehicles,
                         config,
                         world,
                         debug_mode,
                         criteria_enable=criteria_enable)

    def _initialize_actors(self, config):
        """
        Custom initialization
        """
        ego_location = config.trigger_points[0].location
        self._ego_wp = CarlaDataProvider.get_map().get_waypoint(ego_location)

        # Get the junction
        starting_wp = self._ego_wp
        ego_junction_dist = 0
        while not starting_wp.is_junction:
            starting_wps = starting_wp.next(1.0)
            if len(starting_wps) == 0:
                raise ValueError("Failed to find a junction")
            starting_wp = starting_wps[0]
            ego_junction_dist += 1
        junction = starting_wp.get_junction()

        # Get the source entry lane wp
        entry_wps, _ = get_junction_topology(junction)
        source_entry_wps = filter_junction_wp_direction(starting_wp, entry_wps, self._direction)
        if not source_entry_wps:
            raise ValueError("Trying to find a lane in the {} direction but none was found".format(self._direction))

        # Get the rightmost lane
        source_entry_wp = source_entry_wps[0]
        while True:
            right_wp = source_entry_wp.get_right_lane()
            if not right_wp or right_wp.lane_type != carla.LaneType.Driving:
                break
            source_entry_wp = right_wp

        # Get the source transform
        source_wp = source_entry_wp
        source_junction_dist = 0
        while source_junction_dist < self._source_dist:
            source_wps = source_wp.previous(5)
            if len(source_wps) == 0:
                raise ValueError("Failed to find a source location")
            if source_wps[0].is_junction:
                break
            source_wp = source_wps[0]
            source_junction_dist += 5

        self._source_wp = source_wp
        source_transform = self._source_wp.transform

        # Get the sink location
        sink_exit_wp = generate_target_waypoint(self._map.get_waypoint(source_transform.location), 0)
        sink_wps = sink_exit_wp.next(self._sink_dist)
        if len(sink_wps) == 0:
            raise ValueError("Failed to find a sink location")
        self._sink_wp = sink_wps[0]

    def _create_behavior(self):
        """
        Hero vehicle is turning right in an urban area, at a signalized intersection,
        while other actor coming straight from the left. The ego has to avoid colliding with it
        """

        sequence = py_trees.composites.Sequence(name="JunctionRightTurn")
        if self.route_mode:
            sequence.add_child(HandleJunctionScenario(
                clear_junction=True,
                clear_ego_entry=True,
                remove_entries=get_same_dir_lanes(self._source_wp),
                remove_exits=[],
                stop_entries=True,
                extend_road_exit=self._sink_dist
            ))

        root = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        root.add_child(WaitEndIntersection(self.ego_vehicles[0]))
        root.add_child(ActorFlow(
            self._source_wp, self._sink_wp, self._source_dist_interval, 2, self._flow_speed))
        root.add_child(ScenarioTimeout(self._scenario_timeout, self.config.name))

        sequence.add_child(root)
        return sequence

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = [ScenarioTimeoutTest(self.ego_vehicles[0], self.config.name)]
        if not self.route_mode:
            criteria.append(CollisionTest(self.ego_vehicles[0]))
        return criteria

    def __del__(self):
        """
        Remove all actors upon deletion
        """
        self.remove_all_actors()
