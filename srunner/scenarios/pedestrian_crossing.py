#!/usr/bin/env python
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Pedestrians crossing through the middle of the lane.
"""

from __future__ import print_function

import py_trees
import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import WalkerFlow, AIWalkerBehavior
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import WaitEndIntersection
from srunner.scenarios.basic_scenario import BasicScenario

from srunner.tools.background_manager import HandleJunctionScenario


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


class PedestrianCrossing(BasicScenario):

    """
    This class holds everything required for a group of natual pedestrians crossing the road.
    The ego vehicle is passing through a road,
    And encounters a group of pedestrians crossing the road.

    This is a single ego vehicle scenario.

    Notice that the initial pedestrian will walk from the start of the junction ahead to end_walker_flow_1.
    """

    def __init__(self, world, ego_vehicles, config, debug_mode=False, criteria_enable=True, timeout=60):
        """
        Setup all relevant parameters and create scenario
        """
        self._wmap = CarlaDataProvider.get_map()
        self._trigger_location = config.trigger_points[0].location
        self._reference_waypoint = self._wmap.get_waypoint(
            self._trigger_location)

        # Get the start point of the initial pedestrian
        sidewalk_waypoint = self._reference_waypoint
        while sidewalk_waypoint.lane_type != carla.LaneType.Sidewalk:
            right_wp = sidewalk_waypoint.get_right_lane()
            if right_wp is None:
                raise RuntimeError("Can't find sidewalk to spawn pedestrian")
            sidewalk_waypoint = right_wp
        self._init_walker_start = sidewalk_waypoint.next_until_lane_end(
            1)[-1].transform.location
        self._init_walker_start.z += 1

        # The initial pedestrian will walk to end_walker_flow_1
        self._init_walker_end = convert_dict_to_location(
            config.other_parameters['end_walker_flow_1'])

        self._start_walker_flow = convert_dict_to_location(
            config.other_parameters['start_walker_flow'])
        self._sink_locations = []
        self._sink_locations_prob = []

        end_walker_flow_list = [
            v for k, v in config.other_parameters.items() if 'end_walker_flow' in k]

        for item in end_walker_flow_list:
            self._sink_locations.append(convert_dict_to_location(item))
            prob = float(item['p'])  if 'p' in item else 0.5
            self._sink_locations_prob.append(prob)

        if 'source_dist_interval' in config.other_parameters:
            self._source_dist_interval = [
                float(config.other_parameters['source_dist_interval']['from']),
                float(config.other_parameters['source_dist_interval']['to'])
            ]
        else:
            self._source_dist_interval = [2, 8]  # m

        self.timeout = timeout

        super(PedestrianCrossing, self).__init__("PedestrianCrossing",
                                                 ego_vehicles,
                                                 config,
                                                 world,
                                                 debug_mode,
                                                 criteria_enable=criteria_enable)

    def _create_behavior(self):
        """
        After invoking this scenario, pedestrians will start to walk to two different destinations randomly, and some of them will cross the road.
        Pedestrians will be removed when they arrive at their destinations.

        Ego is expected to cross the junction when there is enough space to go through.
        Ego is not expected to wait for pedestrians crossing the road for a long time.
        """
        sequence = py_trees.composites.Sequence(name="CrossingPedestrian")
        if self.route_mode:
            sequence.add_child(HandleJunctionScenario(
                clear_junction=True,
                clear_ego_entry=True,
                remove_entries=[],
                remove_exits=[],
                stop_entries=True,
                extend_road_exit=0
            ))

        # Move the adversary
        root = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL)

        root.add_child(AIWalkerBehavior(
            self._init_walker_start, self._init_walker_end))

        walker_root = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE, name="PedestrianMove")
        walker_root.add_child(WalkerFlow(
            self._start_walker_flow, self._sink_locations, self._sink_locations_prob, self._source_dist_interval))
        walker_root.add_child(WaitEndIntersection(self.ego_vehicles[0]))

        root.add_child(walker_root)

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
