#!/usr/bin/env python

# Copyright (c) 2018-2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides the Scenario and ScenarioManager implementations.
These must not be modified and are for reference only!
"""

from __future__ import print_function
import sys
import time
import threading

import py_trees

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.result_writer import ResultOutputProvider
from srunner.scenariomanager.timer import GameTime, TimeOut
from srunner.scenariomanager.traffic_events import TrafficEventType


class Scenario(object):

    """
    Basic scenario class. This class holds the behavior_tree describing the
    scenario and the test criteria.

    The user must not modify this class.

    Important parameters:
    - behavior: User defined scenario with py_tree
    - criteria_list: List of user defined test criteria with py_tree
    - timeout (default = 60s): Timeout of the scenario in seconds
    - terminate_on_failure: Terminate scenario on first failure
    """

    def __init__(self, behavior, criteria, name, timeout=60, terminate_on_failure=False):
        self.behavior = behavior
        self.test_criteria = criteria
        self.timeout = timeout

        if self.test_criteria is not None and not isinstance(self.test_criteria, py_trees.composites.Parallel):
            # list of nodes
            for criterion in self.test_criteria:
                criterion.terminate_on_failure = terminate_on_failure

            # Create py_tree for test criteria
            self.criteria_tree = py_trees.composites.Parallel(name="Test Criteria")
            self.criteria_tree.add_children(self.test_criteria)
            self.criteria_tree.setup(timeout=1)
        else:
            self.criteria_tree = criteria

        # Create node for timeout
        self.timeout_node = TimeOut(self.timeout, name="TimeOut")

        # Create overall py_tree
        self.scenario_tree = py_trees.composites.Parallel(name, policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        self.scenario_tree.add_child(self.behavior)
        self.scenario_tree.add_child(self.timeout_node)
        if criteria is not None:
            self.scenario_tree.add_child(self.criteria_tree)
        self.scenario_tree.setup(timeout=1)

    def terminate(self):
        """
        This function sets the status of all leaves in the scenario tree to INVALID
        """
        # Get list of all leaves in the tree
        node_list = [self.scenario_tree]
        more_nodes_exist = True
        while more_nodes_exist:
            more_nodes_exist = False
            for node in node_list:
                if node.children:
                    node_list.remove(node)
                    more_nodes_exist = True
                    for child in node.children:
                        node_list.append(child)

        # Set status to INVALID
        for node in node_list:
            node.terminate(py_trees.common.Status.INVALID)


class ScenarioManager(object):

    """
    Basic scenario manager class. This class holds all functionality
    required to start, and analyze a scenario.

    The user must not modify this class.

    To use the ScenarioManager:
    1. Create an object via manager = ScenarioManager()
    2. Load a scenario via manager.load_scenario()
    3. Trigger the execution of the scenario manager.execute()
       This function is designed to explicitly control start and end of
       the scenario execution
    4. Trigger a result evaluation with manager.analyze()
    5. Cleanup with manager.stop_scenario()
    """

    def __init__(self, world, debug_mode=False):
        """
        Init requires scenario as input
        """
        self.scenario = None
        self.scenario_tree = None
        self.scenario_class = None
        self.ego_vehicle = None
        self.other_actors = None

        self._debug_mode = debug_mode
        self.agent = None
        self._autonomous_agent_plugged = False
        self._running = False
        self._timestamp_last_run = 0.0
        self._my_lock = threading.Lock()

        self.scenario_duration_system = 0.0
        self.scenario_duration_game = 0.0
        self.start_system_time = None
        self.end_system_time = None

        world.on_tick(self._tick_scenario)

    def load_scenario(self, scenario):
        """
        Load a new scenario
        """
        self.restart()
        self.scenario_class = scenario
        self.scenario = scenario.scenario
        self.scenario_tree = self.scenario.scenario_tree
        self.ego_vehicle = scenario.ego_vehicle
        self.other_actors = scenario.other_actors

        CarlaDataProvider.register_actor(self.ego_vehicle)
        CarlaDataProvider.register_actors(self.other_actors)
        # To print the scenario tree uncomment the next line
        # py_trees.display.render_dot_tree(self.scenario_tree)

    def restart(self):
        """
        Reset all parameters
        """
        self._running = False
        self._timestamp_last_run = 0.0
        self.scenario_duration_system = 0.0
        self.scenario_duration_game = 0.0
        self.start_system_time = None
        self.end_system_time = None
        GameTime.restart()

    def run_scenario(self, agent=None):
        """
        Trigger the start of the scenario and wait for it to finish/fail
        """
        self.agent = agent
        print("ScenarioManager: Running scenario {}".format(self.scenario_tree.name))
        self.start_system_time = time.time()
        start_game_time = GameTime.get_time()

        self._running = True

        while self._running:
            time.sleep(0.5)

        self.end_system_time = time.time()
        end_game_time = GameTime.get_time()

        self.scenario_duration_system = self.end_system_time - \
            self.start_system_time
        self.scenario_duration_game = end_game_time - start_game_time

        if self.scenario_tree.status == py_trees.common.Status.FAILURE:
            print("ScenarioManager: Terminated due to failure")

    def _tick_scenario(self, timestamp):
        """
        Run next tick of scenario
        This function is a callback for world.on_tick()

        Important:
        - It hast to be ensured that the scenario has not yet completed/failed
          and that the time moved forward.
        - A thread lock should be used to avoid that the scenario tick is performed
          multiple times in parallel.
        """
        with self._my_lock:
            if self._running and self._timestamp_last_run < timestamp.elapsed_seconds:
                self._timestamp_last_run = timestamp.elapsed_seconds

                if self._debug_mode:
                    print("\n--------- Tick ---------\n")

                # Update game time and actor information
                GameTime.on_carla_tick(timestamp)
                CarlaDataProvider.on_carla_tick()

                # Tick scenario
                self.scenario_tree.tick_once()

                if self.agent:
                    # Invoke agent
                    action = self.agent()
                    action = self.scenario_class.change_control(action)
                    self.ego_vehicle.apply_control(action)

                if self._debug_mode:
                    print("\n")
                    py_trees.display.print_ascii_tree(
                        self.scenario_tree, show_status=True)
                    sys.stdout.flush()

                if self.scenario_tree.status != py_trees.common.Status.RUNNING:
                    self._running = False

    def stop_scenario(self):
        """
        This function triggers a proper termination of a scenario
        """
        if self.scenario is not None:
            self.scenario.terminate()

        CarlaDataProvider.cleanup()

    def analyze_scenario(self, stdout, filename, junit):
        """
        This function is intended to be called from outside and provide
        statistics about the scenario (human-readable, in form of a junit
        report, etc.)
        """

        failure = False
        timeout = False
        result = "SUCCESS"

        if self.scenario.test_criteria is None:
            return True

        if isinstance(self.scenario.test_criteria, py_trees.composites.Parallel):
            if self.scenario.test_criteria.status == py_trees.common.Status.FAILURE:
                failure = True
                result = "FAILURE"
        else:
            for criterion in self.scenario.test_criteria:
                if (not criterion.optional and
                        criterion.test_status != "SUCCESS" and
                        criterion.test_status != "ACCEPTABLE"):
                    failure = True
                    result = "FAILURE"
                elif criterion.test_status == "ACCEPTABLE":
                    result = "ACCEPTABLE"

        if self.scenario.timeout_node.timeout and not failure:
            timeout = True
            result = "TIMEOUT"

        output = ResultOutputProvider(self, result, stdout, filename, junit)
        output.write()

        return failure or timeout

    def analyze_scenario_challenge(self):
        """
        This function is intended to be called from outside and provide
        statistics about the scenario (human-readable, for the CARLA challenge.)
        """
        PENALTY_COLLISION_STATIC = 10       # pylint: disable=invalid-name
        PENALTY_COLLISION_VEHICLE = 10      # pylint: disable=invalid-name
        PENALTY_COLLISION_PEDESTRIAN = 30   # pylint: disable=invalid-name
        PENALTY_TRAFFIC_LIGHT = 10          # pylint: disable=invalid-name
        PENALTY_WRONG_WAY = 5               # pylint: disable=invalid-name
        PENALTY_SIDEWALK_INVASION = 5       # pylint: disable=invalid-name
        PENALTY_STOP = 7                    # pylint: disable=invalid-name

        target_reached = False
        failure = False
        result = "SUCCESS"
        final_score = 0.0
        score_penalty = 0.0
        score_route = 0.0
        return_message = ""

        if isinstance(self.scenario.test_criteria, py_trees.composites.Parallel):
            if self.scenario.test_criteria.status == py_trees.common.Status.FAILURE:
                failure = True
                result = "FAILURE"
            if self.scenario.timeout_node.timeout and not failure:
                result = "TIMEOUT"

            list_traffic_events = []
            for node in self.scenario.test_criteria.children:
                if node.list_traffic_events:
                    list_traffic_events.extend(node.list_traffic_events)

            list_collisions = []
            list_red_lights = []
            list_wrong_way = []
            list_route_dev = []
            list_sidewalk_inv = []
            list_stop_inf = []
            # analyze all traffic events
            for event in list_traffic_events:
                if event.get_type() == TrafficEventType.COLLISION_STATIC:
                    score_penalty += PENALTY_COLLISION_STATIC
                    msg = event.get_message()
                    if msg:
                        list_collisions.append(event.get_message())

                elif event.get_type() == TrafficEventType.COLLISION_VEHICLE:
                    score_penalty += PENALTY_COLLISION_VEHICLE
                    msg = event.get_message()
                    if msg:
                        list_collisions.append(event.get_message())

                elif event.get_type() == TrafficEventType.COLLISION_PEDESTRIAN:
                    score_penalty += PENALTY_COLLISION_PEDESTRIAN
                    msg = event.get_message()
                    if msg:
                        list_collisions.append(event.get_message())

                elif event.get_type() == TrafficEventType.TRAFFIC_LIGHT_INFRACTION:
                    score_penalty += PENALTY_TRAFFIC_LIGHT
                    msg = event.get_message()
                    if msg:
                        list_red_lights.append(event.get_message())

                elif event.get_type() == TrafficEventType.WRONG_WAY_INFRACTION:
                    score_penalty += PENALTY_WRONG_WAY
                    msg = event.get_message()
                    if msg:
                        list_wrong_way.append(event.get_message())

                elif event.get_type() == TrafficEventType.ROUTE_DEVIATION:
                    msg = event.get_message()
                    if msg:
                        list_route_dev.append(event.get_message())

                elif event.get_type() == TrafficEventType.ON_SIDEWALK_INFRACTION:
                    score_penalty += PENALTY_SIDEWALK_INVASION
                    msg = event.get_message()
                    if msg:
                        list_sidewalk_inv.append(event.get_message())

                elif event.get_type() == TrafficEventType.STOP_INFRACTION:
                    score_penalty += PENALTY_STOP
                    msg = event.get_message()
                    if msg:
                        list_stop_inf.append(event.get_message())

                elif event.get_type() == TrafficEventType.ROUTE_COMPLETED:
                    score_route = 100.0
                    target_reached = True
                elif event.get_type() == TrafficEventType.ROUTE_COMPLETION:
                    if not target_reached:
                        score_route = event.get_dict()['route_completed']

            final_score = max(score_route - score_penalty, 0)

            return_message += "\n=================================="
            return_message += "\n==[{}] [Score = {:.2f} : (route_score={}, infractions=-{})]".format(result,
                                                                                                     final_score,
                                                                                                     score_route,
                                                                                                     score_penalty)
            if list_collisions:
                return_message += "\n===== Collisions:"
                for item in list_collisions:
                    return_message += "\n========== {}".format(item)

            if list_red_lights:
                return_message += "\n===== Red lights:"
                for item in list_red_lights:
                    return_message += "\n========== {}".format(item)

            if list_stop_inf:
                return_message += "\n===== STOP infractions:"
                for item in list_stop_inf:
                    return_message += "\n========== {}".format(item)

            if list_wrong_way:
                return_message += "\n===== Wrong way:"
                for item in list_wrong_way:
                    return_message += "\n========== {}".format(item)

            if list_sidewalk_inv:
                return_message += "\n===== Sidewalk invasions:"
                for item in list_sidewalk_inv:
                    return_message += "\n========== {}".format(item)

            if list_route_dev:
                return_message += "\n===== Route deviation:"
                for item in list_route_dev:
                    return_message += "\n========== {}".format(item)

            return_message += "\n=================================="

        return result, final_score, return_message
