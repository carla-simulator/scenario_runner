#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: Fabian Oboril (fabian.oboril@intel.com)
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

import srunner
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.result_writer import ResultOutputProvider
from srunner.scenariomanager.timer import GameTime, TimeOut


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

        if not isinstance(self.test_criteria, py_trees.composites.Parallel):
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

    scenario = None
    scenario_tree = None
    ego_vehicle = None
    other_actors = None

    def __init__(self, world, debug_mode=False):
        """
        Init requires scenario as input
        """
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

    def analyze_scenario_challenge(self, stdout, filename, junit):
        """
        This function is intended to be called from outside and provide
        statistics about the scenario (human-readable, for the CARLA challenge.)
        """

        failure = False
        result = "SUCCESS"
        score = 0.0
        return_message = []

        if isinstance(self.scenario.test_criteria, py_trees.composites.Parallel):
            if self.scenario.test_criteria.status == py_trees.common.Status.FAILURE:
                failure = True
                result = "FAILURE"

            target_reached = False
            collisions = False
            for node in self.scenario.test_criteria.children:
                if node.return_message:
                    return_message.append(node.return_message)
                if isinstance(node, srunner.scenariomanager.atomic_scenario_criteria.RouteCompletionTest):
                    percentage_completed_route = node.score
                elif isinstance(node, srunner.scenariomanager.atomic_scenario_criteria.CollisionTest):
                    collisions = (node.test_status == "FAILURE")
                elif isinstance(node, srunner.scenariomanager.atomic_scenario_criteria.InRadiusRegionTest):
                    target_reached = (node.test_status == "SUCCESS")
                elif isinstance(node, srunner.scenariomanager.atomic_scenario_criteria.InRouteTest):
                    offroute = (node.test_status == "FAILURE")

            if target_reached:
                score = 100.0
            else:
                score = percentage_completed_route


        if self.scenario.timeout_node.timeout and not failure:
            result = "TIMEOUT"


        return result, score, return_message