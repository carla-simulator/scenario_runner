#!/usr/bin/env python

#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides the Scenario and ScenarioManager implementations.
These must not be modified and are for reference only!
"""

import sys
import time
import threading

import py_trees

from ScenarioManager import timer


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

        for criterion in self.test_criteria:
            criterion.terminate_on_failure = terminate_on_failure

        # Create py_tree for test criteria
        self.criteria_tree = py_trees.composites.Parallel(name="Test Criteria")
        self.criteria_tree.add_children(self.test_criteria)
        self.criteria_tree.setup(timeout=1)

        # Create node for timeout
        self.timeout_node = timer.TimeOut(timeout, name="TimeOut")

        # Create overall py_tree
        self.scenario = py_trees.composites.Parallel(
            name,
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        self.scenario.add_child(self.behavior)
        self.scenario.add_child(self.timeout_node)
        self.scenario.add_child(self.criteria_tree)
        self.scenario.setup(timeout=1)

    def terminate(self):
        """
        This function sets the status of all leaves in the scenario tree to INVALID
        """
        # Get list of all leaves in the tree
        node_list = [self.scenario]
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
    2. Trigger the execution of the scenario manager.execute()
       This function is designed to explicitly control start and end of
       the scenario execution
    3. Trigger a result evaluation with manager.analyze()
    """

    def __init__(self, world, scenario, debug_mode):
        """
        Init requires scenario as input
        """
        self.scenario = scenario
        self.scenario_tree = scenario.scenario
        self.scenario_duration_system = 0.0
        self.scenario_duration_game = 0.0
        self.debug_mode = debug_mode
        self.running = False
        self.timestamp_last_run = 0.0
        world.on_tick(self.tick_scenario)

        # To print the scenario tree uncomment the next line
        py_trees.display.render_dot_tree(self.scenario_tree)

    def run_scenario(self):
        """
        Trigger the start of the scenario and wait for it to finish/fail
        """
        start_system_time = time.time()
        start_game_time = timer.GameTime.get_time()

        self.running = True

        while self.running:
            time.sleep(0.5)

        end_system_time = time.time()
        end_game_time = timer.GameTime.get_time()

        self.scenario_duration_system = end_system_time - start_system_time
        self.scenario_duration_game = end_game_time - start_game_time

        if self.scenario_tree.status == py_trees.common.Status.FAILURE:
            print("Terminated due to failure")

    def tick_scenario(self, timestamp):
        """
        Run next tick of scenario
        This function is a callback for world.on_tick()

        Important:
        - It hast to be ensured that the scenario has not yet completed/failed
          and that the time moved forward.
        - A thread lock should be used to avoid that the scenario tick is performed
          multiple times in parallel.
        """
        with threading.Lock():
            if self.running and self.timestamp_last_run < timestamp.elapsed_seconds:
                self.timestamp_last_run = timestamp.elapsed_seconds

                if self.debug_mode:
                    print("\n--------- Tick ---------\n")

                self.scenario_tree.tick_once()

                if self.debug_mode:
                    print("\n")
                    py_trees.display.print_ascii_tree(
                        self.scenario_tree, show_status=True)
                    sys.stdout.flush()

                if self.scenario_tree.status != py_trees.common.Status.RUNNING:
                    self.running = False

    def stop_scenario(self):
        """
        This function sets all entries in the
        """
        self.scenario.terminate()

    def analyze_scenario(self):
        """
        This function is intended to be called from outside and provide
        statistics about the scenario (human-readable, in form of a junit
        report, etc.)
        """
        failure = False
        print("Scenario duration: System Time %5.2fs --- Game Time %5.2fs" %
              (self.scenario_duration_system, self.scenario_duration_game))
        for criterion in self.scenario.test_criteria:
            if criterion.get_test_status() == "FAILURE":
                print("Criterion %s failed with %s" %
                      (criterion.name, criterion.get_test_metric()))
                failure = True
            else:
                print("Criterion %s successful with %s" %
                      (criterion.name, criterion.get_test_metric()))

        if ((self.scenario.timeout_node.status == py_trees.common.Status.SUCCESS)
                and (self.scenario_tree.tip().status != py_trees.common.Status.SUCCESS)
                and not failure):
            print("Timeout")
            failure = True
        return failure
