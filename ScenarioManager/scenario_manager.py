#!/usr/bin/env python

#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

#
# This file contains the basic scenario and scenario manager classes
# These must not be modified and are for reference only!

"""
This module provides the Scenario and ScenarioManager implementations.
"""

import time

import py_trees


class Scenario(object):

    """
    Basic scenario class. This class holds the behavior_tree describing the
    scenario and the test criteria.

    The user must not modify this function.

    Important parameters:
    - behavior: User defined scenario with py_tree
    - criteria_list: List of user defined test criteria with py_tree
    - timeout (default = 60s): Timeout of the scenario in seconds
    - terminate_on_failure: Terminate scenario on first failure
    """

    def __init__(self, behavior, criteria, timeout=60, terminate_on_failure=False):
        self.behavior = behavior
        self.test_criteria = criteria

        for criterion in self.test_criteria:
            criterion.terminate_on_failure = terminate_on_failure

        # Create py_tree for test criteria
        self.criteria_tree = py_trees.composites.Parallel(name="Test Criteria")
        self.criteria_tree.add_children(self.test_criteria)
        self.criteria_tree.setup(timeout=1)

        # Create node for timeout
        self.timeout_node = py_trees.timers.Timer(
            name="TimeOut", duration=timeout)

        # Create overall py_tree
        self.scenario = py_trees.composites.Parallel(
            name="Scenario with TimeOut and TestCriteria",
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

    The user must not modify this function.
    """

    def __init__(self, scenario):
        """
        Init requires scenario as input
        """
        self.scenario = scenario
        self.scenario_tree = scenario.scenario
        self.scenario_duration = 0

        # To print the scenario tree uncomment the next line
        # py_trees.display.render_dot_tree(self.scenario_tree)

    def run_scenario(self):
        """
        Execute scenario
        """
        start = time.time()
        while self.scenario_tree.status != py_trees.common.Status.SUCCESS:
            # print("\n--------- Tick ---------\n")
            self.scenario_tree.tick_once()
            # print("\n")
            # py_trees.display.print_ascii_tree(
            #     self.scenario_tree, show_status=True)

            if self.scenario_tree.status == py_trees.common.Status.FAILURE:
                print("Terminated due to failure")
                break

            # Sleep for a small time to avoid high cpu load
            time.sleep(0.1)

        end = time.time()
        self.scenario_duration = end - start

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
        print("Scenario duration: {}".format(self.scenario_duration))
        for criterion in self.scenario.test_criteria:
            if criterion.get_test_status() == "FAILURE":
                print("Criterion {} failed with {}".format(
                    criterion.name, criterion.get_test_metric()))
                failure = True
            else:
                print("Criterion {} successful with {}".format(
                    criterion.name, criterion.get_test_metric()))

        if ((self.scenario.timeout_node.status == py_trees.common.Status.SUCCESS)
                and (self.scenario_tree.tip().status != py_trees.common.Status.SUCCESS)):
            print("Timeout")
            failure = True
        return failure
