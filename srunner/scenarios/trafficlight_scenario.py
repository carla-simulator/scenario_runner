#!/usr/bin/env python

#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Scenario spawning elements to make the town dynamic and interesting
"""

import py_trees

from srunner.scenariomanager.scenarioatomics.atomic_behaviors import TrafficLightManipulator
from srunner.scenarios.basic_scenario import BasicScenario


class TrafficLightScenario(BasicScenario):

    """
    This scenario controls traffic lights at intersection to create interesting situations, e.g.:
      - vehicles running red lights
      - yielding to traffic

    This is a single ego vehicle scenario
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, timeout=35 * 60):
        """
        Setup all relevant parameters and create scenario
        """
        self.config = config
        self.debug = debug_mode

        self.timeout = timeout  # Timeout of scenario in seconds

        super(TrafficLightScenario, self).__init__("TrafficLightScenario",
                                                   ego_vehicles,
                                                   config,
                                                   world,
                                                   debug_mode,
                                                   terminate_on_failure=True,
                                                   criteria_enable=True)

    def _create_behavior(self):
        """
        Basic behavior just checks for the next traffic intersection to manipulate randomly
        """

        # Build behavior tree
        sequence = py_trees.composites.Sequence("TrafficLightManipulator")
        traffic_manipulator = TrafficLightManipulator(self.ego_vehicles[0], subtype=None, debug=self.debug)
        sequence.add_child(traffic_manipulator)

        return sequence

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        pass

    def __del__(self):
        """
        Remove all actors upon deletion
        """
        self.remove_all_actors()
