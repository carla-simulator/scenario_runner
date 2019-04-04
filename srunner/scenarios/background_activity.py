#!/usr/bin/env python

#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Scenario spawning elements to make the town dynamic and interesting
"""

import py_trees

from srunner.scenariomanager.atomic_scenario_behavior import *
from srunner.scenariomanager.atomic_scenario_criteria import *
from srunner.scenarios.basic_scenario import *


BACKGROUND_ACTIVITY_SCENARIOS = ["BackgroundActivity"]


class BackgroundActivity(BasicScenario):

    """
    Implementation of a dummy scenario
    """

    category = "BackgroundActivity"
    radius = 10.0           # meters
    timeout = 300           # Timeout of scenario in seconds

    def __init__(self, world, ego_vehicle, config, randomize=False, debug_mode=False):
        """
        Setup all relevant parameters and create scenario
        """
        self.config = config

        super(BackgroundActivity, self).__init__("BackgroundActivity",
                                             ego_vehicle,
                                             config,
                                             world,
                                             debug_mode,
                                             terminate_on_failure=True,
                                             criteria_enable=True)

    def _initialize_actors(self, config):
        for actor in config.other_actors:
            new_actors = CarlaActorPool.request_new_batch_actors(actor.model,
                                                                 actor.amount,
                                                                 actor.transform,
                                                                 hero=False,
                                                                 autopilot=actor.autopilot,
                                                                 random_location=actor.random_location)
            if new_actors is None:
                raise Exception("Error: Unable to add actor {} at {}".format(actor.model, actor.transform))

            for actor in new_actors:
                self.other_actors.append(actor)

    def _create_behavior(self):
        """
        Basic behavior do nothing, i.e. Idle
        """

        # Build behavior tree
        sequence = py_trees.composites.Sequence("Sequence Behavior")
        check_collisions = CheckCollisions(self.ego_vehicle)
        sequence.add_child(check_collisions)

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
