#!/usr/bin/env python

#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Basic CARLA Autonomous Driving training scenario
"""

import py_trees

from srunner.scenariomanager.atomic_scenario_behavior import *
from srunner.scenariomanager.atomic_scenario_criteria import *
from srunner.scenarios.basic_scenario import *


CHALLENGE_BASIC_SCENARIOS = ["ChallengeBasic"]


class ChallengeBasic(BasicScenario):

    """
    Implementation of a dummy scenario
    """

    category = "ChallengeBasic"
    radius = 10.0           # meters
    timeout = 1000            # Timeout of scenario in seconds

    def __init__(self, world, ego_vehicle, other_actors, town, randomize=False, debug_mode=False, config=None):
        """
        Setup all relevant parameters and create scenario
        """
        self.config = config
        self.target = None
        self.route = None

        if hasattr(self.config, 'target'):
            self.target = self.config.target
        if hasattr(self.config, 'route'):
            self.route = self.config.route

        super(ChallengeBasic, self).__init__("ChallengeBasic", ego_vehicle, other_actors, town, world, debug_mode, True)

    def _create_behavior(self):
        """
        """
        # Build behavior tree
        sequence = py_trees.composites.Sequence("Sequence Behavior")
        idle_behavior = Idle()
        sequence.add_child(idle_behavior)

        return sequence

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """

        collision_criterion = CollisionTest(self.ego_vehicle, terminate_on_failure=True)
        target_criterion = InRadiusRegionTest(self.ego_vehicle,
                                             x=self.target.transform.location.x ,
                                             y=self.target.transform.location.y,
                                             radius=self.radius)

        route_criterion = InRouteTest(self.ego_vehicle,
                                      radius=30.0,
                                      route=self.route,
                                      offroad_max=20,
                                      terminate_on_failure=True)

        completion_criterion = RouteCompletionTest(self.ego_vehicle, route=self.route)

        parallel_criteria = py_trees.composites.Parallel("group_criteria",
                                                         policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)

        parallel_criteria.add_child(completion_criterion)
        parallel_criteria.add_child(collision_criterion)
        parallel_criteria.add_child(target_criterion)
        parallel_criteria.add_child(route_criterion)

        return parallel_criteria
