#!/usr/bin/env python

#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Master scenario for the Carla Challenge.
"""

from srunner.scenariomanager.atomic_scenario_criteria import (CollisionTest, InRouteTest)
from srunner.scenarios.master_scenario import MasterScenario


CHALLENGE_MASTER_SCENARIO = ["ChallengeMasterScenario"]


class ChallengeMasterScenario(MasterScenario):
    """
    Master scenario for the Carla Challenge.
    """

    def __init__(self, world, ego_vehicle, config, debug_mode=False, criteria_enable=True,
                 timeout=300):
        """
        Setup all relevant parameters and create scenario
        """
        super(ChallengeMasterScenario, self).__init__(name="ChallengeMasterScenario", world=world,
                                                      ego_vehicle=ego_vehicle, config=config, debug_mode=debug_mode,
                                                      criteria_enable=criteria_enable, timeout=timeout)

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        parallel_criteria = super(ChallengeMasterScenario, self)._create_test_criteria()

        route_criterion = InRouteTest(self.ego_vehicle,
                                      radius=30.0,
                                      route=self.route,
                                      offroad_max=20,
                                      terminate_on_failure=True)
        collision_criterion = CollisionTest(self.ego_vehicle, terminate_on_failure=True)

        parallel_criteria.add_child(route_criterion)
        parallel_criteria.add_child(collision_criterion)

        return parallel_criteria
