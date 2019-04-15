#!/usr/bin/env python

#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Master scenario for the Carla Challenge.
"""

from srunner.scenariomanager.atomic_scenario_criteria import (CollisionTest, WrongLaneTest, OnSidewalkTest,
                                                              RunningRedLightTest, RunningStopTest)
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

        collision_criterion = CollisionTest(self.ego_vehicle, terminate_on_failure=True)
        wrong_way_criterion = WrongLaneTest(self.ego_vehicle)
        onsidewalk_criterion = OnSidewalkTest(self.ego_vehicle)
        red_light_criterion = RunningRedLightTest(self.ego_vehicle)
        stop_criterion = RunningStopTest(self.ego_vehicle)

        parallel_criteria.add_child(collision_criterion)
        parallel_criteria.add_child(wrong_way_criterion)
        parallel_criteria.add_child(onsidewalk_criterion)
        parallel_criteria.add_child(red_light_criterion)
        parallel_criteria.add_child(stop_criterion)

        return parallel_criteria
