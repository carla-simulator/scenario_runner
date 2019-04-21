#!/usr/bin/env python

#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Master Scenario for the CoRL 2017 / Carla 100 benchmarks.
"""

from srunner.scenarios.master_scenario import MasterScenario
from srunner.scenariomanager.atomic_scenario_criteria import CollisionTest


BENCHMARK_MASTER_SCENARIO = ["BenchmarkMasterScenario"]


class BenchmarkMasterScenario(MasterScenario):
    """
    Master scenario for the CoRL 2017 / Carla 100 benchmarks.

    Differences from the original benchmarks for 0.8.x:
      - Waypoints are provided and the episode terminates if the vehicle strays from the waypoints.
        By contrast, originally only turn-by-turn indications were provided.
      - The route plan is constructed by agents.navigation.global_route_planner.GlobalRoutePlanner
        instead of carla.planner.CityTrack.compute_route. I believe they produce similar directions
        but there are some implementation differences.
      - For Carla 100, two wheeled vehicles are included, which were excluded in the original benchmark.
      - Pedestrians are not currently supported in 0.9.x (to be added soon).

    Differences form ChallengeMasterScenario:
      - No extra termination conditions, e.g. when crashing or running a red light.
    """

    def __init__(self, world, ego_vehicle, config, debug_mode=False, criteria_enable=True, timeout=300):
        """
        Setup all relevant parameters and create scenario
        """
        super(BenchmarkMasterScenario, self).__init__(name="BenchmarkMasterScenario", world=world,
                                                      ego_vehicle=ego_vehicle, config=config, debug_mode=debug_mode,
                                                      criteria_enable=criteria_enable, timeout=timeout)

    def _create_test_criteria(self):
        parallel_criteria = super(BenchmarkMasterScenario, self)._create_test_criteria()
        collision_criterion = CollisionTest(self.ego_vehicle, terminate_on_failure=False)
        parallel_criteria.add_child(collision_criterion)

        return parallel_criteria