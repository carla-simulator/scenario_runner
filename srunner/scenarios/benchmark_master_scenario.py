#!/usr/bin/env python

#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Master Scenario for the CoRL 2017 / Carla 100 benchmarks.
"""

import numpy as np

from srunner.scenarios.master_scenario import MasterScenario


BENCHMARK_MASTER_SCENARIO = ["BenchmarkMasterScenario"]


def _path_distance(trajectory):
    """Sum of L2 distance between points in trajectory."""
    trajectory = np.array([np.array([wp[0].x, wp[0].y, wp[0].z]) for wp in trajectory])
    deltas = trajectory[1:, :] - trajectory[:-1, :]
    sum_of_squares = (deltas * deltas).sum(axis=1)
    return np.sqrt(sum_of_squares).sum()


class BenchmarkMasterScenario(MasterScenario):
    """
    Master scenario for the CoRL 2017 / Carla 100 benchmarks.

    Differences from the original benchmarks for 0.8.x:
      - Waypoints are provided and the episode terminates if the vehicle strays from the waypoints.
        By contrast, originally only turn-by-turn indications were provided.

    Differences form ChallengeMasterScenario:
      - No extra termination conditions, e.g. when crashing or running a red light.
      - Timeout is computed dynamically based on route distance rather than being fixed.
    """

    def __init__(self, world, ego_vehicle, config, debug_mode=False, criteria_enable=True, timeout=300):
        """
        Setup all relevant parameters and create scenario
        """
        super(BenchmarkMasterScenario, self).__init__(name="BenchmarkMasterScenario", world=world,
                                                      ego_vehicle=ego_vehicle, config=config, debug_mode=debug_mode,
                                                      criteria_enable=criteria_enable, timeout=timeout)

        km_distance = _path_distance(self.route) / 1000.0
        min_speed = 10.0 / 3600.0  # 10 km/h
        # Time in seconds to drive km_distance at min_speed, plus 10s leeway
        self.timeout = min(self.timeout, km_distance / min_speed + 10.0)
