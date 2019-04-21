#!/usr/bin/env python

#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Basic CARLA Autonomous Driving training scenario
"""

import py_trees

from srunner.tools.config_parser import RouteConfiguration, TargetConfiguration
from srunner.scenariomanager.atomic_scenario_behavior import Idle
from srunner.scenariomanager.atomic_scenario_criteria import (InRadiusRegionTest, OnSidewalkTest, RouteCompletionTest,
                                                              RunningRedLightTest, RunningStopTest, WrongLaneTest)
from srunner.scenarios.basic_scenario import BasicScenario


class MasterScenario(BasicScenario):
    """
    Implementation of a Master scenario that controls the route.
    """

    category = "Master"
    radius = 10.0           # meters

    def __init__(self, world, ego_vehicle, config, debug_mode=False, criteria_enable=True,
                 name="MasterScenario", timeout=300):
        """
        Setup all relevant parameters and create scenario
        """
        self.config = config
        self._target = None
        self._route = None
        # Timeout of scenario in seconds
        self.timeout = timeout

        if hasattr(self.config, 'target'):
            self._target = self.config.target
        else:
            raise ValueError("Master scenario must have a target")

        if hasattr(self.config, 'route'):
            self._route = self.config.route
        else:
            raise ValueError("Master scenario must have a route")

        super(MasterScenario, self).__init__(name=name, ego_vehicle=ego_vehicle,
                                             config=config, world=world, debug_mode=debug_mode,
                                             terminate_on_failure=True, criteria_enable=criteria_enable)

    @property
    def route(self):
        if isinstance(self._route, RouteConfiguration):
            return self._route.data
        else:
            return self._route

    @property
    def target(self):
        if isinstance(self._target, TargetConfiguration):
            return self._target.transform.location
        else:
            return self._target.location

    def _create_behavior(self):
        """
        Basic behavior do nothing, i.e. Idle
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
        target_criterion = InRadiusRegionTest(self.ego_vehicle,
                                              x=self.target.x,
                                              y=self.target.y,
                                              radius=self.radius)
        completion_criterion = RouteCompletionTest(self.ego_vehicle, route=self.route)
        wrong_way_criterion = WrongLaneTest(self.ego_vehicle)
        onsidewalk_criterion = OnSidewalkTest(self.ego_vehicle)
        red_light_criterion = RunningRedLightTest(self.ego_vehicle)
        stop_criterion = RunningStopTest(self.ego_vehicle)

        parallel_criteria = py_trees.composites.Parallel("group_criteria",
                                                         policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        parallel_criteria.add_child(target_criterion)
        parallel_criteria.add_child(completion_criterion)
        parallel_criteria.add_child(wrong_way_criterion)
        parallel_criteria.add_child(onsidewalk_criterion)
        parallel_criteria.add_child(red_light_criterion)
        parallel_criteria.add_child(stop_criterion)

        return parallel_criteria

    def __del__(self):
        """
        Remove all actors upon deletion
        """
        self.remove_all_actors()
