#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Module used to parse all the route and scenario configuration parameters.
"""

from __future__ import print_function

import py_trees

from srunner.scenarios.basic_scenario import BasicScenario
from srunner.tools.background_manager import (ChangeRoadBehavior,
                                              ChangeOppositeBehavior,
                                              ChangeJunctionBehavior)

class BackgroundActivityParametrizer(BasicScenario):
    """
    This class holds everything required to change the parameters of the background activity.
    Mainly used to change its behavior when, for example, moving from a highway into the city,
    where we might want a different BA behavior.
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=180):
        """
        Setup all relevant parameters and create scenario
        and instantiate scenario manager
        """
        # Road
        self._num_front_vehicles = None
        if 'num_front_vehicles' in config.other_parameters:
            self._num_front_vehicles = float(config.other_parameters['num_front_vehicles']['value'])
        self._num_back_vehicles = None
        if 'num_back_vehicles' in config.other_parameters:
            self._num_back_vehicles = float(config.other_parameters['num_back_vehicles']['value'])
        self._road_spawn_dist = None
        if 'road_spawn_dist' in config.other_parameters:
            self._road_spawn_dist = float(config.other_parameters['road_spawn_dist']['value'])

        # Opposite
        self._opposite_source_dist = None
        if 'opposite_source_dist' in config.other_parameters:
            self._opposite_source_dist = float(config.other_parameters['opposite_source_dist']['value'])
        self._opposite_max_actors = None
        if 'opposite_max_actors' in config.other_parameters:
            self._opposite_max_actors = float(config.other_parameters['opposite_max_actors']['value'])
        self._opposite_spawn_dist = None
        if 'opposite_spawn_dist' in config.other_parameters:
            self._opposite_spawn_dist = float(config.other_parameters['opposite_spawn_dist']['value'])
        self._opposite_active = None
        if 'opposite_active' in config.other_parameters:
            self._opposite_active = float(config.other_parameters['opposite_active']['value'])

        # Junction
        self._junction_source_dist = None
        if 'junction_source_dist' in config.other_parameters:
            self._junction_source_dist = float(config.other_parameters['junction_source_dist']['value'])
        self._junction_max_actors = None
        if 'junction_max_actors' in config.other_parameters:
            self._junction_max_actors = float(config.other_parameters['junction_max_actors']['value'])
        self._junction_spawn_dist = None
        if 'junction_spawn_dist' in config.other_parameters:
            self._junction_spawn_dist = float(config.other_parameters['junction_spawn_dist']['value'])
        self._junction_source_perc = None
        if 'junction_source_perc' in config.other_parameters:
            self._junction_source_perc = float(config.other_parameters['junction_source_perc']['value'])

        super().__init__("BackgroundActivityParametrizer",
                          ego_vehicles,
                          config,
                          world,
                          debug_mode,
                          criteria_enable=criteria_enable)

    def _create_behavior(self):
        """
        Hero vehicle is entering a junction in an urban area, at a signalized intersection,
        while another actor runs a red lift, forcing the ego to break.
        """

        sequence = py_trees.composites.Sequence()
        sequence.add_child(ChangeRoadBehavior(self._num_front_vehicles, self._num_back_vehicles, self._road_spawn_dist))
        sequence.add_child(ChangeJunctionBehavior(
            self._junction_source_dist, self._junction_max_actors, self._junction_spawn_dist, self._junction_source_perc))
        sequence.add_child(ChangeOppositeBehavior(
            self._opposite_source_dist, self._opposite_spawn_dist, self._opposite_active))

        return sequence

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        return []

    def __del__(self):
        """
        Remove all actors and traffic lights upon deletion
        """
        self.remove_all_actors()

