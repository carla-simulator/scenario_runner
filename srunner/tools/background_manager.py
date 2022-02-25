#!/usr/bin/env python

#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Several atomic behaviors to help with the communication with the background activity,
removing its interference with other scenarios
"""

import py_trees
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import AtomicBehavior


class ChangeRoadBehavior(AtomicBehavior):
    """
    Updates the blackboard to change the parameters of the road behavior.
    None values imply that these values won't be changed.

    Args:
        num_front_vehicles (int): Amount of vehicles in front of the ego. Can't be negative
        num_back_vehicles (int): Amount of vehicles behind it. Can't be negative
        vehicle_dist (float): Minimum distance between the road vehicles. Must between 0 and 'spawn_dist'
        spawn_dist (float): Minimum distance between spawned vehicles. Must be positive
        switch_source (bool): (De)activatea the road sources.
    """

    def __init__(self, num_front_vehicles=None, num_back_vehicles=None,
                 vehicle_dist=None, spawn_dist=None, switch_source=None, name="ChangeRoadBehavior"):
        self._num_front_vehicles = num_front_vehicles
        self._num_back_vehicles = num_back_vehicles
        self._vehicle_dist = vehicle_dist
        self._spawn_dist = spawn_dist
        self._switch_source = switch_source
        super(ChangeRoadBehavior, self).__init__(name)

    def update(self):
        py_trees.blackboard.Blackboard().set(
            "BA_ChangeRoadBehavior",
            [self._num_front_vehicles, self._num_back_vehicles, self._vehicle_dist, self._spawn_dist, self._switch_source],
            overwrite=True
        )
        return py_trees.common.Status.SUCCESS


class ChangeOppositeBehavior(AtomicBehavior):
    """
    Updates the blackboard to change the parameters of the opposite road behavior.
    None values imply that these values won't be changed

    Args:
        source_dist (float): Distance between the opposite sources and the ego vehicle. Must be positive
        vehicle_dist (float) Minimum distance between the opposite vehicles. Must between 0 and 'spawn_dist'
        spawn_dist (float): Minimum distance between spawned vehicles. Must be positive
        max_actors (int): Max amount of concurrent alive actors spawned by the same source. Can't be negative
    """

    def __init__(self, source_dist=None, vehicle_dist=None, spawn_dist=None,
                 max_actors=None, name="ChangeOppositeBehavior"):
        self._source_dist = source_dist
        self._vehicle_dist = vehicle_dist
        self._spawn_dist = spawn_dist
        self._max_actors = max_actors
        super(ChangeOppositeBehavior, self).__init__(name)

    def update(self):
        py_trees.blackboard.Blackboard().set(
            "BA_ChangeOppositeBehavior",
            [self._source_dist, self._vehicle_dist, self._spawn_dist, self._max_actors],
            overwrite=True
        )
        return py_trees.common.Status.SUCCESS


class ChangeJunctionBehavior(AtomicBehavior):
    """
    Updates the blackboard to change the parameters of the junction behavior.
    None values imply that these values won't be changed

    Args:
        source_dist (float): Distance between the junctiob sources and the junction entry. Must be positive
        vehicle_dist (float) Minimum distance between the junction vehicles. Must between 0 and 'spawn_dist'
        spawn_dist (float): Minimum distance between spawned vehicles. Must be positive
        max_actors (int): Max amount of concurrent alive actors spawned by the same source. Can't be negative

    """

    def __init__(self, source_dist=None, vehicle_dist=None, spawn_dist=None,
                 max_actors=None, name="ChangeJunctionBehavior"):
        self._source_dist = source_dist
        self._vehicle_dist = vehicle_dist
        self._spawn_dist = spawn_dist
        self._max_actors = max_actors
        super(ChangeJunctionBehavior, self).__init__(name)

    def update(self):
        py_trees.blackboard.Blackboard().set(
            "BA_ChangeJunctionBehavior",
            [self._source_dist, self._vehicle_dist, self._spawn_dist, self._max_actors],
            overwrite=True
        )
        return py_trees.common.Status.SUCCESS


class ActivateHardBreakScenario(AtomicBehavior):
    """
    Updates the blackboard to tell the background activity that a HardBreak scenario has to be triggered.
    'stop_duration' is the amount of time, in seconds, the vehicles will be stopped
    """

    def __init__(self, stop_duration=10, name="ActivateHardBreakScenario"):
        self._stop_duration = stop_duration
        super(ActivateHardBreakScenario, self).__init__(name)

    def update(self):
        py_trees.blackboard.Blackboard().set("BA_ActivateHardBreakScenario", self._stop_duration, overwrite=True)
        return py_trees.common.Status.SUCCESS


class HandleCrossingActor(AtomicBehavior):
    """
    Updates the blackboard to tell the background activity that a crossing actor has been triggered.
    'crossing_dist' is the distance between the crossing actor and the junction
    """

    def __init__(self, crossing_dist=10, name="HandleCrossingActor"):
        self._crossing_dist = crossing_dist
        super(HandleCrossingActor, self).__init__(name)

    def update(self):
        """Updates the blackboard and succeds"""
        py_trees.blackboard.Blackboard().set("BA_HandleCrossingActor", self._crossing_dist, overwrite=True)
        return py_trees.common.Status.SUCCESS


class JunctionScenarioManager(AtomicBehavior):
    """
    Updates the blackboard to tell the background activity that a JunctionScenarioManager has been triggered
    'entry_direction' is the direction from which the incoming traffic enters the junction. It should be
    something like 'left', 'right' or 'opposite'
    """

    def __init__(self, entry_direction, remove_entry, remove_exit, remove_middle, name="Scenario7Manager"):
        self._entry_direction = entry_direction
        self._remove_entry = remove_entry
        self._remove_exit = remove_exit
        self._remove_middle = remove_middle
        super(JunctionScenarioManager, self).__init__(name)

    def update(self):
        """Updates the blackboard and succeds"""
        py_trees.blackboard.Blackboard().set(
            "BA_JunctionScenario",
            [self._entry_direction, self._remove_entry, self._remove_exit, self._remove_middle],
            overwrite=True
        )
        return py_trees.common.Status.SUCCESS

class RoadInitialiser(AtomicBehavior):
    """
    Updates the blackboard to tell the background activity that the road behavior has to be initialized
    """
    def __init__(self, name="RoadInitialiser"):
        super(RoadInitialiser, self).__init__(name)

    def update(self):
        """Updates the blackboard and succeds"""
        py_trees.blackboard.Blackboard().set("BA_RoadInitialiser", True, overwrite=True)
        return py_trees.common.Status.SUCCESS

class HandleAccidentScenario(AtomicBehavior):
    """
    Updates the blackboard to tell the background activity that the road behavior has to be initialized
    """
    def __init__(self, accident_wp, distance, name="HandleAccidentScenario"):
        self._accident_wp = accident_wp
        self._distance = distance
        super(HandleAccidentScenario, self).__init__(name)

    def update(self):
        """Updates the blackboard and succeds"""
        py_trees.blackboard.Blackboard().set("BA_HandleAccidentScenario", [self._accident_wp, self._distance], overwrite=True)
        return py_trees.common.Status.SUCCESS
