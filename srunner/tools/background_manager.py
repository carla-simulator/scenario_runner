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


class ChangeGeneralBehavior(AtomicBehavior):
    """
    Updates the blackboard to change the general parameters.
    None values imply that these values won't be changed.

    Args:
        spawn_dist (float): Minimum distance between spawned vehicles. Must be positive
        target_speed (float): Target speed of all BA vehicles
    """

    def __init__(self, spawn_dist=None, target_speed=None, name="ChangeGeneralBehavior"):
        self._spawn_dist = spawn_dist
        self._target_speed = target_speed
        super().__init__(name)

    def update(self):
        py_trees.blackboard.Blackboard().set(
            "BA_ChangeGeneralBehavior", [self._spawn_dist, self._target_speed], overwrite=True
        )
        return py_trees.common.Status.SUCCESS


class ChangeRoadBehavior(AtomicBehavior):
    """
    Updates the blackboard to change the parameters of the road behavior.
    None values imply that these values won't be changed.

    Args:
        num_front_vehicles (int): Amount of vehicles in front of the ego. Can't be negative
        num_back_vehicles (int): Amount of vehicles behind it. Can't be negative
        switch_source (bool): (De)activatea the road sources.
    """

    def __init__(self, num_front_vehicles=None, num_back_vehicles=None, name="ChangeRoadBehavior"):
        self._num_front_vehicles = num_front_vehicles
        self._num_back_vehicles = num_back_vehicles
        super().__init__(name)

    def update(self):
        py_trees.blackboard.Blackboard().set(
            "BA_ChangeRoadBehavior", [self._num_front_vehicles, self._num_back_vehicles], overwrite=True
        )
        return py_trees.common.Status.SUCCESS


class ChangeOppositeBehavior(AtomicBehavior):
    """
    Updates the blackboard to change the parameters of the opposite road behavior.
    None values imply that these values won't be changed

    Args:
        source_dist (float): Distance between the opposite sources and the ego vehicle. Must be positive
        max_actors (int): Max amount of concurrent alive actors spawned by the same source. Can't be negative
    """

    def __init__(self, source_dist=None, max_actors=None, spawn_dist=None, active=None, name="ChangeOppositeBehavior"):
        self._source_dist = source_dist
        self._max_actors = max_actors
        self._spawn_dist = spawn_dist
        self._active = active
        super().__init__(name)

    def update(self):
        py_trees.blackboard.Blackboard().set(
            "BA_ChangeOppositeBehavior", [self._source_dist, self._max_actors, self._spawn_dist, self._active], overwrite=True
        )
        return py_trees.common.Status.SUCCESS


class ChangeJunctionBehavior(AtomicBehavior):
    """
    Updates the blackboard to change the parameters of the junction behavior.
    None values imply that these values won't be changed

    Args:
        source_dist (float): Distance between the junctiob sources and the junction entry. Must be positive
        max_actors (int): Max amount of concurrent alive actors spawned by the same source. Can't be negative
    """

    def __init__(self, source_dist=None, vehicle_dist=None, spawn_dist=None,
                 max_actors=None, name="ChangeJunctionBehavior"):
        self._source_dist = source_dist
        self._max_actors = max_actors
        super().__init__(name)

    def update(self):
        py_trees.blackboard.Blackboard().set(
            "BA_ChangeJunctionBehavior", [self._source_dist, self._max_actors], overwrite=True
        )
        return py_trees.common.Status.SUCCESS


class StopFrontVehicles(AtomicBehavior):
    """
    Updates the blackboard to tell the background activity that a HardBreak scenario has to be triggered.
    'stop_duration' is the amount of time, in seconds, the vehicles will be stopped
    """

    def __init__(self, name="StopFrontVehicles"):
        super().__init__(name)

    def update(self):
        py_trees.blackboard.Blackboard().set("BA_StopFrontVehicles", True, overwrite=True)
        return py_trees.common.Status.SUCCESS


class StartFrontVehicles(AtomicBehavior):
    """
    Updates the blackboard to tell the background activity that a HardBreak scenario has to be triggered.
    'stop_duration' is the amount of time, in seconds, the vehicles will be stopped
    """

    def __init__(self, name="StartFrontVehicles"):
        super().__init__(name)

    def update(self):
        py_trees.blackboard.Blackboard().set("BA_StartFrontVehicles", True, overwrite=True)
        return py_trees.common.Status.SUCCESS


class JunctionScenarioManager(AtomicBehavior):
    """
    Updates the blackboard to tell the background activity that a JunctionScenarioManager has been triggered
    'entry_direction' is the direction from which the incoming traffic enters the junction. It should be
    something like 'left', 'right' or 'opposite'
    """

    def __init__(self, entry_direction, remove_exit=True, name="JunctionScenarioManager"):
        self._entry_direction = entry_direction
        self._remove_exit = remove_exit
        super().__init__(name)

    def update(self):
        """Updates the blackboard and succeds"""
        py_trees.blackboard.Blackboard().set(
            "BA_JunctionScenario",
            [self._entry_direction, self._remove_exit],
            overwrite=True
        )
        return py_trees.common.Status.SUCCESS


class ExtentExitRoadSpace(AtomicBehavior):
    """
    Updates the blackboard to tell the background activity that an exit road needs more space
    """
    def __init__(self, distance, name="ExtentExitRoadSpace"):
        self._distance = distance
        super().__init__(name)

    def update(self):
        """Updates the blackboard and succeds"""
        py_trees.blackboard.Blackboard().set("BA_ExtentExitRoadSpace", self._distance, overwrite=True)
        return py_trees.common.Status.SUCCESS


class LeaveSpaceInFront(AtomicBehavior):
    """
    Updates the blackboard to tell the background activity that the ego needs more space in front.
    This only works at roads, not junctions.
    """
    def __init__(self, space, name="LeaveSpaceInFront"):
        self._space = space
        super().__init__(name)

    def update(self):
        """Updates the blackboard and succeds"""
        py_trees.blackboard.Blackboard().set("BA_LeaveSpaceInFront", self._space, overwrite=True)
        return py_trees.common.Status.SUCCESS


class SwitchRouteSources(AtomicBehavior):
    """
    Updates the blackboard to tell the background activity to (de)activate all route sources
    """
    def __init__(self, enabled=True, name="SwitchRouteSources"):
        self._enabled = enabled
        super().__init__(name)

    def update(self):
        """Updates the blackboard and succeds"""
        py_trees.blackboard.Blackboard().set("BA_SwitchRouteSources", self._enabled, overwrite=True)
        return py_trees.common.Status.SUCCESS


class HandleStartAccidentScenario(AtomicBehavior):
    """
    Updates the blackboard to tell the background activity that the road behavior has to be initialized
    """
    def __init__(self, accident_wp, distance, name="HandleStartAccidentScenario"):
        self._accident_wp = accident_wp
        self._distance = distance
        super().__init__(name)

    def update(self):
        """Updates the blackboard and succeds"""
        py_trees.blackboard.Blackboard().set("BA_HandleStartAccidentScenario", [self._accident_wp, self._distance], overwrite=True)
        return py_trees.common.Status.SUCCESS


class HandleEndAccidentScenario(AtomicBehavior):
    """
    Updates the blackboard to tell the background activity that the road behavior has to be initialized
    """
    def __init__(self, name="HandleEndAccidentScenario"):
        super().__init__(name)

    def update(self):
        """Updates the blackboard and succeds"""
        py_trees.blackboard.Blackboard().set("BA_HandleEndAccidentScenario", True, overwrite=True)
        return py_trees.common.Status.SUCCESS

class SwitchLane(AtomicBehavior):
    """
    Updates the blackboard to tell the background activity to remove its actors from the given lane 
    and stop generating new ones on this lane, or recover from stopping.

    Args:
        lane_id (str): A carla.Waypoint.lane_id
        active (bool)
    """
    def __init__(self, lane_id=None, active=True, name="SwitchLane"):
        self._lane_id = lane_id
        self._active = active
        super().__init__(name)

    def update(self):
        """Updates the blackboard and succeds"""
        py_trees.blackboard.Blackboard().set("BA_SwitchLane", [self._lane_id, self._active], overwrite=True)
        return py_trees.common.Status.SUCCESS

class RemoveJunctionEntry(AtomicBehavior):
    """
    Updates the blackboard to tell the background activity to remove its actors from the given lane,
    and stop generating new ones on this lane.

    Args:
        wp (carla.Waypoint): A waypoint used as reference to the entry lane
        all_road_entries (bool): Boolean to remove all entries part of the same road, or just one
    """
    def __init__(self, wp, all_road_entries=False, name="RemoveJunctionEntry"):
        self._wp = wp
        self._all_road_entries = all_road_entries
        super().__init__(name)

    def update(self):
        """Updates the blackboard and succeds"""
        py_trees.blackboard.Blackboard().set("BA_RemoveJunctionEntry", [self._wp, self._all_road_entries], overwrite=True)
        return py_trees.common.Status.SUCCESS


class ClearJunction(AtomicBehavior):
    """
    Updates the blackboard to tell the background activity to remove all actors inside the junction,
    and stop the ones that are about to enter it, leaving an empty space inside the junction.
    """

    def __init__(self, name="ClearJunction"):
        super().__init__(name)

    def update(self):
        """Updates the blackboard and succeds"""
        py_trees.blackboard.Blackboard().set("BA_ClearJunction", True, overwrite=True)
        return py_trees.common.Status.SUCCESS
