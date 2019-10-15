#!/usr/bin/env python

# Copyright (c) 2018-2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides all atomic scenario behaviors that reflect
trigger conditions to either activate another behavior, or to stop
another behavior.

For example, such a condition could be "InTriggerRegion", which checks
that a given actor reached a certain region on the map, and then starts/stops
a behavior of this actor.

The atomics are implemented with py_trees and make use of the AtomicBehavior
base class
"""

from __future__ import print_function

import operator
import py_trees

from agents.navigation.basic_agent import *
from agents.navigation.roaming_agent import *

from srunner.scenariomanager.scenarioatomics.atomic_behaviors import AtomicBehavior, calculate_distance
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime
from srunner.tools.scenario_helper import get_distance_along_route

EPSILON = 0.001


class StandStill(AtomicBehavior):

    """
    This class contains a standstill behavior of a scenario

    Important parameters:
    - actor: CARLA actor to execute the behavior
    - name: Name of the condition
    - duration: Duration of the behavior in seconds

    The condition terminates with SUCCESS, when the actor does not move
    """

    def __init__(self, actor, name, duration=float("inf")):
        """
        Setup actor
        """
        super(StandStill, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._actor = actor

        self._duration = duration
        self._start_time = 0

    def initialise(self):
        """
        Initialize the start time of this condition
        """
        self._start_time = GameTime.get_time()
        super(StandStill, self).initialise()

    def update(self):
        """
        Check if the _actor stands still (v=0)
        """
        new_status = py_trees.common.Status.RUNNING

        velocity = CarlaDataProvider.get_velocity(self._actor)

        if velocity < EPSILON:
            new_status = py_trees.common.Status.SUCCESS

        if GameTime.get_time() - self._start_time > self._duration:
            new_status = py_trees.common.Status.SUCCESS

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status


class TriggerVelocity(AtomicBehavior):

    """
    This class contains the trigger velocity (condition) of a scenario

    Important parameters:
    - actor: CARLA actor to execute the behavior
    - name: Name of the condition
    - target_velocity: The behavior is successful, if the actor is at least as fast as target_velocity in m/s

    The condition terminates with SUCCESS, when the actor reached the target_velocity
    """

    def __init__(self, actor, target_velocity, name="TriggerVelocity"):
        """
        Setup trigger velocity
        """
        super(TriggerVelocity, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._actor = actor
        self._target_velocity = target_velocity

    def update(self):
        """
        Check if the actor has the trigger velocity
        """
        new_status = py_trees.common.Status.RUNNING

        delta_velocity = self._target_velocity - CarlaDataProvider.get_velocity(self._actor)
        if delta_velocity < EPSILON:
            new_status = py_trees.common.Status.SUCCESS

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status


class AtStartCondition(AtomicBehavior):

    """
    This class contains a check if a named story element has started.

    Important parameters:
    - element_name: The story element's name attribute
    - element_type: The element type [act,scene,maneuver,event,action]

    The condition terminates with SUCCESS, when the named story element starts
    """

    def __init__(self, element_type, element_name):
        """
        Setup element details
        """
        super(AtStartCondition, self).__init__("AtStartCondition")
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._element_type = element_type
        self._element_name = element_name
        self._start_time = None
        self._blackboard = py_trees.blackboard.Blackboard()

    def initialise(self):
        """
        Initialize the start time of this condition
        """
        self._start_time = GameTime.get_time()
        super(AtStartCondition, self).initialise()

    def update(self):
        """
        Check if the specified story element has started since the beginning of the condition
        """
        new_status = py_trees.common.Status.RUNNING

        blackboard_variable_name = "({}){}-{}".format(self._element_type.upper(), self._element_name, "START")
        element_start_time = self._blackboard.get(blackboard_variable_name)
        if element_start_time and element_start_time >= self._start_time:
            new_status = py_trees.common.Status.SUCCESS

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status


class AfterTerminationCondition(AtomicBehavior):

    """
    This class contains a check if a named story element has terminated.

    Important parameters:
    - element_name: The story element's name attribute
    - element_type: The element type [act,scene,maneuver,event,action]

    The condition terminates with SUCCESS, when the named story element ends
    """

    def __init__(self, element_type, element_name, rule):
        """
        Setup element details
        """
        super(AfterTerminationCondition, self).__init__("AfterTerminationCondition")
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._element_type = element_type.upper()
        self._element_name = element_name
        self._rule = rule.upper()
        self._start_time = GameTime.get_time()
        self._blackboard = py_trees.blackboard.Blackboard()

    def update(self):
        """
        Check if the specified story element has ended since the beginning of the condition
        """
        new_status = py_trees.common.Status.RUNNING
        if self._rule == "ANY":
            rules = ["END", "CANCEL"]
        else:
            rules = [self._rule]

        for rule in rules:
            if new_status == py_trees.common.Status.RUNNING:
                blackboard_variable_name = "({}){}-{}".format(self._element_type, self._element_name, rule)
                element_start_time = self._blackboard.get(blackboard_variable_name)
                if element_start_time and element_start_time >= self._start_time:
                    new_status = py_trees.common.Status.SUCCESS

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status


class InTriggerRegion(AtomicBehavior):

    """
    This class contains the trigger region (condition) of a scenario

    Important parameters:
    - actor: CARLA actor to execute the behavior
    - name: Name of the condition
    - min_x, max_x, min_y, max_y: bounding box of the trigger region

    The condition terminates with SUCCESS, when the actor reached the target region
    """

    def __init__(self, actor, min_x, max_x, min_y, max_y, name="TriggerRegion"):
        """
        Setup trigger region (rectangle provided by
        [min_x,min_y] and [max_x,max_y]
        """
        super(InTriggerRegion, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._actor = actor
        self._min_x = min_x
        self._max_x = max_x
        self._min_y = min_y
        self._max_y = max_y

    def update(self):
        """
        Check if the _actor location is within trigger region
        """
        new_status = py_trees.common.Status.RUNNING

        location = CarlaDataProvider.get_location(self._actor)

        if location is None:
            return new_status

        not_in_region = (location.x < self._min_x or location.x > self._max_x) or (
            location.y < self._min_y or location.y > self._max_y)
        if not not_in_region:
            new_status = py_trees.common.Status.SUCCESS

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status


class InTriggerDistanceToVehicle(AtomicBehavior):

    """
    This class contains the trigger distance (condition) between to actors
    of a scenario

    Important parameters:
    - actor: CARLA actor to execute the behavior
    - other_actor: Reference CARLA actor
    - name: Name of the condition
    - distance: Trigger distance between the two actors in meters

    The condition terminates with SUCCESS, when the actor reached the target distance to the other actor
    """

    def __init__(self, other_actor, actor, distance, comparison_operator=operator.lt, name="TriggerDistanceToVehicle"):
        """
        Setup trigger distance
        """
        super(InTriggerDistanceToVehicle, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._other_actor = other_actor
        self._actor = actor
        self._distance = distance
        self._comparison_operator = comparison_operator

    def update(self):
        """
        Check if the ego vehicle is within trigger distance to other actor
        """
        new_status = py_trees.common.Status.RUNNING

        ego_location = CarlaDataProvider.get_location(self._actor)
        other_location = CarlaDataProvider.get_location(self._other_actor)

        if ego_location is None or other_location is None:
            return new_status

        if self._comparison_operator(calculate_distance(ego_location, other_location), self._distance):
            new_status = py_trees.common.Status.SUCCESS

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status


class InTriggerDistanceToLocation(AtomicBehavior):

    """
    This class contains the trigger (condition) for a distance to a fixed
    location of a scenario

    Important parameters:
    - actor: CARLA actor to execute the behavior
    - target_location: Reference location (carla.location)
    - name: Name of the condition
    - distance: Trigger distance between the actor and the target location in meters

    The condition terminates with SUCCESS, when the actor reached the target distance to the given location
    """

    def __init__(self,
                 actor,
                 target_location,
                 distance,
                 comparison_operator=operator.lt,
                 name="InTriggerDistanceToLocation"):
        """
        Setup trigger distance
        """
        super(InTriggerDistanceToLocation, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._target_location = target_location
        self._actor = actor
        self._distance = distance
        self._comparison_operator = comparison_operator

    def update(self):
        """
        Check if the actor is within trigger distance to the target location
        """
        new_status = py_trees.common.Status.RUNNING

        location = CarlaDataProvider.get_location(self._actor)

        if location is None:
            return new_status

        if self._comparison_operator(calculate_distance(
                location, self._target_location), self._distance):
            new_status = py_trees.common.Status.SUCCESS

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status


class InTriggerDistanceToNextIntersection(AtomicBehavior):

    """
    This class contains the trigger (condition) for a distance to the
    next intersection of a scenario

    Important parameters:
    - actor: CARLA actor to execute the behavior
    - name: Name of the condition
    - distance: Trigger distance between the actor and the next intersection in meters

    The condition terminates with SUCCESS, when the actor reached the target distance to the next intersection
    """

    def __init__(self, actor, distance, name="InTriggerDistanceToNextIntersection"):
        """
        Setup trigger distance
        """
        super(InTriggerDistanceToNextIntersection, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._actor = actor
        self._distance = distance
        self._map = CarlaDataProvider.get_map()

        waypoint = self._map.get_waypoint(self._actor.get_location())
        while waypoint and not waypoint.is_intersection:
            waypoint = waypoint.next(1)[-1]

        self._final_location = waypoint.transform.location

    def update(self):
        """
        Check if the actor is within trigger distance to the intersection
        """
        new_status = py_trees.common.Status.RUNNING

        current_waypoint = self._map.get_waypoint(CarlaDataProvider.get_location(self._actor))
        distance = calculate_distance(current_waypoint.transform.location, self._final_location)

        if distance < self._distance:
            new_status = py_trees.common.Status.SUCCESS

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status


class InTriggerDistanceToLocationAlongRoute(AtomicBehavior):

    """
    Implementation for a behavior that will check if a given actor
    is within a given distance to a given location considering a given route

    Important parameters:
    - actor: CARLA actor to execute the behavior
    - name: Name of the condition
    - distance: Trigger distance between the actor and the next intersection in meters
    - route: Route to be checked
    - location: Location on the route to be checked

    The condition terminates with SUCCESS, when the actor reached the target distance
    along its route to the given location
    """

    def __init__(self, actor, route, location, distance, name="InTriggerDistanceToLocationAlongRoute"):
        """
        Setup class members
        """
        super(InTriggerDistanceToLocationAlongRoute, self).__init__(name)
        self._map = CarlaDataProvider.get_map()
        self._actor = actor
        self._location = location
        self._route = route
        self._distance = distance

        self._location_distance, _ = get_distance_along_route(self._route, self._location)

    def update(self):
        new_status = py_trees.common.Status.RUNNING

        current_location = CarlaDataProvider.get_location(self._actor)

        if current_location is None:
            return new_status

        if current_location.distance(self._location) < self._distance + 20:

            actor_distance, _ = get_distance_along_route(self._route, current_location)

            if (self._location_distance < actor_distance + self._distance and
                actor_distance < self._location_distance) or \
                    self._location_distance < 1.0:
                new_status = py_trees.common.Status.SUCCESS

        return new_status


class InTimeToArrivalToLocation(AtomicBehavior):

    """
    This class contains a check if a actor arrives within a given time
    at a given location.

    Important parameters:
    - actor: CARLA actor to execute the behavior
    - name: Name of the condition
    - time: The behavior is successful, if TTA is less than _time_ in seconds
    - location: Location to be checked in this behavior

    The condition terminates with SUCCESS, when the actor can reach the target location within the given time
    """

    _max_time_to_arrival = float('inf')  # time to arrival in seconds

    def __init__(self, actor, time, location, comparison_operator=operator.lt, name="TimeToArrival"):
        """
        Setup parameters
        """
        super(InTimeToArrivalToLocation, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._actor = actor
        self._time = time
        self._target_location = location
        self._comparison_operator = comparison_operator

    def update(self):
        """
        Check if the actor can arrive at target_location within time
        """
        new_status = py_trees.common.Status.RUNNING

        current_location = CarlaDataProvider.get_location(self._actor)

        if current_location is None:
            return new_status

        distance = calculate_distance(current_location, self._target_location)
        velocity = CarlaDataProvider.get_velocity(self._actor)

        # if velocity is too small, simply use a large time to arrival
        time_to_arrival = self._max_time_to_arrival
        if velocity > EPSILON:
            time_to_arrival = distance / velocity

        if self._comparison_operator(time_to_arrival, self._time):
            new_status = py_trees.common.Status.SUCCESS

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status


class InTimeToArrivalToVehicle(AtomicBehavior):

    """
    This class contains a check if a actor arrives within a given time
    at another actor.

    Important parameters:
    - actor: CARLA actor to execute the behavior
    - name: Name of the condition
    - time: The behavior is successful, if TTA is less than _time_ in seconds
    - other_actor: Reference actor used in this behavior

    The condition terminates with SUCCESS, when the actor can reach the other vehicle within the given time
    """

    _max_time_to_arrival = float('inf')  # time to arrival in seconds

    def __init__(self, other_actor, actor, time, comparison_operator=operator.lt, name="TimeToArrival"):
        """
        Setup parameters
        """
        super(InTimeToArrivalToVehicle, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._other_actor = other_actor
        self._actor = actor
        self._time = time
        self._comparison_operator = comparison_operator

    def update(self):
        """
        Check if the ego vehicle can arrive at other actor within time
        """
        new_status = py_trees.common.Status.RUNNING

        current_location = CarlaDataProvider.get_location(self._actor)
        target_location = CarlaDataProvider.get_location(self._other_actor)

        if current_location is None or target_location is None:
            return new_status

        distance = calculate_distance(current_location, target_location)
        current_velocity = CarlaDataProvider.get_velocity(self._actor)
        other_velocity = CarlaDataProvider.get_velocity(self._other_actor)

        # if velocity is too small, simply use a large time to arrival
        time_to_arrival = self._max_time_to_arrival

        if current_velocity > other_velocity:
            time_to_arrival = 2 * distance / (current_velocity - other_velocity)

        if self._comparison_operator(time_to_arrival, self._time):
            new_status = py_trees.common.Status.SUCCESS

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status


class DriveDistance(AtomicBehavior):

    """
    This class contains an atomic behavior to drive a certain distance.

    Important parameters:
    - actor: CARLA actor to execute the condition
    - distance: Distance for this condition in meters

    The condition terminates with SUCCESS, when the actor drove at least the given distance
    """

    def __init__(self, actor, distance, name="DriveDistance"):
        """
        Setup parameters
        """
        super(DriveDistance, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._target_distance = distance
        self._distance = 0
        self._location = None
        self._actor = actor

    def initialise(self):
        self._location = CarlaDataProvider.get_location(self._actor)
        super(DriveDistance, self).initialise()

    def update(self):
        """
        Check driven distance
        """
        new_status = py_trees.common.Status.RUNNING

        new_location = CarlaDataProvider.get_location(self._actor)
        self._distance += calculate_distance(self._location, new_location)
        self._location = new_location

        if self._distance > self._target_distance:
            new_status = py_trees.common.Status.SUCCESS

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))
        return new_status


class WaitForTrafficLightState(AtomicBehavior):

    """
    This class contains an atomic behavior to wait for a given traffic light
    to have the desired state.

    Important parameters:
    - traffic_light: CARLA traffic light to execute the condition
    - state: State to be checked in this condition

    The condition terminates with SUCCESS, when the traffic light switches to the desired state
    """

    def __init__(self, traffic_light, state, name="WaitForTrafficLightState"):
        """
        Setup traffic_light
        """
        super(WaitForTrafficLightState, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._traffic_light = traffic_light
        self._traffic_light_state = state

    def update(self):
        """
        Set status to SUCCESS, when traffic light state is RED
        """
        new_status = py_trees.common.Status.RUNNING

        # the next line may throw, if self._traffic_light is not a traffic
        # light, but another actor. This is intended.
        if str(self._traffic_light.state) == self._traffic_light_state:
            new_status = py_trees.common.Status.SUCCESS

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status

    def terminate(self, new_status):
        self._traffic_light = None
        super(WaitForTrafficLightState, self).terminate(new_status)
