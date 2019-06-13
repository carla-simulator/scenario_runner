#!/usr/bin/env python

# Copyright (c) 2018-2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides all atomic scenario behaviors required to realize
complex, realistic scenarios such as "follow a leading vehicle", "lane change",
etc.

The atomic behaviors are implemented with py_trees.
"""

from __future__ import print_function

import random

import numpy as np
import py_trees
from py_trees.blackboard import Blackboard

import carla
from agents.navigation.basic_agent import *
from agents.navigation.roaming_agent import *

from srunner.scenariomanager.carla_data_provider import CarlaActorPool, CarlaDataProvider
from srunner.scenariomanager.timer import GameTime
from srunner.tools.scenario_helper import detect_lane_obstacle, get_distance_along_route

EPSILON = 0.001


def calculate_distance(location, other_location):
    """
    Method to calculate the distance between to locations

    Note: It uses the direct distance between the current location and the
          target location to estimate the time to arrival.
          To be accurate, it would have to use the distance along the
          (shortest) route between the two locations.
    """
    return location.distance(other_location)


def get_actor_control(actor):
    """
    Method to return the type of control to the actor.
    """
    control = None
    actor_type = actor.type_id.split('.')[0]
    if actor.type_id.split('.')[0] == 'vehicle':
        control = carla.VehicleControl()
        control.steering = 0
    elif actor.type_id.split('.')[0] == 'walker':
        control = carla.WalkerControl()

    return control, actor_type


class AtomicBehavior(py_trees.behaviour.Behaviour):

    """
    Base class for all atomic behaviors used to setup a scenario

    Important parameters:
    - name: Name of the atomic behavior
    """

    def __init__(self, name):
        super(AtomicBehavior, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self.name = name

    def setup(self, unused_timeout=15):
        self.logger.debug("%s.setup()" % (self.__class__.__name__))
        return True

    def initialise(self):
        self.logger.debug("%s.initialise()" % (self.__class__.__name__))

    def terminate(self, new_status):
        self.logger.debug("%s.terminate()[%s->%s]" % (self.__class__.__name__, self.status, new_status))


class StandStill(AtomicBehavior):

    """
    This class contains a standstill behavior of a scenario
    """

    def __init__(self, actor, name):
        """
        Setup actor
        """
        super(StandStill, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._actor = actor

    def update(self):
        """
        Check if the _actor stands still (v=0)
        """
        new_status = py_trees.common.Status.RUNNING

        velocity = CarlaDataProvider.get_velocity(self._actor)

        if velocity < EPSILON:
            new_status = py_trees.common.Status.SUCCESS

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status


class InTriggerRegion(AtomicBehavior):

    """
    This class contains the trigger region (condition) of a scenario
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
    """

    def __init__(self, other_actor, actor, distance, name="TriggerDistanceToVehicle"):
        """
        Setup trigger distance
        """
        super(InTriggerDistanceToVehicle, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._other_actor = other_actor
        self._actor = actor
        self._distance = distance

    def update(self):
        """
        Check if the ego vehicle is within trigger distance to other actor
        """
        new_status = py_trees.common.Status.RUNNING

        ego_location = CarlaDataProvider.get_location(self._actor)
        other_location = CarlaDataProvider.get_location(self._other_actor)

        if ego_location is None or other_location is None:
            return new_status

        if calculate_distance(ego_location, other_location) < self._distance:
            new_status = py_trees.common.Status.SUCCESS

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status


class InTriggerDistanceToLocation(AtomicBehavior):

    """
    This class contains the trigger (condition) for a distance to a fixed
    location of a scenario
    """

    def __init__(self, actor, target_location, distance, name="InTriggerDistanceToLocation"):
        """
        Setup trigger distance
        """
        super(InTriggerDistanceToLocation, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._target_location = target_location
        self._actor = actor
        self._distance = distance

    def update(self):
        """
        Check if the actor is within trigger distance to the target location
        """
        new_status = py_trees.common.Status.RUNNING

        location = CarlaDataProvider.get_location(self._actor)

        if location is None:
            return new_status

        if calculate_distance(
                location, self._target_location) < self._distance:
            new_status = py_trees.common.Status.SUCCESS

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status


class InTriggerDistanceToNextIntersection(AtomicBehavior):

    """
    This class contains the trigger (condition) for a distance to the
    next intersection of a scenario
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


class TriggerVelocity(AtomicBehavior):

    """
    This class contains the trigger velocity (condition) of a scenario

    The behavior is successful, if the actor is at least as fast as requested
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


class InTimeToArrivalToLocation(AtomicBehavior):

    """
    This class contains a check if a actor arrives within a given time
    at a given location.
    """

    _max_time_to_arrival = float('inf')  # time to arrival in seconds

    def __init__(self, actor, time, location, name="TimeToArrival"):
        """
        Setup parameters
        """
        super(InTimeToArrivalToLocation, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._actor = actor
        self._time = time
        self._target_location = location

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

        if time_to_arrival < self._time:
            new_status = py_trees.common.Status.SUCCESS

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status


class InTimeToArrivalToVehicle(AtomicBehavior):

    """
    This class contains a check if a actor arrives within a given time
    at another actor.
    """

    _max_time_to_arrival = float('inf')  # time to arrival in seconds

    def __init__(self, other_actor, actor, time, name="TimeToArrival"):
        """
        Setup parameters
        """
        super(InTimeToArrivalToVehicle, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._other_actor = other_actor
        self._actor = actor
        self._time = time

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

        if time_to_arrival < self._time:
            new_status = py_trees.common.Status.SUCCESS

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status


class InTriggerDistanceToLocationAlongRoute(AtomicBehavior):

    """
    Implementation for a behavior that will check if a given actor
    is within a given distance to a given location considering a given route
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


class AccelerateToVelocity(AtomicBehavior):

    """
    This class contains an atomic acceleration behavior. The controlled
    traffic participant will accelerate with _throttle_value_ until reaching
    a given _target_velocity_
    """

    def __init__(self, actor, throttle_value, target_velocity, walker_direction=0, name="Acceleration"):
        """
        Setup parameters including acceleration value (via throttle_value)
        and target velocity
        """
        super(AccelerateToVelocity, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._control, self._type = get_actor_control(actor)
        if self._type == 'walker':
            self._control.speed = target_velocity
            self._control.direction = carla.Rotation(0, walker_direction, 0).get_forward_vector()
        self._actor = actor
        self._throttle_value = throttle_value
        self._target_velocity = target_velocity

    def update(self):
        """
        Set throttle to throttle_value, as long as velocity is < target_velocity
        """
        new_status = py_trees.common.Status.RUNNING

        if self._type == 'vehicle':
            if CarlaDataProvider.get_velocity(self._actor) < self._target_velocity:
                self._control.throttle = self._throttle_value
            else:
                new_status = py_trees.common.Status.SUCCESS
                self._control.throttle = 0

        self._actor.apply_control(self._control)
        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status


class KeepVelocity(AtomicBehavior):

    """
    This class contains an atomic behavior to keep the provided velocity.
    The controlled traffic participant will accelerate as fast as possible
    until reaching a given _target_velocity_, which is then maintained for
    as long as this behavior is active.

    Note: In parallel to this behavior a termination behavior has to be used
          to keep the velocity either for a certain duration, or for a certain
          distance, etc.
    """

    def __init__(self, actor, target_velocity, walker_direction=0, name="KeepVelocity"):
        """
        Setup parameters including acceleration value (via throttle_value)
        and target velocity
        """
        super(KeepVelocity, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._actor = actor
        self._target_velocity = target_velocity

        self._control, self._type = get_actor_control(actor)
        self._map = self._actor.get_world().get_map()
        self._waypoint = self._map.get_waypoint(self._actor.get_location())
        if self._type == 'walker':
            self._control.speed = target_velocity
            self._control.direction = carla.Rotation(0, walker_direction, 0).get_forward_vector()

    def update(self):
        """
        Set throttle to throttle_value, as long as velocity is < target_velocity
        """
        new_status = py_trees.common.Status.RUNNING

        if self._type == 'vehicle':
            if CarlaDataProvider.get_velocity(self._actor) < self._target_velocity:
                self._control.throttle = 1.0
            else:
                self._control.throttle = 0.0
        self._actor.apply_control(self._control)

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status

    def terminate(self, new_status):
        """
        On termination of this behavior, the throttle should be set back to 0.,
        to avoid further acceleration.
        """
        self._control.throttle = 0.0
        if self._actor is not None and self._actor.is_alive:
            self._actor.apply_control(self._control)
        super(KeepVelocity, self).terminate(new_status)


class DriveDistance(AtomicBehavior):

    """
    This class contains an atomic behavior to drive a certain distance.
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


class UseAutoPilot(AtomicBehavior):

    """
    This class contains an atomic behavior to use the auto pilot.

    Note: In parallel to this behavior a termination behavior has to be used
          to terminate this behavior after a certain duration, or after a
          certain distance, etc.
    """

    def __init__(self, actor, name="UseAutoPilot"):
        """
        Setup parameters
        """
        super(UseAutoPilot, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._actor = actor

    def update(self):
        """
        Activate autopilot
        """
        new_status = py_trees.common.Status.RUNNING

        self._actor.set_autopilot(True)

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))
        return new_status

    def terminate(self, new_status):
        """
        Deactivate autopilot
        """
        self._actor.set_autopilot(False)
        super(UseAutoPilot, self).terminate(new_status)


class StopVehicle(AtomicBehavior):

    """
    This class contains an atomic stopping behavior. The controlled traffic
    participant will decelerate with _bake_value_ until reaching a full stop.
    """

    def __init__(self, actor, brake_value, name="Stopping"):
        """
        Setup _actor and maximum braking value
        """
        super(StopVehicle, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._control, self._type = get_actor_control(actor)
        if self._type == 'walker':
            self._control.speed = 0
        self._actor = actor
        self._brake_value = brake_value

    def update(self):
        """
        Set brake to brake_value until reaching full stop
        """
        new_status = py_trees.common.Status.RUNNING

        if self._type == 'vehicle':
            if CarlaDataProvider.get_velocity(self._actor) > EPSILON:
                self._control.brake = self._brake_value
            else:
                new_status = py_trees.common.Status.SUCCESS
                self._control.brake = 0
            self._actor.apply_control(self._control)
        else:
            new_status = py_trees.common.Status.SUCCESS
            self._actor.apply_control(self._control)

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status


class WaitForTrafficLightState(AtomicBehavior):

    """
    This class contains an atomic behavior to wait for a given traffic light
    to have the desired state.
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


class SyncArrival(AtomicBehavior):

    """
    This class contains an atomic behavior to
    set velocity of actor so that it reaches location at the same time as
    actor_reference. The behavior assumes that the two actors are moving
    towards location in a straight line.
    Note: In parallel to this behavior a termination behavior has to be used
          to keep continue synchronization for a certain duration, or for a
          certain distance, etc.
    """

    def __init__(self, actor, actor_reference, target_location, gain=1, name="SyncArrival"):
        """
        actor : actor to be controlled
        actor_ reference : reference actor with which arrival has to be
                             synchronized
        gain : coefficient for actor's throttle and break
               controls
        """
        super(SyncArrival, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._control = carla.VehicleControl()
        self._actor = actor
        self._actor_reference = actor_reference
        self._target_location = target_location
        self._gain = gain

        self._control.steering = 0

    def update(self):
        """
        Dynamic control update for actor velocity
        """
        new_status = py_trees.common.Status.RUNNING

        distance_reference = calculate_distance(CarlaDataProvider.get_location(self._actor_reference),
                                                self._target_location)
        distance = calculate_distance(CarlaDataProvider.get_location(self._actor),
                                      self._target_location)

        velocity_reference = CarlaDataProvider.get_velocity(self._actor_reference)
        time_reference = float('inf')
        if velocity_reference > 0:
            time_reference = distance_reference / velocity_reference

        velocity_current = CarlaDataProvider.get_velocity(self._actor)
        time_current = float('inf')
        if velocity_current > 0:
            time_current = distance / velocity_current

        control_value = (self._gain) * (time_current - time_reference)

        if control_value > 0:
            self._control.throttle = min([control_value, 1])
            self._control.brake = 0
        else:
            self._control.throttle = 0
            self._control.brake = min([abs(control_value), 1])

        self._actor.apply_control(self._control)
        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))
        return new_status

    def terminate(self, new_status):
        """
        On termination of this behavior, the throttle should be set back to 0.,
        to avoid further acceleration.
        """
        self._control.throttle = 0.0
        self._control.brake = 0.0
        self._actor.apply_control(self._control)
        super(SyncArrival, self).terminate(new_status)


class AddNoiseToVehicle(AtomicBehavior):

    """
    This class contains an atomic jitter behavior.
    To add noise to steer as well as throttle of the vehicle.
    """

    def __init__(self, actor, steer_value, throttle_value, name="Jittering"):
        """
        Setup actor , maximum steer value and throttle value
        """
        super(AddNoiseToVehicle, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._control = carla.VehicleControl()
        self._actor = actor
        self._steer_value = steer_value
        self._throttle_value = throttle_value

    def update(self):
        """
        Set steer to steer_value and throttle to throttle_value until reaching full stop
        """
        self._control = self._actor.get_control()
        self._control.steer = self._steer_value
        self._control.throttle = self._throttle_value
        new_status = py_trees.common.Status.SUCCESS

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))
        self._actor.apply_control(self._control)

        return new_status


class ChangeNoiseParameters(AtomicBehavior):

    """
    This class contains an atomic jitter behavior.
    To add noise to steer as well as throttle of the vehicle.
    """

    def __init__(self, new_steer_noise, new_throttle_noise,
                 noise_mean, noise_std, dynamic_mean_for_steer, dynamic_mean_for_throttle, name="ChangeJittering"):
        """
        Setup actor , maximum steer value and throttle value
        """
        super(ChangeNoiseParameters, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._new_steer_noise = new_steer_noise
        self._new_throttle_noise = new_throttle_noise
        self._noise_mean = noise_mean
        self._noise_std = noise_std
        self._dynamic_mean_for_steer = dynamic_mean_for_steer
        self._dynamic_mean_for_throttle = dynamic_mean_for_throttle

        self._noise_to_apply = abs(random.gauss(self._noise_mean, self._noise_std))

    def update(self):
        """
        Change the noise parameters from the structure copy that it receives.
        """

        self._new_steer_noise[0] = min(0, -(self._noise_to_apply - self._dynamic_mean_for_steer))
        self._new_throttle_noise[0] = min(self._noise_to_apply + self._dynamic_mean_for_throttle, 1)

        new_status = py_trees.common.Status.SUCCESS
        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))
        return new_status


class BasicAgentBehavior(AtomicBehavior):

    """
    This class contains an atomic behavior, which uses the
    basic_agent from CARLA to control the actor until
    reaching a target location.
    """

    _acceptable_target_distance = 2

    def __init__(self, actor, target_location, name="BasicAgentBehavior"):
        """
        Setup actor and maximum steer value
        """
        super(BasicAgentBehavior, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._agent = BasicAgent(actor)
        self._agent.set_destination((target_location.x, target_location.y, target_location.z))
        self._control = carla.VehicleControl()
        self._actor = actor
        self._target_location = target_location

    def update(self):
        new_status = py_trees.common.Status.RUNNING

        self._control = self._agent.run_step()

        location = CarlaDataProvider.get_location(self._actor)
        if calculate_distance(location, self._target_location) < self._acceptable_target_distance:
            new_status = py_trees.common.Status.SUCCESS

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))
        self._actor.apply_control(self._control)

        return new_status

    def terminate(self, new_status):
        self._control.throttle = 0.0
        self._control.brake = 0.0
        self._actor.apply_control(self._control)
        super(BasicAgentBehavior, self).terminate(new_status)


class TrafficJamChecker(AtomicBehavior):

    """
    Atomic behavior that performs the followin actions:
       1. Instantiates a set of vehicles managed by a server autopilot
       2. Check for possible traffic jams
       3. Destroy the NPC actors (in autopilot mode) involved in the traffic jam

    This scenario stops when blackboard.get('master_scenario_command') == scenarios_stop_request
    """
    SOFT_NUMBER_BLOCKS = 10  # 10 seconds
    HARD_NUMBER_BLOCKS = 30  # 30 seconds

    MINIMUM_DISTANCE = 5.0  # meters

    def __init__(self, ego_vehicle, debug=False, name="TrafficJamChecker"):
        super(TrafficJamChecker, self).__init__(name)
        self.ego_vehicle = ego_vehicle
        self.debug = debug
        self.blackboard = Blackboard()
        self.world = CarlaDataProvider.get_world()
        self.map = CarlaDataProvider.get_map()
        self.list_intersection_waypoints = []

        # remove initial collisions during setup
        list_actors = list(CarlaActorPool.get_actors())
        for _, actor in list_actors:
            if actor.attributes['role_name'] == 'autopilot':
                if detect_lane_obstacle(actor, margin=0.2):
                    CarlaActorPool.remove_actor_by_id(actor.id)

        # prepare a table to check for stalled vehicles during the execution of the scenario
        self.table_blocked_actors = {}
        current_game_time = GameTime.get_time()
        for actor_id, actor in CarlaActorPool.get_actors():
            if actor.attributes['role_name'] == 'autopilot':
                actor.set_autopilot(True)
                self.table_blocked_actors[actor_id] = {'location': actor.get_location(),
                                                       'time': current_game_time
                                                       }

        self.logger.debug("%s.__init__()" % (self.__class__.__name__))

    def update(self):
        master_scenario_command = self.blackboard.get('master_scenario_command')
        if master_scenario_command and master_scenario_command == 'scenarios_stop_request':
            new_status = py_trees.common.Status.SUCCESS
            return new_status
        else:
            new_status = py_trees.common.Status.RUNNING

        current_game_time = GameTime.get_time()

        list_actors_to_destroy = []
        for actor_id, actor in CarlaActorPool.get_actors():
            if actor.attributes['role_name'] == 'autopilot':
                block_info = self.table_blocked_actors[actor_id]
                current_location = actor.get_location()
                distance = current_location.distance(block_info['location'])

                # if vehicle is moving we reset the current time
                if distance >= self.MINIMUM_DISTANCE:
                    self.table_blocked_actors[actor_id]['location'] = current_location
                    self.table_blocked_actors[actor_id]['time'] = current_game_time

                # if the vehicle is on a trigger box than it should have the time reset
                if actor.is_at_traffic_light():
                    self.table_blocked_actors[actor_id]['location'] = current_location
                    self.table_blocked_actors[actor_id]['time'] = current_game_time
                    if self.debug:
                        self.world.debug.draw_point(current_location,
                                                    size=1.3,
                                                    color=carla.Color(0, 255, 0),
                                                    life_time=5)

                # if vehicle has been static for a long time we get rid of it
                if (current_game_time - self.table_blocked_actors[actor_id]['time']) > self.HARD_NUMBER_BLOCKS:
                    list_actors_to_destroy.append(actor_id)

                    if self.debug:
                        self.world.debug.draw_point(current_location,
                                                    size=1.3,
                                                    color=carla.Color(255, 0, 0),
                                                    life_time=5)

                # if the vehicle has been static for a short period of time...
                elif (current_game_time - self.table_blocked_actors[actor_id]['time']) > self.SOFT_NUMBER_BLOCKS:
                    # check if this vehicle is at an intersection
                    current_waypoint = self.map.get_waypoint(current_location)

                    # is it blocked at an intersection? Then we need to get rid of it!
                    if current_waypoint.is_intersection:
                        if self.debug:
                            self.world.debug.draw_point(current_location,
                                                        size=1.3,
                                                        color=carla.Color(0, 0, 255),
                                                        life_time=5)
                        list_actors_to_destroy.append(actor_id)

        for actor_id in list_actors_to_destroy:
            CarlaActorPool.remove_actor_by_id(actor_id)
            self.table_blocked_actors[actor_id] = None

        return new_status


class Idle(AtomicBehavior):

    """
    This class contains an idle behavior scenario
    """

    def __init__(self, name="Idle"):
        """
        Setup actor
        """
        super(Idle, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))

    def update(self):
        new_status = py_trees.common.Status.RUNNING

        return new_status


class WaypointFollower(AtomicBehavior):

    """
    This is an atomic behavior to follow waypoints indefinitely
    while maintaining a given speed or if given a waypoint plan,
    follows the given plan
    """

    def __init__(self, actor, target_speed, plan=None, blackboard_queue_name=None,
                 avoid_collision=False, name="FollowWaypoints"):
        """
        Set up actor and local planner
        """
        super(WaypointFollower, self).__init__(name)
        self._actor_list = []
        self._actor_list.append(actor)
        self._target_speed = target_speed
        self._local_planner_list = []
        self._plan = plan
        self._blackboard_queue_name = blackboard_queue_name
        if blackboard_queue_name is not None:
            self._queue = Blackboard().get(blackboard_queue_name)
        self._args_lateral_dict = {'K_P': 1.0, 'K_D': 0.01, 'K_I': 0.0, 'dt': 0.05}
        self._avoid_collision = avoid_collision

    def setup(self, timeout=5):
        """
        Delayed one-time initialization
        """
        for actor in self._actor_list:
            self._apply_local_planner(actor)

        return True

    def _apply_local_planner(self, actor):
        local_planner = LocalPlanner(
            actor, opt_dict={
                'target_speed': self._target_speed,
                'lateral_control_dict': self._args_lateral_dict})
        if self._plan is not None:
            local_planner.set_global_plan(self._plan)
        self._local_planner_list.append(local_planner)

    def update(self):
        """
        Run local planner, obtain and apply control to actor
        """

        new_status = py_trees.common.Status.RUNNING

        if self._blackboard_queue_name is not None:
            while not self._queue.empty():
                actor = self._queue.get()
                if actor is not None and actor not in self._actor_list:
                    self._actor_list.append(actor)
                    self._apply_local_planner(actor)

        for actor, local_planner in zip(self._actor_list, self._local_planner_list):
            if actor is not None and actor.is_alive and local_planner is not None:
                control = local_planner.run_step(debug=False)
                if self._avoid_collision and detect_lane_obstacle(actor):
                    control.throttle = 0.0
                    control.brake = 1.0
                actor.apply_control(control)

        return new_status

    def terminate(self, new_status):
        """
        On termination of this behavior,
        the throttle, brake and steer should be set back to 0.
        """
        control = carla.VehicleControl()
        control.throttle = 0.0
        control.brake = 0.0
        control.steer = 0.0
        for actor, local_planner in zip(self._actor_list, self._local_planner_list):
            if actor is not None and actor.is_alive:
                actor.apply_control(control)
            if local_planner is not None:
                local_planner.reset_vehicle()
                local_planner = None
        super(WaypointFollower, self).terminate(new_status)


class HandBrakeVehicle(AtomicBehavior):

    """
    This class contains an atomic hand brake behavior.
    To set the hand brake value of the vehicle.
    """

    def __init__(self, vehicle, hand_brake_value, name="Braking"):
        """
        Setup vehicle control and brake value
        """
        super(HandBrakeVehicle, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._vehicle = vehicle
        self._control, self._type = get_actor_control(vehicle)
        self._hand_brake_value = hand_brake_value

    def update(self):
        """
        Set handbrake
        """
        new_status = py_trees.common.Status.SUCCESS
        if self._type == 'vehicle':
            self._control.hand_brake = self._hand_brake_value
            self._vehicle.apply_control(self._control)
        else:
            self._hand_brake_value = None
            self.logger.debug("%s.update()[%s->%s]" %
                              (self.__class__.__name__, self.status, new_status))
            self._vehicle.apply_control(self._control)

        return new_status


class ActorDestroy(AtomicBehavior):

    """
    This class contains an actor destroy behavior.
    Given a actor this behavior will delete it.
    """

    def __init__(self, actor, name="ActorDestroy"):
        """
        Setup actor
        """
        super(ActorDestroy, self).__init__(name)
        self._actor = actor
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))

    def update(self):
        new_status = py_trees.common.Status.RUNNING
        if self._actor:
            CarlaActorPool.remove_actor_by_id(self._actor.id)
            self._actor = None
            new_status = py_trees.common.Status.SUCCESS

        return new_status


class ActorTransformSetter(AtomicBehavior):

    """
    This class contains an atomic behavior to set the transform
    of an actor.
    """

    def __init__(self, actor, transform, physics=True, name="ActorTransformSetter"):
        """
        Init
        """
        super(ActorTransformSetter, self).__init__(name)
        self._actor = actor
        self._transform = transform
        self._physics = physics
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))

    def update(self):
        """
        Transform actor
        """
        new_status = py_trees.common.Status.RUNNING
        if self._actor.is_alive:
            self._actor.set_velocity(carla.Vector3D(0, 0, 0))
            self._actor.set_angular_velocity(carla.Vector3D(0, 0, 0))
            self._actor.set_transform(self._transform)
            if self._physics:
                self._actor.set_simulate_physics(enabled=True)
            new_status = py_trees.common.Status.SUCCESS
        else:
            # For some reason the actor is gone...
            new_status = py_trees.common.Status.FAILURE
        return new_status


class ActorSource(AtomicBehavior):

    """
    Implementation for a behavior that will indefinitely create actors
    at a given transform if no other actor exists in a given radius
    from the transform.
    """

    def __init__(self, world, actor_type_list, transform, threshold, blackboard_queue_name,
                 actor_limit=7, name="ActorSource"):
        """
        Setup class members
        """
        super(ActorSource, self).__init__(name)
        self._world = world
        self._actor_types = actor_type_list
        self._spawn_point = transform
        self._threshold = threshold
        self._queue = Blackboard().get(blackboard_queue_name)
        self._actor_limit = actor_limit
        self._last_blocking_actor = None

    def update(self):
        new_status = py_trees.common.Status.RUNNING
        if self._actor_limit > 0:
            world_actors = self._world.get_actors()
            spawn_point_blocked = False
            if (self._last_blocking_actor and
                    self._spawn_point.location.distance(self._last_blocking_actor.get_location()) < self._threshold):
                spawn_point_blocked = True

            if not spawn_point_blocked:
                for actor in world_actors:
                    if self._spawn_point.location.distance(actor.get_location()) < self._threshold:
                        spawn_point_blocked = True
                        self._last_blocking_actor = actor
                        break

            if not spawn_point_blocked:
                try:
                    new_actor = CarlaActorPool.request_new_actor(np.random.choice(self._actor_types), self._spawn_point)
                    self._actor_limit -= 1
                    self._queue.put(new_actor)
                except:
                    print("ActorSource unable to spawn actor")
        return new_status


class ActorSink(AtomicBehavior):

    """
    Implementation for a behavior that will indefinitely destroy actors
    that wander near a given location within a specified threshold.
    """

    def __init__(self, world, sink_location, threshold, name="ActorSink"):
        """
        Setup class members
        """
        super(ActorSink, self).__init__(name)
        self._world = world
        self._sink_location = sink_location
        self._threshold = threshold

    def update(self):
        new_status = py_trees.common.Status.RUNNING
        CarlaActorPool.remove_actors_in_surrounding(self._sink_location, self._threshold)
        return new_status


class TrafficLightManipulator(AtomicBehavior):

    """
    Atomic behavior that manipulate traffic lights to simulate TS07, TS08 and TS09
    This scenario stops when blackboard.get('master_scenario_command') == scenarios_stop_request
    """

    MAX_DISTANCE_TRAFFIC_LIGHT = 15
    RANDOM_VALUE_INTERVENTION = 0.4
    RED = carla.TrafficLightState.Red
    GREEN = carla.TrafficLightState.Green

    INT_CONF_OPP = {'ego': GREEN, 'ref': GREEN, 'left': RED, 'right': RED, 'opposite': GREEN}
    INT_CONF_LFT = {'ego': GREEN, 'ref': GREEN, 'left': GREEN, 'right': RED, 'opposite': RED}
    INT_CONF_RGT = {'ego': GREEN, 'ref': GREEN, 'left': RED, 'right': GREEN, 'opposite': RED}
    INTERSECTION_CONFIGURATIONS = [INT_CONF_OPP, INT_CONF_LFT, INT_CONF_RGT]

    def __init__(self, ego_vehicle, debug=False, name="TrafficLightManipulator"):
        super(TrafficLightManipulator, self).__init__(name)
        self.ego_vehicle = ego_vehicle
        self.debug = debug
        self.blackboard = Blackboard()
        self.target_traffic_light = None
        self.annotations = None
        self.reset_annotations = None
        self.intervention = False
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))

    def update(self):
        master_scenario_command = self.blackboard.get('master_scenario_command')
        if master_scenario_command and master_scenario_command == 'scenarios_stop_request':
            new_status = py_trees.common.Status.SUCCESS
            return new_status
        else:
            new_status = py_trees.common.Status.RUNNING

        # find a suitable target
        if not self.target_traffic_light:
            traffic_light = CarlaDataProvider.get_next_traffic_light(self.ego_vehicle, use_cached_location=False)
            if not traffic_light:
                # nothing else to do in this iteration...
                return new_status

            base_transform = traffic_light.get_transform()
            area_loc = carla.Location(base_transform.transform(traffic_light.trigger_volume.location))
            distance_to_traffic_light = area_loc.distance(self.ego_vehicle.get_location())

            if self.debug:
                print("[{}] distance={}".format(traffic_light.id, distance_to_traffic_light))

            if distance_to_traffic_light < self.MAX_DISTANCE_TRAFFIC_LIGHT:
                self.target_traffic_light = traffic_light
                self.intervention = random.random() > self.RANDOM_VALUE_INTERVENTION

                if self.intervention:
                    if self.debug:
                        print("--- We are going to affect the following intersection")
                        loc = self.target_traffic_light.get_location()
                        CarlaDataProvider.get_world().debug.draw_point(loc + carla.Location(z=1.0),
                                                                       size=0.5, color=carla.Color(255, 255, 0),
                                                                       life_time=50000)
                    self.annotations = CarlaDataProvider.annotate_trafficlight_in_group(self.target_traffic_light)
        else:
            if not self.reset_annotations:
                if self.intervention:
                    # the light has not been manipulated yet
                    choice = random.choice(self.INTERSECTION_CONFIGURATIONS)
                    self.reset_annotations = CarlaDataProvider.update_light_states(
                        self.target_traffic_light,
                        self.annotations,
                        choice,
                        freeze=True)

            else:
                # the traffic light has been manipulated...
                base_transform =  self.target_traffic_light.get_transform()
                area_loc = carla.Location(base_transform.transform( self.target_traffic_light.trigger_volume.location))
                distance_to_traffic_light = area_loc.distance(self.ego_vehicle.get_location())

                if self.debug:
                    print("++ distance={}".format(distance_to_traffic_light))

                if distance_to_traffic_light > self.MAX_DISTANCE_TRAFFIC_LIGHT:
                    if self.reset_annotations:
                        CarlaDataProvider.reset_lights(self.reset_annotations)
                        self.target_traffic_light = None
                        self.reset_annotations = None
                        self.annotations = None
                        self.intervention = False

        return new_status
