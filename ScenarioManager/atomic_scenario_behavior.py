#!/usr/bin/env python

# Copyright (c) 2018-2019 Intel Labs.
# authors: Fabian Oboril (fabian.oboril@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides all atomic scenario behaviors required to realize
complex, realistic scenarios such as "follow a leading vehicle", "lane change",
etc.
The atomic behaviors are implemented with py_trees.
"""

import math

import numpy as np
import py_trees
import carla
from agents.navigation.roaming_agent import *
from agents.navigation.basic_agent import *
from agents.navigation.local_planner import LocalPlanner, RoadOption

from ScenarioManager.carla_data_provider import CarlaDataProvider

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
        self._map = self._actor.get_world().get_map()

        waypoint = self._map.get_waypoint(self._actor.get_location())
        while not waypoint.is_intersection:
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


class AccelerateToVelocity(AtomicBehavior):

    """
    This class contains an atomic acceleration behavior. The controlled
    traffic participant will accelerate with _throttle_value_ until reaching
    a given _target_velocity_
    """

    def __init__(self, actor, throttle_value, target_velocity, name="Acceleration"):
        """
        Setup parameters including acceleration value (via throttle_value)
        and target velocity
        """
        super(AccelerateToVelocity, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._control = carla.VehicleControl()
        self._actor = actor
        self._throttle_value = throttle_value
        self._target_velocity = target_velocity

        self._control.steering = 0

    def update(self):
        """
        Set throttle to throttle_value, as long as velocity is < target_velocity
        """
        new_status = py_trees.common.Status.RUNNING

        if CarlaDataProvider.get_velocity(self._actor) < self._target_velocity:
            self._control.throttle = self._throttle_value
        else:
            new_status = py_trees.common.Status.SUCCESS
            self._control.throttle = 0

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))
        self._actor.apply_control(self._control)

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

    def __init__(self, actor, target_velocity, name="KeepVelocity"):
        """
        Setup parameters including acceleration value (via throttle_value)
        and target velocity
        """
        super(KeepVelocity, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._control = carla.VehicleControl()
        self._actor = actor
        self._target_velocity = target_velocity

        self._control.steering = 0

    def update(self):
        """
        Set throttle to throttle_value, as long as velocity is < target_velocity
        """
        new_status = py_trees.common.Status.RUNNING

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
        self._control = carla.VehicleControl()
        self._actor = actor
        self._brake_value = brake_value

        self._control.steering = 0

    def update(self):
        """
        Set brake to brake_value until reaching full stop
        """
        new_status = py_trees.common.Status.RUNNING

        if CarlaDataProvider.get_velocity(self._actor) > EPSILON:
            self._control.brake = self._brake_value
        else:
            new_status = py_trees.common.Status.SUCCESS
            self._control.brake = 0

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))
        self._actor.apply_control(self._control)

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
    actor_reference. The behaviour assumes that the two actors are moving
    towards location in a straight line.
    Note: In parallel to this behavior a termination behavior has to be used
          to keep continue scynhronisation for a certain duration, or for a
          certain distance, etc.
    """

    def __init__(self, actor, actor_reference, target_location, gain=1, name="SyncArrival"):
        """
        actor : actor to be controlled
        actor_ reference : reference actor with which arrival has to be
                             synchronised
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


class SteerVehicle(AtomicBehavior):

    """
    This class contains an atomic steer behavior.
    To set the steer value of the actor.
    """

    def __init__(self, actor, steer_value, name="Steering"):
        """
        Setup actor and maximum steer value
        """
        super(SteerVehicle, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._control = carla.VehicleControl()
        self._actor = actor
        self._steer_value = steer_value

    def update(self):
        """
        Set steer to steer_value until reaching full stop
        """
        self._control.steer = self._steer_value
        new_status = py_trees.common.Status.SUCCESS

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))
        self._actor.apply_control(self._control)

        return new_status


class BasicAgentBehavior(AtomicBehavior):

    """
    This class contains an atomic behavior, which uses the
    basic_agent from CARLA to control the vehicle until
    reaching a target location.
    """

    _acceptable_target_distance = 2

    def __init__(self, vehicle, target_location, name="BasicAgentBehavior"):
        """
        Setup vehicle and maximum steer value
        """
        super(BasicAgentBehavior, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._agent = BasicAgent(vehicle)
        self._agent.set_destination(
            (target_location.x, target_location.y, target_location.z))
        self._control = carla.VehicleControl()
        self._vehicle = vehicle
        self._target_location = target_location

    def update(self):
        new_status = py_trees.common.Status.RUNNING

        self._control = self._agent.run_step()

        location = CarlaDataProvider.get_location(self._vehicle)
        if calculate_distance(location, self._target_location) < self._acceptable_target_distance:
            new_status = py_trees.common.Status.SUCCESS

        self.logger.debug("%s.update()[%s->%s]" %
                          (self.__class__.__name__, self.status, new_status))
        self._vehicle.apply_control(self._control)

        return new_status

    def terminate(self, new_status):
        self._control.throttle = 0.0
        self._control.brake = 0.0
        self._vehicle.apply_control(self._control)
        super(BasicAgentBehavior, self).terminate(new_status)


class HandBrakeVehicle(AtomicBehavior):

    """
    This class contains an atomic brake behavior.
    To set the brake value of the vehicle.
    """

    def __init__(self, vehicle, hand_brake_value, name="Braking"):
        """
        Setup vehicle control and brake value
        """
        super(HandBrakeVehicle, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._control = carla.VehicleControl()
        self._vehicle = vehicle
        self._hand_brake_value = hand_brake_value

    def update(self):
        """
        Set handbrake
        """
        self._control.hand_brake = self._hand_brake_value
        new_status = py_trees.common.Status.SUCCESS

        self.logger.debug("%s.update()[%s->%s]" %
                          (self.__class__.__name__, self.status, new_status))
        self._vehicle.apply_control(self._control)

        return new_status

class TurnVehicle(AtomicBehavior):
    """
    This atomic behaviour performs either a right or left turn
    at the next junction
    """

    def __init__(self, vehicle, speed, turn, name="TurnVehicle"):
        """
        Setting initialization parameters.
        vehicle     :   Vehicle to be turned
        speed       :   Speed at which to execute the turn in kmph
        turn        :   Turn index. LEFT -> 1, RIGHT -> -1
        """
        super(TurnVehicle, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._vehicle = vehicle
        self._map = self._vehicle.get_world().get_map()
        self._turn = turn
        self._next_wp = None
        self._previous_wp = None
        self._wp_list = []
        self._threshold = 2 #   degree
        self._target_speed = speed
        self._sampling_radius = 0.5 * self._target_speed / 3.6
        self._minimum_distance = 0.9 * self._sampling_radius
        self._road_option = RoadOption.LEFT if turn > 0 else RoadOption.RIGHT
        # Generate tragectoy
        self._get_tragectory()
        self._local_planner = LocalPlanner(
            self._vehicle, opt_dict={'target_speed' : self._target_speed})
        self._local_planner.set_global_plan(self._wp_list)

    def update(self):
        """
        Follow waypoints to a junction and choose path based on turn input,
        then execute the turn
        """

        new_status = py_trees.common.Status.RUNNING

        #   Generate controls and apply
        target_location = self._wp_list[-1][0].transform.location
        if self._vehicle.get_location().distance(
                target_location) > self._minimum_distance:
            control = self._local_planner.run_step(debug=False)
            self._vehicle.apply_control(control)
        else:
            new_status = py_trees.common.Status.SUCCESS

        return new_status

    def _get_tragectory(self):
        """
        Generates waypoint list to execute the turn
        """

        reached_junction = False
        angle_wp = float('inf')
        while angle_wp > np.radians(self._threshold):
            #   Retrieve vehicle state
            current_transform = self._vehicle.get_transform()
            current_location = current_transform.location
            projected_location = current_location + \
                carla.Location(
                    x=math.cos(math.radians(current_transform.rotation.yaw)),
                    y=math.sin(math.radians(current_transform.rotation.yaw)))
            v_actual = self._vector(current_location, projected_location)
            if not self._next_wp:
                self._next_wp = self._map.get_waypoint(current_location)
            wp_choice = self._next_wp.next(self._sampling_radius)
            #   Choose path at intersection
            if not reached_junction and len(wp_choice) > 1:
                reached_junction = True
                select_criteria = float('inf')
                for wp_select in wp_choice:
                    v_select = self._vector(
                        current_location, wp_select.transform.location)
                    cross = self._turn*np.cross(v_actual, v_select)[-1]
                    if cross < select_criteria:
                        select_criteria = cross
                        self._next_wp = wp_select
            else:
                self._next_wp = wp_choice[0]
            self._wp_list.append((self._next_wp, self._road_option))

            #   Update loop condition
            if len(self._wp_list) >= 3:
                v_1 = self._vector(
                    self._wp_list[-2][0].transform.location,
                    self._wp_list[-1][0].transform.location)
                v_2 = self._vector(
                    self._wp_list[-3][0].transform.location,
                    self._wp_list[-2][0].transform.location)
                angle_wp = math.acos(
                    np.dot(v_1, v_2)/\
                        (np.linalg.norm(v_1)*np.linalg.norm(v_2)))

    def _vector(self, l_1, l_2):
        """
        Returns the unit vector from l_1 to l_2
        l_1, l_2    :   carla.Location objects
        """
        x = l_2.x-l_1.x
        y = l_2.y-l_1.y
        norm = np.linalg.norm([x, y])
        return [x/norm, y/norm, 0]

    def _draw(self, begin, end):
        """
        Function to draw arrows in the simulator for debugging
        """
        begin = begin.transform.location
        end = end.transform.location
        world = self._vehicle.get_world()
        world.debug.draw_arrow(begin, end, arrow_size=0.3, life_time=1.0)

    def terminate(self, new_status):
        """
        On termination of this behavior, the throttle should be set back to 0.,
        to avoid further acceleration.
        """
        control = carla.VehicleControl()
        control.throttle = 0.0
        control.brake = 0.0
        control.steer = 0.0
        self._vehicle.apply_control(control)
        super(TurnVehicle, self).terminate(new_status)
