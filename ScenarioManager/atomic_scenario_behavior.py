#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
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

import py_trees
import carla

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
        self.logger.debug("%s.terminate()[%s->%s]" % (
            self.__class__.__name__, self.status, new_status))


class InTriggerRegion(AtomicBehavior):

    """
    This class contains the trigger region (condition) of a scenario
    """

    def __init__(self, vehicle, min_x, max_x, min_y,
                 max_y, name="TriggerRegion"):
        """
        Setup trigger region (rectangle provided by
        [min_x,min_y] and [max_x,max_y]
        """
        super(InTriggerRegion, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._vehicle = vehicle
        self._min_x = min_x
        self._max_x = max_x
        self._min_y = min_y
        self._max_y = max_y

    def update(self):
        """
        Check if the _vehicle location is within trigger region
        """
        new_status = py_trees.common.Status.RUNNING

        location = CarlaDataProvider.get_location(self._vehicle)

        if location is None:
            return new_status

        not_in_region = (location.x < self._min_x or location.x > self._max_x) or (
            location.y < self._min_y or location.y > self._max_y)
        if not not_in_region:
            new_status = py_trees.common.Status.SUCCESS

        self.logger.debug("%s.update()[%s->%s]" %
                          (self.__class__.__name__, self.status, new_status))

        return new_status


class InTriggerDistanceToVehicle(AtomicBehavior):

    """
    This class contains the trigger distance (condition) between to vehicles
    of a scenario
    """

    def __init__(self, other_vehicle, ego_vehicle, distance,
                 name="TriggerDistanceToVehicle"):
        """
        Setup trigger distance
        """
        super(InTriggerDistanceToVehicle, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._other_vehicle = other_vehicle
        self._ego_vehicle = ego_vehicle
        self._distance = distance

    def update(self):
        """
        Check if the ego vehicle is within trigger distance to other vehicle
        """
        new_status = py_trees.common.Status.RUNNING

        ego_location = CarlaDataProvider.get_location(self._ego_vehicle)
        other_location = CarlaDataProvider.get_location(self._other_vehicle)

        if ego_location is None or other_location is None:
            return new_status

        if calculate_distance(ego_location, other_location) < self._distance:
            new_status = py_trees.common.Status.SUCCESS

        self.logger.debug("%s.update()[%s->%s]" %
                          (self.__class__.__name__, self.status, new_status))

        return new_status


class InTriggerDistanceToLocation(AtomicBehavior):

    """
    This class contains the trigger (condition) for a distance to a fixed
    location of a scenario
    """

    def __init__(self, vehicle, target_location, distance,
                 name="InTriggerDistanceToLocation"):
        """
        Setup trigger distance
        """
        super(InTriggerDistanceToLocation, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._target_location = target_location
        self._vehicle = vehicle
        self._distance = distance

    def update(self):
        """
        Check if the vehicle is within trigger distance to the target location
        """
        new_status = py_trees.common.Status.RUNNING

        location = CarlaDataProvider.get_location(self._vehicle)

        if location is None:
            return new_status

        if calculate_distance(
                location, self._target_location) < self._distance:
            new_status = py_trees.common.Status.SUCCESS

        self.logger.debug("%s.update()[%s->%s]" %
                          (self.__class__.__name__, self.status, new_status))

        return new_status


class TriggerVelocity(AtomicBehavior):

    """
    This class contains the trigger velocity (condition) of a scenario
    """

    def __init__(self, vehicle, target_velocity, name="TriggerVelocity"):
        """
        Setup trigger velocity
        """
        super(TriggerVelocity, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._vehicle = vehicle
        self._target_velocity = target_velocity

    def update(self):
        """
        Check if the vehicle has the trigger velocity
        """
        new_status = py_trees.common.Status.RUNNING

        delta_velocity = CarlaDataProvider.get_velocity(
            self._vehicle) - self._target_velocity
        if delta_velocity < EPSILON:
            new_status = py_trees.common.Status.SUCCESS

        self.logger.debug("%s.update()[%s->%s]" %
                          (self.__class__.__name__, self.status, new_status))

        return new_status


class InTimeToArrivalToLocation(AtomicBehavior):

    """
    This class contains a check if a vehicle arrives within a given time
    at a given location.
    """

    _max_time_to_arrival = float('inf')  # time to arrival in seconds

    def __init__(self, vehicle, time, location, name="TimeToArrival"):
        """
        Setup parameters
        """
        super(InTimeToArrivalToLocation, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._vehicle = vehicle
        self._time = time
        self._target_location = location

    def update(self):
        """
        Check if the vehicle can arrive at target_location within time
        """
        new_status = py_trees.common.Status.RUNNING

        current_location = CarlaDataProvider.get_location(self._vehicle)

        if current_location is None:
            return new_status

        distance = calculate_distance(current_location, self._target_location)
        velocity = CarlaDataProvider.get_velocity(self._vehicle)

        # if velocity is too small, simply use a large time to arrival
        time_to_arrival = self._max_time_to_arrival
        if velocity > EPSILON:
            time_to_arrival = distance / velocity

        if time_to_arrival < self._time:
            new_status = py_trees.common.Status.SUCCESS

        self.logger.debug("%s.update()[%s->%s]" %
                          (self.__class__.__name__, self.status, new_status))

        return new_status


class InTimeToArrivalToVehicle(AtomicBehavior):

    """
    This class contains a check if a vehicle arrives within a given time
    at another vehicle.
    """

    _max_time_to_arrival = float('inf')  # time to arrival in seconds

    def __init__(self, other_vehicle, ego_vehicle, time, name="TimeToArrival"):
        """
        Setup parameters
        """
        super(InTimeToArrivalToVehicle, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._other_vehicle = other_vehicle
        self._ego_vehicle = ego_vehicle
        self._time = time

    def update(self):
        """
        Check if the ego vehicle can arrive at other vehicle within time
        """
        new_status = py_trees.common.Status.RUNNING

        current_location = CarlaDataProvider.get_location(self._ego_vehicle)
        target_location = CarlaDataProvider.get_location(self._other_vehicle)

        if current_location is None or target_location is None:
            return new_status

        distance = calculate_distance(current_location, target_location)
        current_velocity = CarlaDataProvider.get_velocity(self._ego_vehicle)
        other_velocity = CarlaDataProvider.get_velocity(self._other_vehicle)

        # if velocity is too small, simply use a large time to arrival
        time_to_arrival = self._max_time_to_arrival

        if current_velocity > other_velocity:
            time_to_arrival = 2 * distance / \
                (current_velocity - other_velocity)

        if time_to_arrival < self._time:
            new_status = py_trees.common.Status.SUCCESS

        self.logger.debug("%s.update()[%s->%s]" %
                          (self.__class__.__name__, self.status, new_status))

        return new_status


class AccelerateToVelocity(AtomicBehavior):

    """
    This class contains an atomic acceleration behavior. The controlled
    traffic participant will accelerate with _throttle_value_ until reaching
    a given _target_velocity_
    """

    def __init__(self, vehicle, throttle_value,
                 target_velocity, name="Acceleration"):
        """
        Setup parameters including acceleration value (via throttle_value)
        and target velocity
        """
        super(AccelerateToVelocity, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._control = carla.VehicleControl()
        self._vehicle = vehicle
        self._throttle_value = throttle_value
        self._target_velocity = target_velocity

        self._control.steering = 0

    def update(self):
        """
        Set throttle to throttle_value, as long as velocity is < target_velocity
        """
        new_status = py_trees.common.Status.RUNNING

        if CarlaDataProvider.get_velocity(
                self._vehicle) < self._target_velocity:
            self._control.throttle = self._throttle_value
        else:
            new_status = py_trees.common.Status.SUCCESS
            self._control.throttle = 0

        self.logger.debug("%s.update()[%s->%s]" %
                          (self.__class__.__name__, self.status, new_status))
        self._vehicle.apply_control(self._control)

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

    def __init__(self, vehicle, target_velocity, name="KeepVelocity"):
        """
        Setup parameters including acceleration value (via throttle_value)
        and target velocity
        """
        super(KeepVelocity, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._control = carla.VehicleControl()
        self._vehicle = vehicle
        self._target_velocity = target_velocity

        self._control.steering = 0

    def update(self):
        """
        Set throttle to throttle_value, as long as velocity is < target_velocity
        """
        new_status = py_trees.common.Status.RUNNING

        if CarlaDataProvider.get_velocity(
                self._vehicle) < self._target_velocity:
            self._control.throttle = 1.0
        else:
            self._control.throttle = 0.0

        self._vehicle.apply_control(self._control)
        self.logger.debug("%s.update()[%s->%s]" %
                          (self.__class__.__name__, self.status, new_status))
        return new_status

    def terminate(self, new_status):
        """
        On termination of this behavior, the throttle should be set back to 0.,
        to avoid further acceleration.
        """
        self._control.throttle = 0.0
        self._vehicle.apply_control(self._control)
        super(KeepVelocity, self).terminate(new_status)


class DriveDistance(AtomicBehavior):

    """
    This class contains an atomic behavior to drive a certain distance.
    """

    def __init__(self, vehicle, distance, name="DriveDistance"):
        """
        Setup parameters
        """
        super(DriveDistance, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._target_distance = distance
        self._distance = 0
        self._location = None
        self._vehicle = vehicle

    def initialise(self):
        self._location = CarlaDataProvider.get_location(self._vehicle)
        super(DriveDistance, self).initialise()

    def update(self):
        """
        Check driven distance
        """
        new_status = py_trees.common.Status.RUNNING

        new_location = CarlaDataProvider.get_location(self._vehicle)
        self._distance += calculate_distance(self._location, new_location)
        self._location = new_location

        if self._distance > self._target_distance:
            new_status = py_trees.common.Status.SUCCESS

        self.logger.debug("%s.update()[%s->%s]" %
                          (self.__class__.__name__, self.status, new_status))
        return new_status


class UseAutoPilot(AtomicBehavior):

    """
    This class contains an atomic behavior to use the auto pilot.

    Note: In parallel to this behavior a termination behavior has to be used
          to terminate this behavior after a certain duration, or after a
          certain distance, etc.
    """

    def __init__(self, vehicle, name="UseAutoPilot"):
        """
        Setup parameters
        """
        super(UseAutoPilot, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._vehicle = vehicle

    def update(self):
        """
        Activate autopilot
        """
        new_status = py_trees.common.Status.RUNNING

        self._vehicle.set_autopilot(True)

        self.logger.debug("%s.update()[%s->%s]" %
                          (self.__class__.__name__, self.status, new_status))
        return new_status

    def terminate(self, new_status):
        """
        Deactivate autopilot
        """
        self._vehicle.set_autopilot(False)
        super(UseAutoPilot, self).terminate(new_status)


class StopVehicle(AtomicBehavior):

    """
    This class contains an atomic stopping behavior. The controlled traffic
    participant will decelerate with _bake_value_ until reaching a full stop.
    """

    def __init__(self, vehicle, brake_value, name="Stopping"):
        """
        Setup _vehicle and maximum braking value
        """
        super(StopVehicle, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._control = carla.VehicleControl()
        self._vehicle = vehicle
        self._brake_value = brake_value

        self._control.steering = 0

    def update(self):
        """
        Set brake to brake_value until reaching full stop
        """
        new_status = py_trees.common.Status.RUNNING

        if CarlaDataProvider.get_velocity(self._vehicle) > EPSILON:
            self._control.brake = self._brake_value
        else:
            new_status = py_trees.common.Status.SUCCESS
            self._control.brake = 0

        self.logger.debug("%s.update()[%s->%s]" %
                          (self.__class__.__name__, self.status, new_status))
        self._vehicle.apply_control(self._control)

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

        self.logger.debug("%s.update()[%s->%s]" %
                          (self.__class__.__name__, self.status, new_status))

        return new_status

    def terminate(self, new_status):
        self._traffic_light = None
        super(WaitForTrafficLightState, self).terminate(new_status)


class SyncArrival(AtomicBehavior):

    """
    This class contains an atomic behavior to
    set velocity of vehicle so that it reaches location at the same time as
    vehicle_reference. The behaviour assumes that the two vehicles are moving
    towards location in a straight line.
    Note: In parallel to this behavior a termination behavior has to be used
          to keep continue scynhronisation for a certain duration, or for a
          certain distance, etc.
    """

    def __init__(self, vehicle, vehicle_reference, target_location, gain=1, name="SyncArrival"):
        """
        vehicle : vehicle to be controlled
        vehicle_ reference : reference vehicle with which arrival has to be
                             synchronised
        gain : coefficient for vehicle's throttle and break
               controls
        """
        super(SyncArrival, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._control = carla.VehicleControl()
        self._vehicle = vehicle
        self._vehicle_reference = vehicle_reference
        self._target_location = target_location
        self._gain = gain

        self._control.steering = 0

    def update(self):
        """
        Dynamic control update for vehicle velocity
        """
        new_status = py_trees.common.Status.RUNNING

        distance_reference = calculate_distance(
            CarlaDataProvider.get_location(self._vehicle_reference),
            self._target_location)
        distance = calculate_distance(
            CarlaDataProvider.get_location(self._vehicle),
            self._target_location)

        velocity_reference = CarlaDataProvider.get_velocity(
            self._vehicle_reference)
        time_reference = float('inf')
        if velocity_reference > 0:
            time_reference = distance_reference / velocity_reference

        velocity_current = CarlaDataProvider.get_velocity(self._vehicle)
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

        self._vehicle.apply_control(self._control)
        self.logger.debug("%s.update()[%s->%s]" %
                          (self.__class__.__name__, self.status, new_status))
        return new_status

    def terminate(self, new_status):
        """
        On termination of this behavior, the throttle should be set back to 0.,
        to avoid further acceleration.
        """
        self._control.throttle = 0.0
        self._control.brake = 0.0
        self._vehicle.apply_control(self._control)
        super(SyncArrival, self).terminate(new_status)

class SteerVehicle(AtomicBehavior):

    """
    This class contains an atomic steer behavior.
    To set the steer value of the vehicle.
    """

    def __init__(self, vehicle, steer_value, name="Steering"):
        """
        Setup vehicle and maximum steer value
        """
        super(SteerVehicle, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._control = carla.VehicleControl()
        self._vehicle = vehicle
        self._steer_value = steer_value

    def update(self):
        """
        Set steer to steer_value until reaching full stop
        """
        self._control.steer = self._steer_value
        new_status = py_trees.common.Status.SUCCESS

        self.logger.debug("%s.update()[%s->%s]" %
                          (self.__class__.__name__, self.status, new_status))
        self._vehicle.apply_control(self._control)

        return new_status
