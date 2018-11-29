#!/usr/bin/env python

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

    def __init__(self, vehicle, min_x, max_x, min_y, max_y, name="TriggerRegion"):
        """
        Setup trigger region (rectangle provided by
        [min_x,min_y] and [max_x,max_y]
        """
        super(InTriggerRegion, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self.vehicle = vehicle
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y

    def update(self):
        """
        Check if the vehicle location is within trigger region
        """
        new_status = py_trees.common.Status.RUNNING

        location = CarlaDataProvider.get_location(self.vehicle)

        if location is None:
            return new_status

        not_in_region = (location.x < self.min_x or location.x > self.max_x) or (
            location.y < self.min_y or location.y > self.max_y)
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

    def __init__(self, other_vehicle, ego_vehicle, distance, name="TriggerDistanceToVehicle"):
        """
        Setup trigger distance
        """
        super(InTriggerDistanceToVehicle, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self.other_vehicle = other_vehicle
        self.ego_vehicle = ego_vehicle
        self.distance = distance

    def update(self):
        """
        Check if the ego vehicle is within trigger distance to other vehicle
        """
        new_status = py_trees.common.Status.RUNNING

        ego_location = CarlaDataProvider.get_location(self.ego_vehicle)
        other_location = CarlaDataProvider.get_location(self.other_vehicle)

        if ego_location is None or other_location is None:
            return new_status

        if calculate_distance(ego_location, other_location) < self.distance:
            new_status = py_trees.common.Status.SUCCESS

        self.logger.debug("%s.update()[%s->%s]" %
                          (self.__class__.__name__, self.status, new_status))

        return new_status


class InTriggerDistanceToLocation(AtomicBehavior):

    """
    This class contains the trigger (condition) for a distance to a fixed
    location of a scenario
    """

    def __init__(self, vehicle, target_location, distance, name="InTriggerDistanceToLocation"):
        """
        Setup trigger distance
        """
        super(InTriggerDistanceToLocation, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self.target_location = target_location
        self.vehicle = vehicle
        self.distance = distance

    def update(self):
        """
        Check if the vehicle is within trigger distance to the target location
        """
        new_status = py_trees.common.Status.RUNNING

        location = CarlaDataProvider.get_location(self.vehicle)

        if location is None:
            return new_status

        if calculate_distance(location, self.target_location) < self.distance:
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
        self.vehicle = vehicle
        self.target_velocity = target_velocity

    def update(self):
        """
        Check if the vehicle has the trigger velocity
        """
        new_status = py_trees.common.Status.RUNNING

        if (CarlaDataProvider.get_velocity(self.vehicle) - self.target_velocity) < EPSILON:
            new_status = py_trees.common.Status.SUCCESS

        self.logger.debug("%s.update()[%s->%s]" %
                          (self.__class__.__name__, self.status, new_status))

        return new_status


class InTimeToArrivalToLocation(AtomicBehavior):

    """
    This class contains a check if a vehicle arrives within a given time
    at a given location.
    """

    max_time_to_arrival = float('inf')  # time to arrival in seconds

    def __init__(self, vehicle, time, location, name="TimeToArrival"):
        """
        Setup parameters
        """
        super(InTimeToArrivalToLocation, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self.vehicle = vehicle
        self.time = time
        self.target_location = location

    def update(self):
        """
        Check if the vehicle can arrive at target_location within time
        """
        new_status = py_trees.common.Status.RUNNING

        current_location = CarlaDataProvider.get_location(self.vehicle)

        if current_location is None:
            return new_status

        distance = calculate_distance(current_location, self.target_location)
        velocity = CarlaDataProvider.get_velocity(self.vehicle)

        # if velocity is too small, simply use a large time to arrival
        time_to_arrival = self.max_time_to_arrival
        if velocity > EPSILON:
            time_to_arrival = distance / velocity

        if time_to_arrival < self.time:
            new_status = py_trees.common.Status.SUCCESS

        self.logger.debug("%s.update()[%s->%s]" %
                          (self.__class__.__name__, self.status, new_status))

        return new_status


class InTimeToArrivalToVehicle(AtomicBehavior):

    """
    This class contains a check if a vehicle arrives within a given time
    at another vehicle.
    """

    max_time_to_arrival = float('inf')  # time to arrival in seconds

    def __init__(self, other_vehicle, ego_vehicle, time, name="TimeToArrival"):
        """
        Setup parameters
        """
        super(InTimeToArrivalToVehicle, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self.other_vehicle = other_vehicle
        self.ego_vehicle = ego_vehicle
        self.time = time

    def update(self):
        """
        Check if the ego vehicle can arrive at other vehicle within time
        """
        new_status = py_trees.common.Status.RUNNING

        current_location = CarlaDataProvider.get_location(self.ego_vehicle)
        target_location = CarlaDataProvider.get_location(self.other_vehicle)

        if current_location is None or target_location is None:
            return new_status

        distance = calculate_distance(current_location, target_location)
        current_velocity = CarlaDataProvider.get_velocity(self.ego_vehicle)
        other_velocity = CarlaDataProvider.get_velocity(self.other_vehicle)

        # if velocity is too small, simply use a large time to arrival
        time_to_arrival = self.max_time_to_arrival
        if current_velocity > other_velocity:
            time_to_arrival = 2 * distance / (current_velocity - other_velocity)

        if time_to_arrival < self.time:
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

    def __init__(self, vehicle, throttle_value, target_velocity, name="Acceleration"):
        """
        Setup parameters including acceleration value (via throttle_value)
        and target velocity
        """
        super(AccelerateToVelocity, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self.control = carla.VehicleControl()
        self.vehicle = vehicle
        self.throttle_value = throttle_value
        self.target_velocity = target_velocity

    def update(self):
        """
        Set throttle to throttle_value, as long as velocity is < target_velocity
        """
        new_status = py_trees.common.Status.RUNNING

        if CarlaDataProvider.get_velocity(self.vehicle) < self.target_velocity:
            self.control.throttle = self.throttle_value
        else:
            new_status = py_trees.common.Status.SUCCESS
            self.control.throttle = 0

        self.logger.debug("%s.update()[%s->%s]" %
                          (self.__class__.__name__, self.status, new_status))
        self.vehicle.apply_control(self.control)

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
        self.control = carla.VehicleControl()
        self.vehicle = vehicle
        self.target_velocity = target_velocity

    def update(self):
        """
        Set throttle to throttle_value, as long as velocity is < target_velocity
        """
        new_status = py_trees.common.Status.RUNNING

        if CarlaDataProvider.get_velocity(self.vehicle) < self.target_velocity:
            self.control.throttle = 1.0
        else:
            self.control.throttle = 0.0

        self.vehicle.apply_control(self.control)
        self.logger.debug("%s.update()[%s->%s]" %
                          (self.__class__.__name__, self.status, new_status))
        return new_status

    def terminate(self, new_status):
        """
        On termination of this behavior, the throttle should be set back to 0.,
        to avoid further acceleration.
        """
        self.control.throttle = 0.0
        self.vehicle.apply_control(self.control)
        super(KeepVelocity, self).terminate(new_status)


class StopVehicle(AtomicBehavior):

    """
    This class contains an atomic stopping behavior. The controlled traffic
    participant will decelerate with _bake_value_ until reaching a full stop.
    """

    def __init__(self, vehicle, brake_value, name="Stopping"):
        """
        Setup vehicle and maximum braking value
        """
        super(StopVehicle, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self.control = carla.VehicleControl()
        self.vehicle = vehicle
        self.brake_value = brake_value

    def update(self):
        """
        Set brake to brake_value until reaching full stop
        """
        new_status = py_trees.common.Status.RUNNING

        if CarlaDataProvider.get_velocity(self.vehicle) > EPSILON:
            self.control.brake = self.brake_value
        else:
            new_status = py_trees.common.Status.SUCCESS
            self.control.brake = 0

        self.logger.debug("%s.update()[%s->%s]" %
                          (self.__class__.__name__, self.status, new_status))
        self.vehicle.apply_control(self.control)

        return new_status
