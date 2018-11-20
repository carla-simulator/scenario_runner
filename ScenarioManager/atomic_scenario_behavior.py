#!/usr/bin/env python

#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

#
# This file contains all atomic behaviors required to perform a scenario,
# e.g. accelerate, stop, keep velocity, etc.
#

"""
This module provides all atomic scenario behaviors required to realize
complex, realistic scenarios such as "follow a leading vehicle", "lane change",
etc.

The atomic behaviors are implemented with py_trees.
"""

import math
import time

import py_trees
import carla


class InTriggerRegion(py_trees.behaviour.Behaviour):

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

    def setup(self, unused_timeout=15):
        self.logger.debug("%s.setup()" % (self.__class__.__name__))
        return True

    def initialise(self):
        self.logger.debug("%s.initialise()" % (self.__class__.__name__))

    def update(self):
        """
        Check if the vehicle location is within trigger region
        """
        location = self.vehicle.get_location()
        not_in_region = (location.x < self.min_x or location.x > self.max_x) or (
            location.y < self.min_y or location.y > self.max_y)
        if not_in_region:
            new_status = py_trees.common.Status.RUNNING
        else:
            new_status = py_trees.common.Status.SUCCESS

        self.logger.debug("%s.update()[%s->%s]" %
                          (self.__class__.__name__, self.status, new_status))

        return new_status

    def terminate(self, new_status):
        self.vehicle = None
        self.logger.debug("%s.terminate()[%s->%s]" % (
            self.__class__.__name__, self.status, new_status))


class InTriggerDistance(py_trees.behaviour.Behaviour):

    """
    This class contains the trigger distance (condition) of a scenario
    """

    def __init__(self, other_vehicle, ego_vehicle, distance, name="TriggerDistance"):
        """
        Setup trigger distance
        """
        super(InTriggerDistance, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self.other_vehicle = other_vehicle
        self.ego_vehicle = ego_vehicle
        self.distance = distance

    def setup(self, unused_timeout=15):
        self.logger.debug("%s.setup()" % (self.__class__.__name__))
        return True

    def initialise(self):
        self.logger.debug("%s.initialise()" % (self.__class__.__name__))

    def update(self):
        """
        Check if the ego vehicle is within trigger distance to other vehicle
        """
        ego_location = self.ego_vehicle.get_location()
        other_location = self.other_vehicle.get_location()

        distance_squared = (ego_location.x - other_location.x)**2
        distance_squared += (ego_location.y - other_location.y)**2

        if distance_squared > self.distance**2:
            new_status = py_trees.common.Status.RUNNING
        else:
            new_status = py_trees.common.Status.SUCCESS

        self.logger.debug("%s.update()[%s->%s]" %
                          (self.__class__.__name__, self.status, new_status))

        return new_status

    def terminate(self, new_status):
        self.ego_vehicle = None
        self.other_vehicle = None
        self.logger.debug("%s.terminate()[%s->%s]" % (
            self.__class__.__name__, self.status, new_status))


class AccelerateToVelocity(py_trees.behaviour.Behaviour):

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

    def setup(self, unused_timeout=15):
        self.logger.debug("%s.setup()" % (self.__class__.__name__))
        return True

    def initialise(self):
        self.logger.debug("%s.initialise()" % (self.__class__.__name__))

    def update(self):
        """
        Set throttle to throttle_value, as long as velocity is < target_velocity
        """
        velocity_squared = math.pow(self.vehicle.get_velocity().x, 2)
        velocity_squared += math.pow(self.vehicle.get_velocity().y, 2)

        if velocity_squared < self.target_velocity * self.target_velocity:
            new_status = py_trees.common.Status.RUNNING
            self.control.throttle = self.throttle_value
        else:
            new_status = py_trees.common.Status.SUCCESS
            self.control.throttle = 0

        self.logger.debug("%s.update()[%s->%s]" %
                          (self.__class__.__name__, self.status, new_status))
        self.vehicle.apply_control(self.control)

        return new_status

    def terminate(self, new_status):
        self.vehicle = None
        self.logger.debug("%s.terminate()[%s->%s]" % (
            self.__class__.__name__, self.status, new_status))


class KeepVelocity(py_trees.behaviour.Behaviour):

    """
    This class contains an atomic behavior to keep the provided velocity.
    The controlled traffic participant will accelerate as fast as possible
    until reaching a given _target_velocity_, which is maintained for a
    given duration _duration_
    """

    def __init__(self, vehicle, target_velocity, duration, name="KeepVelocity"):
        """
        Setup parameters including acceleration value (via throttle_value)
        and target velocity
        """
        super(KeepVelocity, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self.control = carla.VehicleControl()
        self.vehicle = vehicle
        self.target_velocity = target_velocity
        self.duration = duration
        self.elapsed_time = 0
        self.start_time = 0

    def setup(self, unused_timeout=15):
        self.logger.debug("%s.setup()" % (self.__class__.__name__))
        return True

    def initialise(self):
        self.logger.debug("%s.initialise()" % (self.__class__.__name__))

    def update(self):
        """
        Set throttle to throttle_value, as long as velocity is < target_velocity
        """
        if self.start_time == 0:
            self.start_time = time.time()

        self.elapsed_time = time.time() - self.start_time
        if self.elapsed_time < self.duration:
            new_status = py_trees.common.Status.RUNNING
            velocity_squared = math.pow(self.vehicle.get_velocity().x, 2)
            velocity_squared += math.pow(self.vehicle.get_velocity().y, 2)

            if velocity_squared < self.target_velocity * self.target_velocity:
                self.control.throttle = 1.0
            else:
                self.control.throttle = 0.0
        else:
            # If duration is elapsed, the throttle value is set to 0 (to avoid
            # further acceleration)
            self.control.throttle = 0.0
            new_status = py_trees.common.Status.SUCCESS

        self.vehicle.apply_control(self.control)
        self.logger.debug("%s.update()[%s->%s]" %
                          (self.__class__.__name__, self.status, new_status))
        return new_status

    def terminate(self, new_status):
        self.vehicle = None
        self.logger.debug("%s.terminate()[%s->%s]" % (
            self.__class__.__name__, self.status, new_status))


class StopVehicle(py_trees.behaviour.Behaviour):

    """
    This class contains an atomic stopping behavior. The controlled traffic
    participant will decelerate with _bake_value_ until reaching a full stop.
    """

    epsilon = 0.001  # velocities lower than epsilon are considered as 0.

    def __init__(self, vehicle, brake_value, name="Stopping"):
        """
        Setup vehicle and maximum braking value
        """
        super(StopVehicle, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self.control = carla.VehicleControl()
        self.vehicle = vehicle
        self.brake_value = brake_value

    def setup(self, unused_timeout=15):
        self.logger.debug("%s.setup()" % (self.__class__.__name__))
        return True

    def initialise(self):
        self.logger.debug("%s.initialise()" % (self.__class__.__name__))

    def update(self):
        """
        Set brake to brake_value until reaching full stop
        """
        velocity_squared = math.pow(self.vehicle.get_velocity().x, 2)
        velocity_squared += math.pow(self.vehicle.get_velocity().y, 2)

        if velocity_squared > self.epsilon:
            new_status = py_trees.common.Status.RUNNING
            self.control.brake = self.brake_value
        else:
            new_status = py_trees.common.Status.SUCCESS
            self.control.brake = 0

        self.logger.debug("%s.update()[%s->%s]" %
                          (self.__class__.__name__, self.status, new_status))
        self.vehicle.apply_control(self.control)

        return new_status

    def terminate(self, new_status):
        self.vehicle = None
        self.logger.debug("%s.terminate()[%s->%s]" % (
            self.__class__.__name__, self.status, new_status))
