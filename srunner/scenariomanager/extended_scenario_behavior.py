#!/usr/bin/env python

# Copyright (c) 2019 Aptiv.
# authors: Tomasz Sulkowski (tomasz.sulkowski@aptiv.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides extended scenario behaviors required to implement traffic
movement like "drifting onto oncoming lane" or "driving out of driveway" not
covered by atomic_scenario_behavior

All the actions found in this module are continuous - i.e. they do not reset car
controls on termination by design
"""

import math
import py_trees
import carla
from srunner.scenariomanager.atomic_scenario_behavior import *
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider


def apply_longitudal_control(current_control, vehicle, max_speed=None,
                             max_throttle=None, max_braking=None, use_reverse=None):
    """
    Method to apply all kinds of longitudal vehicle control
    """

    speed_diff = None if (max_speed is None) else (
        max_speed - CarlaDataProvider.get_velocity(vehicle))
    new_throttle = (max_throttle or 1.0) if (speed_diff is None) else (
        max(0.0, min(speed_diff+1/3, max_throttle or 1.0)))
    new_brake = (max_braking) if (speed_diff is None) else (
        max(0.0, min(-speed_diff, max_braking or 1.0)))

    if new_throttle is not None:
        current_control.throttle = new_throttle
    if new_brake is not None:
        current_control.brake = new_brake
    if use_reverse is not None:
        current_control.reverse = use_reverse

    return current_control


def apply_lateral_control_to_point(current_control, vehicle, target_location, max_steering, use_reverse=False):
    """
    Method to steer vehicle to a location
    """

    # Navigation
    ego_location = CarlaDataProvider.get_location(vehicle)
    angle_to_target = math.degrees(math.atan2(target_location.y-ego_location.y,
                                              target_location.x-ego_location.x))
    # Steering
    ego_yaw = vehicle.get_transform().rotation.yaw
    relative_angle = (angle_to_target - ego_yaw) if (not use_reverse) else (
        -angle_to_target + ego_yaw - 180)
    # Normalize angle to (-180,180)
    relative_angle = ((relative_angle+180.0) % 360) - 180
    # Limit steering to max_steering
    new_steering = max(-max_steering, min(relative_angle/20.0, max_steering))
    # Smooth steering outward, prevents tank slapper
    current_control.steer = new_steering if abs(new_steering) < abs(
        current_control.steer) else (9*current_control.steer+new_steering)/10.0

    return current_control


class DriveVehicleContinuous(AtomicBehavior):
    """
    This class contains basic drive behavior. Controlled vehicle will have the
    specified control values applied to itself. Optional speed parameter will
    limit acceleration once speed is achieved.
    """

    def __init__(self, vehicle, max_speed=None, max_throttle=None, max_braking=None,
                 use_reverse=None, steering=None, name="DriveVehicleContinuous"):
        """
        Sets up controls to drive vehicle
        """
        super(DriveVehicleContinuous, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))

        self.vehicle = vehicle
        self.max_speed = max_speed
        self.max_throttle = None if max_throttle is None else min(
            max_throttle, 1.0)
        self.max_braking = None if max_braking is None else min(
            max_braking, 1.0)
        self.steering = None if steering is None else max(
            -1.0, min(1.0, steering))
        self.use_reverse = use_reverse

        self.new_control = carla.VehicleControl()

    def update(self):
        """
        Applies longitudal vehicle control indefinitely
        """
        self.new_control = self.vehicle.get_control()
        self.new_control = apply_longitudal_control(self.new_control, self.vehicle,
                                                    self.max_speed, self.max_throttle,
                                                    self.max_braking, self.use_reverse)

        if self.steering is not None:
            self.new_control.steer = self.steering

        self.vehicle.apply_control(self.new_control)

        new_status = py_trees.common.Status.RUNNING
        self.logger.debug(
            "%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))
        return new_status


class DriveToLocationContinuous(AtomicBehavior):
    """
    This class contains simple 'drive to a location' behavior. Controlled
    vehicle will have the specified longitudal control applied to itself.
    Optional speed parameter will limit acceleration once speed is achieved.

    Lateral control is also applied so that the vehicle will point to
    target_location. Optional max_steering parameter will limit maximum steering
    applied to the vehicle
    """

    def __init__(self, vehicle, target_location, max_speed=None, max_throttle=None,
                 max_braking=None, use_reverse=None, max_steering=None, name="DriveToLocationContinuous"):
        """
        Sets up controls to drive vehicle
        """
        super(DriveToLocationContinuous, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))

        self.vehicle = vehicle
        self.max_speed = max_speed
        self.max_throttle = None if max_throttle is None else min(
            max_throttle, 1.0)
        self.max_braking = None if max_braking is None else min(
            max_braking, 1.0)
        self.use_reverse = use_reverse

        self.new_control = carla.VehicleControl()

        self.target_location = target_location
        self.max_steering = max_steering or 1.0

    def update(self):
        """
        Applies longitudal and lateral vehicle control indefinitely
        """
        if not isinstance(self.target_location, carla.Location):
            new_status = py_trees.common.Status.INVALID
            self.logger.debug(
                "%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))
            return new_status

        self.new_control = self.vehicle.get_control()
        self.new_control = apply_lateral_control_to_point(
            self.new_control, self.vehicle, self.target_location, self.max_steering, self.use_reverse)
        self.new_control = apply_longitudal_control(
            self.new_control, self.vehicle, self.max_speed, self.max_throttle, self.max_braking, self.use_reverse)

        self.vehicle.apply_control(self.new_control)

        new_status = py_trees.common.Status.RUNNING
        self.logger.debug(
            "%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))
        return new_status


class FollowVehicleContinuous(AtomicBehavior):
    """
    This class contains simple 'follow the vehicle' behavior. Controlled
    vehicle will have the specified longitudal control applied to itself.
    Optional speed parameter will limit acceleration once speed is achieved

    Lateral control is also applied so that the vehicle will steer towards
    target vehicle. Optional max_steering parameter will limit maximum steering
    applied to the vehicle
    """

    def __init__(self, vehicle, target_vehicle, max_speed=None, max_throttle=None,
                 max_braking=None, use_reverse=None, max_steering=None,
                 min_distance=None, name="DriveToLocationContinuous"):
        """
        Sets up controls to drive vehicle
        """
        super(FollowVehicleContinuous, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))

        self.vehicle = vehicle
        self.target_vehicle = target_vehicle
        self.max_speed = max_speed
        self.max_throttle = None if max_throttle is None else min(
            max_throttle, 1.0)
        self.max_braking = None if max_braking is None else min(
            max_braking, 1.0)
        self.use_reverse = use_reverse
        self.max_steering = max_steering or 1.0
        self.min_distance = min_distance or 6

        self.new_control = carla.VehicleControl()
        self.target_location = CarlaDataProvider.get_location(
            self.target_vehicle)

    def update(self):
        """
        Applies longitudal and lateral vehicle control indefinitely
        """
        ego_location = CarlaDataProvider.get_location(self.vehicle)
        ego_speed = CarlaDataProvider.get_velocity(self.vehicle)
        target_location = CarlaDataProvider.get_location(self.target_vehicle)
        target_speed = CarlaDataProvider.get_velocity(self.target_vehicle)

        ego_yaw_mltpl = 5+min(6, ego_speed)*5
        target_yaw_mltpl = 2+min(6, target_speed)*1/2

        target_veh_yaw = self.target_vehicle.get_transform().rotation.yaw
        ego_yaw = self.vehicle.get_transform().rotation.yaw
        # To not cut corners target location to drive to is calculated behind
        # target_vehicle based on its speed and in front of ego car to provoke
        # larger corner radius the bigger the speeds
        self.target_location = carla.Location(
            x=target_location.x
            - target_yaw_mltpl*math.cos(math.radians(target_veh_yaw))
            + ego_yaw_mltpl*math.cos(math.radians(ego_yaw)), y=target_location.y
            - target_yaw_mltpl*math.sin(math.radians(target_veh_yaw))
            + ego_yaw_mltpl*math.sin(math.radians(ego_yaw)), z=target_location.z)

        max_speed = min(self.max_speed or 999, calculate_distance(
            ego_location, target_location)-self.min_distance)

        self.new_control = self.vehicle.get_control()
        self.new_control = apply_lateral_control_to_point(
            self.new_control, self.vehicle, self.target_location, self.max_steering, self.use_reverse)
        self.new_control = apply_longitudal_control(
            self.new_control, self.vehicle, max_speed, self.max_throttle, self.max_braking, self.use_reverse)

        self.vehicle.apply_control(self.new_control)

        new_status = py_trees.common.Status.RUNNING
        self.logger.debug(
            "%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))
        return new_status

class TriggerOnStatusChange(AtomicBehavior):
    """
    This class contains the status change trigger (condition) of a scenario

    The behavior is successful, if any of the actor properties has passed
    specified transform values. Those include: location.x, location.y,
    location.z, rotation.roll, rotation.pitch, rotation.yaw
    """

    def __init__(self, vehicle, transform, name="TriggerOnStatusChange"):
        """
        Setup trigger values
        """
        super(TriggerOnStatusChange, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self.vehicle = vehicle
        self.target_x = None if transform.location is None else transform.location.x
        self.target_y = None if transform.location is None else transform.location.y
        self.target_z = None if transform.location is None else transform.location.z
        self.target_roll = None if transform.rotation is None else transform.rotation.roll
        self.target_pitch = None if transform.rotation is None else transform.rotation.pitch
        self.target_yaw = None if transform.rotation is None else transform.rotation.yaw

        self.old_location = None
        self.old_rotation = None

    def update(self):
        """
        Check if previous rotation or location values passed one of the parameters
        """
        current_location = CarlaDataProvider.get_location(self.vehicle)
        current_rotation = self.vehicle.get_transform().rotation

        if (self.old_location is not None) and (self.old_rotation is not None):
            changed_x = ((self.target_x is not None) and (
                current_location.x - self.target_x)*(self.old_location.x-self.target_x) <= 0)
            changed_y = ((self.target_y is not None) and (
                current_location.y-self.target_y)*(self.old_location.y-self.target_y) <= 0)
            changed_z = ((self.target_z is not None) and (
                current_location.z-self.target_z)*(self.old_location.z-self.target_z) <= 0)
            changed_yaw = ((self.target_yaw is not None) and (
                current_rotation.yaw-self.target_yaw)*(self.old_rotation.yaw-self.target_yaw) <= 0)
            changed_roll = ((self.target_roll is not None) and (
                current_rotation.roll-self.target_roll)*(self.old_rotation.roll-self.target_roll) <= 0)
            changed_pitch = ((self.target_pitch is not None) and (
                current_rotation.pitch-self.target_pitch)*(self.old_rotation.pitch-self.target_pitch) <= 0)

            # Trigger on any car parameter passing target value
            if any(changed_x, changed_y, changed_z, changed_yaw, changed_roll, changed_pitch):
                return py_trees.common.Status.SUCCESS

        self.old_location = current_location
        self.old_rotation = current_rotation

        self.logger.debug(
            "%s.update()[%s->%s]" % (self.__class__.__name__, self.status, py_trees.common.Status.RUNNING))
        return py_trees.common.Status.RUNNING
