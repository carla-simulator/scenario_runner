#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

from collections import deque

import carla
from Tools.misc import *

class VehiclePIDController():
    """
    VehiclePIDController is the combination of two PID controllers (lateral and longitudinal) to perform the
    low level control a vehicle from client side
    """
    def __init__(self, vehicle,
                 args_lateral={'K_P': 1.0, 'K_D': 0.0, 'K_I': 0.0},
                 args_longitudinal={'K_P': 1.0, 'K_D': 0.0, 'K_I': 0.0} ):
        """
        :param vehicle: actor to apply to local planner logic onto
        :param args_lateral: dictionary of arguments to set the lateral PID controller using the following semantics:
                             K_P -- Proportional term
                             K_D -- Differential term
                             K_I -- Integral term
        :param args_longitudinal: dictionary of arguments to set the longitudinal PID controller using the following
        semantics:
                             K_P -- Proportional term
                             K_D -- Differential term
                             K_I -- Integral term
        """
        self._vehicle = vehicle
        self._world = self._vehicle.get_world()
        self._long_controller = PIDLongitudinalController(self._vehicle, **args_longitudinal)
        self._later_controller = PIDLateralController(self._vehicle, **args_lateral)

    def run_step(self, target_speed, waypoint):
        """
        Execute one step of control invoking both lateral and longitudinal PID controllers to reach a target waypoint
        at a given target_speed.

        :param target_speed: desired vehicle speed
        :param waypoint: target location encoded as a waypoint
        :return: distance (in meters) to the waypoint
        """
        throttle = self._long_controller.run_step(target_speed)
        steering = self._later_controller.run_step(waypoint)

        control = carla.VehicleControl()
        control.steer = steering
        control.throttle = throttle
        control.brake = 0.0
        control.hand_brake = False
        self._vehicle.apply_control(control)

        vehicle_transform = self._vehicle.get_transform()
        return distance_vehicle(waypoint, vehicle_transform)

    def run_iter(self, target_speed, waypoint, radius, max_iters):
        """
        Execute max_iters step(s) of control invoking both lateral and longitudinal PID controllers to reach a
        target waypoint within a distance specified by radius at a given target_speed.

        :param target_speed: desired vehicle speed
        :param waypoint: target location encoded as a waypoint
        :param radius: termination criterion based on distance between the vehicle and the target waypoint
        :param max_iters: termination criterion based on number of iterations (<0 --> no limit)
        :return: distance (in meters) to the waypoint
        """
        _buffer = []
        iters = 0
        if max_iters < 0:
            max_iters = math.inf
        control = carla.VehicleControl()
        vehicle_transform = self._vehicle.get_transform()
        while  distance_vehicle(waypoint, vehicle_transform) > radius and iters < max_iters:
            throttle = self._long_controller.run_step(target_speed)
            steering = self._later_controller.run_step(waypoint)

            control.steer = steering
            control.throttle = throttle
            control.brake = 0.0
            control.hand_brake = False
            self._vehicle.apply_control(control)

            vehicle_transform = self._vehicle.get_transform()
            loc = vehicle_transform.location
            dx = waypoint.transform.location.x - loc.x
            dy = waypoint.transform.location.y - loc.y
            _error = math.sqrt(dx * dx + dy * dy)
            _buffer.append(_error)
            iters += 1

        vehicle_transform = self._vehicle.get_transform()
        return distance_vehicle(waypoint, vehicle_transform)


    def warmup(self):
        """
        Setting the vehicle to change gears and become reactive
        :return:
        """
        speed = get_speed(self._vehicle)
        while speed < 0.5:
            if not self._world.wait_for_tick(10.0):
                continue

            control = carla.VehicleControl()
            control.steer = 0.0
            control.throttle = 1.0
            control.brake = 0.0
            control.hand_brake = False
            self._vehicle.apply_control(control)

        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.3
        control.brake = 0.0
        control.hand_brake = False
        self._vehicle.apply_control(control)

class PIDLongitudinalController():
    """
    PIDLongitudinalController implements longitudinal control using a PID.
    """
    def __init__(self, vehicle, K_P=1.0, K_D=0.0, K_I=0.0, dt=0.03):
        """
        :param vehicle: actor to apply to local planner logic onto
        :param K_P: Proportional term
        :param K_D: Differential term
        :param K_I: Integral term
        :param dt: time differential in seconds
        """
        self._vehicle = vehicle
        self._K_P = K_P
        self._K_D = K_D
        self._K_I = K_I
        self._dt = dt
        self._e_buffer = deque(maxlen=30)

    def run_step(self, target_speed, debug=False):
        """
        Execute one step of longitudinal control to reach a given target speed.

        :param target_speed: target speed in Km/h
        :return: throttle control in the range [0, 1]
        """
        current_speed = get_speed(self._vehicle)

        if debug:
            print('Current speed = {}'.format(current_speed))

        return self._pid_control(target_speed, current_speed)


    def _pid_control(self, target_speed, current_speed):
        """
        Estimate the throttle of the vehicle based on the PID equations

        :param target_speed:  target speed in Km/h
        :param current_speed: current speed of the vehicle in Km/h
        :return: throttle control in the range [0, 1]
        """
        _e = (target_speed - current_speed)
        self._e_buffer.append(_e)

        if len(self._e_buffer) >= 2:
            _de = (self._e_buffer[-1] - self._e_buffer[-2]) / self._dt
            _ie = sum(self._e_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        return np.clip( (self._K_P * _e) + (self._K_D * _de / self._dt) + (self._K_I * _ie * self._dt), 0.0, 1.0)



class PIDLateralController():
    """
    PIDLateralController implements lateral control using a PID.
    """
    def __init__(self, vehicle, K_P=1.0, K_D=0.0, K_I=0.0, dt=0.03):
        """
        :param vehicle: actor to apply to local planner logic onto
        :param K_P: Proportional term
        :param K_D: Differential term
        :param K_I: Integral term
        :param dt: time differential in seconds
        """
        self._vehicle = vehicle
        self._K_P = K_P
        self._K_D = K_D
        self._K_I = K_I
        self._dt = dt
        self._eps = -0.1
        self._e_buffer = deque(maxlen=30)

    def run_step(self, waypoint):
        """
        Execute one step of lateral control to steer the vehicle towards a certain waypoin.

        :param waypoint: target waypoint
        :return: steering control in the range [-1, 1] where:
            -1 represent maximum steering to left
            +1 maximum steering to right
        """
        return self._pid_control(waypoint, self._vehicle.get_transform())

    def _pid_control(self, waypoint, vehicle_transform):
        """
        Estimate the steering angle of the vehicle based on the PID equations

        :param waypoint: target waypoint
        :param vehicle_transform: current transform of the vehicle
        :return: steering control in the range [-1, 1]
        """
        v_begin = vehicle_transform.location
        v_end = v_begin + carla.Location(x=math.cos(math.radians(vehicle_transform.rotation.yaw)),
                                         y=math.sin(math.radians(vehicle_transform.rotation.yaw)))

        v_vec = np.array([v_end.x - v_begin.x, v_end.y - v_begin.y, 0.0])
        w_vec = np.array([waypoint.transform.location.x - v_begin.x, waypoint.transform.location.y - v_begin.y, 0.0])
        _dot = math.acos(np.dot(w_vec, v_vec) / (np.linalg.norm(w_vec) * np.linalg.norm(v_vec)))
        _cross = np.cross(v_vec, w_vec)
        if _cross[2] < -self._eps:
            _dot *= -1.0

        self._e_buffer.append(_dot)
        if len(self._e_buffer) >= 2:
            _de = (self._e_buffer[-1] - self._e_buffer[-2]) / self._dt
            _ie = sum(self._e_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        return np.clip( (self._K_P * _dot) + (self._K_D * _de / self._dt) + (self._K_I * _ie * self._dt), -1.0, 1.0)
