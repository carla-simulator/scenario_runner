import datetime
from collections import deque

import math
import numpy as np

import carla

from ScenarioManager.timer import GameTime
import pdb

def distance_vehicle(waypoint, vehicle_transform):
    loc = vehicle_transform.location
    dx = waypoint.transform.location.x - loc.x
    dy = waypoint.transform.location.y - loc.y

    return math.sqrt(dx * dx + dy * dy)

def distance(waypointA, waypointB):
    dx = waypointB.transform.location.x - waypointA.transform.location.x
    dy = waypointB.transform.location.y - waypointA.transform.location.y

    return math.sqrt(dx * dx + dy * dy)

def get_speed(vehicle):
    """
    Compute speed of a vehicle in Kmh
    :param vehicle: the vehicle for which speed is calculated
    :return: speed as a float in Kmh
    """
    vel = vehicle.get_velocity()
    return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)

class VehiclePIDController():
    def __init__(self, vehicle,
                 args_lateral={'K_P': 1.0, 'K_D': 0.0, 'K_I': 0.0},
                 args_longitudinal={'K_P': 1.0, 'K_D': 0.0, 'K_I': 0.0} ):
        self._vehicle = vehicle
        self._long_controller = PIDLongitudinalController(self._vehicle, **args_longitudinal)
        self._later_controller = PIDLateralController(self._vehicle, **args_lateral)

    def run_step(self, target_speed, waypoint):
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

            # debug
            loc = vehicle_transform.location
            dx = waypoint.transform.location.x - loc.x
            dy = waypoint.transform.location.y - loc.y
            _error = math.sqrt(dx * dx + dy * dy)
            _buffer.append(_error)
            iters += 1

        vehicle_transform = self._vehicle.get_transform()
        return distance_vehicle(waypoint, vehicle_transform)


    def warmup(self):
        speed = get_speed(self._vehicle)
        while speed < 0.5:
            control = carla.VehicleControl()
            control.steer = 0.0
            control.throttle = 1.0
            control.brake = 0.0
            control.hand_brake = False
            self._vehicle.apply_control(control)
            # print('speed = {}'.format(speed))

        # print('speed = {}'.format(speed))
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.3
        control.brake = 0.0
        control.hand_brake = False
        self._vehicle.apply_control(control)

class PIDLongitudinalController():
    def __init__(self, vehicle, K_P=1.0, K_D=0.0, K_I=0.0, dt=0.03):
        self._vehicle = vehicle
        self._K_P = K_P
        self._K_D = K_D
        self._K_I = K_I
        self._dt = dt

        #GameTime(self._vehicle.get_world())
        self._last_time = GameTime.get_time()
        self._e_buffer = deque(maxlen=30)

    def run_step(self, target_speed):
        current_speed = get_speed(self._vehicle)
        #print('Current speed = {}'.format(current_speed))
        #current_time = GameTime.get_time()
        #self._dt = current_time - self._last_time
        #self._last_time = current_time

        return self._pid_control(target_speed, current_speed)


    def _pid_control(self, target_speed, current_speed):
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
    def __init__(self, vehicle, K_P=1.0, K_D=0.0, K_I=0.0, dt=0.03):
        self._vehicle = vehicle
        self._K_P = K_P
        self._K_D = K_D
        self._K_I = K_I
        self._dt = dt
        self._last_time = datetime.datetime.now()

        self._eps = -0.1
        self._e_buffer = deque(maxlen=30)

    def run_step(self, waypoint):
        #current_time = datetime.datetime.now()
        #self._dt = (current_time - self._last_time).total_seconds()
        #self._last_time = current_time

        #print('== DT = {} | FPS = {}'.format(self._dt, 1.0/self._dt))
        return self._pid_control(waypoint, self._vehicle.get_transform())

    def _pid_control(self, waypoint, vehicle_transform):
        # loc = vehicle_transform.location
        # yaw = -m.radians(vehicle_transform.rotation.yaw)
        # rel_v = np.array([1.0, 0.0, 0.0])
        #
        # _x = waypoint.transform.location.x - loc.x
        # _y = waypoint.transform.location.y - loc.y
        # _xr = m.cos(yaw) * _x - m.sin(yaw) * _y
        # _yr = m.sin(yaw) * _x + m.cos(yaw) * _y
        # rel_w = np.array([_xr, _yr, 0.0])
        #
        # _cross = np.cross(rel_v, rel_w)
        # _dot = m.acos(np.dot(rel_v, rel_w) / (np.linalg.norm(rel_v) * np.linalg.norm(rel_w)))
        # if _cross[2] < -self._eps:
        #     _dot *= -1.0

        v_begin = vehicle_transform.location
        v_end = v_begin + carla.Location(x=math.cos(math.radians(vehicle_transform.rotation.yaw)),
                                         y=math.sin(math.radians(vehicle_transform.rotation.yaw)))

        v_vec = np.array([v_end.x - v_begin.x, v_end.y - v_begin.y, 0.0])
        w_vec = np.array([waypoint.transform.location.x - v_begin.x, waypoint.transform.location.y - v_begin.y, 0.0])
        _dot = math.acos(np.dot(w_vec, v_vec) / (np.linalg.norm(w_vec) * np.linalg.norm(v_vec)))
        _cross = np.cross(v_vec, w_vec)
        if _cross[2] < -self._eps:
            _dot *= -1.0

        #print(_dot)
        self._e_buffer.append(_dot)
        if len(self._e_buffer) >= 2:
            _de = (self._e_buffer[-1] - self._e_buffer[-2]) / self._dt
            _ie = sum(self._e_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        return np.clip( (self._K_P * _dot) + (self._K_D * _de / self._dt) + (self._K_I * _ie * self._dt), -1.0, 1.0)
