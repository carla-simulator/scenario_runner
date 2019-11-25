import copy
import logging
import numpy as np
import os
import time
from threading import Thread

import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.challenge.envs.scene_layout_sensors import SceneLayoutMeasurement, ObjectMeasurements, threaded


class HDMapMeasurement(object):
    def __init__(self, data, frame):
        self.data = data
        self.frame = frame


class HDMapReader(object):
    def __init__(self, vehicle, reading_frequency=1.0):
        self._vehicle = vehicle
        self._reading_frequency = reading_frequency
        self._CARLA_ROOT = os.getenv('CARLA_ROOT', "./")
        self._callback = None
        self._frame = 0
        self._run_ps = True
        self.run()

    def __call__(self):
        map_name = os.path.basename(CarlaDataProvider.get_map().name)
        transform = self._vehicle.get_transform()

        return {'map_file': "{}/HDMaps/{}.pcd".format(self._CARLA_ROOT, map_name),
                'opendrive': CarlaDataProvider.get_map().to_opendrive(),
                'transform': {'x': transform.location.x,
                              'y': transform.location.y,
                              'z': transform.location.z,
                              'yaw': transform.rotation.yaw,
                              'pitch': transform.rotation.pitch,
                              'roll': transform.rotation.roll}
                }

    @threaded
    def run(self):
        latest_read = time.time()
        while self._run_ps:
            if self._callback is not None:
                capture = time.time()
                if capture - latest_read > (1 / self._reading_frequency):
                    self._callback(HDMapMeasurement(self.__call__(), self._frame))
                    self._frame += 1
                    latest_read = time.time()
                else:
                    time.sleep(0.001)

    def listen(self, callback):
        # Tell that this function receives what the producer does.
        self._callback = callback

    def stop(self):
        self._run_ps = False

    def destroy(self):
        self._run_ps = False


class CANBusMeasurement(object):
    def __init__(self, data, frame):
        self.data = data
        self.frame = frame


class CANBusSensor(object):
    """
    CAN BUS pseudo sensor that gets to read all the vehicle proprieties including speed.
    This sensor is not placed at the CARLA environment. It is
    only an asynchronous interface to the forward speed.
    """
    MAX_CONNECTION_ATTEMPTS = 10

    def __init__(self, vehicle, reading_frequency):
        # The vehicle where the class reads the speed
        self._vehicle = vehicle
        # How often do you look at your speedometer in hz
        self._reading_frequency = reading_frequency
        self._callback = None
        #  Counts the frames
        self._frame = 0
        self._run_ps = True
        self.read_CAN_Bus()

    def _get_forward_speed(self, transform=None, velocity=None):
        """ Convert the vehicle transform directly to forward speed """
        if not velocity:
            velocity = self._vehicle.get_velocity()
        if not transform:
            transform = self._vehicle.get_transform()

        vel_np = np.array([velocity.x, velocity.y, velocity.z])
        pitch = np.deg2rad(transform.rotation.pitch)
        yaw = np.deg2rad(transform.rotation.yaw)
        orientation = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
        speed = np.dot(vel_np, orientation)
        return speed


    def _get_rotation_matrix(self, transform=None):
        """
        from: https://github.com/xmyqsh/scenario_runner/tree/development
        Generate the rotation matrix from Euler angles (actually, Tait-Bryan angles)
        with intrinsic sequence ZYX
        """
        if not transform:
            transform = self._vehicle.get_transform()

        roll  = np.deg2rad(-transform.rotation.roll)
        pitch = np.deg2rad(-transform.rotation.pitch)
        yaw = np.deg2rad(transform.rotation.yaw)
        sr, cr = np.sin(roll),  np.cos(roll)
        sp, cp = np.sin(pitch), np.cos(pitch)
        sy, cy = np.sin(yaw),   np.cos(yaw)
        rotation_matrix= np.array([[cy * cp,      -sy * sr + cy * sp * sr,  cy * sp * cr +      sy * sr],
                                   [sy * cp,  cy * sp * sr +      cy * sr,      -cy * sr + sy * sp * cr],
                                   [    -sp,                      cp * sr,                      cp * cr]])
        return rotation_matrix

    def _get_linear_velocity(self, velocity=None):
        """
        from: https://github.com/xmyqsh/scenario_runner/tree/development
        Convert linear velocity from world frame to vehicle reference frame
        """

        if not velocity:
            velocity = self._vehicle.get_velocity()

        rotation_matrix = self._get_rotation_matrix()
        linear_velocity_vrf = rotation_matrix.transpose().dot(velocity)
        return linear_velocity_vrf

    def _get_linear_acceleration(self, acceleration=None):
        """
        from: https://github.com/xmyqsh/scenario_runner/tree/development
        Convert linear acceleration from world frame to vehicle reference frame
        """

        if not acceleration:
            acceleration = self._vehicle.get_acceleration()

        rotation_matrix = self._get_rotation_matrix()
        linear_acceleration_vrf = rotation_matrix.transpose().dot(acceleration)
        return linear_acceleration_vrf

    def _get_angular_velocity(self, angular_velocity=None):
        """ Convert angular velocity from world frame to vehicle reference frame """

        if not angular_velocity:
            angular_velocity = self._vehicle.get_angular_velocity()

        rotation_matrix = self._get_rotation_matrix()
        angular_velocity_vrf = rotation_matrix.transpose().dot(angular_velocity)
        return angular_velocity_vrf


    def __call__(self):

        """ We convert the vehicle physics information into a convenient dictionary """

        # protect this access against timeout
        attempts = 0
        while attempts < self.MAX_CONNECTION_ATTEMPTS:
            try:
                vehicle_physics = self._vehicle.get_physics_control()
                velocity = self._vehicle.get_velocity()
                angular_velocity = self._vehicle.get_angular_velocity()
                transform = self._vehicle.get_transform()
                acceleration = self._vehicle.get_acceleration()
                break
            except Exception:
                attempts += 1
                print('======[WARNING] The server is frozen [{}/{} attempts]!!'.format(attempts,
                                                                                       self.MAX_CONNECTION_ATTEMPTS))
                time.sleep(1.0)
                continue


        wheels_list_dict = []
        for wheel in vehicle_physics.wheels:
            wheels_list_dict.append(
                {'tire_friction': wheel.tire_friction,
                 'damping_rate': wheel.damping_rate,
                 'steer_angle': wheel.max_steer_angle
                 }
            )

        torque_curve = []
        for point in vehicle_physics.torque_curve:
            torque_curve.append({'x': point.x,
                                 'y': point.y
                                 })
        steering_curve = []
        for point in vehicle_physics.steering_curve:
            steering_curve.append({'x': point.x,
                                   'y': point.y
                                   })

        return {
            'transform': transform,
            'dimensions': {'length': self._vehicle.bounding_box.extent.x,
                           'width': self._vehicle.bounding_box.extent.y,
                           'height': self._vehicle.bounding_box.extent.z},
            'speed': self._get_forward_speed(transform=transform, velocity=velocity),
            'lateral_speed': self._get_angular_velocity(angular_velocity=angular_velocity),
            'linear_velocity': self._get_linear_velocity(velocity=velocity),
            'linear_acceleration': self._get_linear_acceleration(acceleration=acceleration),
            'torque_curve': torque_curve,
            'max_rpm': vehicle_physics.max_rpm,
            'moi': vehicle_physics.moi,
            'damping_rate_full_throttle': vehicle_physics.damping_rate_full_throttle,
            'damping_rate_zero_throttle_clutch_disengaged':
                vehicle_physics.damping_rate_zero_throttle_clutch_disengaged,
            'use_gear_autobox': vehicle_physics.use_gear_autobox,
            'clutch_strength': vehicle_physics.clutch_strength,
            'mass': vehicle_physics.mass,
            'drag_coefficient': vehicle_physics.drag_coefficient,
            'center_of_mass': {'x': vehicle_physics.center_of_mass.x,
                               'y': vehicle_physics.center_of_mass.y,
                               'z': vehicle_physics.center_of_mass.z
                               },
            'steering_curve': steering_curve,
            'wheels': wheels_list_dict
        }


    @threaded
    def read_CAN_Bus(self):
        latest_speed_read = time.time()
        while self._run_ps:
            if self._callback is not None:
                capture = time.time()
                if capture - latest_speed_read > (1 / self._reading_frequency):
                    self._callback(CANBusMeasurement(self.__call__(), self._frame))
                    self._frame += 1
                    latest_speed_read = time.time()
                else:
                    time.sleep(0.001)

    def listen(self, callback):
        # Tell that this function receives what the producer does.
        self._callback = callback

    def stop(self):
        self._run_ps = False

    def destroy(self):
        self._run_ps = False


class CallBack(object):
    def __init__(self, tag, sensor, data_provider):
        self._tag = tag
        self._data_provider = data_provider

        self._data_provider.register_sensor(tag, sensor)

    def __call__(self, data):
        if isinstance(data, carla.Image):
            self._parse_image_cb(data, self._tag)
        elif isinstance(data, carla.LidarMeasurement):
            self._parse_lidar_cb(data, self._tag)
        elif isinstance(data, carla.GnssMeasurement):
            self._parse_gnss_cb(data, self._tag)
        elif isinstance(data, CANBusMeasurement) or isinstance(data, HDMapMeasurement) \
                or isinstance(data, SceneLayoutMeasurement) or isinstance(data, ObjectMeasurements):
            self._parse_pseudosensor(data, self._tag)
        else:
            logging.error('No callback method for this sensor.')

    # Parsing CARLA physical Sensors
    def _parse_image_cb(self, image, tag):
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = copy.deepcopy(array)
        array = np.reshape(array, (image.height, image.width, 4))
        self._data_provider.update_sensor(tag, array, image.frame)

    def _parse_lidar_cb(self, lidar_data, tag):
        points = np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4'))
        points = copy.deepcopy(points)
        points = np.reshape(points, (int(points.shape[0] / 3), 3))
        self._data_provider.update_sensor(tag, points, lidar_data.frame)

    def _parse_gnss_cb(self, gnss_data, tag):
        array = np.array([gnss_data.latitude,
                          gnss_data.longitude,
                          gnss_data.altitude], dtype=np.float64)
        self._data_provider.update_sensor(tag, array, gnss_data.frame)

    # The pseudo sensors already come properly parsed, so we can basically use a single function
    def _parse_pseudosensor(self, package, tag):
        self._data_provider.update_sensor(tag, package.data, package.frame)


class SensorInterface(object):
    def __init__(self):
        self._sensors_objects = {}
        self._data_buffers = {}
        self._timestamps = {}

    def register_sensor(self, tag, sensor):
        if tag in self._sensors_objects:
            raise ValueError("Duplicated sensor tag [{}]".format(tag))

        self._sensors_objects[tag] = sensor
        self._data_buffers[tag] = None
        self._timestamps[tag] = -1

    def update_sensor(self, tag, data, timestamp):
        if tag not in self._sensors_objects:
            raise ValueError("The sensor with tag [{}] has not been created!".format(tag))
        self._data_buffers[tag] = data
        self._timestamps[tag] = timestamp

    def all_sensors_ready(self):
        for key in self._sensors_objects.keys():
            if self._data_buffers[key] is None:
                return False
        return True

    def get_data(self):
        data_dict = {}
        for key in self._sensors_objects.keys():
            data_dict[key] = (self._timestamps[key], self._data_buffers[key])
        return data_dict
