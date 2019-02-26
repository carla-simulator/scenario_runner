import copy
import logging
import numpy as np
import time
from threading import Thread

import carla


def threaded(fn):
    def wrapper(*args, **kwargs):
        thread = Thread(target=fn, args=args, kwargs=kwargs)
        thread.setDaemon(True)
        thread.start()

        return thread
    return wrapper


class SpeedMeasurement(object):
    def __init__(self, data, frame_number):
        self.data = data
        self.frame_number = frame_number


class Speedometer(object):
    """
    Speed pseudo sensor that gets the current speed of the vehicle.
    This sensor is not placed at the CARLA environment. It is
    only an asynchronous interface to the forward speed.
    """

    def __init__(self, vehicle, reading_frequency):
        # The vehicle where the class reads the speed
        self._vehicle = vehicle
        # How often do you look at your speedometer in hz
        self._reading_frequency = reading_frequency
        self._callback = None
        #  Counts the frames
        self._frame_number = 0
        self._run_ps = True
        self.produce_speed()

    def _get_forward_speed(self):
        """ Convert the vehicle transform directly to forward speed """

        velocity = self._vehicle.get_velocity()
        transform = self._vehicle.get_transform()
        vel_np = np.array([velocity.x, velocity.y, velocity.z])
        pitch = np.deg2rad(transform.rotation.pitch)
        yaw = np.deg2rad(transform.rotation.yaw)
        orientation = np.array(
            [np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
        speed = np.dot(vel_np, orientation)
        return speed

    @threaded
    def produce_speed(self):
        latest_speed_read = time.time()
        while self._run_ps:
            if self._callback is not None:
                capture = time.time()
                if capture - latest_speed_read > (1 / self._reading_frequency):
                    self._callback(SpeedMeasurement(self._get_forward_speed(), self._frame_number))
                    self._frame_number += 1
                    latest_speed_read = time.time()
                else:
                    time.sleep(0.001)

    def listen(self, callback):
        # Tell that this function receives what the producer does.
        self._callback = callback

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
        elif isinstance(data, carla.GnssEvent):
            self._parse_gnss_cb(data, self._tag)
        elif isinstance(data, SpeedMeasurement):
            self._parse_speedometer(data, self._tag)
        else:
            logging.error('No callback method for this sensor.')

    def _parse_image_cb(self, image, tag):
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = copy.deepcopy(array)
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        self._data_provider.update_sensor(tag, array, image.frame_number)

    def _parse_lidar_cb(self, lidar_data, tag):
        points = np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4'))
        points = copy.deepcopy(points)
        points = np.reshape(points, (int(points.shape[0] / 3), 3))
        self._data_provider.update_sensor(tag, points, lidar_data.frame_number)

    def _parse_gnss_cb(self, gnss_data, tag):
        array = np.array([gnss_data.latitude,
                          gnss_data.longitude,
                          gnss_data.altitude], dtype=np.float32)
        self._data_provider.update_sensor(tag, array, gnss_data.frame_number)

    def _parse_speedometer(self, speed, tag):
        self._data_provider.update_sensor(tag, speed.data, speed.frame_number)


class SensorInterface(object):
    def __init__(self):
        self._sensors_objects = {}
        self._data_buffers = {}
        self._timestamps = {}

    def register_sensor(self, tag, sensor):
        if tag  in self._sensors_objects:
            raise ValueError("Duplicated sensor tag [{}]".format(tag))

        self._sensors_objects[tag] = sensor
        self._data_buffers[tag] = None
        self._timestamps[tag] = -1

    def update_sensor(self, tag, data, timestamp):
        if tag  not in self._sensors_objects:
            raise ValueError("The sensor with tag [{}] has not been "
                             "created!".format(tag))
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
            data_dict[key] = (self._timestamps[key],
                              copy.deepcopy(self._data_buffers[key]))
        return data_dict