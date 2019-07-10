
import time

from threading import Thread
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
import scene_layout as scene_layout_parser  # This should come from CARLA path


def threaded(fn):
    def wrapper(*args, **kwargs):
        thread = Thread(target=fn, args=args, kwargs=kwargs)
        thread.setDaemon(True)
        thread.start()

        return thread
    return wrapper


class SceneLayoutMeasurement(object):
    def __init__(self, data, frame):
        self.data = data
        self.frame = frame


class SceneLayoutReader(object):
    def __init__(self, world):
        """
            The scene layout just requires a reference to world where you will
            extract all the necessary information.
        """

        # The static scene dictionary of all the entire scene layout.

        self.static_scene_dict = scene_layout_parser.get_scene_layout(CarlaDataProvider.get_map())
        print("Map loaded. Number of waypoints:  ", len(self.static_scene_dict))

        # Callback attribute to set the function being used.
        self._callback = None
        # Just connect the scene layout directly with the sensors
        self.read_scene_layout()

    def __call__(self):
        return self.static_scene_dict

    @threaded  # This thread just produces the callback once and dies.
    def read_scene_layout(self):
        while True:
            # We will wait for the callback to be defined, produce a layout and then die.
            if self._callback is not None:
                self._callback(SceneLayoutMeasurement(self.__call__(), 0))
                break
            else:
                time.sleep(0.01)

    def listen(self, callback):
        # Tell that this function receives what the producer does.
        self._callback = callback

    def stop(self):
        # Nothing to stop here.
        pass

    def destroy(self):
        # Nothing to destroy here.
        pass


class ObjectMeasurements(object):
    def __init__(self, data, frame):
        self.data = data
        self.frame = frame


class ObjectFinder(object):
    """
    Pseudo sensor that gives you the position of all the other dynamic objects and their states
    """

    def __init__(self,  world, reading_frequency):
        """
            The object finder is used to give you the positions of all the
            other dynamic objects in the scene and their properties.
        """
        # Give the entire access there
        # The vehicle where the class reads the speed
        self._world = world
        # Map used by the object finder
        self._map = CarlaDataProvider.get_map()
        # How often do you look at your speedometer in hz
        self._reading_frequency = reading_frequency
        self._callback = None
        #  Counts the frames
        self._frame = 0
        self._run_ps = True
        self.find_objects()

    def __call__(self):
        """ We here work into getting all the dynamic objects """
        return scene_layout_parser.get_dynamic_objects(self._world, self._map)

    @threaded
    def find_objects(self):
        latest_speed_read = time.time()
        while self._run_ps:
            if self._callback is not None:
                capture = time.time()
                if capture - latest_speed_read > (1 / self._reading_frequency):
                    self._callback(ObjectMeasurements(self.__call__(), self._frame))
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

