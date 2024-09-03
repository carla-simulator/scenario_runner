"""Scenic world model for traffic scenarios in CARLA.

The model currently supports vehicles, pedestrians, and props. It implements the
basic `Car` and `Pedestrian` classes from the :obj:`scenic.domains.driving` domain,
while also providing convenience classes for specific types of objects like bicycles,
traffic cones, etc. Vehicles and pedestrians support the basic actions and behaviors
from the driving domain; several more are automatically imported from
:obj:`scenic.simulators.carla.actions` and :obj:`scenic.simulators.carla.behaviors`.

The model defines several global parameters, whose default values can be overridden
in scenarios using the ``param`` statement or on the command line using the
:option:`--param` option:

Global Parameters:
    carla_map (str): Name of the CARLA map to use, e.g. 'Town01'. Can also be set
        to ``None``, in which case CARLA will attempt to create a world from the
        **map** file used in the scenario (which must be an ``.xodr`` file).
    timestep (float): Timestep to use for simulations (i.e., how frequently Scenic
        interrupts CARLA to run behaviors, check requirements, etc.), in seconds. Default
        is 0.1 seconds.
    snapToGroundDefault (bool): Default value for :prop:`snapToGround` on `CarlaActor` objects.
        Default is True if :ref:`2D compatibility mode` is enabled and False otherwise. 

    weather (str or dict): Weather to use for the simulation. Can be either a
        string identifying one of the CARLA weather presets (e.g. 'ClearSunset') or a
        dictionary specifying all the weather parameters (see `carla.WeatherParameters`_).
        Default is a uniform distribution over all the weather presets.

    address (str): IP address at which to connect to CARLA. Default is localhost
        (127.0.0.1).
    port (int): Port on which to connect to CARLA. Default is 2000.
    timeout (float): Maximum time to wait when attempting to connect to CARLA, in
        seconds. Default is 10.

    render (int): Whether or not to have CARLA create a window showing the
        simulations from the point of view of the ego object: 1 for yes, 0
        for no. Default 1.
    record (str): If nonempty, folder in which to save CARLA record files for
        replaying the simulations.

.. _carla.WeatherParameters: https://carla.readthedocs.io/en/latest/python_api/#carlaweatherparameters

"""
import pathlib
from scenic.domains.driving.model import *

import srunner.scenic.models.blueprints as blueprints
from srunner.scenic.models.behaviors import *
from scenic.simulators.utils.colors import Color

try:
    from srunner.scenic.models.simulator import CarlaSimulator    # for use in scenarios
    from srunner.scenic.models.actions import *
    from srunner.scenic.models.actions import _CarlaVehicle, _CarlaPedestrian
    import srunner.scenic.models.utils.utils as _utils
except ModuleNotFoundError:
    # for convenience when testing without the carla package
    from scenic.core.simulators import SimulatorInterfaceWarning
    import warnings
    warnings.warn('the "carla" package is not installed; '
                  'will not be able to run dynamic simulations',
                  SimulatorInterfaceWarning)

    def CarlaSimulator(*args, **kwargs):
        """Dummy simulator to allow compilation without the 'carla' package.

        :meta private:
        """
        raise RuntimeError('the "carla" package is required to run simulations '
                           'from this scenario')

    class _CarlaVehicle: pass
    class _CarlaPedestrian: pass

map_town = pathlib.Path(globalParameters.map).stem
param carla_map = map_town
param address = '127.0.0.1'
param port = 2000
param timeout = 10
param render = 1
if globalParameters.render not in [0, 1]:
    raise ValueError('render param must be either 0 or 1')
param record = ''
param timestep = 0.1
param weather = Uniform(
    'ClearNoon',
    'CloudyNoon',
    'WetNoon',
    'WetCloudyNoon',
    'SoftRainNoon',
    'MidRainyNoon',
    'HardRainNoon',
    'ClearSunset',
    'CloudySunset',
    'WetSunset',
    'WetCloudySunset',
    'SoftRainSunset',
    'MidRainSunset',
    'HardRainSunset'
)
param snapToGroundDefault = is2DMode()

simulator CarlaSimulator(
    carla_map=globalParameters.carla_map,
    map_path=globalParameters.map,
    address=globalParameters.address,
    port=int(globalParameters.port),
    timeout=int(globalParameters.timeout),
    render=bool(globalParameters.render),
    record=globalParameters.record,
    timestep=float(globalParameters.timestep)
)

class CarlaActor(DrivingObject):
    """Abstract class for CARLA objects.

    Properties:
        carlaActor (dynamic): Set during simulations to the ``carla.Actor`` representing this
            object.
        blueprint (str): Identifier of the CARLA blueprint specifying the type of object.
        rolename (str): Can be used to differentiate specific actors during runtime. Default
            value ``None``.
        physics (bool): Whether physics is enabled for this object in CARLA. Default true.
        snapToGround (bool): Whether or not to snap this object to the ground when placed in CARLA.
            The default is set by the ``snapToGroundDefault`` global parameter above.
    """
    carlaActor: None
    blueprint: None
    rolename: None
    color: None
    physics: True
    snapToGround: globalParameters.snapToGroundDefault

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._control = None    # used internally to accumulate control updates

    @property
    def control(self):
        if self._control is None:
            self._control = self.carlaActor.get_control()
        return self._control

    def setPosition(self, pos, elevation):
        self.carlaActor.set_location(_utils.scenicToCarlaLocation(pos, elevation))

    def setVelocity(self, vel):
        cvel = _utils.scenicToCarlaVector3D(*vel)
        if hasattr(self.carlaActor, 'set_target_velocity'):
            self.carlaActor.set_target_velocity(cvel)
        else:
            self.carlaActor.set_velocity(cvel)

class Vehicle(Vehicle, CarlaActor, Steers, _CarlaVehicle):
    """Abstract class for steerable vehicles."""

    def setThrottle(self, throttle):
        self.control.throttle = throttle

    def setSteering(self, steering):
        self.control.steer = steering

    def setBraking(self, braking):
        self.control.brake = braking

    def setHandbrake(self, handbrake):
        self.control.hand_brake = handbrake

    def setReverse(self, reverse):
        self.control.reverse = reverse

    def _getClosestTrafficLight(self, distance=100):
        return _getClosestTrafficLight(self, distance)

class Car(Vehicle):
    """A car.

    The default ``blueprint`` (see `CarlaActor`) is a uniform distribution over the
    blueprints listed in :obj:`scenic.simulators.carla.blueprints.carModels`.
    """
    blueprint: Uniform(*blueprints.carModels)

    @property
    def isCar(self):
        return True

class NPCCar(Car):  # no distinction between these in CARLA
    pass

class Bicycle(Vehicle):
    width: 1
    length: 2
    blueprint: Uniform(*blueprints.bicycleModels)


class Motorcycle(Vehicle):
    width: 1
    length:2
    blueprint: Uniform(*blueprints.motorcycleModels)


class Truck(Vehicle):
    width: 3
    length: 7
    blueprint: Uniform(*blueprints.truckModels)


class Pedestrian(Pedestrian, CarlaActor, Walks, _CarlaPedestrian):
    """A pedestrian.

    The default ``blueprint`` (see `CarlaActor`) is a uniform distribution over the
    blueprints listed in :obj:`scenic.simulators.carla.blueprints.walkerModels`.
    """
    width: 0.5
    length: 0.5
    blueprint: Uniform(*blueprints.walkerModels)
    carlaController: None

    def setWalkingDirection(self, heading):
        direction = Vector(0, 1, 0).rotatedBy(heading)
        self.control.direction = _utils.scenicToCarlaVector3D(*direction)

    def setWalkingSpeed(self, speed):
        self.control.speed = speed


class Prop(CarlaActor):
    """Abstract class for props, i.e. non-moving objects.

    Properties:
        parentOrientation (Orientation): Default value overridden to have uniformly random yaw.
        physics (bool): Default value overridden to be false.
    """
    regionContainedIn: road
    position: new Point on road
    parentOrientation: Range(0, 360) deg
    width: 0.5
    length: 0.5
    physics: False

class Trash(Prop):
    blueprint: Uniform(*blueprints.trashModels)


class Cone(Prop):
    blueprint: Uniform(*blueprints.coneModels)


class Debris(Prop):
    blueprint: Uniform(*blueprints.debrisModels)


class VendingMachine(Prop):
    blueprint: Uniform(*blueprints.vendingMachineModels)


class Chair(Prop):
    blueprint: Uniform(*blueprints.chairModels)


class BusStop(Prop):
    blueprint: Uniform(*blueprints.busStopModels)


class Advertisement(Prop):
    blueprint: Uniform(*blueprints.advertisementModels)


class Garbage(Prop):
    blueprint: Uniform(*blueprints.garbageModels)


class Container(Prop):
    blueprint: Uniform(*blueprints.containerModels)


class Table(Prop):
    blueprint: Uniform(*blueprints.tableModels)


class Barrier(Prop):
    blueprint: Uniform(*blueprints.barrierModels)


class PlantPot(Prop):
    blueprint: Uniform(*blueprints.plantpotModels)


class Mailbox(Prop):
    blueprint: Uniform(*blueprints.mailboxModels)


class Gnome(Prop):
    blueprint: Uniform(*blueprints.gnomeModels)


class CreasedBox(Prop):
    blueprint: Uniform(*blueprints.creasedboxModels)


class Case(Prop):
    blueprint: Uniform(*blueprints.caseModels)


class Box(Prop):
    blueprint: Uniform(*blueprints.boxModels)


class Bench(Prop):
    blueprint: Uniform(*blueprints.benchModels)


class Barrel(Prop):
    blueprint: Uniform(*blueprints.barrelModels)


class ATM(Prop):
    blueprint: Uniform(*blueprints.atmModels)


class Kiosk(Prop):
    blueprint: Uniform(*blueprints.kioskModels)


class IronPlate(Prop):
    blueprint: Uniform(*blueprints.ironplateModels)


class TrafficWarning(Prop):
    blueprint: Uniform(*blueprints.trafficwarningModels)


## Utility functions

def freezeTrafficLights():
    """Freezes all traffic lights in the scene.

    Frozen traffic lights can be modified by the user
    but the time will not update them until unfrozen.
    """
    simulation().world.freeze_all_traffic_lights(True)

def unfreezeTrafficLights():
    """Unfreezes all traffic lights in the scene."""
    simulation().world.freeze_all_traffic_lights(False)

def setAllIntersectionTrafficLightsStatus(intersection, color):
    for signal in intersection.signals:
        if signal.isTrafficLight:
            setTrafficLightStatus(signal, color)

def setTrafficLightStatus(signal, color):
    if not signal.isTrafficLight:
        raise RuntimeError('The provided signal is not a traffic light')
    color = utils.scenicToCarlaTrafficLightStatus(color)
    if color is None:
        raise RuntimeError('Color must be red/yellow/green/off/unknown.')
    landmarks = simulation().map.get_all_landmarks_from_id(signal.openDriveID)
    if landmarks:
        traffic_light = simulation().world.get_traffic_light(landmarks[0])
        if traffic_light is not None:
            traffic_light.set_state(color)

def getTrafficLightStatus(signal):
    if not signal.isTrafficLight:
        raise RuntimeError('The provided signal is not a traffic light')
    landmarks = simulation().map.get_all_landmarks_from_id(signal.openDriveID)
    if landmarks:
        traffic_light = simulation().world.get_traffic_light(landmarks[0])
        if traffic_light is not None:
            return utils.carlaToScenicTrafficLightStatus(traffic_light.state)
    return "None"

def _getClosestLandmark(vehicle, type, distance=100):
    if vehicle._intersection is not None:
        return None

    waypoint = simulation().map.get_waypoint(vehicle.carlaActor.get_transform().location)
    landmarks = waypoint.get_landmarks_of_type(distance, type)

    if landmarks:
        return min(landmarks, key=lambda l: l.distance)
    return None

def _getClosestTrafficLight(vehicle, distance=100):
    """Returns the closest traffic light affecting 'vehicle', up to a maximum of 'distance'"""
    landmark = _getClosestLandmark(vehicle, type="1000001", distance=distance)
    if landmark is not None:
        return simulation().world.get_traffic_light(landmark)
    return None

def withinDistanceToRedYellowTrafficLight(vehicle, thresholdDistance):
    traffic_light = _getClosestTrafficLight(vehicle, distance=thresholdDistance)
    if traffic_light is not None and str(traffic_light.state) in ("Red", "Yellow"):
        return True
    return False

def withinDistanceToTrafficLight(vehicle, thresholdDistance):
    traffic_light = _getClosestTrafficLight(vehicle, distance=thresholdDistance)
    if traffic_light is not None:
        return True
    return False

def getClosestTrafficLightStatus(vehicle, distance=100):
    traffic_light = _getClosestTrafficLight(vehicle, distance)
    if traffic_light is not None:
        return _utils.carlaToScenicTrafficLightStatus(traffic_light.state)
    return "None"

def setClosestTrafficLightStatus(vehicle, color, distance=100):
    color = _utils.scenicToCarlaTrafficLightStatus(color)
    if color is None:
        raise RuntimeError('Color must be red/yellow/green/off/unknown.')
    
    traffic_light = _getClosestTrafficLight(vehicle, distance)
    if traffic_light is not None:
        traffic_light.set_state(color)
