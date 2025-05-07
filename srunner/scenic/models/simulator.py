"""Simulator interface for CARLA."""

try:
    import carla
except ImportError as e:
    raise ModuleNotFoundError('CARLA scenarios require the "carla" Python package') from e

import math
import os
import warnings

import scenic.core.errors as errors

if errors.verbosityLevel == 0:  # suppress pygame advertisement at zero verbosity
    os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

import pygame

from scenic.core.simulators import SimulationCreationError
from scenic.domains.driving.simulators import DrivingSimulation, DrivingSimulator
from srunner.scenic.models.blueprints import oldBlueprintNames
import srunner.scenic.models.utils.utils as utils
import srunner.scenic.models.utils.visuals as visuals
from scenic.syntax.veneer import verbosePrint
from scenic.domains.driving.controllers import (
    PIDLateralController,
    PIDLongitudinalController,
)


class CarlaSimulator(DrivingSimulator):
    """Implementation of `Simulator` for CARLA."""

    def __init__(
        self,
        carla_map,
        map_path,
        address="127.0.0.1",
        port=2000,
        timeout=10,
        render=True,
        record="",
        timestep=0.05,
        traffic_manager_port=8000,
    ):
        super().__init__()
        verbosePrint(f"Connecting to CARLA on port {port}")
        self.client = carla.Client(address, port)
        self.client.set_timeout(timeout)  # limits networking operations (seconds)
        if carla_map is not None:
            try:
                self.world = self.client.load_world(carla_map)
            except Exception as e:
                raise RuntimeError(f"CARLA could not load world '{carla_map}'") from e
        else:
            if str(map_path).endswith(".xodr"):
                with open(map_path) as odr_file:
                    self.world = self.client.generate_opendrive_world(odr_file.read())
            else:
                raise RuntimeError("CARLA only supports OpenDrive maps")
        self.timestep = timestep

        self.tm = self.client.get_trafficmanager(traffic_manager_port)
        self.tm.set_synchronous_mode(True)

        # Set to synchronous with fixed timestep
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = timestep  # NOTE: Should not exceed 0.1
        self.world.apply_settings(settings)
        verbosePrint("Map loaded in simulator.")

        self.render = render  # visualization mode ON/OFF
        self.record = record  # whether to use the carla recorder
        self.scenario_number = 0  # Number of the scenario executed

    def createSimulation(self, scene, *, timestep, **kwargs):
        if timestep is not None and timestep != self.timestep:
            raise RuntimeError(
                "cannot customize timestep for individual CARLA simulations; "
                "set timestep when creating the CarlaSimulator instead"
            )

        self.scenario_number += 1
        return CarlaSimulation(
            scene,
            self.client,
            self.tm,
            self.render,
            self.record,
            self.scenario_number,
            timestep=self.timestep,
            **kwargs,
        )

    def destroy(self):
        super().destroy()
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        self.world.apply_settings(settings)
        self.tm.set_synchronous_mode(False)


class CarlaSimulation(DrivingSimulation):
    def __init__(self, scene, client, tm, render, record, scenario_number, **kwargs):
        self.client = client
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        self.blueprintLib = self.world.get_blueprint_library()
        self.tm = tm
        self.render = render
        self.record = record
        self.scenario_number = scenario_number
        self.cameraManager = None

        super().__init__(scene, **kwargs)

    def setup(self):

        # Setup HUD
        if self.render:
            self.displayDim = (1280, 720)
            self.displayClock = pygame.time.Clock()
            self.camTransform = 0
            pygame.init()
            pygame.font.init()
            self.hud = visuals.HUD(*self.displayDim)
            self.display = pygame.display.set_mode(
                self.displayDim, pygame.HWSURFACE | pygame.DOUBLEBUF
            )
            self.cameraManager = None

        if self.record:
            if not os.path.exists(self.record):
                os.mkdir(self.record)
            name = "{}/scenario{}.log".format(self.record, self.scenario_number)
            # Carla is looking for an absolute path, so convert it if necessary.
            name = os.path.abspath(name)
            self.client.start_recorder(name)

        # Create objects.
        super().setup()

        # Set up camera manager and collision sensor for ego
        if self.render:
            camIndex = 0
            camPosIndex = 0
            egoActor = self.objects[0].carlaActor
            self.cameraManager = visuals.CameraManager(self.world, egoActor, self.hud)
            self.cameraManager._transform_index = camPosIndex
            self.cameraManager.set_sensor(camIndex)
            # self.cameraManager.set_transform(self.camTransform)

        self.world.tick()  ## allowing manualgearshift to take effect    # TODO still need this?

        for obj in self.objects:
            if isinstance(obj.carlaActor, carla.Vehicle):
                obj.carlaActor.apply_control(
                    carla.VehicleControl(manual_gear_shift=False)
                )

        self.world.tick()

        for obj in self.objects:
            if obj.speed is not None and obj.speed != 0:
                raise RuntimeError(
                    f"object {obj} cannot have a nonzero initial speed "
                    "(this is not yet possible in CARLA)"
                )

    def createObjectInSimulator(self, obj):
        # Extract blueprint
        try:
            blueprint = self.blueprintLib.find(obj.blueprint)
        except IndexError as e:
            found = False
            if obj.blueprint in oldBlueprintNames:
                for oldName in oldBlueprintNames[obj.blueprint]:
                    try:
                        blueprint = self.blueprintLib.find(oldName)
                        found = True
                        warnings.warn(
                            f"CARLA blueprint {obj.blueprint} not found; "
                            f"using older version {oldName}"
                        )
                        obj.blueprint = oldName
                        break
                    except IndexError:
                        continue
            if not found:
                raise SimulationCreationError(
                    f"Unable to find blueprint {obj.blueprint}" f" for object {obj}"
                ) from e
        if obj.rolename is not None:
            blueprint.set_attribute("role_name", obj.rolename)

        # set walker as not invincible
        if blueprint.has_attribute("is_invincible"):
            blueprint.set_attribute("is_invincible", "False")

        # Set up transform
        loc = utils.scenicToCarlaLocation(
            obj.position,
            world=self.world,
            blueprint=obj.blueprint,
            snapToGround=obj.snapToGround,
        )
        rot = utils.scenicToCarlaRotation(obj.orientation)
        transform = carla.Transform(loc, rot)

        # Color, cannot be set for Pedestrians
        if blueprint.has_attribute("color") and obj.color is not None:
            c = obj.color
            c_str = f"{int(c.r*255)},{int(c.g*255)},{int(c.b*255)}"
            blueprint.set_attribute("color", c_str)

        # Create Carla actor
        carlaActor = self.world.try_spawn_actor(blueprint, transform)
        if carlaActor is None:
            raise SimulationCreationError(f"Unable to spawn object {obj}")
        obj.carlaActor = carlaActor

        carlaActor.set_simulate_physics(obj.physics)

        if isinstance(carlaActor, carla.Vehicle):
            # TODO should get dimensions at compile time, not simulation time
            extent = carlaActor.bounding_box.extent
            ex, ey, ez = extent.x, extent.y, extent.z
            # Ensure each extent is positive to work around CARLA issue #5841
            obj.width = ey * 2 if ey > 0 else obj.width
            obj.length = ex * 2 if ex > 0 else obj.length
            obj.height = ez * 2 if ez > 0 else obj.height
            carlaActor.apply_control(carla.VehicleControl(manual_gear_shift=True, gear=1))
        elif isinstance(carlaActor, carla.Walker):
            carlaActor.apply_control(carla.WalkerControl())
            # spawn walker controller
            controller_bp = self.blueprintLib.find("controller.ai.walker")
            controller = self.world.try_spawn_actor(
                controller_bp, carla.Transform(), carlaActor
            )
            if controller is None:
                raise SimulationCreationError(
                    f"Unable to spawn carla controller for object {obj}"
                )
            obj.carlaController = controller
        return carlaActor

    def executeActions(self, allActions):
        super().executeActions(allActions)

        # Apply control updates which were accumulated while executing the actions
        for obj in self.agents:
            ctrl = obj._control
            if ctrl is not None:
                obj.carlaActor.apply_control(ctrl)
                obj._control = None

    def step(self):
        # Run simulation for one timestep
        self.world.tick()

        # Render simulation
        if self.render:
            self.cameraManager.render(self.display)
            pygame.display.flip()

    def getProperties(self, obj, properties):
        # Extract Carla properties
        carlaActor = obj.carlaActor
        currTransform = carlaActor.get_transform()
        currLoc = currTransform.location
        currRot = currTransform.rotation
        currVel = carlaActor.get_velocity()
        currAngVel = carlaActor.get_angular_velocity()

        # Prepare Scenic object properties
        position = utils.carlaToScenicPosition(currLoc)
        velocity = utils.carlaToScenicPosition(currVel)
        speed = math.hypot(*velocity)
        angularSpeed = utils.carlaToScenicAngularSpeed(currAngVel)
        angularVelocity = utils.carlaToScenicAngularVel(currAngVel)
        globalOrientation = utils.carlaToScenicOrientation(currRot)
        yaw, pitch, roll = obj.parentOrientation.localAnglesFor(globalOrientation)
        elevation = utils.carlaToScenicElevation(currLoc)

        values = dict(
            position=position,
            velocity=velocity,
            speed=speed,
            angularSpeed=angularSpeed,
            angularVelocity=angularVelocity,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            elevation=elevation,
        )
        return values

    def destroy(self):
        for obj in self.objects:
            if obj.carlaActor is not None:
                if isinstance(obj.carlaActor, carla.Vehicle):
                    obj.carlaActor.set_autopilot(False, self.tm.get_port())
                if isinstance(obj.carlaActor, carla.Walker):
                    obj.carlaController.stop()
                    obj.carlaController.destroy()
                obj.carlaActor.destroy()
        if self.render and self.cameraManager:
            self.cameraManager.destroy_sensor()

        self.client.stop_recorder()

        self.world.tick()
        super().destroy()

    def getLaneFollowingControllers(self, agent):
        """Get longitudinal and lateral controllers for lane following."""
        dt = self.timestep
        lon_controller = PIDLongitudinalController(K_P=1.0, K_D=0.2, K_I=1.4, dt=dt)
        lat_controller = PIDLateralController(K_P=0.8, K_D=0.2, K_I=0.0, dt=dt)
        return lon_controller, lat_controller

    def getTurningControllers(self, agent):
        """Get longitudinal and lateral controllers for turning."""
        dt = self.timestep
        lon_controller = PIDLongitudinalController(K_P=1.0, K_D=0.2, K_I=1.4, dt=dt)
        lat_controller = PIDLateralController(K_P=2.0, K_D=0.5, K_I=0.0, dt=dt)
        return lon_controller, lat_controller

    def getLaneChangingControllers(self, agent):
        """Get longitudinal and lateral controllers for lane changing."""
        dt = self.timestep
        lon_controller = PIDLongitudinalController(K_P=1.0, K_D=0.2, K_I=1.4, dt=dt)
        lat_controller = PIDLateralController(K_P=0.8, K_D=0.2, K_I=0.0, dt=dt)
        return lon_controller, lat_controller
