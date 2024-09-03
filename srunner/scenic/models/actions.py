"""Actions for dynamic agents in CARLA scenarios."""

import math as _math

import carla as _carla

from scenic.domains.driving.actions import *
import srunner.scenic.models.utils.utils as _utils

################################################
# Actions available to all carla.Actor objects #
################################################

SetLocationAction = SetPositionAction  # TODO refactor


class SetAngularVelocityAction(Action):
    def __init__(self, angularVel):
        self.angularVel = angularVel

    def applyTo(self, obj, sim):
        xAngularVel = self.angularVel * _math.cos(obj.heading)
        yAngularVel = self.angularVel * _math.sin(obj.heading)
        newAngularVel = _utils.scalarToCarlaVector3D(xAngularVel, yAngularVel)
        obj.carlaActor.set_angular_velocity(newAngularVel)


class SetTransformAction(Action):  # TODO eliminate
    def __init__(self, pos, heading):
        self.pos = pos
        self.heading = heading

    def applyTo(self, obj, sim):
        loc = _utils.scenicToCarlaLocation(self.pos, z=obj.elevation)
        rot = _utils.scenicToCarlaRotation(self.heading)
        transform = _carla.Transform(loc, rot)
        obj.carlaActor.set_transform(transform)


#############################################
# Actions specific to carla.Vehicle objects #
#############################################


class _CarlaVehicle:
    # Mixin identifying CARLA vehicles.
    # Used to avoid importing the Vehicle class from the CARLA model, which is
    # a Scenic module not importable from Python.
    pass


class VehicleAction(Action):
    def canBeTakenBy(self, agent):
        return isinstance(agent, _CarlaVehicle)


class SetManualGearShiftAction(VehicleAction):
    def __init__(self, manualGearShift):
        if not isinstance(manualGearShift, bool):
            raise RuntimeError("Manual gear shift must be a boolean.")
        self.manualGearShift = manualGearShift

    def applyTo(self, obj, sim):
        vehicle = obj.carlaActor
        ctrl = vehicle.get_control()
        ctrl.manual_gear_shift = self.manualGearShift
        vehicle.apply_control(ctrl)


class SetGearAction(VehicleAction):
    def __init__(self, gear):
        if not isinstance(gear, int):
            raise RuntimeError("Gear must be an int.")
        self.gear = gear

    def applyTo(self, obj, sim):
        vehicle = obj.carlaActor
        ctrl = vehicle.get_control()
        ctrl.gear = self.gear
        vehicle.apply_control(ctrl)


class SetManualFirstGearShiftAction(VehicleAction):  # TODO eliminate
    def applyTo(self, obj, sim):
        ctrl = _carla.VehicleControl(manual_gear_shift=True, gear=1)
        obj.carlaActor.apply_control(ctrl)


class SetTrafficLightAction(VehicleAction):
    """Set the traffic light to desired color. It will only take
    effect if the car is within a given distance of the traffic light.

    Arguments:
        color: the string red/yellow/green/off/unknown
        distance: the maximum distance to search for traffic lights from the current position
    """

    def __init__(self, color, distance=100, group=False):
        self.color = _utils.scenicToCarlaTrafficLightStatus(color)
        if color is None:
            raise RuntimeError("Color must be red/yellow/green/off/unknown.")
        self.distance = distance

    def applyTo(self, obj, sim):
        traffic_light = obj._getClosestTrafficLight(self.distance)
        if traffic_light is not None:
            traffic_light.set_state(self.color)


class SetAutopilotAction(VehicleAction):
    def __init__(self, enabled):
        if not isinstance(enabled, bool):
            raise RuntimeError("Enabled must be a boolean.")
        self.enabled = enabled

    def applyTo(self, obj, sim):
        vehicle = obj.carlaActor
        vehicle.set_autopilot(self.enabled, sim.tm.get_port())


class SetVehicleLightStateAction(VehicleAction):
    """Set the vehicle lights' states.

    Arguments:
        vehicleLightState: Which lights are on.
    """

    def __init__(self, vehicleLightState):
        self.vehicleLightState = vehicleLightState

    def applyTo(self, obj, sim):
        obj.carlaActor.set_light_state(self.vehicleLightState)


#################################################
# Actions available to all carla.Walker objects #
#################################################


class _CarlaPedestrian:
    # Mixin identifying CARLA pedestrians. (see _CarlaVehicle)
    pass


class PedestrianAction(Action):
    def canBeTakenBy(self, agent):
        return isinstance(agent, _CarlaPedestrian)


class SetJumpAction(PedestrianAction):
    def __init__(self, jump):
        if not isinstance(jump, bool):
            raise RuntimeError("Jump must be a boolean.")
        self.jump = jump

    def applyTo(self, obj, sim):
        walker = obj.carlaActor
        ctrl = walker.get_control()
        ctrl.jump = self.jump
        walker.apply_control(ctrl)


class SetWalkAction(PedestrianAction):
    def __init__(self, enabled, maxSpeed=1.4):
        if not isinstance(enabled, bool):
            raise RuntimeError("Enabled must be a boolean.")
        self.enabled = enabled
        self.maxSpeed = maxSpeed

    def applyTo(self, obj, sim):
        controller = obj.carlaController
        if self.enabled:
            controller.start()
            controller.go_to_location(sim.world.get_random_location_from_navigation())
            controller.set_max_speed(self.maxSpeed)
        else:
            controller.stop()


class TrackWaypointsAction(Action):
    def __init__(self, waypoints, cruising_speed=10):
        self.waypoints = np.array(waypoints)
        self.curr_index = 1
        self.cruising_speed = cruising_speed

    def canBeTakenBy(self, agent):
        # return agent.lgsvlAgentType is lgsvl.AgentType.EGO
        return True

    def LQR(v_target, wheelbase, Q, R):
        A = np.matrix([[0, v_target * (5.0 / 18.0)], [0, 0]])
        B = np.matrix([[0], [(v_target / wheelbase) * (5.0 / 18.0)]])
        V = np.matrix(linalg.solve_continuous_are(A, B, Q, R))
        K = np.matrix(linalg.inv(R) * (B.T * V))
        return K

    def applyTo(self, obj, sim):
        carlaObj = obj.carlaActor
        transform = carlaObj.get_transform()
        pos = transform.location
        rot = transform.rotation
        velocity = carlaObj.get_velocity()
        th, x, y, v = (
            rot.y / 180.0 * np.pi,
            pos.x,
            pos.z,
            (velocity.x**2 + velocity.z**2) ** 0.5,
        )
        # print('state:', th, x, y, v)
        PREDICTIVE_LENGTH = 3
        MIN_SPEED = 1
        WHEEL_BASE = 3
        v = max(MIN_SPEED, v)

        x = x + PREDICTIVE_LENGTH * np.cos(-th + np.pi / 2)
        y = y + PREDICTIVE_LENGTH * np.sin(-th + np.pi / 2)
        # print('car front:', x, y)
        dists = np.linalg.norm(self.waypoints - np.array([x, y]), axis=1)
        dist_pos = np.argpartition(dists, 1)
        index = dist_pos[0]
        if index > self.curr_index and index < len(self.waypoints) - 1:
            self.curr_index = index
        p1, p2, p3 = (
            self.waypoints[self.curr_index - 1],
            self.waypoints[self.curr_index],
            self.waypoints[self.curr_index + 1],
        )

        p1_a = np.linalg.norm(p1 - np.array([x, y]))
        p3_a = np.linalg.norm(p3 - np.array([x, y]))
        p1_p2 = np.linalg.norm(p1 - p2)
        p3_p2 = np.linalg.norm(p3 - p2)

        if p1_a - p1_p2 > p3_a - p3_p2:
            p1 = p2
            p2 = p3

        # print('points:',p1, p2)
        x1, y1, x2, y2 = p1[0], p1[1], p2[0], p2[1]
        th_n = -math.atan2(y2 - y1, x2 - x1) + np.pi / 2
        d_th = (th - th_n + 3 * np.pi) % (2 * np.pi) - np.pi
        d_x = (x2 - x1) * y - (y2 - y1) * x + y2 * x1 - y1 * x2
        d_x /= np.linalg.norm(np.array([x1, y1]) - np.array([x2, y2]))
        # print('d_th, d_x:',d_th, d_x)

        K = TrackWaypoints.LQR(
            v, WHEEL_BASE, np.array([[1, 0], [0, 3]]), np.array([[10]])
        )
        u = -K * np.matrix([[-d_x], [d_th]])
        u = np.double(u)
        u_steering = min(max(u, -1), 1)

        K = 1
        u = -K * (v - self.cruising_speed)
        u_thrust = min(max(u, -1), 1)

        # print('u:', u_thrust, u_steering)

        ctrl = carlaObj.get_control()
        ctrl.steering = u_steering
        if u_thrust > 0:
            ctrl.throttle = u_thrust
        elif u_thrust < 0.1:
            ctrl.braking = -u_thrust
        carlaObj.apply_control(ctrl)
