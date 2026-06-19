#!/usr/bin/env python3

import math
import time
import threading

import carla
import py_trees

from srunner.scenarios.basic_scenario import BasicScenario
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest


HERO_TARGET_SPEED_KMH = 80.0
LEAD_TARGET_SPEED_KMH = 60.0
BRAKE_TRIGGER_DISTANCE_M = 35.0

SCENARIO_DURATION_S = 120.0
CONTROL_DT = 0.05

STEER_LOOKAHEAD_M = 12.0
MAX_STEER = 0.55

SPEED_KP = 0.075
SPEED_KI = 0.010
SPEED_KD = 0.004

STEER_KP = 1.30
STEER_KI = 0.0
STEER_KD = 0.10

HERO_EMERGENCY_DISTANCE_M = 18.0
HERO_CAUTION_DISTANCE_M = 32.0

SPECTATOR_DISTANCE = 18.0
SPECTATOR_HEIGHT = 8.0

# Town04 straight segment found earlier.
HERO_X = 65.346
HERO_Y = 9.848
HERO_Z = 11.993
HERO_YAW_DEG = -179.767349

LEAD_X = 20.346
LEAD_Y = 9.665
LEAD_Z = 12.000
LEAD_YAW_DEG = -179.767349

HERO_MODEL = "vehicle.tesla.model3"
LEAD_MODEL = "vehicle.audi.etron"


class PIDState:
    def __init__(self):
        self.integral = 0.0
        self.previous_error = 0.0
        self.initialized = False


def clamp(value, low, high):
    return max(low, min(high, value))


def normalize_angle_rad(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def get_speed_kmh(actor):
    velocity = actor.get_velocity()
    speed_ms = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
    return 3.6 * speed_ms


def make_transform(x, y, z, yaw_deg):
    return carla.Transform(
        carla.Location(x=x, y=y, z=z),
        carla.Rotation(pitch=0.0, yaw=yaw_deg, roll=0.0),
    )


def set_actor_velocity_kmh(actor, speed_kmh, yaw_deg):
    """
    Set an approximate initial world velocity for a CARLA actor.

    This makes the scenario start with moving vehicles instead of waiting
    for both cars to accelerate from rest.
    """
    speed_ms = speed_kmh / 3.6
    yaw_rad = math.radians(yaw_deg)

    velocity = carla.Vector3D(
        x=speed_ms * math.cos(yaw_rad),
        y=speed_ms * math.sin(yaw_rad),
        z=0.0,
    )

    actor.set_target_velocity(velocity)


def pid_step(error, dt, state, kp, ki, kd):
    if not state.initialized:
        state.previous_error = error
        state.initialized = True

    state.integral += error * dt
    derivative = (error - state.previous_error) / max(dt, 1e-6)
    state.previous_error = error

    return kp * error + ki * state.integral + kd * derivative


class RearEndBrakingBehavior(py_trees.behaviour.Behaviour):
    """
    Main behavior:
    - hero tries to reach 80 km/h;
    - lead_vehicle cruises at 60 km/h;
    - when distance < 35 m, lead_vehicle brakes hard;
    - hero reacts with caution/emergency braking.
    """

    def __init__(self, world, hero, lead):
        super().__init__("RearEndBrakingBehavior")

        self.world = world
        self.world_map = world.get_map()
        self.hero = hero
        self.lead = lead

        self.speed_pid = PIDState()
        self.steer_pid = PIDState()

        self.start_time = None
        self.previous_time = None
        self.trigger_has_fired = False
        self.min_distance = float("inf")
        self.trigger_time = None

    def initialise(self):
        self.start_time = time.time()
        self.previous_time = self.start_time
        self.trigger_has_fired = False
        self.min_distance = float("inf")
        self.trigger_time = None

        print("")
        print("[Town04RearEndBraking] Scenario behavior started.")
        print("[Town04RearEndBraking] hero target speed: 80 km/h")
        print("[Town04RearEndBraking] lead target speed: 60 km/h")
        print("[Town04RearEndBraking] brake trigger distance: 35 m")
        print("")

    def update(self):
        now = time.time()
        elapsed_s = now - self.start_time
        dt = max(now - self.previous_time, CONTROL_DT)
        self.previous_time = now

        if elapsed_s > SCENARIO_DURATION_S:
            print("")
            print("[Town04RearEndBraking] Scenario duration reached.")
            print(f"[Town04RearEndBraking] Minimum distance: {self.min_distance:.2f} m")
            return py_trees.common.Status.SUCCESS

        hero_loc = self.hero.get_location()
        lead_loc = self.lead.get_location()
        distance = hero_loc.distance(lead_loc)

        self.min_distance = min(self.min_distance, distance)

        if distance <= BRAKE_TRIGGER_DISTANCE_M and not self.trigger_has_fired:
            self.trigger_has_fired = True
            self.trigger_time = elapsed_s
            print("")
            print(f"[Town04RearEndBraking] TRIGGER FIRED at t={elapsed_s:.2f}s, distance={distance:.2f} m")

        lead_status = self._control_lead(distance)
        hero_status = self._control_hero(distance, dt)
        self._move_spectator()

        hero_speed = get_speed_kmh(self.hero)
        lead_speed = get_speed_kmh(self.lead)
        trigger_text = "TRIGGERED" if self.trigger_has_fired else "waiting"

        print(
            f"\r[Town04RearEndBraking] "
            f"t={elapsed_s:5.1f}s | "
            f"dist={distance:5.1f} m | "
            f"hero={hero_speed:5.1f} km/h | "
            f"lead={lead_speed:5.1f} km/h | "
            f"trigger={trigger_text} | "
            f"{lead_status} | {hero_status}",
            end="",
            flush=True,
        )

        return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        print("")
        print("[Town04RearEndBraking] Behavior terminated.")
        print(f"[Town04RearEndBraking] Final status: {new_status}")
        print(f"[Town04RearEndBraking] Minimum distance: {self.min_distance:.2f} m")
        if self.trigger_time is not None:
            print(f"[Town04RearEndBraking] Trigger time: {self.trigger_time:.2f} s")

    def _control_lead(self, distance_to_hero_m):
        speed_kmh = get_speed_kmh(self.lead)

        if distance_to_hero_m > BRAKE_TRIGGER_DISTANCE_M:
            target_speed = LEAD_TARGET_SPEED_KMH

            if speed_kmh < target_speed - 5.0:
                self.lead.apply_control(carla.VehicleControl(throttle=0.85, brake=0.0, steer=0.0))
            elif speed_kmh < target_speed - 2.0:
                self.lead.apply_control(carla.VehicleControl(throttle=0.55, brake=0.0, steer=0.0))
            elif speed_kmh > target_speed + 3.0:
                self.lead.apply_control(carla.VehicleControl(throttle=0.0, brake=0.20, steer=0.0))
            else:
                self.lead.apply_control(carla.VehicleControl(throttle=0.25, brake=0.0, steer=0.0))

            return "lead cruising at 60 km/h"

        self.lead.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0))
        return "lead braking"

    def _compute_hero_speed_target(self, distance_to_lead_m):
        if distance_to_lead_m < HERO_EMERGENCY_DISTANCE_M:
            return 0.0

        if distance_to_lead_m < HERO_CAUTION_DISTANCE_M:
            ratio = (distance_to_lead_m - HERO_EMERGENCY_DISTANCE_M) / (
                HERO_CAUTION_DISTANCE_M - HERO_EMERGENCY_DISTANCE_M
            )
            return clamp(ratio, 0.0, 1.0) * 35.0

        return HERO_TARGET_SPEED_KMH

    def _control_hero(self, distance_to_lead_m, dt):
        current_speed = get_speed_kmh(self.hero)
        target_speed = self._compute_hero_speed_target(distance_to_lead_m)

        speed_error = target_speed - current_speed

        accel_cmd = pid_step(
            error=speed_error,
            dt=dt,
            state=self.speed_pid,
            kp=SPEED_KP,
            ki=SPEED_KI,
            kd=SPEED_KD,
        )

        steer = self._compute_steering(dt)

        if distance_to_lead_m < HERO_EMERGENCY_DISTANCE_M:
            throttle = 0.0
            brake = 1.0
            status = "hero emergency braking"
        elif accel_cmd >= 0.0:
            throttle = clamp(accel_cmd, 0.0, 0.75)
            brake = 0.0
            status = "hero cruising"
        else:
            throttle = 0.0
            brake = clamp(-accel_cmd, 0.0, 0.8)
            status = "hero slowing"

        self.hero.apply_control(
            carla.VehicleControl(
                throttle=throttle,
                brake=brake,
                steer=steer,
                hand_brake=False,
                reverse=False,
            )
        )

        return status

    def _compute_steering(self, dt):
        hero_location = self.hero.get_location()

        current_wp = self.world_map.get_waypoint(
            hero_location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )

        if current_wp is None:
            return 0.0

        next_wps = current_wp.next(STEER_LOOKAHEAD_M)
        target_wp = next_wps[0] if next_wps else current_wp

        hero_tf = self.hero.get_transform()
        hero_loc = hero_tf.location
        target_loc = target_wp.transform.location

        dx = target_loc.x - hero_loc.x
        dy = target_loc.y - hero_loc.y

        target_yaw = math.atan2(dy, dx)
        hero_yaw = math.radians(hero_tf.rotation.yaw)

        heading_error = normalize_angle_rad(target_yaw - hero_yaw)

        steer = pid_step(
            error=heading_error,
            dt=dt,
            state=self.steer_pid,
            kp=STEER_KP,
            ki=STEER_KI,
            kd=STEER_KD,
        )

        return clamp(steer, -MAX_STEER, MAX_STEER)

    def _move_spectator(self):
        spectator = self.world.get_spectator()

        hero_tf = self.hero.get_transform()
        yaw_rad = math.radians(hero_tf.rotation.yaw)

        backward = carla.Vector3D(
            x=-math.cos(yaw_rad),
            y=-math.sin(yaw_rad),
            z=0.0,
        )

        camera_location = hero_tf.location + carla.Location(
            x=backward.x * SPECTATOR_DISTANCE,
            y=backward.y * SPECTATOR_DISTANCE,
            z=SPECTATOR_HEIGHT,
        )

        camera_rotation = carla.Rotation(
            pitch=-20.0,
            yaw=hero_tf.rotation.yaw,
            roll=0.0,
        )

        spectator.set_transform(carla.Transform(camera_location, camera_rotation))



class KeepRunning(py_trees.behaviour.Behaviour):
    """
    Keeps Scenario Runner alive while the vehicle-control thread runs.

    It returns SUCCESS when the internal scenario logic says the episode
    is complete, for example: lead braked, ego stopped, no collision.
    """

    def __init__(self, scenario_ref):
        super().__init__("KeepRunning")
        self.scenario_ref = scenario_ref

    def update(self):
        if getattr(self.scenario_ref, "_scenario_completed", False):
            return py_trees.common.Status.SUCCESS

        return py_trees.common.Status.RUNNING


class Town04RearEndBraking(BasicScenario):
    """
    Native Scenario Runner implementation of the Town04 rear-end braking scenario.
    """

    timeout = SCENARIO_DURATION_S

    def __init__(
        self,
        world,
        ego_vehicles,
        config,
        randomize=False,
        debug_mode=False,
        criteria_enable=True,
        timeout=SCENARIO_DURATION_S,
    ):
        self.timeout = timeout
        self._lead_vehicle = None
        self._control_thread = None
        self._stop_thread = threading.Event()
        self._thread_behavior = None
        self._scenario_completed = False
        self._stopped_since = None

        super(Town04RearEndBraking, self).__init__(
            "Town04RearEndBraking",
            ego_vehicles,
            config,
            world,
            debug_mode,
            criteria_enable=criteria_enable,
        )

    def _initialize_actors(self, config):
        """
        Spawn both vehicles directly.

        This avoids relying on XML actor parsing because the previous version
        loaded the scenario but did not make the vehicles appear reliably.
        """
        print("[Town04RearEndBraking] Initializing actors directly from Python scenario.")

        hero_transform = make_transform(HERO_X, HERO_Y, HERO_Z, HERO_YAW_DEG)
        lead_transform = make_transform(LEAD_X, LEAD_Y, LEAD_Z, LEAD_YAW_DEG)

        # If Scenario Runner already provided an ego vehicle, use it.
        # Otherwise, spawn hero directly.
        if self.ego_vehicles:
            hero = self.ego_vehicles[0]
            hero.set_transform(hero_transform)
            print(
                f"[Town04RearEndBraking] Using existing ego vehicle: "
                f"id={hero.id}, type={hero.type_id}, role={hero.attributes.get('role_name', 'SEM_ROLE')}"
            )
        else:
            hero = CarlaDataProvider.request_new_actor(
                HERO_MODEL,
                hero_transform,
                rolename="hero",
            )

            if hero is None:
                raise RuntimeError("Failed to spawn hero vehicle.")

            self.ego_vehicles.append(hero)
            print(
                f"[Town04RearEndBraking] Spawned hero directly: "
                f"id={hero.id}, type={hero.type_id}, role={hero.attributes.get('role_name', 'SEM_ROLE')}"
            )

        self._lead_vehicle = CarlaDataProvider.request_new_actor(
            LEAD_MODEL,
            lead_transform,
            rolename="lead_vehicle",
        )

        if self._lead_vehicle is None:
            raise RuntimeError("Failed to spawn lead_vehicle.")

        self.other_actors.append(self._lead_vehicle)

        print(
            f"[Town04RearEndBraking] Spawned lead_vehicle directly: "
            f"id={self._lead_vehicle.id}, type={self._lead_vehicle.type_id}, "
            f"role={self._lead_vehicle.attributes.get('role_name', 'SEM_ROLE')}"
        )

        weather = carla.WeatherParameters(
            cloudiness=100.0,
            precipitation=80.0,
            precipitation_deposits=80.0,
            wind_intensity=30.0,
            sun_azimuth_angle=0.0,
            sun_altitude_angle=-10.0,
            fog_density=20.0,
            fog_distance=120.0,
            wetness=80.0,
        )

        self.world.set_weather(weather)
        print("[Town04RearEndBraking] Weather set: rainy/night-like condition.")

        # Start the scenario with both vehicles already moving.
        # This is closer to the intended Option B:
        # hero at 80 km/h behind, lead_vehicle at 60 km/h ahead.
        set_actor_velocity_kmh(self.ego_vehicles[0], HERO_TARGET_SPEED_KMH, HERO_YAW_DEG)
        set_actor_velocity_kmh(self._lead_vehicle, LEAD_TARGET_SPEED_KMH, LEAD_YAW_DEG)

        print(
            f"[Town04RearEndBraking] Initial velocity set: "
            f"hero={HERO_TARGET_SPEED_KMH:.1f} km/h, "
            f"lead_vehicle={LEAD_TARGET_SPEED_KMH:.1f} km/h"
        )

        self._start_control_thread()

    def _start_control_thread(self):
        print("[Town04RearEndBraking] Starting vehicle control thread.")

        self._stop_thread.clear()
        self._thread_behavior = RearEndBrakingBehavior(
            world=self.world,
            hero=self.ego_vehicles[0],
            lead=self._lead_vehicle,
        )

        self._control_thread = threading.Thread(
            target=self._control_loop,
            name="Town04RearEndBrakingControlThread",
            daemon=True,
        )
        self._control_thread.start()

    def _control_loop(self):
        print("[Town04RearEndBraking] Control loop started.")

        self._thread_behavior.initialise()

        while not self._stop_thread.is_set():
            status = self._thread_behavior.update()

            if status != py_trees.common.Status.RUNNING:
                print(f"\n[Town04RearEndBraking] Control loop ended with status: {status}")
                break

            # Automatic success condition:
            # after the lead vehicle has braked, both vehicles are stopped,
            # no collision occurred, and a safe residual gap remains.
            try:
                hero_actor_for_stop = self.ego_vehicles[0]
                lead_actor_for_stop = self._lead_vehicle

                if (
                    hero_actor_for_stop is None
                    or lead_actor_for_stop is None
                    or not hero_actor_for_stop.is_alive
                    or not lead_actor_for_stop.is_alive
                ):
                    print("")
                    print("[Town04RearEndBraking] Actor destroyed or unavailable. Stopping control loop.")
                    break

                current_distance = hero_actor_for_stop.get_location().distance(
                    lead_actor_for_stop.get_location()
                )

                hero_speed_for_stop = get_speed_kmh(hero_actor_for_stop)
                lead_speed_for_stop = get_speed_kmh(lead_actor_for_stop)

            except RuntimeError as exc:
                print("")
                print(f"[Town04RearEndBraking] RuntimeError while checking stop condition: {exc}")
                break

            if (
                self._thread_behavior.trigger_has_fired
                and current_distance > 3.0
                and hero_speed_for_stop < 1.0
                and lead_speed_for_stop < 1.0
            ):
                if self._stopped_since is None:
                    self._stopped_since = time.time()

                stopped_duration = time.time() - self._stopped_since

                if stopped_duration >= 2.0:
                    print("")
                    print(
                        f"[Town04RearEndBraking] SUCCESS condition reached: "
                        f"both vehicles stopped for {stopped_duration:.1f}s, "
                        f"final distance={current_distance:.2f} m"
                    )
                    self._scenario_completed = True
                    break
            else:
                self._stopped_since = None

            time.sleep(CONTROL_DT)

        self._thread_behavior.terminate(py_trees.common.Status.SUCCESS if self._scenario_completed else py_trees.common.Status.INVALID)
        print("[Town04RearEndBraking] Control loop stopped.")

    def _stop_control_thread(self):
        if self._control_thread is not None and self._control_thread.is_alive():
            print("[Town04RearEndBraking] Stopping vehicle control thread.")
            self._stop_thread.set()
            self._control_thread.join(timeout=2.0)

    def remove_all_actors(self):
        self._stop_control_thread()
        super(Town04RearEndBraking, self).remove_all_actors()

    def _create_behavior(self):
        return KeepRunning(self)

    def _create_test_criteria(self):
        criteria = []

        collision_criterion = CollisionTest(
            self.ego_vehicles[0],
            terminate_on_failure=True,
        )

        criteria.append(collision_criterion)

        return criteria
