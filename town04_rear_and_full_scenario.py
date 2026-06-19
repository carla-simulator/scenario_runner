#!/usr/bin/env python3
"""
Full Python CARLA rear-end collision-risk scenario in Town04.

This script replaces:
- town04_rear_end_python_scenario.py
- pid_ego_agent.py
- follow_ego_camera.py

Scenario logic:
- Town04
- rainy/night-like condition
- hero starts behind, target speed = 80 km/h
- lead_vehicle starts ahead, target speed = 60 km/h
- when hero gets closer than 35 m, lead_vehicle brakes hard
- hero is controlled by internal PID lane-following controller
- spectator camera follows hero
"""

import math
import signal
import sys
import time
from dataclasses import dataclass
from typing import Optional

import carla


HOST = "127.0.0.1"
PORT = 2000
TARGET_TOWN = "Town04"

HERO_BLUEPRINT = "vehicle.tesla.model3"
LEAD_BLUEPRINT = "vehicle.audi.etron"

HERO_ROLE = "hero"
LEAD_ROLE = "lead_vehicle"

# Town04 straight segment found earlier.
# Direction is approximately toward negative X.
HERO_X = 65.346
HERO_Y = 9.848
HERO_Z = 11.993
HERO_YAW_DEG = -179.767349

LEAD_X = 20.346
LEAD_Y = 9.665
LEAD_Z = 12.000
LEAD_YAW_DEG = -179.767349

# Option B: physically intuitive rear-end risk.
HERO_TARGET_SPEED_KMH = 80.0
LEAD_TARGET_SPEED_KMH = 60.0
BRAKE_TRIGGER_DISTANCE_M = 35.0

SCENARIO_DURATION_S = 120.0
CONTROL_DT = 0.05

# Hero controller.
STEER_LOOKAHEAD_M = 12.0
MAX_STEER = 0.55

SPEED_KP = 0.075
SPEED_KI = 0.010
SPEED_KD = 0.004

STEER_KP = 1.30
STEER_KI = 0.000
STEER_KD = 0.10

# Emergency behavior for hero after lead brakes.
HERO_EMERGENCY_DISTANCE_M = 14.0
HERO_CAUTION_DISTANCE_M = 25.0

# Camera.
SPECTATOR_DISTANCE = 18.0
SPECTATOR_HEIGHT = 8.0


stop_requested = False


@dataclass
class PIDState:
    integral: float = 0.0
    previous_error: float = 0.0
    initialized: bool = False


@dataclass
class ScenarioActors:
    hero: carla.Actor
    lead: carla.Actor


def signal_handler(_signum, _frame):
    global stop_requested
    stop_requested = True


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def normalize_angle_rad(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def get_speed_kmh(actor: carla.Actor) -> float:
    velocity = actor.get_velocity()
    speed_ms = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
    return 3.6 * speed_ms


def make_transform(x: float, y: float, z: float, yaw_deg: float) -> carla.Transform:
    return carla.Transform(
        carla.Location(x=x, y=y, z=z),
        carla.Rotation(pitch=0.0, yaw=yaw_deg, roll=0.0),
    )


def setup_world(client: carla.Client) -> carla.World:
    world = client.get_world()
    current_map = world.get_map().name
    print(f"[scenario] Current map: {current_map}")

    if not current_map.endswith(TARGET_TOWN):
        print(f"[scenario] Loading {TARGET_TOWN}...")
        world = client.load_world(TARGET_TOWN)
        time.sleep(3.0)
        print(f"[scenario] Loaded map: {world.get_map().name}")

    return world


def destroy_old_test_vehicles(world: carla.World) -> None:
    roles_to_destroy = {"hero", "Ego", "LeadVehicle", "lead_vehicle"}

    old_actors = []
    for actor in world.get_actors().filter("vehicle.*"):
        role_name = actor.attributes.get("role_name", "")
        if role_name in roles_to_destroy:
            old_actors.append(actor)

    if not old_actors:
        print("[scenario] No old test vehicles to destroy.")
        return

    print(f"[scenario] Destroying {len(old_actors)} old test vehicle(s).")
    for actor in old_actors:
        print(
            f"[scenario] Destroying id={actor.id}, "
            f"type={actor.type_id}, role={actor.attributes.get('role_name', 'SEM_ROLE')}"
        )
        actor.destroy()

    time.sleep(1.0)


def set_weather(world: carla.World) -> None:
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
    world.set_weather(weather)
    print("[scenario] Weather set: rainy/night-like condition.")


def spawn_vehicle(
    world: carla.World,
    blueprint_id: str,
    role_name: str,
    transform: carla.Transform,
    color: str,
) -> carla.Actor:
    blueprint_library = world.get_blueprint_library()
    blueprint = blueprint_library.find(blueprint_id)

    if blueprint.has_attribute("role_name"):
        blueprint.set_attribute("role_name", role_name)

    if blueprint.has_attribute("color"):
        blueprint.set_attribute("color", color)

    actor = world.try_spawn_actor(blueprint, transform)

    if actor is None:
        raise RuntimeError(
            f"CARLA refused to spawn {role_name} at {transform.location}."
        )

    print(
        f"[scenario] Spawned {role_name}: id={actor.id}, "
        f"type={actor.type_id}, location={actor.get_location()}"
    )

    return actor


def spawn_scenario_actors(world: carla.World) -> ScenarioActors:
    hero_tf = make_transform(HERO_X, HERO_Y, HERO_Z, HERO_YAW_DEG)
    lead_tf = make_transform(LEAD_X, LEAD_Y, LEAD_Z, LEAD_YAW_DEG)

    hero = spawn_vehicle(
        world=world,
        blueprint_id=HERO_BLUEPRINT,
        role_name=HERO_ROLE,
        transform=hero_tf,
        color="255,0,0",
    )

    lead = spawn_vehicle(
        world=world,
        blueprint_id=LEAD_BLUEPRINT,
        role_name=LEAD_ROLE,
        transform=lead_tf,
        color="0,0,255",
    )

    time.sleep(1.0)

    return ScenarioActors(hero=hero, lead=lead)


def print_vehicle_summary(world: carla.World) -> None:
    vehicles = list(world.get_actors().filter("vehicle.*"))

    print("[scenario] Current vehicles:")
    for actor in vehicles:
        print(
            f"  id={actor.id}, type={actor.type_id}, "
            f"role={actor.attributes.get('role_name', 'SEM_ROLE')}, "
            f"location={actor.get_location()}"
        )


def pid_step(error: float, dt: float, state: PIDState, kp: float, ki: float, kd: float) -> float:
    if not state.initialized:
        state.previous_error = error
        state.initialized = True

    state.integral += error * dt
    derivative = (error - state.previous_error) / max(dt, 1e-6)
    state.previous_error = error

    return kp * error + ki * state.integral + kd * derivative


def get_target_waypoint(world_map: carla.Map, hero: carla.Actor) -> Optional[carla.Waypoint]:
    hero_location = hero.get_location()
    current_wp = world_map.get_waypoint(
        hero_location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )

    if current_wp is None:
        return None

    next_wps = current_wp.next(STEER_LOOKAHEAD_M)
    if not next_wps:
        return current_wp

    return next_wps[0]


def compute_steering_control(
    world_map: carla.Map,
    hero: carla.Actor,
    dt: float,
    steer_pid: PIDState,
) -> float:
    target_wp = get_target_waypoint(world_map, hero)
    if target_wp is None:
        return 0.0

    hero_tf = hero.get_transform()
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
        state=steer_pid,
        kp=STEER_KP,
        ki=STEER_KI,
        kd=STEER_KD,
    )

    return clamp(steer, -MAX_STEER, MAX_STEER)


def compute_hero_speed_target(distance_to_lead_m: float) -> float:
    """
    Hero normally tries 80 km/h.

    After the lead brakes, if the distance becomes dangerous,
    the controller reduces the target speed or brakes hard.
    """
    if distance_to_lead_m < HERO_EMERGENCY_DISTANCE_M:
        return 0.0

    if distance_to_lead_m < HERO_CAUTION_DISTANCE_M:
        # Linear slowdown between 14 m and 25 m.
        ratio = (distance_to_lead_m - HERO_EMERGENCY_DISTANCE_M) / (
            HERO_CAUTION_DISTANCE_M - HERO_EMERGENCY_DISTANCE_M
        )
        return clamp(ratio, 0.0, 1.0) * 35.0

    return HERO_TARGET_SPEED_KMH


def control_hero(
    world_map: carla.Map,
    hero: carla.Actor,
    distance_to_lead_m: float,
    dt: float,
    speed_pid: PIDState,
    steer_pid: PIDState,
) -> str:
    current_speed = get_speed_kmh(hero)
    target_speed = compute_hero_speed_target(distance_to_lead_m)

    speed_error = target_speed - current_speed

    accel_cmd = pid_step(
        error=speed_error,
        dt=dt,
        state=speed_pid,
        kp=SPEED_KP,
        ki=SPEED_KI,
        kd=SPEED_KD,
    )

    steer = compute_steering_control(world_map, hero, dt, steer_pid)

    if distance_to_lead_m < HERO_EMERGENCY_DISTANCE_M:
        throttle = 0.0
        brake = 1.0
        status = "hero emergency braking"
    elif accel_cmd >= 0.0:
        throttle = clamp(accel_cmd, 0.0, 0.75)
        brake = 0.0
        status = "hero cruising toward 80 km/h"
    else:
        throttle = 0.0
        brake = clamp(-accel_cmd, 0.0, 0.8)
        status = "hero slowing"

    hero.apply_control(
        carla.VehicleControl(
            throttle=throttle,
            brake=brake,
            steer=steer,
            hand_brake=False,
            reverse=False,
        )
    )

    return status


def control_lead_vehicle(lead: carla.Actor, distance_to_hero_m: float) -> str:
    speed_kmh = get_speed_kmh(lead)

    if distance_to_hero_m > BRAKE_TRIGGER_DISTANCE_M:
        target_speed = LEAD_TARGET_SPEED_KMH

        if speed_kmh < target_speed - 2.0:
            lead.apply_control(carla.VehicleControl(throttle=0.55, brake=0.0, steer=0.0))
        elif speed_kmh > target_speed + 2.0:
            lead.apply_control(carla.VehicleControl(throttle=0.0, brake=0.15, steer=0.0))
        else:
            lead.apply_control(carla.VehicleControl(throttle=0.25, brake=0.0, steer=0.0))

        return "lead cruising at 60 km/h"

    lead.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0))
    return "lead braking: hero closer than 35 m"


def move_spectator(world: carla.World, hero: carla.Actor) -> None:
    spectator = world.get_spectator()

    hero_tf = hero.get_transform()
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


def main() -> int:
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    client = carla.Client(HOST, PORT)
    client.set_timeout(120.0)

    world = setup_world(client)
    world_map = world.get_map()

    destroy_old_test_vehicles(world)
    set_weather(world)

    actors = spawn_scenario_actors(world)
    print_vehicle_summary(world)

    speed_pid = PIDState()
    steer_pid = PIDState()

    print("")
    print("[scenario] Full scenario is running.")
    print("[scenario] Do NOT run pid_ego_agent.py separately.")
    print("[scenario] Do NOT run follow_ego_camera.py separately.")
    print("[scenario] This script controls hero, lead_vehicle, and camera.")
    print("[scenario] Press Ctrl+C when finished.")
    print("")

    start_time = time.time()
    previous_time = start_time
    trigger_has_fired = False

    try:
        while not stop_requested:
            now = time.time()
            elapsed_s = now - start_time
            dt = max(now - previous_time, CONTROL_DT)
            previous_time = now

            if elapsed_s > SCENARIO_DURATION_S:
                print("\n[scenario] Scenario duration reached.")
                break

            hero_loc = actors.hero.get_location()
            lead_loc = actors.lead.get_location()
            distance = hero_loc.distance(lead_loc)

            if distance <= BRAKE_TRIGGER_DISTANCE_M:
                trigger_has_fired = True

            lead_status = control_lead_vehicle(actors.lead, distance)
            hero_status = control_hero(
                world_map=world_map,
                hero=actors.hero,
                distance_to_lead_m=distance,
                dt=dt,
                speed_pid=speed_pid,
                steer_pid=steer_pid,
            )

            move_spectator(world, actors.hero)

            hero_speed = get_speed_kmh(actors.hero)
            lead_speed = get_speed_kmh(actors.lead)

            trigger_text = "TRIGGERED" if trigger_has_fired else "waiting"

            print(
                f"\r[scenario] t={elapsed_s:5.1f}s | "
                f"dist={distance:5.1f} m | "
                f"hero={hero_speed:5.1f} km/h | "
                f"lead={lead_speed:5.1f} km/h | "
                f"trigger={trigger_text} | "
                f"{lead_status} | {hero_status}",
                end="",
                flush=True,
            )

            time.sleep(CONTROL_DT)

    finally:
        print("\n[scenario] Stopping. Destroying scenario vehicles.")

        for actor in [actors.hero, actors.lead]:
            try:
                if actor.is_alive:
                    actor.destroy()
            except RuntimeError:
                pass

    return 0


if __name__ == "__main__":
    sys.exit(main())