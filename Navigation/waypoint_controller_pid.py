#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Allows controlling a vehicle with a keyboard. For a simpler and more
# documented example, please take a look at tutorial.py.

"""

Use ARROWS or WASD keys for control.

    W            : throttle
    S            : brake
    AD           : steer
    Q            : toggle reverse
    Space        : hand-brake
    P            : toggle autopilot

    TAB          : change sensor position
    `            : next sensor
    [1-9]        : change to sensor [1-9]
    C            : change weather (Shift+C reverse)
    Backspace    : change vehicle

    R            : toggle recording images to disk

    H/?          : toggle help
    ESC          : quit
"""
from __future__ import print_function


# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================



from collections import deque

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================


import carla

from carla import ColorConverter as cc

import argparse
import logging
import random
import re
import weakref
import pdb
import pyrr

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_LEFT
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_c
    from pygame.locals import K_d
    from pygame.locals import K_h
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_w
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')


# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================


def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


class World(object):
    def __init__(self, carla_world, hud):
        self.world = carla_world
        self.hud = hud
        blueprints = self.world.get_blueprint_library().filter('vehicle')
        blueprint = [e for i, e in enumerate(blueprints) if e.id == 'vehicle.lincoln.mkz2017'][0]

        spawn_points = self.world.get_map().get_spawn_points()
        spawn_point = spawn_points[1] #random.choice(spawn_points) if spawn_points else carla.Transform()
        self.vehicle = self.world.spawn_actor(blueprint, spawn_point)
        self.collision_sensor = CollisionSensor(self.vehicle, self.hud)
        self.camera_manager = CameraManager(self.vehicle, self.hud)
        self.camera_manager.set_sensor(0, notify=False)
        self.controller = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0

    def restart(self):
        cam_index = self.camera_manager._index
        cam_pos_index = self.camera_manager._transform_index
        start_pose = self.vehicle.get_transform()
        start_pose.location.z += 2.0
        start_pose.rotation.roll = 0.0
        start_pose.rotation.pitch = 0.0
        blueprint = self._get_random_blueprint()
        self.destroy()
        self.vehicle = self.world.spawn_actor(blueprint, start_pose)
        self.collision_sensor = CollisionSensor(self.vehicle, self.hud)
        self.camera_manager = CameraManager(self.vehicle, self.hud)
        self.camera_manager._transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = ' '.join(self.vehicle.type_id.replace('_', '.').title().split('.')[1:])
        self.hud.notification(actor_type)

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.vehicle.get_world().set_weather(preset[0])

    def tick(self, clock):
        self.hud.tick(self, clock)

    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy(self):
        for actor in [self.camera_manager.sensor, self.collision_sensor.sensor, self.vehicle]:
            if actor is not None:
                actor.destroy()

    def _get_random_blueprint(self):
        bp = random.choice(self.world.get_blueprint_library().filter('tesla'))
        if bp.has_attribute('color'):
            color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)
        return bp


# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================


class KeyboardControl(object):
    def __init__(self, world, start_in_autopilot):
        self._autopilot_enabled = start_in_autopilot
        self._control = carla.VehicleControl()
        self._steer_cache = 0.0
        world.vehicle.set_autopilot(self._autopilot_enabled)
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

    def parse_events(self, world, clock):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == K_BACKSPACE:
                    world.restart()
                elif event.key == K_h or (event.key == K_SLASH and pygame.key.get_mods() & KMOD_SHIFT):
                    world.hud.help.toggle()
                elif event.key == K_TAB:
                    world.camera_manager.toggle_camera()
                elif event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_weather(reverse=True)
                elif event.key == K_c:
                    world.next_weather()
                elif event.key == K_BACKQUOTE:
                    world.camera_manager.next_sensor()
                elif event.key > K_0 and event.key <= K_9:
                    world.camera_manager.set_sensor(event.key - 1 - K_0)
                elif event.key == K_r:
                    world.camera_manager.toggle_recording()
                elif event.key == K_q:
                    self._control.reverse = not self._control.reverse
                elif event.key == K_p:
                    self._autopilot_enabled = not self._autopilot_enabled
                    world.vehicle.set_autopilot(self._autopilot_enabled)
                    world.hud.notification('Autopilot %s' % ('On' if self._autopilot_enabled else 'Off'))
        if not self._autopilot_enabled:
            self._parse_keys(pygame.key.get_pressed(), clock.get_time())
            world.vehicle.apply_control(self._control)

    def _parse_keys(self, keys, milliseconds):
        self._control.throttle = 1.0 if keys[K_UP] or keys[K_w] else 0.0
        steer_increment = 5e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        self._control.steer = round(self._steer_cache, 1)
        self._control.brake = 1.0 if keys[K_DOWN] or keys[K_s] else 0.0
        self._control.hand_brake = keys[K_SPACE]

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        mono = next(x for x in pygame.font.get_fonts() if 'mono' in x) # hope for the best...
        mono = pygame.font.match_font(mono, bold=True)
        self._font_mono = pygame.font.Font(mono, 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 24), width, height)
        self.client_fps = 0
        self.server_fps = 0

    def tick(self, world, clock):
        self.client_fps = clock.get_fps()
        self._notifications.tick(world, clock)

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        self._notifications.render(display)
        self.help.render(display)
        fps_text = 'client: %02d FPS; server: %02d FPS' % (self.client_fps, self.server_fps)
        fps = self._font_mono.render(fps_text, True, (60, 60, 60))
        display.blit(fps, (6, 4))


# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)


# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    def __init__(self, font, width, height):
        lines = __doc__.split('\n')
        self.font = font
        self.dim = (680, len(lines) * 22 + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * 22))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        self._render = not self._render

    def render(self, display):
        if self._render:
            display.blit(self.surface, self.pos)


# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================


class CollisionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._parent = parent_actor
        self._hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = ' '.join(event.other_actor.type_id.replace('_', '.').title().split('.')[1:])
        self._hud.notification('Collision with %r' % actor_type)


# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._surface = None
        self._parent = parent_actor
        self._hud = hud
        self._recording = False
        self._camera_transforms = [
            carla.Transform(carla.Location(x=1.6, z=1.7)),
            carla.Transform(carla.Location(x=24, z=28.0), carla.Rotation(roll=-90, pitch=-90)),
            carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15))]
        self._transform_index = 1
        self._sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB'],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)'],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)'],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)'],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)'],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette, 'Camera Semantic Segmentation (CityScapes Palette)'],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)']]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self._sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))
            item.append(bp)
        self._index = None
        self._server_clock = pygame.time.Clock()

    def toggle_camera(self):
        self._transform_index = (self._transform_index + 1) % len(self._camera_transforms)
        self.sensor.set_transform(self._camera_transforms[self._transform_index])

    def set_sensor(self, index, notify=True):
        index = index % len(self._sensors)
        needs_respawn = True if self._index is None \
            else self._sensors[index][0] != self._sensors[self._index][0]
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self._surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self._sensors[index][-1],
                self._camera_transforms[self._transform_index],
                attach_to=self._parent)
            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self._hud.notification(self._sensors[index][2])
        self._index = index

    def next_sensor(self):
        self.set_sensor(self._index + 1)

    def toggle_recording(self):
        self._recording = not self._recording
        self._hud.notification('Recording %s' % ('On' if self._recording else 'Off'))

    def render(self, display):
        if self._surface is not None:
            display.blit(self._surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        self._server_clock.tick()
        self._hud.server_fps = self._server_clock.get_fps()
        if self._sensors[self._index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0]/3), 3))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self._hud.dim) / 100.0
            lidar_data += (0.5 * self._hud.dim[0], 0.5 * self._hud.dim[1])
            lidar_data = np.fabs(lidar_data)
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self._hud.dim[0], self._hud.dim[1], 3)
            lidar_img = np.zeros(lidar_img_size)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self._surface = pygame.surfarray.make_surface(lidar_img)
        else:
            image.convert(self._sensors[self._index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self._surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self._recording:
            image.save_to_disk('_out/%08d' % image.frame_number)


# ==============================================================================
# -- PID Controller-------------------------------------------------------------
# ==============================================================================

import math as m
import numpy as np
import matplotlib.pyplot as plt

class VehiclePIDController():
    def __init__(self, vehicle,
                 args_lateral={'K_P': 1.0, 'K_D': 0.0, 'K_I': 0.0},
                 args_longitudinal={'K_P': 1.0, 'K_D': 0.0, 'K_I': 0.0} ):
        self._vehicle = vehicle
        self._long_controller = PIDLongitudinalController(self._vehicle, **args_longitudinal)
        self._later_controller = PIDLateralController(self._vehicle, **args_lateral)
        #self._counter = 0
        #plt.axis([0, 10, 0.4, 1.5])
        #plt.ion()
        #plt.show()

    def run_step(self, target_speed, waypoint):
        throttle = self._long_controller.run_step(target_speed)
        steering = self._later_controller.run_step(waypoint)

        control = carla.VehicleControl()
        control.steer = steering
        control.throttle = throttle
        control.brake = 0.0
        control.hand_brake = False
        self._vehicle.apply_control(control)

    def run_iter(self, target_speed, waypoint, radius, max_iters):
        _buffer = []
        iters = 0
        control = carla.VehicleControl()
        vehicle_transform = self._vehicle.get_transform()
        while not self._under(waypoint, vehicle_transform, radius) and iters < max_iters:
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
            _error = m.sqrt(dx * dx + dy * dy)
            _buffer.append(_error)
            iters += 1

            # self._counter += 1
            # if self._counter % 20 == 0:
            #     self._counter = 0
            #     plt.plot(range(len(_buffer)), _buffer)
            #     plt.draw()
            #     plt.pause(0.001)

    def _under(self, waypoint, vehicle_transform, radius):
        loc = vehicle_transform.location
        dx = waypoint.transform.location.x - loc.x
        dy = waypoint.transform.location.y - loc.y

        #print('Distance to wp = {}'.format(m.sqrt(dx*dx + dy*dy)))
        return m.sqrt(dx*dx + dy*dy) < radius

    def warmup(self):
        vel = self._vehicle.get_velocity()
        speed = m.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
        while speed < 0.5:
            control = carla.VehicleControl()
            control.steer = 0.0
            control.throttle = 1.0
            control.brake = 0.0
            control.hand_brake = False
            self._vehicle.apply_control(control)
            # print('speed = {}'.format(speed))

            vel = self._vehicle.get_velocity()
            speed = m.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)

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

        self._e_buffer = deque(maxlen=5)

    def run_step(self, target_speed):
        vel = self._vehicle.get_velocity()
        current_speed = 3.6 * m.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
        #print('Current speed = {}'.format(current_speed))
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

        self._eps = -0.1
        self._e_buffer = deque(maxlen=5)

    def run_step(self, waypoint):
        return self._pid_control(waypoint, self._vehicle.get_transform())

    def _pid_control(self, waypoint, vehicle_transform):
        loc = vehicle_transform.location
        yaw = -m.radians(vehicle_transform.rotation.yaw)
        rel_v = np.array([1.0, 0.0, 0.0])

        _x = waypoint.transform.location.x - loc.x
        _y = waypoint.transform.location.y - loc.y
        _xr = m.cos(yaw) * _x - m.sin(yaw) * _y
        _yr = m.sin(yaw) * _x + m.cos(yaw) * _y
        rel_w = np.array([_xr, _yr, 0.0])

        _cross = np.cross(rel_v, rel_w)
        _dot = m.acos(np.dot(rel_v, rel_w) / (np.linalg.norm(rel_v) * np.linalg.norm(rel_w)))
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

# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================
import math

def draw_waypoints(world, waypoint, depth=6):
    if depth < 0:
        return
    for w in waypoint.next(4.0):
        t = w.transform
        begin = t.location + carla.Location(z=0.5)
        angle = math.radians(t.rotation.yaw)
        end = begin + carla.Location(x=math.cos(angle), y=math.sin(angle))
        world.debug.draw_arrow(begin, end, arrow_size=0.3, life_time=1.0)
        draw_waypoints(world, w, depth - 1)

from enum import Enum
class TOPO_OPTIONS(Enum):
        VOID = -1
        LEFT = 1
        RIGHT = 2
        STRAIGHT = 3


def retrieve_options(list_waypoints, current_waypoint, topology_table):
    options = []
    for next_waypoint in list_waypoints:
        print("WP.road_id = {}".format(next_waypoint.road_id))
        options.append(TOPO_OPTIONS(topology_table[current_waypoint.road_id, next_waypoint.road_id]))
    return options

def compute_connection(current_waypoint, next_waypoint):
    loc = current_waypoint.transform.location
    yaw = -m.radians(current_waypoint.transform.rotation.yaw)
    rel_v = np.array([1.0, 0.0, 0.0])

    _x = next_waypoint.transform.location.x - loc.x
    _y = next_waypoint.transform.location.y - loc.y
    _xr = m.cos(yaw) * _x - m.sin(yaw) * _y
    _yr = m.sin(yaw) * _x + m.cos(yaw) * _y
    rel_w = np.array([_xr, _yr, 0.0])

    _cross = np.cross(rel_v, rel_w)
    print("({}, {}) --> ({}, {}) = {}".format(loc.x, loc.y, next_waypoint.transform.location.x, next_waypoint.transform.location.y, _cross))
    if _cross[2] < -1.0e-7:
        return TOPO_OPTIONS.LEFT
    elif _cross[2] > 1.0e-7:
        return TOPO_OPTIONS.RIGHT
    else:
        return TOPO_OPTIONS.STRAIGHT

def parse_topology(list_connections):
    max_road_id = -1
    for connection in list_connections:
        start, end = connection

        if start.road_id > max_road_id:
            max_road_id = start.road_id
        if end.road_id > max_road_id:
            max_road_id = end.road_id

    table_topology = -1 * np.ones((max_road_id+1, max_road_id+1))
    for connection in list_connections:
        start, end = connection
        #if start.road_id == 4 and end.road_id == 100:
        #    pass
        table_topology[start.road_id, end.road_id] = compute_connection(start, end).value

    return table_topology


import time
def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None
    DT = 1.0/61.0
    TARGET_SPEED = 10.0 # Km/h
    NEXT_RADIUS = 6.0 #6.0 * DT * TARGET_SPEED / 3.6
    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(4.0)

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        hud = HUD(args.width, args.height)
        world = World(client.get_world(), hud)
        controller = KeyboardControl(world, args.autopilot)

        time.sleep(1)

        m = world.world.get_map()
        topology_table = parse_topology(m.get_topology())
        current_waypoint = m.get_waypoint(world.vehicle.get_location())
        lane_id = current_waypoint.lane_id
        road_id = current_waypoint.road_id

        nexts = list(current_waypoint.next(NEXT_RADIUS))

        clock = pygame.time.Clock()
        count = 0
        vehicle_controller = VehiclePIDController(world.vehicle,
                                                  args_lateral={'K_P': 1.7, 'K_D': 0., 'K_I': 0., 'dt':DT},
                                                  args_longitudinal={'K_P': 0.6, 'K_D': 0.0, 'K_I': 0., 'dt':DT})
        vehicle_controller.warmup()
        print('======== WARMUP done =========')

        random_choice_made = False
        while True:
            #print("Next radius = {}".format(NEXT_RADIUS))
            clock.tick_busy_loop(60)
            if controller.parse_events(world, clock):
                return
            world.tick(clock)
            world.render(display)
            pygame.display.flip()

            if not nexts:
                raise RuntimeError("No more waypoints!")

            if len(nexts) == 1:
                current_waypoint = nexts[0]
                random_choice_made = False

            elif len(nexts) > 1:
                if random_choice_made:
                    for waypoint in nexts:
                        if waypoint.road_id == road_id and waypoint.lane_id == lane_id:
                            current_waypoint = waypoint
                            break
                else:
                    # make random choice
                    topo_options = retrieve_options(nexts, current_waypoint, topology_table)

                    # preference to straight
                    if TOPO_OPTIONS.STRAIGHT in topo_options:
                        next_waypoint = nexts[topo_options.index(TOPO_OPTIONS.STRAIGHT)]
                        print('!! SELECTED STRAIGHT')
                    elif TOPO_OPTIONS.LEFT in topo_options:
                        next_waypoint = nexts[topo_options.index(TOPO_OPTIONS.LEFT)]
                        print('<<<< SELECTED LEFT')
                    else:
                        next_waypoint = nexts[topo_options.index(TOPO_OPTIONS.RIGHT)]
                        print('>>>> SELECTED RIGHT')


                    print(">>>>>>>>>>")
                    for e in topo_options:
                        print("{}, ".format(e))
                    print("<<<<<<<<<<<")
                    current_waypoint = next_waypoint
                    random_choice_made = True
                    #pdb.set_trace()

            lane_id = current_waypoint.lane_id
            road_id = current_waypoint.road_id

            # if count % 10 == 0:
            #     draw_waypoints(world.world, w)
            #     count = 0

            vehicle_controller.run_iter(TARGET_SPEED, current_waypoint, 0.2, 15)

            # new waypoints based on current location
            current_waypoint = m.get_waypoint(world.vehicle.get_location())
            nexts = list(current_waypoint.next(NEXT_RADIUS))

            count += 1

    finally:

        if world is not None:
            world.destroy()

        pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:

        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
    except Exception as error:
        logging.exception(error)


if __name__ == '__main__':

    main()
