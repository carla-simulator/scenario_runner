import math
import pdb

import carla

from Navigation.local_planner import *


class RoamingAgent(object):
    def __init__(self, vehicle):
        """

        :param vehicle:
        """
        self._vehicle = vehicle
        self._world = self._vehicle.get_world()
        self._map = self._vehicle.get_world().get_map()
        self._proximity_threshold = 10.0
        self._local_planner = LocalPlanner(self._vehicle)

    def run_step(self):
        hazard_detected = False
        current_location = self._vehicle.get_location()
        vehicle_waypoint = self._map.get_waypoint(self._vehicle.get_location())

        actor_list = self._world.get_actors()
        vehicle_list = actor_list.filter("*vehicle*")
        lights_list = actor_list.filter("*traffic_light*")

        for object in vehicle_list:
            # do not account for the ego vehicle
            if object.id == self._vehicle.id:
                continue
            # if the object is not in our lane it's not an obstacle
            object_waypoint = self._map.get_waypoint(object.get_location())
            if object_waypoint.road_id != vehicle_waypoint.road_id or object_waypoint.lane_id != vehicle_waypoint.lane_id:
                continue

            loc = object.get_location()
            if self.within_distance_ahead(loc, current_location, self._vehicle.get_transform().rotation.yaw,
                                          self._proximity_threshold):
                print('!!! HAZARD [{}] ==> (x={}, y={})'.format(object.id, loc.x, loc.y))
                hazard_detected = True
                break

        for object in lights_list:
            object_waypoint = self._map.get_waypoint(object.get_location())
            if object_waypoint.road_id != vehicle_waypoint.road_id or object_waypoint.lane_id != vehicle_waypoint.lane_id:
                continue

            loc = object.get_location()
            if self.within_distance_ahead(loc, current_location, self._vehicle.get_transform().rotation.yaw,
                                          self._proximity_threshold):
                if object.state == carla.libcarla.TrafficLightState.Red:
                    print('=== RED LIGHT AHEAD [{}] ==> (x={}, y={})'.format(object.id, loc.x, loc.y))
                    hazard_detected = True
                    break

        if hazard_detected:
            self.emergency_stop()
        else:
            self._local_planner.run_step()

    def emergency_stop(self):
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 1.0
        control.hand_brake = False
        self._vehicle.apply_control(control)


    def distance_location(self, locA, locB):
        dx = locA.x - locB.x
        dy = locA.y - locB.y

        return math.sqrt(dx * dx + dy * dy)

    def within_distance_ahead(self, target_location, current_location, orientation, max_distance):
        target_vector = np.array([target_location.x - current_location.x, target_location.y - current_location.y])
        norm_target = np.linalg.norm(target_vector)
        if norm_target > max_distance:
            return False

        forward_vector = np.array([math.cos(math.radians(orientation)), math.sin(math.radians(orientation))])
        d_angle = math.degrees(math.acos(np.dot(forward_vector, target_vector) / norm_target))

        return d_angle < 90.0
