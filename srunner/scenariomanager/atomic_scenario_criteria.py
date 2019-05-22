#!/usr/bin/env python

# Copyright (c) 2018-2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides all atomic evaluation criteria required to analyze if a
scenario was completed successfully or failed.

The atomic criteria are implemented with py_trees.
"""

import weakref
import math
import numpy as np
import py_trees
import shapely

import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.traffic_events import TrafficEvent, TrafficEventType


class Criterion(py_trees.behaviour.Behaviour):

    """
    Base class for all criteria used to evaluate a scenario for success/failure

    Important parameters (PUBLIC):
    - name: Name of the criterion
    - expected_value_success:    Result in case of success
                                 (e.g. max_speed, zero collisions, ...)
    - expected_value_acceptable: Result that does not mean a failure,
                                 but is not good enough for a success
    - actual_value: Actual result after running the scenario
    - test_status: Used to access the result of the criterion
    - optional: Indicates if a criterion is optional (not used for overall analysis)
    """

    def __init__(self,
                 name,
                 actor,
                 expected_value_success,
                 expected_value_acceptable=None,
                 optional=False,
                 terminate_on_failure=False):
        super(Criterion, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._terminate_on_failure = terminate_on_failure

        self.name = name
        self.actor = actor
        self.test_status = "INIT"
        self.expected_value_success = expected_value_success
        self.expected_value_acceptable = expected_value_acceptable
        self.actual_value = 0
        self.optional = optional
        self.list_traffic_events = []

    def setup(self, unused_timeout=15):
        self.logger.debug("%s.setup()" % (self.__class__.__name__))
        return True

    def initialise(self):
        self.logger.debug("%s.initialise()" % (self.__class__.__name__))

    def terminate(self, new_status):
        self.logger.debug("%s.terminate()[%s->%s]" % (self.__class__.__name__, self.status, new_status))


class MaxVelocityTest(Criterion):

    """
    This class contains an atomic test for maximum velocity.
    """

    def __init__(self, actor, max_velocity_allowed, optional=False, name="CheckMaximumVelocity"):
        """
        Setup actor and maximum allowed velovity
        """
        super(MaxVelocityTest, self).__init__(name, actor, max_velocity_allowed, None, optional)

    def update(self):
        """
        Check velocity
        """
        new_status = py_trees.common.Status.RUNNING

        if self.actor is None:
            return new_status

        velocity = CarlaDataProvider.get_velocity(self.actor)

        self.actual_value = max(velocity, self.actual_value)

        if velocity > self.expected_value_success:
            self.test_status = "FAILURE"
        else:
            self.test_status = "SUCCESS"

        if self._terminate_on_failure and (self.test_status == "FAILURE"):
            new_status = py_trees.common.Status.FAILURE

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status


class DrivenDistanceTest(Criterion):

    """
    This class contains an atomic test to check the driven distance
    """

    def __init__(self,
                 actor,
                 distance_success,
                 distance_acceptable=None,
                 optional=False,
                 name="CheckDrivenDistance"):
        """
        Setup actor
        """
        super(DrivenDistanceTest, self).__init__(name, actor, distance_success, distance_acceptable, optional)
        self._last_location = None

    def initialise(self):
        self._last_location = CarlaDataProvider.get_location(self.actor)
        super(DrivenDistanceTest, self).initialise()

    def update(self):
        """
        Check distance
        """
        new_status = py_trees.common.Status.RUNNING

        if self.actor is None:
            return new_status

        location = CarlaDataProvider.get_location(self.actor)

        if location is None:
            return new_status

        if self._last_location is None:
            self._last_location = location
            return new_status

        self.actual_value += location.distance(self._last_location)
        self._last_location = location

        if self.actual_value > self.expected_value_success:
            self.test_status = "SUCCESS"
        elif (self.expected_value_acceptable is not None and
              self.actual_value > self.expected_value_acceptable):
            self.test_status = "ACCEPTABLE"
        else:
            self.test_status = "RUNNING"

        if self._terminate_on_failure and (self.test_status == "FAILURE"):
            new_status = py_trees.common.Status.FAILURE

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status


class AverageVelocityTest(Criterion):

    """
    This class contains an atomic test for average velocity.
    """

    def __init__(self,
                 actor,
                 avg_velocity_success,
                 avg_velocity_acceptable=None,
                 optional=False,
                 name="CheckAverageVelocity"):
        """
        Setup actor and average velovity expected
        """
        super(AverageVelocityTest, self).__init__(name, actor,
                                                  avg_velocity_success,
                                                  avg_velocity_acceptable,
                                                  optional)
        self._last_location = None
        self._distance = 0.0

    def initialise(self):
        self._last_location = CarlaDataProvider.get_location(self.actor)
        super(AverageVelocityTest, self).initialise()

    def update(self):
        """
        Check velocity
        """
        new_status = py_trees.common.Status.RUNNING

        if self.actor is None:
            return new_status

        location = CarlaDataProvider.get_location(self.actor)

        if location is None:
            return new_status

        if self._last_location is None:
            self._last_location = location
            return new_status

        self._distance += location.distance(self._last_location)
        self._last_location = location

        elapsed_time = GameTime.get_time()
        if elapsed_time > 0.0:
            self.actual_value = self._distance / elapsed_time

        if self.actual_value > self.expected_value_success:
            self.test_status = "SUCCESS"
        elif (self.expected_value_acceptable is not None and
              self.actual_value > self.expected_value_acceptable):
            self.test_status = "ACCEPTABLE"
        else:
            self.test_status = "RUNNING"

        if self._terminate_on_failure and (self.test_status == "FAILURE"):
            new_status = py_trees.common.Status.FAILURE

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status


class CollisionTest(Criterion):

    """
    This class contains an atomic test for collisions.
    """

    def __init__(self, actor, optional=False, name="CheckCollisions", terminate_on_failure=False):
        """
        Construction with sensor setup
        """
        super(CollisionTest, self).__init__(name, actor, 0, None, optional, terminate_on_failure)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))

        world = self.actor.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.collision')
        self._collision_sensor = world.spawn_actor(blueprint, carla.Transform(), attach_to=self.actor)
        self._collision_sensor.listen(lambda event: self._count_collisions(weakref.ref(self), event))

    def update(self):
        """
        Check collision count
        """
        new_status = py_trees.common.Status.RUNNING

        if self._terminate_on_failure and (self.test_status == "FAILURE"):
            new_status = py_trees.common.Status.FAILURE

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status

    def terminate(self, new_status):
        """
        Cleanup sensor
        """
        if self._collision_sensor is not None:
            self._collision_sensor.destroy()
        self._collision_sensor = None
        super(CollisionTest, self).terminate(new_status)

    @staticmethod
    def _count_collisions(weak_self, event):
        """
        Callback to update collision count
        """
        self = weak_self()
        if not self:
            return

        registered = False
        actor_type = None
        if 'static' in event.other_actor.type_id:
            actor_type = TrafficEventType.COLLISION_STATIC
            self.test_status = "FAILURE"
        elif 'vehicle' in event.other_actor.type_id:
            for traffic_event in self.list_traffic_events:
                if traffic_event.get_type() == TrafficEventType.COLLISION_VEHICLE \
                    and traffic_event.get_dict()['id'] == event.other_actor.id:
                        registered = True
            actor_type = TrafficEventType.COLLISION_VEHICLE
        elif 'walker' in event.other_actor.type_id:
            for traffic_event in self.list_traffic_events:
                if traffic_event.get_type() == TrafficEventType.COLLISION_PEDESTRIAN \
                        and traffic_event.get_dict()['id'] == event.other_actor.id:
                    registered = True
            actor_type = TrafficEventType.COLLISION_PEDESTRIAN

        if not registered:
            collision_event = TrafficEvent(event_type=actor_type)
            collision_event.set_dict({'type': event.other_actor.type_id, 'id': event.other_actor.id})
            collision_event.set_message("Agent collided against object with type={} and id={}".format(
                event.other_actor.type_id, event.other_actor.id))
            self.list_traffic_events.append(collision_event)


class KeepLaneTest(Criterion):

    """
    This class contains an atomic test for keeping lane.
    """

    def __init__(self, actor, optional=False, name="CheckKeepLane"):
        """
        Construction with sensor setup
        """
        super(KeepLaneTest, self).__init__(name, actor, 0, None, optional)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))

        world = self.actor.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self._lane_sensor = world.spawn_actor(blueprint, carla.Transform(), attach_to=self.actor)
        self._lane_sensor.listen(lambda event: self._count_lane_invasion(weakref.ref(self), event))

    def update(self):
        """
        Check lane invasion count
        """
        new_status = py_trees.common.Status.RUNNING

        if self.actual_value > 0:
            self.test_status = "FAILURE"
        else:
            self.test_status = "SUCCESS"

        if self._terminate_on_failure and (self.test_status == "FAILURE"):
            new_status = py_trees.common.Status.FAILURE

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status

    def terminate(self, new_status):
        """
        Cleanup sensor
        """
        if self._lane_sensor is not None:
            self._lane_sensor.destroy()
        self._lane_sensor = None
        super(KeepLaneTest, self).terminate(new_status)

    @staticmethod
    def _count_lane_invasion(weak_self, event):
        """
        Callback to update lane invasion count
        """
        self = weak_self()
        if not self:
            return
        self.actual_value += 1


class ReachedRegionTest(Criterion):

    """
    This class contains the reached region test
    The test is a success if the actor reaches a specified region
    """

    def __init__(self, actor, min_x, max_x, min_y, max_y, name="ReachedRegionTest"):
        """
        Setup trigger region (rectangle provided by
        [min_x,min_y] and [max_x,max_y]
        """
        super(ReachedRegionTest, self).__init__(name, actor, 0)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._actor = actor
        self._min_x = min_x
        self._max_x = max_x
        self._min_y = min_y
        self._max_y = max_y

    def update(self):
        """
        Check if the actor location is within trigger region
        """
        new_status = py_trees.common.Status.RUNNING

        location = CarlaDataProvider.get_location(self._actor)
        if location is None:
            return new_status

        in_region = False
        if self.test_status != "SUCCESS":
            in_region = (location.x > self._min_x and location.x < self._max_x) and (
                location.y > self._min_y and location.y < self._max_y)
            if in_region:
                self.test_status = "SUCCESS"
            else:
                self.test_status = "RUNNING"

        if self.test_status == "SUCCESS":
            new_status = py_trees.common.Status.SUCCESS

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status

class OnSidewalkTest(Criterion):

    """
    This class contains an atomic test to detect sidewalk invasions.
    """

    def __init__(self, actor, optional=False, name="WrongLaneTest"):
        """
        Construction with sensor setup
        """
        super(OnSidewalkTest, self).__init__(name, actor, 0, None, optional)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))

        self._actor = actor
        self._map = CarlaDataProvider.get_map()
        self._onsidewalk_active = False

        self.positive_shift = shapely.geometry.LineString([(0, 0), (0.0, 1.2)])
        self.negative_shift = shapely.geometry.LineString([(0, 0), (0.0, -1.2)])


    def update(self):
        """
        Check lane invasion count
        """
        new_status = py_trees.common.Status.RUNNING

        if self._terminate_on_failure and (self.test_status == "FAILURE"):
            new_status = py_trees.common.Status.FAILURE

        current_transform = self._actor.get_transform()
        current_location = current_transform.location
        current_yaw = current_transform.rotation.yaw


        rot_x = shapely.affinity.rotate(self.positive_shift, angle=current_yaw, origin=shapely.geometry.Point(0, 0))
        rot_nx = shapely.affinity.rotate(self.negative_shift, angle=current_yaw, origin=shapely.geometry.Point(0, 0))

        sample_point_right = current_location + carla.Location(x=rot_x.coords[1][0], y=rot_x.coords[1][1])
        sample_point_left = current_location + carla.Location(x=rot_nx.coords[1][0], y=rot_nx.coords[1][1])

        closest_waypoint_right = self._map.get_waypoint(sample_point_right, lane_type=carla.LaneType.Any)
        closest_waypoint_left = self._map.get_waypoint(sample_point_left, lane_type=carla.LaneType.Any)

        if closest_waypoint_right and closest_waypoint_left \
                and closest_waypoint_right.lane_type != carla.LaneType.Sidewalk \
                and closest_waypoint_left.lane_type != carla.LaneType.Sidewalk:
            # we are not on a sidewalk
            self._onsidewalk_active = False

        else:
            if not self._onsidewalk_active:
                onsidewalk_event = TrafficEvent(event_type=TrafficEventType.ON_SIDEWALK_INFRACTION)
                onsidewalk_event.set_message('Agent invaded the sidewalk')
                onsidewalk_event.set_dict({'x': current_location.x, 'y': current_location.y})
                self.list_traffic_events.append(onsidewalk_event)

                self.test_status = "FAILURE"
                self._onsidewalk_active = True

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status


class WrongLaneTest(Criterion):

    """
    This class contains an atomic test to detect invasions to wrong direction lanes.
    """
    MAX_ALLOWED_ANGLE = 140.0

    def __init__(self, actor, optional=False, name="WrongLaneTest"):
        """
        Construction with sensor setup
        """
        super(WrongLaneTest, self).__init__(name, actor, 0, None, optional)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))

        self._world = self.actor.get_world()
        self._actor = actor
        self._map = CarlaDataProvider.get_map()
        self._infractions = 0
        self._last_lane_id = None
        self._last_road_id = None

        blueprint = self._world.get_blueprint_library().find('sensor.other.lane_invasion')
        self._lane_sensor = self._world.spawn_actor(blueprint, carla.Transform(), attach_to=self.actor)
        self._lane_sensor.listen(lambda event: self._lane_change(weakref.ref(self), event))

    def update(self):
        """
        Check lane invasion count
        """
        new_status = py_trees.common.Status.RUNNING

        if self._terminate_on_failure and (self.test_status == "FAILURE"):
            new_status = py_trees.common.Status.FAILURE

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status

    def terminate(self, new_status):
        """
        Cleanup sensor
        """
        if self._lane_sensor is not None:
            self._lane_sensor.destroy()
        self._lane_sensor = None
        super(WrongLaneTest, self).terminate(new_status)

    @staticmethod
    def _lane_change(weak_self, event):
        """
        Callback to update lane invasion count
        """
        # pylint: disable=protected-access

        self = weak_self()
        if not self:
            return

        # check the lane direction
        lane_waypoint = self._map.get_waypoint(self._actor.get_location())
        current_lane_id = lane_waypoint.lane_id
        current_road_id = lane_waypoint.road_id

        if not (self._last_road_id == current_road_id and self._last_lane_id == current_lane_id):
            next_waypoint = lane_waypoint.next(2.0)[0]

            if not next_waypoint:
                return

            vector_wp = np.array([next_waypoint.transform.location.x - lane_waypoint.transform.location.x,
                                  next_waypoint.transform.location.y - lane_waypoint.transform.location.y])

            vector_actor = np.array([math.cos(math.radians(self._actor.get_transform().rotation.yaw)),
                                     math.sin(math.radians(self._actor.get_transform().rotation.yaw))])

            ang = math.degrees(
                math.acos(np.clip(np.dot(vector_actor, vector_wp) / (np.linalg.norm(vector_wp)), -1.0, 1.0)))
            if ang > self.MAX_ALLOWED_ANGLE:
                self.test_status = "FAILURE"
                # is there a difference of orientation greater than MAX_ALLOWED_ANGLE deg with respect of the lane
                # direction?
                self._infractions += 1

                wrong_way_event = TrafficEvent(event_type=TrafficEventType.WRONG_WAY_INFRACTION)
                wrong_way_event.set_message('Agent invaded a lane in opposite direction: road_id={}, lane_id={}'.format(
                    current_road_id, current_lane_id))
                wrong_way_event.set_dict({'road_id': current_road_id, 'lane_id': current_lane_id})
                self.list_traffic_events.append(wrong_way_event)

        # remember the current lane and road
        self._last_lane_id = current_lane_id
        self._last_road_id = current_road_id


class InRadiusRegionTest(Criterion):

    """
    The test is a success if the actor is within a given radius of a specified region
    """

    def __init__(self, actor, x, y, radius, name="InRadiusRegionTest"):
        """
        """
        super(InRadiusRegionTest, self).__init__(name, actor, 0)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._actor = actor
        self._x = x     # pylint: disable=invalid-name
        self._y = y     # pylint: disable=invalid-name
        self._radius = radius

    def update(self):
        """
        Check if the actor location is within trigger region
        """
        new_status = py_trees.common.Status.RUNNING

        location = CarlaDataProvider.get_location(self._actor)
        if location is None:
            return new_status

        if self.test_status != "SUCCESS":
            in_radius = math.sqrt(((location.x - self._x)**2) + ((location.y - self._y)**2)) < self._radius
            if in_radius:
                route_completion_event = TrafficEvent(event_type=TrafficEventType.ROUTE_COMPLETED)
                route_completion_event.set_message("Destination was successfully reached")
                self.list_traffic_events.append(route_completion_event)
                self.test_status = "SUCCESS"
            else:
                self.test_status = "RUNNING"

        if self.test_status == "SUCCESS":
            new_status = py_trees.common.Status.SUCCESS

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status

class InRouteTest(Criterion):

    """
    The test is a success if the actor is never outside route
    """
    DISTANCE_THRESHOLD = 15.0  # meters
    WINDOWS_SIZE = 3

    def __init__(self, actor, radius, route, offroad_max, name="InRouteTest", terminate_on_failure=False):
        """
        """
        super(InRouteTest, self).__init__(name, actor, 0, terminate_on_failure=terminate_on_failure)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._actor = actor
        self._route = route

        self._wsize = self.WINDOWS_SIZE
        self._waypoints, _ = zip(*self._route)
        self._route_length = len(self._route)
        self._current_index = 0

    def update(self):
        """
        Check if the actor location is within trigger region
        """
        new_status = py_trees.common.Status.RUNNING

        location = CarlaDataProvider.get_location(self._actor)
        if location is None:
            return new_status

        if self._terminate_on_failure and (self.test_status == "FAILURE"):
            new_status = py_trees.common.Status.FAILURE

        elif self.test_status == "RUNNING" or self.test_status == "INIT":
            # are we too far away from the route waypoints (i.e., off route)?
            off_route = True

            shortest_distance = float('inf')
            for index in range(max(0, self._current_index - self._wsize),
                               min(self._current_index + self._wsize + 1, self._route_length)):
                # look for the distance to the current waipoint + windows_size
                ref_waypoint = self._waypoints[index]
                distance = math.sqrt(((location.x - ref_waypoint.x) ** 2) + ((location.y - ref_waypoint.y) ** 2))
                if distance < self.DISTANCE_THRESHOLD \
                        and distance <= shortest_distance \
                        and index >= self._current_index:
                    shortest_distance = distance
                    self._current_index = index
                    off_route = False
            if off_route:
                route_deviation_event = TrafficEvent(event_type=TrafficEventType.ROUTE_DEVIATION)
                route_deviation_event.set_message("Agent deviated from the route at (x={}, y={}, z={})".format(
                    location.x, location.y, location.z))
                route_deviation_event.set_dict({'x': location.x, 'y': location.y, 'z': location.z})
                self.list_traffic_events.append(route_deviation_event)

                self.test_status = "FAILURE"
                new_status = py_trees.common.Status.FAILURE

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status


class RouteCompletionTest(Criterion):

    """
    Check at which stage of the route is the actor at each tick
    """
    DISTANCE_THRESHOLD = 10.0 # meters
    WINDOWS_SIZE = 2

    def __init__(self, actor, route, name="RouteCompletionTest", terminate_on_failure=False):
        """
        """
        super(RouteCompletionTest, self).__init__(name, actor, 0, terminate_on_failure=terminate_on_failure)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._actor = actor
        self._route = route

        self._wsize = self.WINDOWS_SIZE
        self._current_index = 0
        self._route_length = len(self._route)
        self._waypoints, _ = zip(*self._route)
        self.target = self._waypoints[-1]

        self._accum_meters = []
        prev_wp = self._waypoints[0]
        for i, wp in enumerate(self._waypoints):
            d = wp.distance(prev_wp)
            if i > 0:
                accum = self._accum_meters[i-1]
            else:
                accum = 0

            self._accum_meters.append(d + accum)
            prev_wp = wp

        self._traffic_event = TrafficEvent(event_type=TrafficEventType.ROUTE_COMPLETION)
        self.list_traffic_events.append(self._traffic_event)
        self._percentage_route_completed = 0.0

    def update(self):
        """
        Check if the actor location is within trigger region
        """
        new_status = py_trees.common.Status.RUNNING

        location = CarlaDataProvider.get_location(self._actor)
        if location is None:
            return new_status

        if self._terminate_on_failure and (self.test_status == "FAILURE"):
            new_status = py_trees.common.Status.FAILURE

        elif self.test_status == "RUNNING" or self.test_status == "INIT":

            for index in range(self._current_index, min(self._current_index + self._wsize + 1, self._route_length)):
                # look for the distance to the current waipoint + windows_size
                ref_waypoint = self._waypoints[index]
                distance = math.sqrt(((location.x - ref_waypoint.x) ** 2) + ((location.y - ref_waypoint.y) ** 2))
                if distance < self.DISTANCE_THRESHOLD:
                    # good! segment completed!
                    self._current_index = index
                    self._percentage_route_completed = 100.0 * float(self._accum_meters[self._current_index]) \
                                                       / float(self._accum_meters[-1])
                    self._traffic_event.set_dict({'route_completed': self._percentage_route_completed})
                    self._traffic_event.set_message(
                        "Agent has completed > {:.2f}% of the route".format(self._percentage_route_completed))

            if self._percentage_route_completed > 99.0 and location.distance(self.target) < self.DISTANCE_THRESHOLD:
                route_completion_event = TrafficEvent(event_type=TrafficEventType.ROUTE_COMPLETED)
                route_completion_event.set_message("Destination was successfully reached")
                self.list_traffic_events.append(route_completion_event)
                self.test_status = "SUCCESS"

        elif self.test_status == "SUCCESS":
            new_status = py_trees.common.Status.SUCCESS

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status


class RunningRedLightTest(Criterion):

    """
    Check if an actor is running a red light
    """
    DISTANCE_LIGHT = 10 # m

    def __init__(self, actor, name="RunningRedLightTest", terminate_on_failure=False):
        """
        Init
        """
        super(RunningRedLightTest, self).__init__(name, actor, 0, terminate_on_failure=terminate_on_failure)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._actor = actor
        self._world = actor.get_world()
        self._map = CarlaDataProvider.get_map()
        self._list_traffic_lights = []
        self._last_red_light_id = None
        self.debug = False

        all_actors = self._world.get_actors()
        for _actor in all_actors:
            if 'traffic_light' in _actor.type_id:
                center, area = self.get_traffic_light_area(_actor)
                waypoints = []
                for pt in area:
                    waypoints.append(self._map.get_waypoint(pt))
                self._list_traffic_lights.append((_actor, center, area, waypoints))


    def is_vehicle_crossing_line(self, seg1, seg2):
        line1 = shapely.geometry.LineString([ (seg1[0].x, seg1[0].y), (seg1[1].x, seg1[1].y) ])
        line2 = shapely.geometry.LineString([ (seg2[0].x, seg2[0].y), (seg2[1].x, seg2[1].y) ])
        inter = line1.intersection(line2)

        return not inter.is_empty


    def update(self):
        """
        Check if the actor is running a red light
        """
        new_status = py_trees.common.Status.RUNNING

        location = self._actor.get_transform().location
        if location is None:
            return new_status

        ego_waypoint = self._map.get_waypoint(location)

        tail_pt0 = self.rotate_point(carla.Vector3D(-1.0, 0.0, location.z), self._actor.get_transform().rotation.yaw)
        tail_pt0 = location + carla.Location(tail_pt0)

        tail_pt1 = self.rotate_point(carla.Vector3D(-4.0, 0.0, location.z), self._actor.get_transform().rotation.yaw)
        tail_pt1 = location + carla.Location(tail_pt1)

        for traffic_light, center, area, waypoints in self._list_traffic_lights:

            if self.debug:
                Z = 2.1
                if traffic_light.state == carla.TrafficLightState.Red:
                    color = carla.Color(255, 0, 0)
                elif traffic_light.state == carla.TrafficLightState.Green:
                    color = carla.Color(0, 255, 0)
                else:
                    color = carla.Color(255, 255, 255)
                self._world.debug.draw_point(center + carla.Location(z=Z), size=0.2, color=color, life_time=0.01)
                for pt in area:
                    self._world.debug.draw_point(pt + carla.Location(z=Z), size=0.1, color=color, life_time=0.01)
                for wp in waypoints:
                    text = "{}.{}".format(wp.road_id, wp.lane_id)
                    self._world.debug.draw_string(wp.transform.location, text, draw_shadow=False, color=color,life_time=0.01)

            # logic
            center_loc = carla.Location(center)

            if self._last_red_light_id and self._last_red_light_id == traffic_light.id:
                continue
            if center_loc.distance(location) > self.DISTANCE_LIGHT:
                continue
            if traffic_light.state != carla.TrafficLightState.Red:
                continue

            for wp in waypoints:
                if ego_waypoint.road_id == wp.road_id and ego_waypoint.lane_id == wp.lane_id:
                    # this light is red and is affecting our lane!
                    # is the vehicle traversing the stop line?
                    if self.is_vehicle_crossing_line((tail_pt0, tail_pt1), (area[0], area[-1])):
                        self.test_status = "FAILURE"
                        location = traffic_light.get_transform().location
                        red_light_event = TrafficEvent(event_type=TrafficEventType.TRAFFIC_LIGHT_INFRACTION)
                        red_light_event.set_message("Agent ran a red light {} at (x={}, y={}, x={})".format(
                            traffic_light,
                            location.x,
                            location.y,
                            location.z))
                        red_light_event.set_dict({'id': traffic_light.id, 'x': location.x,
                                                  'y': location.y, 'z': location.z})
                        self.list_traffic_events.append(red_light_event)
                        self._last_red_light_id = traffic_light.id
                        break

        if self._terminate_on_failure and (self.test_status == "FAILURE"):
            new_status = py_trees.common.Status.FAILURE

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status

    def rotate_point(self, pt, angle):
        x_ = math.cos(math.radians(angle))*pt.x - math.sin(math.radians(angle))*pt.y
        y_ = math.sin(math.radians(angle))*pt.x - math.cos(math.radians(angle))*pt.y
        return carla.Vector3D(x_, y_, pt.z)

    def get_traffic_light_area(self, tl):
        base_transform = tl.get_transform()
        base_rot = base_transform.rotation.yaw

        area_loc = base_transform.transform(tl.trigger_volume.location)

        wpx = self._map.get_waypoint(area_loc)
        while not wpx.is_intersection:
            next = wpx.next(1.0)[0]
            if next:
                wpx = next
            else:
                break
        wpx_location = wpx.transform.location
        area_ext = tl.trigger_volume.extent

        area = []
        # why the 0.9 you may ask?... because the triggerboxes are set manually and sometimes they
        # cross to adjacent lanes by accident
        x_values = np.arange(-area_ext.x*0.9, area_ext.x*0.9, 1.0)
        for x in x_values:
            pt = self.rotate_point(carla.Vector3D(x, 0, area_ext.z), base_rot)
            area.append(wpx_location + carla.Location(x=pt.x, y=pt.y))

        return area_loc, area


class RunningStopTest(Criterion):

    """
    Check if an actor is running a stop sign
    """
    PROXIMITY_THRESHOLD = 50.0  # meters
    SPEED_THRESHOLD = 0.1
    WAYPOINT_STEP = 1.0  # meters

    def __init__(self, actor, name="RunningStopTest", terminate_on_failure=False):
        """
        """
        super(RunningStopTest, self).__init__(name, actor, 0, terminate_on_failure=terminate_on_failure)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._actor = actor
        self._world = CarlaDataProvider.get_world()
        self._map = CarlaDataProvider.get_map()
        self._list_stop_signs = []
        self._target_stop_sign = None
        self._stop_completed = False

        all_actors = self._world.get_actors()
        for _actor in all_actors:
            if 'traffic.stop' in _actor.type_id:
                self._list_stop_signs.append(_actor)

    @staticmethod
    def point_inside_boundingbox(point, bb_center, bb_extent):
        """
        X
        :param point:
        :param bb_center:
        :param bb_extent:
        :return:
        """

        # pylint: disable=invalid-name
        A = carla.Vector2D(bb_center.x - bb_extent.x, bb_center.y - bb_extent.y)
        B = carla.Vector2D(bb_center.x + bb_extent.x, bb_center.y - bb_extent.y)
        D = carla.Vector2D(bb_center.x - bb_extent.x, bb_center.y + bb_extent.y)
        M = carla.Vector2D(point.x, point.y)

        AB = B - A
        AD = D - A
        AM = M - A
        am_ab = AM.x * AB.x + AM.y * AB.y
        ab_ab = AB.x * AB.x + AB.y * AB.y
        am_ad = AM.x * AD.x + AM.y * AD.y
        ad_ad = AD.x * AD.x + AD.y * AD.y

        return am_ab > 0 and am_ab < ab_ab and am_ad > 0 and am_ad < ad_ad

    def is_actor_affected_by_stop(self, actor, stop, multi_step=20):
        """
        Check if the given actor is affected by the stop
        """
        affected = False
        # first we run a fast coarse test
        current_location = actor.get_location()
        stop_location = stop.get_transform().location
        if stop_location.distance(current_location) > self.PROXIMITY_THRESHOLD:
            return affected

        # print("Affected by stop!")
        stop_t = stop.get_transform()
        transformed_tv = stop_t.transform(stop.trigger_volume.location)

        # slower and accurate test based on waypoint's horizon and geometric test
        list_locations = [current_location]
        waypoint = self._map.get_waypoint(current_location)
        for _ in range(multi_step):
            if waypoint:
                waypoint = waypoint.next(self.WAYPOINT_STEP)[0]
                if not waypoint:
                    break
                list_locations.append(waypoint.transform.location)

        for actor_location in list_locations:
            if self.point_inside_boundingbox(actor_location, transformed_tv, stop.trigger_volume.extent):
                affected = True

        return affected

    def _scan_for_stop_sign(self):
        target_stop_sign = None
        for stop_sign in self._list_stop_signs:
            if self.is_actor_affected_by_stop(self._actor, stop_sign):
                # this stop sign is affecting the vehicle
                target_stop_sign = stop_sign
                break

        return target_stop_sign

    def update(self):
        """
        Check if the actor is running a red light
        """
        new_status = py_trees.common.Status.RUNNING

        location = self._actor.get_location()
        if location is None:
            return new_status

        if not self._target_stop_sign:
            # scan for stop signs
            self._target_stop_sign = self._scan_for_stop_sign()
        else:
            # we were in the middle of dealing with a stop sign
            if not self.is_actor_affected_by_stop(self._actor, self._target_stop_sign):
                # is the vehicle out of the influence of this stop sign now?
                if not self._stop_completed:
                    # did we stop?
                    self.test_status = "FAILURE"
                    stop_location = self._target_stop_sign.get_transform().location
                    running_stop_event = TrafficEvent(event_type=TrafficEventType.STOP_INFRACTION)
                    running_stop_event.set_message("Agent ran a stop {} at (x={}, y={}, x={})".format(
                        self._target_stop_sign.id,
                        stop_location.x,
                        stop_location.y,
                        stop_location.z))
                    running_stop_event.set_dict({'id': self._target_stop_sign.id,
                                                 'x': stop_location.x,
                                                 'y': stop_location.y,
                                                 'z': stop_location.z})

                    self.list_traffic_events.append(running_stop_event)

                # reset state
                self._target_stop_sign = None
                self._stop_completed = False

        if self._target_stop_sign:
            # we are already dealing with a target stop sign
            #
            # did the ego-vehicle stop?
            current_speed = CarlaDataProvider.get_velocity(self._actor)
            if current_speed < self.SPEED_THRESHOLD:
                self._stop_completed = True

        if self._terminate_on_failure and (self.test_status == "FAILURE"):
            new_status = py_trees.common.Status.FAILURE

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status
