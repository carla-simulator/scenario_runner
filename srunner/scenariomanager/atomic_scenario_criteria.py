#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: Fabian Oboril (fabian.oboril@intel.com)
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

        actor_type = None
        if 'static' in event.other_actor.type_id:
            actor_type = TrafficEventType.COLLISION_STATIC
        elif 'vehicle' in event.other_actor.type_id:
            actor_type = TrafficEventType.COLLISION_VEHICLE
        elif 'walker' in event.other_actor.type_id:
            actor_type = TrafficEventType.COLLISION_PEDESTRIAN

        collision_event = TrafficEvent(type=actor_type)
        collision_event.set_dict({'type':event.other_actor.type_id, 'id': event.other_actor.id})
        collision_event.set_message("Agent collided against object with type={} and id={}".format(
            event.other_actor.type_id, event.other_actor.id))

        self.list_traffic_events.append(collision_event)
        self.actual_value += 1


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
        blueprint = world.get_blueprint_library().find('sensor.other.lane_detector')
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
        self._map = self._world.get_map()
        self._infractions = 0
        self._last_lane_id = None
        self._last_road_id = None

        blueprint = self._world.get_blueprint_library().find('sensor.other.lane_detector')
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
        self = weak_self()
        if not self:
            return

        # check the lane direction
        lane_waypoint = self._map.get_waypoint(self._actor.get_location())
        current_lane_id = lane_waypoint.lane_id
        current_road_id = lane_waypoint.road_id

        if not (self._last_road_id == current_road_id and self._last_lane_id == current_lane_id):
            next_waypoint = lane_waypoint.next(2.0)[0]

            vector_wp = np.array([next_waypoint.transform.location.x - lane_waypoint.transform.location.x,
                                  next_waypoint.transform.location.y - lane_waypoint.transform.location.y])

            vector_actor = np.array([math.cos(math.radians(self._actor.get_transform().rotation.yaw)),
                                     math.sin(math.radians(self._actor.get_transform().rotation.yaw))])

            ang = math.degrees(math.acos(np.clip(np.dot(vector_actor, vector_wp) / (np.linalg.norm(vector_wp)), -1.0, 1.0)))
            if ang > self.MAX_ALLOWED_ANGLE:
                self.test_status = "FAILURE"
                # is there a difference of orientation greater than MAX_ALLOWED_ANGLE deg with respect of the lane
                # direction?
                self._infractions += 1

                wrong_way_event = TrafficEvent(type=TrafficEventType.WRONG_WAY_INFRACTION)
                wrong_way_event.set_message('Agent invaded a lane in opposite direction: road_id={}, lane_id={}'.format(
                    current_road_id, current_lane_id))
                wrong_way_event.set_dict({'road_id':current_road_id, 'lane_id':current_lane_id})
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
        self._x = x
        self._y = y
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
                route_completion_event = TrafficEvent(type=TrafficEventType.ROUTE_COMPLETED)
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

        def __init__(self, actor, radius, route, offroad_max, name="InRouteTest", terminate_on_failure=False):
            """
            """
            super(InRouteTest, self).__init__(name, actor, 0, terminate_on_failure=terminate_on_failure)
            self.logger.debug("%s.__init__()" % (self.__class__.__name__))
            self._actor = actor
            self._radius = radius
            self._route = route
            self._offroad_max = offroad_max

            self._counter_off_route = 0
            self._waypoints, _ = zip(*self._route)

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
                for waypoint in self._waypoints:
                    distance = math.sqrt(((location.x - waypoint.x) ** 2) + ((location.y - waypoint.y) ** 2))
                    if distance < self._radius:
                        off_route = False
                        break
                if off_route:
                    self._counter_off_route += 1

                if self._counter_off_route > self._offroad_max:
                    route_deviation_event = TrafficEvent(type=TrafficEventType.ROUTE_DEVIATION)
                    route_deviation_event.set_message("Agent deviated from the route at (x={}, y={}, z={})".format(
                        location.x, location.y, location.z))
                    route_deviation_event.set_dict({'x':location.x, 'y':location.y, 'z':location.z})
                    self.list_traffic_events.append(route_deviation_event)

                    self.test_status = "FAILURE"
                    new_status = py_trees.common.Status.FAILURE

            self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

            return new_status


class RouteCompletionTest(Criterion):
    """
    Check at which stage of the route is the actor at each tick
    """

    def __init__(self, actor, route, name="RouteCompletionTest", terminate_on_failure=False):
        """
        """
        super(RouteCompletionTest, self).__init__(name, actor, 0, terminate_on_failure=terminate_on_failure)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._actor = actor
        self._route = route

        self._current_index = 0
        self._route_length = len(self._route)
        self._waypoints, _ = zip(*self._route)

        self._traffic_event = TrafficEvent(type=TrafficEventType.ROUTE_COMPLETION)
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
            best_distance = float("inf")
            best_index = self._current_index
            for index in range(self._current_index, self._route_length):
                ref_waypoint = self._waypoints[index]
                distance = math.sqrt(((location.x - ref_waypoint.x) ** 2) + ((location.y - ref_waypoint.y) ** 2))
                if distance < best_distance:
                    best_distance = distance
                    best_index = index
            self._current_index = best_index
            self._percentage_route_completed = 100.0*float(self._current_index) / float(self._route_length)
            self._traffic_event.set_dict({'route_completed': self._percentage_route_completed})
            self._traffic_event.set_message("Agent has completed > {:.2f}% of the route".format(self._percentage_route_completed))
        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status


class RunningRedLightTest(Criterion):
    """
    Check if an actor is running a red light
    """
    def __init__(self, actor, name="RunningRedLightTest", terminate_on_failure=False):
        """
        """
        super(RunningRedLightTest, self).__init__(name, actor, 0, terminate_on_failure=terminate_on_failure)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._actor = actor
        self._world = actor.get_world()
        self._list_traffic_lights = []
        self._target_traffic_light = None
        self._in_red_light = False

        all_actors = self._world.get_actors()
        for actor in all_actors:
            if 'traffic_light' in actor.type_id:
                self._list_traffic_lights.append(actor)

    @staticmethod
    def length(v):
      return math.sqrt(v.x**2 + v.y**2 + v.z**2)

    def update(self):
        """
        Check if the actor is running a red light
        """
        new_status = py_trees.common.Status.RUNNING

        location = self._actor.get_transform().location
        if location is None:
            return new_status

        # were you in affected by a red traffic light and just decided to ignore it?
        if self._in_red_light:
            if self._target_traffic_light.state != carla.TrafficLightState.Red:
                # it is safe now!
                self._in_red_light = False
                self._target_traffic_light = None

            else:
                # still red
                tl_t = self._target_traffic_light.get_transform()
                transformed_tv = tl_t.transform(self._target_traffic_light.trigger_volume.location)
                distance = carla.Location(transformed_tv).distance(location)
                s = self.length(self._target_traffic_light.trigger_volume.extent) + self.length(self._actor.bounding_box.extent)

                if distance > s and self._target_traffic_light.state == carla.TrafficLightState.Red:
                    # you are running a red light
                    self.test_status = "FAILURE"

                    red_light_event = TrafficEvent(type=TrafficEventType.TRAFFIC_LIGHT_INFRACTION)
                    red_light_event.set_message("Agent ran a red light {} at (x={}, y={}, x={})".format(
                        self._target_traffic_light.id,
                        location.x,
                        location.y,
                        location.z))
                    red_light_event.set_dict({'id':self._target_traffic_light.id, 'x': location.x,
                                              'y':location.y, 'z':location.z})
                    self.list_traffic_events.append(red_light_event)


                    # state reset
                    self._in_red_light = False
                    self._target_traffic_light = None


        # scan for red traffic lights
        for traffic_light in self._list_traffic_lights:
            if hasattr(traffic_light, 'trigger_volume'):
                tl_t = traffic_light.get_transform()

                transformed_tv = tl_t.transform(traffic_light.trigger_volume.location)
                distance = carla.Location(transformed_tv).distance(location)
                s = self.length(traffic_light.trigger_volume.extent) + self.length(self._actor.bounding_box.extent)
                if distance <= s:
                    # this traffic light is affecting the vehicle
                    if traffic_light.state == carla.TrafficLightState.Red:
                        self._target_traffic_light = traffic_light
                        self._in_red_light = True
                        break




        if self._terminate_on_failure and (self.test_status == "FAILURE"):
            new_status = py_trees.common.Status.FAILURE

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status