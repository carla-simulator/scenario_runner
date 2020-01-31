#!/usr/bin/env python

# Copyright (c) 2018-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides all atomic evaluation criteria required to analyze if a
scenario was completed successfully or failed.

Criteria should run continuously to monitor the state of a single actor, multiple
actors or environmental parameters. Hence, a termination is not required.

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

    def initialise(self):
        """
        Initialise the criterion. Can be extended by the user-derived class
        """
        self.logger.debug("%s.initialise()" % (self.__class__.__name__))

    def terminate(self, new_status):
        """
        Terminate the criterion. Can be extended by the user-derived class
        """
        if (self.test_status == "RUNNING") or (self.test_status == "INIT"):
            self.test_status = "SUCCESS"

        self.logger.debug("%s.terminate()[%s->%s]" % (self.__class__.__name__, self.status, new_status))


class MaxVelocityTest(Criterion):

    """
    This class contains an atomic test for maximum velocity.

    Important parameters:
    - actor: CARLA actor to be used for this test
    - max_velocity_allowed: maximum allowed velocity in m/s
    - optional [optional]: If True, the result is not considered for an overall pass/fail result
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

    Important parameters:
    - actor: CARLA actor to be used for this test
    - distance_success: If the actor's driven distance is more than this value (in meters),
                        the test result is SUCCESS
    - distance_acceptable: If the actor's driven distance is more than this value (in meters),
                           the test result is ACCEPTABLE
    - optional [optional]: If True, the result is not considered for an overall pass/fail result
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

    def terminate(self, new_status):
        """
        Set final status
        """
        if self.test_status != "SUCCESS":
            self.test_status = "FAILURE"
        super(DrivenDistanceTest, self).terminate(new_status)


class AverageVelocityTest(Criterion):

    """
    This class contains an atomic test for average velocity.

    Important parameters:
    - actor: CARLA actor to be used for this test
    - avg_velocity_success: If the actor's average velocity is more than this value (in m/s),
                            the test result is SUCCESS
    - avg_velocity_acceptable: If the actor's average velocity is more than this value (in m/s),
                               the test result is ACCEPTABLE
    - optional [optional]: If True, the result is not considered for an overall pass/fail result
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

    def terminate(self, new_status):
        """
        Set final status
        """
        if self.test_status == "RUNNING":
            self.test_status = "FAILURE"
        super(AverageVelocityTest, self).terminate(new_status)


class CollisionTest(Criterion):

    """
    This class contains an atomic test for collisions.

    Important parameters:
    - actor: CARLA actor to be used for this test
    - terminate_on_failure [optional]: If True, the complete scenario will terminate upon failure of this test
    - optional [optional]: If True, the result is not considered for an overall pass/fail result
    """

    MIN_AREA_OF_COLLISION = 3
    MAX_AREA_OF_COLLISION = 5       # If further than this distance, the area if forgotten

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
        self.registered_collisions = []

    def update(self):
        """
        Check collision count
        """
        new_status = py_trees.common.Status.RUNNING

        if self._terminate_on_failure and (self.test_status == "FAILURE"):
            new_status = py_trees.common.Status.FAILURE

        actor_location = self.actor.get_location()
        new_registered_collisions = []

        # Loops through all the previous registered collisions
        for collision_location in self.registered_collisions:

            # Get the distance to the collision point
            distance_vector = actor_location - collision_location
            distance = math.sqrt(math.pow(distance_vector.x, 2) + math.pow(distance_vector.y, 2))

            # If far away from a previous collision, forget it
            if distance <= self.MAX_AREA_OF_COLLISION:
                new_registered_collisions.append(collision_location)

        self.registered_collisions = new_registered_collisions

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

        self.test_status = "FAILURE"
        self.actual_value += 1

        actor_location = self.actor.get_location()

        # Loops through all the previous registered collisions
        for collision_location in self.registered_collisions:

            # Get the distance to the collision point
            distance_vector = actor_location - collision_location
            distance = math.sqrt(math.pow(distance_vector.x, 2) + math.pow(distance_vector.y, 2))

            # Ignore the current one if close to a previous one
            if distance <= self.MIN_AREA_OF_COLLISION:
                self.actual_value -= 1
                registered = True
                break

        # Register it if needed
        if not registered:

            if ('static' in event.other_actor.type_id or 'traffic' in event.other_actor.type_id) \
                and 'sidewalk' not in event.other_actor.type_id:
                actor_type = TrafficEventType.COLLISION_STATIC

            elif 'vehicle' in event.other_actor.type_id:
                actor_type = TrafficEventType.COLLISION_VEHICLE

            elif 'walker' in event.other_actor.type_id:
                actor_type = TrafficEventType.COLLISION_PEDESTRIAN

            collision_event = TrafficEvent(event_type=actor_type)
            collision_event.set_dict({
                'type': event.other_actor.type_id,
                'id': event.other_actor.id,
                'x': round(actor_location.x, 3),
                'y': round(actor_location.y, 3),
                'z': round(actor_location.z, 3)})
            collision_event.set_message(
                "Agent collided against object with type={} and id={} at (x={}, y={}, z={})".format(
                event.other_actor.type_id,
                event.other_actor.id,
                round(actor_location.x, 3),
                round(actor_location.y, 3),
                round(actor_location.z, 3)))

            self.registered_collisions.append(actor_location)
            self.list_traffic_events.append(collision_event)


class KeepLaneTest(Criterion):

    """
    This class contains an atomic test for keeping lane.

    Important parameters:
    - actor: CARLA actor to be used for this test
    - optional [optional]: If True, the result is not considered for an overall pass/fail result
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

    Important parameters:
    - actor: CARLA actor to be used for this test
    - min_x, max_x, min_y, max_y: Bounding box of the checked region
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

    Important parameters:
    - actor: CARLA actor to be used for this test
    - optional [optional]: If True, the result is not considered for an overall pass/fail result
    """

    def __init__(self, actor, optional=False, name="OnSidewalkTest"):
        """
        Construction with sensor setup
        """
        super(OnSidewalkTest, self).__init__(name, actor, 0, None, optional)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))

        self._actor = actor
        self._map = CarlaDataProvider.get_map()
        self._onsidewalk_active = False
        self._outside_lane_active = False

        self._actor_location = self._actor.get_location()
        self._wrong_sidewalk_distance = 0
        self._wrong_outside_lane_distance = 0
        self._sidewalk_start_location = None
        self._outside_lane_start_location = None

    def update(self):
        """
        Check lane invasion count
        """
        new_status = py_trees.common.Status.RUNNING

        if self._terminate_on_failure and (self.test_status == "FAILURE"):
            new_status = py_trees.common.Status.FAILURE

        # Some of the vehicle parameters
        current_transform = self._actor.get_transform()
        current_location = current_transform.location
        current_waypoint = self._map.get_waypoint(current_location, lane_type=carla.LaneType.Any)

        # Case 1) Car center is at a sidewalk
        if current_waypoint.lane_type == carla.LaneType.Sidewalk:
            if not self._onsidewalk_active:
                self.test_status = "FAILURE"
                self._onsidewalk_active = True
                self._sidewalk_start_location = current_location

        # Case 2) Not inside allowed zones (Driving and Parking)
        elif current_waypoint.lane_type != carla.LaneType.Driving \
            and current_waypoint.lane_type != carla.LaneType.Parking:

            # Get the vertices of the vehicle
            heading_vector = current_transform.get_forward_vector()
            heading_vector.z = 0
            heading_vector = heading_vector/math.sqrt(math.pow(heading_vector.x, 2) + math.pow(heading_vector.y, 2))
            perpendicular_vector = carla.Vector3D(-heading_vector.y, heading_vector.x, 0)

            extent = self.actor.bounding_box.extent
            x_boundary_vector = heading_vector * extent.x
            y_boundary_vector = perpendicular_vector * extent.y

            bounding_box = [
                current_location + carla.Location(x_boundary_vector - y_boundary_vector),
                current_location + carla.Location(x_boundary_vector + y_boundary_vector),
                current_location + carla.Location(-1*x_boundary_vector - y_boundary_vector),
                current_location + carla.Location(-1*x_boundary_vector + y_boundary_vector)]

            bounding_box_points = [
                self._map.get_waypoint(bounding_box[0], lane_type=carla.LaneType.Any),
                self._map.get_waypoint(bounding_box[1], lane_type=carla.LaneType.Any),
                self._map.get_waypoint(bounding_box[2], lane_type=carla.LaneType.Any),
                self._map.get_waypoint(bounding_box[3], lane_type=carla.LaneType.Any)]

            #Case 2.1) Not quite outside yet
            if bounding_box_points[0].lane_type == (carla.LaneType.Driving or carla.LaneType.Parking) \
                or bounding_box_points[1].lane_type == (carla.LaneType.Driving or carla.LaneType.Parking) \
                or bounding_box_points[2].lane_type == (carla.LaneType.Driving or carla.LaneType.Parking) \
                or bounding_box_points[3].lane_type == (carla.LaneType.Driving or carla.LaneType.Parking):

                self._onsidewalk_active = False
                self._outside_lane_active = False

            # Case 2.2) At the mini Shoulders between Driving and Sidewalk
            elif bounding_box_points[0].lane_type == carla.LaneType.Sidewalk \
                or bounding_box_points[1].lane_type == carla.LaneType.Sidewalk \
                or bounding_box_points[2].lane_type == carla.LaneType.Sidewalk \
                or bounding_box_points[3].lane_type == carla.LaneType.Sidewalk:

                if not self._onsidewalk_active:
                    self.test_status = "FAILURE"
                    self._onsidewalk_active = True
                    self._sidewalk_start_location = current_location


            else:
                distance_vehicle_waypoint = current_location.distance(current_waypoint.transform.location)

                # Case 2.3) Outside lane
                if distance_vehicle_waypoint >= current_waypoint.lane_width / 2:

                    if not self._outside_lane_active:
                        self.test_status = "FAILURE"
                        self._outside_lane_active = True
                        self._outside_lane_start_location = current_location

                # Case 2.4) Very very edge case (but still inside driving lanes)
                else:
                    self._onsidewalk_active = False
                    self._outside_lane_active = False

        # Case 3) Driving and Parking conditions
        else:
            # Check for false positives at junctions
            if current_waypoint.is_junction:
                distance_vehicle_waypoint = math.sqrt(
                    math.pow(current_waypoint.transform.location.x - current_location.x, 2) +
                    math.pow(current_waypoint.transform.location.y - current_location.y, 2))

                if distance_vehicle_waypoint <= current_waypoint.lane_width / 2:
                    self._onsidewalk_active = False
                    self._outside_lane_active = False
                # Else, do nothing, the waypoint is too far to consider it a correct position
            else:

                self._onsidewalk_active = False
                self._outside_lane_active = False

        # Update the distances
        distance_vector = self._actor.get_location() - self._actor_location
        distance = math.sqrt(math.pow(distance_vector.x, 2) + math.pow(distance_vector.y, 2))

        if distance >= 0.02: # Used to avoid micro-changes adding to considerable sums
            self._actor_location = self._actor.get_location()

            if self._onsidewalk_active:
                self._wrong_sidewalk_distance += distance
            elif self._outside_lane_active:
                # Only add if car is outside the lane but ISN'T in a junction
                self._wrong_outside_lane_distance += distance

        # Register the sidewalk event
        if not self._onsidewalk_active and self._wrong_sidewalk_distance > 0:

            self.actual_value += 1

            onsidewalk_event = TrafficEvent(event_type=TrafficEventType.ON_SIDEWALK_INFRACTION)
            onsidewalk_event.set_message(
                'Agent invaded the sidewalk for about {} meters, starting at (x={}, y={}, z={})'.format(
                round(self._wrong_sidewalk_distance, 3),
                round(self._sidewalk_start_location.x, 3),
                round(self._sidewalk_start_location.y, 3),
                round(self._sidewalk_start_location.z, 3)))
            onsidewalk_event.set_dict({
                'x': round(self._sidewalk_start_location.x, 3),
                'y': round(self._sidewalk_start_location.y, 3),
                'z': round(self._sidewalk_start_location.z, 3),
                'distance': round(self._wrong_sidewalk_distance, 3)})

            self._onsidewalk_active = False
            self._wrong_sidewalk_distance = 0
            self.list_traffic_events.append(onsidewalk_event)

        # Register the outside of a lane event
        if not self._outside_lane_active and self._wrong_outside_lane_distance > 0:

            self.actual_value += 1

            outsidelane_event = TrafficEvent(event_type=TrafficEventType.OUTSIDE_LANE_INFRACTION)
            outsidelane_event.set_message(
                'Agent went outside the lane for about {} meters, starting at (x={}, y={}, z={})'.format(
                round(self._wrong_outside_lane_distance, 3),
                round(self._outside_lane_start_location.x, 3),
                round(self._outside_lane_start_location.y, 3),
                round(self._outside_lane_start_location.z, 3)))
            outsidelane_event.set_dict({
                'x': round(self._outside_lane_start_location.x, 3),
                'y': round(self._outside_lane_start_location.y, 3),
                'z': round(self._outside_lane_start_location.z, 3),
                'distance': round(self._wrong_outside_lane_distance, 3)})

            self._outside_lane_active = False
            self._wrong_outside_lane_distance = 0
            self.list_traffic_events.append(outsidelane_event)

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status

    def terminate(self, new_status):
        """
        If there is currently an event running, it is registered
        """
        # If currently at a sidewalk, register the event
        if self._onsidewalk_active:

            self.actual_value += 1

            onsidewalk_event = TrafficEvent(event_type=TrafficEventType.ON_SIDEWALK_INFRACTION)
            onsidewalk_event.set_message(
                'Agent invaded the sidewalk for {} meters, starting at (x={}, y={}, z={})'.format(
                round(self._wrong_sidewalk_distance, 3),
                round(self._sidewalk_start_location.x, 3),
                round(self._sidewalk_start_location.y, 3),
                round(self._sidewalk_start_location.z, 3)))
            onsidewalk_event.set_dict({
                'x': round(self._sidewalk_start_location.x, 3),
                'y': round(self._sidewalk_start_location.y, 3),
                'z': round(self._sidewalk_start_location.z, 3),
                'distance': round(self._wrong_sidewalk_distance, 3)})

            self._onsidewalk_active = False
            self._wrong_sidewalk_distance = 0
            self.list_traffic_events.append(onsidewalk_event)

        # If currently outside of our lane, register the event
        if self._outside_lane_active:

            self.actual_value += 1

            outsidelane_event = TrafficEvent(event_type=TrafficEventType.OUTSIDE_LANE_INFRACTION)
            outsidelane_event.set_message(
                'Agent went outside the lane for about {} meters, starting at (x={}, y={}, z={})'.format(
                round(self._wrong_outside_lane_distance, 3),
                round(self._outside_lane_start_location.x, 3),
                round(self._outside_lane_start_location.y, 3),
                round(self._outside_lane_start_location.z, 3)))
            outsidelane_event.set_dict({
                'x': round(self._outside_lane_start_location.x, 3),
                'y': round(self._outside_lane_start_location.y, 3),
                'z': round(self._outside_lane_start_location.z, 3),
                'distance': round(self._wrong_outside_lane_distance, 3)})

            self._outside_lane_active = False
            self._wrong_outside_lane_distance = 0
            self.list_traffic_events.append(outsidelane_event)


class WrongLaneTest(Criterion):

    """
    This class contains an atomic test to detect invasions to wrong direction lanes.

    Important parameters:
    - actor: CARLA actor to be used for this test
    - optional [optional]: If True, the result is not considered for an overall pass/fail result
    """
    MAX_ALLOWED_ANGLE = 120.0
    MAX_ALLOWED_WAYPOINT_ANGLE = 150.0

    def __init__(self, actor, optional=False, name="WrongLaneTest"):
        """
        Construction with sensor setup
        """
        super(WrongLaneTest, self).__init__(name, actor, 0, None, optional)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))

        self._actor = actor
        self._map = CarlaDataProvider.get_map()
        self._last_lane_id = None
        self._last_road_id = None

        self._in_lane = True
        self._wrong_distance = 0
        self._actor_location = self._actor.get_location()
        self._previous_lane_waypoint = self._map.get_waypoint(self._actor.get_location())
        self._wrong_lane_start_location = None

    def update(self):
        """
        Check lane invasion count
        """

        new_status = py_trees.common.Status.RUNNING

        if self._terminate_on_failure and (self.test_status == "FAILURE"):
            new_status = py_trees.common.Status.FAILURE

        lane_waypoint = self._map.get_waypoint(self._actor.get_location())
        current_lane_id = lane_waypoint.lane_id
        current_road_id = lane_waypoint.road_id

        if (self._last_road_id != current_road_id or self._last_lane_id != current_lane_id) and not lane_waypoint.is_junction:
            next_waypoint = lane_waypoint.next(2.0)[0]

            if not next_waypoint:
                return

            # The waypoint route direction can be considered continuous.
            # Therefore just check for a big gap in waypoint directions.
            previous_lane_direction = self._previous_lane_waypoint.transform.get_forward_vector()
            current_lane_direction = lane_waypoint.transform.get_forward_vector()

            previous_lane_vector = np.array([previous_lane_direction.x, previous_lane_direction.y])
            current_lane_vector = np.array([current_lane_direction.x, current_lane_direction.y])

            waypoint_angle = math.degrees(
                 math.acos(np.clip(np.dot(previous_lane_vector, current_lane_vector) /
                    (np.linalg.norm(previous_lane_vector)*np.linalg.norm(current_lane_vector)), -1.0, 1.0)))

            if waypoint_angle > self.MAX_ALLOWED_WAYPOINT_ANGLE and self._in_lane:

                self.test_status = "FAILURE"
                self._in_lane = False
                self.actual_value += 1
                self._wrong_lane_start_location = self._actor.get_location()

            else:
                # Reset variables
                self._in_lane = True

            # Continuity is broken after a junction so check vehicle-lane angle instead
            if self._previous_lane_waypoint.is_junction:

                vector_wp = np.array([next_waypoint.transform.location.x - lane_waypoint.transform.location.x,
                                  next_waypoint.transform.location.y - lane_waypoint.transform.location.y])

                vector_actor = np.array([math.cos(math.radians(self._actor.get_transform().rotation.yaw)),
                                     math.sin(math.radians(self._actor.get_transform().rotation.yaw))])

                vehicle_lane_angle = math.degrees(
                    math.acos(np.clip(np.dot(vector_actor, vector_wp) / (np.linalg.norm(vector_wp)), -1.0, 1.0)))

                if vehicle_lane_angle > self.MAX_ALLOWED_ANGLE:

                    self.test_status = "FAILURE"
                    self._in_lane = False
                    self.actual_value += 1
                    self._wrong_lane_start_location = self._actor.get_location()

        # Keep adding "meters" to the counter
        distance_vector = self._actor.get_location() - self._actor_location
        distance = math.sqrt(math.pow(distance_vector.x, 2) + math.pow(distance_vector.y, 2))

        if distance >= 0.02: # Used to avoid micro-changes adding add to considerable sums
            self._actor_location = self._actor.get_location()

            if not self._in_lane and not lane_waypoint.is_junction:
                self._wrong_distance += distance

        # Register the event
        if self._in_lane and self._wrong_distance > 0:

            wrong_way_event = TrafficEvent(event_type=TrafficEventType.WRONG_WAY_INFRACTION)
            wrong_way_event.set_message(
                "Agent invaded a lane in opposite direction for {} meters, starting at (x={}, y={}, z={}). "
                "road_id={}, lane_id={}".format(
                round(self._wrong_distance, 3),
                round(self._wrong_lane_start_location.x, 3),
                round(self._wrong_lane_start_location.y, 3),
                round(self._wrong_lane_start_location.z, 3),
                self._last_road_id,
                self._last_lane_id))
            wrong_way_event.set_dict({
                'x': round(self._wrong_lane_start_location.x, 3),
                'y': round(self._wrong_lane_start_location.y, 3),
                'z': round(self._wrong_lane_start_location.y, 3),
                'distance': round(self._wrong_distance, 3),
                'road_id': self._last_road_id,
                'lane_id': self._last_lane_id})

            self.list_traffic_events.append(wrong_way_event)
            self._wrong_distance = 0

        # Remember the last state
        self._last_lane_id = current_lane_id
        self._last_road_id = current_road_id
        self._previous_lane_waypoint = lane_waypoint

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status

    def terminate(self, new_status):
        """
        If there is currently an event running, it is registered
        """
        if not self._in_lane:

            lane_waypoint = self._map.get_waypoint(self._actor.get_location())
            current_lane_id = lane_waypoint.lane_id
            current_road_id = lane_waypoint.road_id

            wrong_way_event = TrafficEvent(event_type=TrafficEventType.WRONG_WAY_INFRACTION)
            wrong_way_event.set_message(
                "Agent invaded a lane in opposite direction for {} meters, starting at (x={}, y={}, z={}). "
                "road_id={}, lane_id={}".format(
                round(self._wrong_distance, 3),
                round(self._wrong_lane_start_location.x, 3),
                round(self._wrong_lane_start_location.y, 3),
                round(self._wrong_lane_start_location.z, 3),
                current_road_id,
                current_lane_id))
            wrong_way_event.set_dict({
                'x': round(self._wrong_lane_start_location.x, 3),
                'y': round(self._wrong_lane_start_location.y, 3),
                'z': round(self._wrong_lane_start_location.y, 3),
                'distance': round(self._wrong_distance, 3),
                'road_id': current_road_id,
                'lane_id': current_lane_id})

            self._wrong_distance = 0
            self._in_lane = True
            self.list_traffic_events.append(wrong_way_event)


class InRadiusRegionTest(Criterion):

    """
    The test is a success if the actor is within a given radius of a specified region

    Important parameters:
    - actor: CARLA actor to be used for this test
    - x, y, radius: Position (x,y) and radius (in meters) used to get the checked region
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

    Important parameters:
    - actor: CARLA actor to be used for this test
    - radius: Allowed radius around the route (meters)
    - route: Route to be checked
    - offroad_max: Maximum allowed distance the actor can deviate from the route, when not driving on a road (meters)
    - terminate_on_failure [optional]: If True, the complete scenario will terminate upon failure of this test
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
                route_deviation_event.set_message(
                    "Agent deviated from the route at (x={}, y={}, z={})".format(
                    round(location.x, 3),
                    round(location.y, 3),
                    round(location.z, 3)))
                route_deviation_event.set_dict({
                    'x': round(location.x, 3),
                    'y': round(location.y, 3),
                    'z': round(location.z, 3)})

                self.list_traffic_events.append(route_deviation_event)

                self.test_status = "FAILURE"
                new_status = py_trees.common.Status.FAILURE

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status


class RouteCompletionTest(Criterion):

    """
    Check at which stage of the route is the actor at each tick

    Important parameters:
    - actor: CARLA actor to be used for this test
    - route: Route to be checked
    - terminate_on_failure [optional]: If True, the complete scenario will terminate upon failure of this test
    """
    DISTANCE_THRESHOLD = 15.0  # meters
    WINDOWS_SIZE = 2

    def __init__(self, actor, route, name="RouteCompletionTest", terminate_on_failure=False):
        """
        """
        super(RouteCompletionTest, self).__init__(name, actor, 100, terminate_on_failure=terminate_on_failure)
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
                accum = self._accum_meters[i - 1]
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
                    self._traffic_event.set_dict({
                        'route_completed': self._percentage_route_completed})
                    self._traffic_event.set_message(
                        "Agent has completed > {:.2f}% of the route".format(
                        self._percentage_route_completed))

            if self._percentage_route_completed > 99.0 and location.distance(self.target) < self.DISTANCE_THRESHOLD:
                route_completion_event = TrafficEvent(event_type=TrafficEventType.ROUTE_COMPLETED)
                route_completion_event.set_message("Destination was successfully reached")
                self.list_traffic_events.append(route_completion_event)
                self.test_status = "SUCCESS"

        elif self.test_status == "SUCCESS":
            new_status = py_trees.common.Status.SUCCESS

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        self.actual_value = self._percentage_route_completed

        return new_status

    def terminate(self, new_status):
        """
        Set test status to failure if not successful and terminate
        """
        if self.test_status == "INIT":
            self.test_status = "FAILURE"
        super(RouteCompletionTest, self).terminate(new_status)


class RunningRedLightTest(Criterion):

    """
    Check if an actor is running a red light

    Important parameters:
    - actor: CARLA actor to be used for this test
    - terminate_on_failure [optional]: If True, the complete scenario will terminate upon failure of this test
    """
    DISTANCE_LIGHT = 10  # m

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

    # pylint: disable=no-self-use
    def is_vehicle_crossing_line(self, seg1, seg2):
        """
        check if vehicle crosses a line segment
        """
        line1 = shapely.geometry.LineString([(seg1[0].x, seg1[0].y), (seg1[1].x, seg1[1].y)])
        line2 = shapely.geometry.LineString([(seg2[0].x, seg2[0].y), (seg2[1].x, seg2[1].y)])
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
                z = 2.1
                if traffic_light.state == carla.TrafficLightState.Red:
                    color = carla.Color(255, 0, 0)
                elif traffic_light.state == carla.TrafficLightState.Green:
                    color = carla.Color(0, 255, 0)
                else:
                    color = carla.Color(255, 255, 255)
                self._world.debug.draw_point(center + carla.Location(z=z), size=0.2, color=color, life_time=0.01)
                for pt in area:
                    self._world.debug.draw_point(pt + carla.Location(z=z), size=0.1, color=color, life_time=0.01)
                for wp in waypoints:
                    text = "{}.{}".format(wp.road_id, wp.lane_id)
                    self._world.debug.draw_string(
                        wp.transform.location, text, draw_shadow=False, color=color, life_time=0.01)

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
                        self.actual_value += 1
                        location = traffic_light.get_transform().location
                        red_light_event = TrafficEvent(event_type=TrafficEventType.TRAFFIC_LIGHT_INFRACTION)
                        red_light_event.set_message(
                            "Agent ran a red light {} at (x={}, y={}, z={})".format(
                            traffic_light.id,
                            round(location.x, 3),
                            round(location.y, 3),
                            round(location.z, 3)))
                        red_light_event.set_dict({
                            'id': traffic_light.id,
                            'x': round(location.x, 3),
                            'y': round(location.y, 3),
                            'z': round(location.z, 3)})

                        self.list_traffic_events.append(red_light_event)
                        self._last_red_light_id = traffic_light.id
                        break

        if self._terminate_on_failure and (self.test_status == "FAILURE"):
            new_status = py_trees.common.Status.FAILURE

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status

    def rotate_point(self, point, angle):
        """
        rotate a given point by a given angle
        """
        x_ = math.cos(math.radians(angle)) * point.x - math.sin(math.radians(angle)) * point.y
        y_ = math.sin(math.radians(angle)) * point.x - math.cos(math.radians(angle)) * point.y
        return carla.Vector3D(x_, y_, point.z)

    def get_traffic_light_area(self, traffic_light):
        """
        get area of a given traffic light
        """
        base_transform = traffic_light.get_transform()
        base_rot = base_transform.rotation.yaw

        area_loc = base_transform.transform(traffic_light.trigger_volume.location)

        wpx = self._map.get_waypoint(area_loc)
        while not wpx.is_intersection:
            next_wp = wpx.next(1.0)[0]
            if next_wp:
                wpx = next_wp
            else:
                break
        wpx_location = wpx.transform.location
        area_ext = traffic_light.trigger_volume.extent

        area = []
        # why the 0.9 you may ask?... because the triggerboxes are set manually and sometimes they
        # cross to adjacent lanes by accident
        x_values = np.arange(-area_ext.x * 0.9, area_ext.x * 0.9, 1.0)
        for x in x_values:
            point = self.rotate_point(carla.Vector3D(x, 0, area_ext.z), base_rot)
            area.append(wpx_location + carla.Location(x=point.x, y=point.y))

        return area_loc, area


class RunningStopTest(Criterion):

    """
    Check if an actor is running a stop sign

    Important parameters:
    - actor: CARLA actor to be used for this test
    - terminate_on_failure [optional]: If True, the complete scenario will terminate upon failure of this test
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
                    running_stop_event.set_message(
                        "Agent ran a stop {} at (x={}, y={}, z={})".format(
                        self._target_stop_sign.id,
                        round(stop_location.x, 3),
                        round(stop_location.y, 3),
                        round(stop_location.z, 3)))
                    running_stop_event.set_dict({
                        'id': self._target_stop_sign.id,
                        'x': round(stop_location.x, 3),
                        'y': round(stop_location.y, 3),
                        'z': round(stop_location.z, 3)})

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
