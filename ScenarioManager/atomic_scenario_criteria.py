#!/usr/bin/env python

#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides all atomic evaluation criteria required to analyze if a
scenario was completed successfully or failed.

The atomic criteria are implemented with py_trees.
"""

import math
import weakref

import py_trees
import carla

from ScenarioManager.timer import GameTime


class Criterion(py_trees.behaviour.Behaviour):

    """
    Base class for all criteria used to evaluate a scenario for success/failure

    Important parameters:
    - name: Name of the criterion
    - expected_value: Result in case of success (e.g. max_speed, zero collisions, ...)
    - actual_value: Actual result after running the scenario
    """

    def __init__(self, name, expected_value):
        super(Criterion, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self.name = name
        self.expected_value = expected_value
        self.actual_value = 0
        self.test_status = "INIT"
        self.terminate_on_failure = False

    def setup(self, unused_timeout=15):
        self.logger.debug("%s.setup()" % (self.__class__.__name__))
        return True

    def initialise(self):
        self.logger.debug("%s.initialise()" % (self.__class__.__name__))


class MaxVelocityTest(Criterion):

    """
    This class contains an atomic test for maximum velocity.
    """

    def __init__(self, vehicle, max_velocity_allowed, name="CheckMaximumVelocity"):
        """
        Setup vehicle and maximum allowed velovity
        """
        super(MaxVelocityTest, self).__init__(name, max_velocity_allowed)
        self.vehicle = vehicle

    def update(self):
        """
        Check velocity
        """
        if self.vehicle is None:
            return self.status

        velocity = math.sqrt(
            self.vehicle.get_velocity().x**2 + self.vehicle.get_velocity().y**2)

        self.actual_value = max(velocity, self.actual_value)

        if velocity > self.expected_value:
            self.test_status = "FAILURE"
        else:
            self.test_status = "SUCCESS"

        if self.terminate_on_failure and (self.test_status == "FAILURE"):
            new_status = py_trees.common.Status.FAILURE
        else:
            new_status = py_trees.common.Status.RUNNING

        self.logger.debug("%s.update()[%s->%s]" %
                          (self.__class__.__name__, self.status, new_status))

        return new_status

    def terminate(self, new_status):
        self.vehicle = None
        self.logger.debug("%s.terminate()[%s->%s]" % (
            self.__class__.__name__, self.status, new_status))


class DrivenDistanceTest(Criterion):

    """
    This class contains an atomic test to check the driven distance
    """

    def __init__(self, vehicle, distance, name="CheckDrivenDistance"):
        """
        Setup vehicle
        """
        super(DrivenDistanceTest, self).__init__(name, distance)
        self.vehicle = vehicle
        self.last_location = vehicle.get_location()

    def update(self):
        """
        Check distance
        """
        if self.vehicle is None:
            return self.status

        location = self.vehicle.get_location()
        self.actual_value += location.distance(self.last_location)
        self.last_location = location

        if self.actual_value > self.expected_value:
            self.test_status = "SUCCESS"
        else:
            self.test_status = "RUNNING"

        if self.terminate_on_failure and (self.test_status == "FAILURE"):
            new_status = py_trees.common.Status.FAILURE
        else:
            new_status = py_trees.common.Status.RUNNING

        self.logger.debug("%s.update()[%s->%s]" %
                          (self.__class__.__name__, self.status, new_status))

        return new_status

    def terminate(self, new_status):
        self.vehicle = None
        self.logger.debug("%s.terminate()[%s->%s]" % (
            self.__class__.__name__, self.status, new_status))


class AverageVelocityTest(Criterion):

    """
    This class contains an atomic test for average velocity.
    """

    def __init__(self, vehicle, avg_velocity, name="CheckAverageVelocity"):
        """
        Setup vehicle and average velovity expected
        """
        super(AverageVelocityTest, self).__init__(name, avg_velocity)
        self.vehicle = vehicle
        self.last_location = vehicle.get_location()
        self.distance = 0.0

    def update(self):
        """
        Check velocity
        """
        if self.vehicle is None:
            return self.status

        location = self.vehicle.get_location()
        self.distance += location.distance(self.last_location)
        self.last_location = location

        elapsed_time = GameTime.get_time()
        if elapsed_time > 0.0:
            self.actual_value = self.distance / elapsed_time

        if self.actual_value > self.expected_value:
            self.test_status = "SUCCESS"
        else:
            self.test_status = "RUNNING"

        if self.terminate_on_failure and (self.test_status == "FAILURE"):
            new_status = py_trees.common.Status.FAILURE
        else:
            new_status = py_trees.common.Status.RUNNING

        self.logger.debug("%s.update()[%s->%s]" %
                          (self.__class__.__name__, self.status, new_status))

        return new_status

    def terminate(self, new_status):
        self.vehicle = None
        self.logger.debug("%s.terminate()[%s->%s]" % (
            self.__class__.__name__, self.status, new_status))


class CollisionTest(Criterion):

    """
    This class contains an atomic test for collisions.
    """

    def __init__(self, vehicle, name="CheckCollisions"):
        """
        Construction with sensor setup
        """
        super(CollisionTest, self).__init__(name, 0)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self.vehicle = vehicle

        world = self.vehicle.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.collision')
        self.collision_sensor = world.spawn_actor(
            blueprint, carla.Transform(), attach_to=self.vehicle)
        self.collision_sensor.listen(
            lambda event: self.count_collisions(weakref.ref(self), event))

    def update(self):
        """
        Check collision count
        """
        if self.actual_value > 0:
            self.test_status = "FAILURE"
        else:
            self.test_status = "SUCCESS"

        if self.terminate_on_failure and (self.test_status == "FAILURE"):
            new_status = py_trees.common.Status.FAILURE
        else:
            new_status = py_trees.common.Status.RUNNING

        self.logger.debug("%s.update()[%s->%s]" %
                          (self.__class__.__name__, self.status, new_status))

        return new_status

    def terminate(self, new_status):
        """
        Cleanup sensor
        """
        if self.collision_sensor is not None:
            self.collision_sensor.destroy()
        self.collision_sensor = None
        self.vehicle = None
        self.logger.debug("%s.terminate()[%s->%s]" % (
            self.__class__.__name__, self.status, new_status))

    @staticmethod
    def count_collisions(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.actual_value += 1


class KeepLaneTest(Criterion):

    """
    This class contains an atomic test for keeping lane.
    """

    def __init__(self, vehicle, name="CheckKeepLane"):
        """
        Construction with sensor setup
        """
        super(KeepLaneTest, self).__init__(name, 0)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self.vehicle = vehicle

        world = self.vehicle.get_world()
        blueprint = world.get_blueprint_library().find(
            'sensor.other.lane_detector')
        self.lane_sensor = world.spawn_actor(
            blueprint, carla.Transform(), attach_to=self.vehicle)
        self.lane_sensor.listen(
            lambda event: self.count_lane_invasion(weakref.ref(self), event))

    def update(self):
        """
        Check lane invasion count
        """
        if self.actual_value > 0:
            self.test_status = "FAILURE"
        else:
            self.test_status = "SUCCESS"

        if self.terminate_on_failure and (self.test_status == "FAILURE"):
            new_status = py_trees.common.Status.FAILURE
        else:
            new_status = py_trees.common.Status.RUNNING

        self.logger.debug("%s.update()[%s->%s]" %
                          (self.__class__.__name__, self.status, new_status))

        return new_status

    def terminate(self, new_status):
        """
        Cleanup sensor
        """
        if self.lane_sensor is not None:
            self.lane_sensor.destroy()
        self.lane_sensor = None
        self.vehicle = None
        self.logger.debug("%s.terminate()[%s->%s]" % (
            self.__class__.__name__, self.status, new_status))

    @staticmethod
    def count_lane_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.actual_value += 1
