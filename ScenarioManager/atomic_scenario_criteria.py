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


class MaxVelocityTest(py_trees.behaviour.Behaviour):

    """
    This class contains an atomic test for maximum velocity.
    """

    def __init__(self, vehicle, max_velocity_allowed, name="CheckMaximumVelocity"):
        """
        Setup vehicle and maximum allowed velovity
        """
        super(MaxVelocityTest, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self.vehicle = vehicle
        self.max_velocity_allowed = max_velocity_allowed
        self.max_velocity_driven = 0
        self.terminate_on_failure = False
        self.test_status = "INIT"

    def setup(self, unused_timeout=15):
        self.logger.debug("%s.setup()" % (self.__class__.__name__))
        return True

    def initialise(self):
        self.logger.debug("%s.initialise()" % (self.__class__.__name__))

    def update(self):
        """
        Check velocity
        """
        velocity = math.sqrt(
            self.vehicle.get_velocity().x**2 + self.vehicle.get_velocity().y**2)

        self.max_velocity_driven = max(velocity, self.max_velocity_driven)

        if velocity > self.max_velocity_allowed:
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

    def get_test_status(self):
        return self.test_status

    def get_test_metric(self):
        return "Value: %5.2f Limit: %5.2f" % (self.max_velocity_driven,
                                              self.max_velocity_allowed)


class CollisionTest(py_trees.behaviour.Behaviour):

    """
    This class contains an atomic test for collisions.
    """

    def __init__(self, vehicle, name="CheckCollisions"):
        """
        Construction with sensor setup
        """
        super(CollisionTest, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self.vehicle = vehicle
        self.collision_count = 0
        self.terminate_on_failure = False
        self.test_status = "INIT"

        world = self.vehicle.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.collision_sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=self.vehicle)
        self.collision_sensor.listen(
            lambda event: self.count_collisions(weakref.ref(self), event))

    def setup(self, unused_timeout=15):
        self.logger.debug("%s.setup()" % (self.__class__.__name__))
        return True

    def initialise(self):
        self.logger.debug("%s.initialise()" % (self.__class__.__name__))

    def update(self):
        """
        Check collision count
        """
        if self.collision_count > 0:
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
        self.collision_count += 1

    def get_test_status(self):
        return self.test_status

    def get_test_metric(self):
        return "Value: %5.2f Limit: %5.2f" % (self.collision_count, 0)


class KeepLaneTest(py_trees.behaviour.Behaviour):

    """
    This class contains an atomic test for keeping lane.
    """

    def __init__(self, vehicle, name="CheckKeepLane"):
        """
        Construction with sensor setup
        """
        super(KeepLaneTest, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self.vehicle = vehicle
        self.violation_count = 0
        self.terminate_on_failure = False
        self.test_status = "INIT"

        world = self.vehicle.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_detector')
        self.lane_sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=self.vehicle)
        self.lane_sensor.listen(
            lambda event: self.count_lane_invasion(weakref.ref(self), event))

    def setup(self, unused_timeout=15):
        self.logger.debug("%s.setup()" % (self.__class__.__name__))
        return True

    def initialise(self):
        self.logger.debug("%s.initialise()" % (self.__class__.__name__))

    def update(self):
        """
        Check lane invasion count
        """
        if self.violation_count > 0:
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
        self.violation_count += 1

    def get_test_status(self):
        return self.test_status

    def get_test_metric(self):
        return "Value: %5.2f Limit: %5.2f" % (self.violation_count, 0)
