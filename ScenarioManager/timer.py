#!/usr/bin/env python

#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides access to the CARLA game time and contains a py_trees
timeout behavior using the CARLA game time
"""

import py_trees
import carla


class GameTime(object):

    """
    This (static) class provides access to the CARLA game time.

    The elapsed game time can be simply retrieved by calling:
    GameTime.get_time()
    """

    current_game_time = 0.0  # Elapsed game time after starting this Timer

    @staticmethod
    def on_carla_tick(timestamp):
        """
        Callback receiving the CARLA time
        """
        GameTime.current_game_time += timestamp.delta_seconds

    @staticmethod
    def restart():
        """
        Reset game timer to 0
        """
        GameTime.current_game_time = 0.0

    @staticmethod
    def get_time():
        """
        Returns elapsed game time
        """
        return GameTime.current_game_time


class TimeOut(py_trees.behaviour.Behaviour):

    """
    This class contains an atomic timeout behavior.
    It uses the CARLA game time, not the system time which is used by
    the py_trees timer.
    """

    def __init__(self, timeout, name="TimeOut"):
        """
        Setup timeout
        """
        super(TimeOut, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self.timeout_value = timeout
        self.start_time = 0.0

    def setup(self, unused_timeout=15):
        self.logger.debug("%s.setup()" % (self.__class__.__name__))
        return True

    def initialise(self):
        self.start_time = GameTime.get_time()
        self.logger.debug("%s.initialise()" % (self.__class__.__name__))

    def update(self):
        """
        Get current game time, and compare it to the timeout value
        Upon reaching the timeout value the status changes to SUCCESS
        """

        elapsed_time = GameTime.get_time() - self.start_time

        if elapsed_time < self.timeout_value:
            new_status = py_trees.common.Status.RUNNING
        else:
            new_status = py_trees.common.Status.SUCCESS

        self.logger.debug("%s.update()[%s->%s]" %
                          (self.__class__.__name__, self.status, new_status))

        return new_status

    def terminate(self, new_status):
        self.logger.debug("%s.terminate()[%s->%s]" % (
            self.__class__.__name__, self.status, new_status))
