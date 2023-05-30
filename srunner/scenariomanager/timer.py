#!/usr/bin/env python

# Copyright (c) 2018-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides access to the CARLA game time and contains a py_trees
timeout behavior using the CARLA game time
"""

import datetime
import operator
import py_trees
from srunner.tools.history import history

class GameTime(object):

    """
    This (static) class provides access to the CARLA game time.

    The elapsed game time can be simply retrieved by calling:
    GameTime.get_time()
    """

    _current_game_time = 0.0  # Elapsed game time after starting this Timer
    _carla_time = 0.0
    _last_frame = 0
    _platform_timestamp = 0
    _init = False

    @staticmethod
    def on_carla_tick(timestamp):
        """
        Callback receiving the CARLA time
        Update time only when frame is more recent that last frame
        """
        if GameTime._last_frame < timestamp.frame:
            frames = timestamp.frame - GameTime._last_frame if GameTime._init else 1
            GameTime._current_game_time += timestamp.delta_seconds * frames
            GameTime._last_frame = timestamp.frame
            GameTime._platform_timestamp = datetime.datetime.now()
            GameTime._init = True
            GameTime._carla_time = timestamp.elapsed_seconds

    @staticmethod
    def restart():
        """
        Reset game timer to 0
        """
        GameTime._current_game_time = 0.0
        GameTime._carla_time = 0.0
        GameTime._last_frame = 0
        GameTime._init = False

    @staticmethod
    def get_time():
        """
        Returns elapsed game time
        """
        return GameTime._current_game_time

    @staticmethod
    def get_carla_time():
        """
        Returns elapsed game time
        """
        return GameTime._carla_time

    @staticmethod
    def get_wallclocktime():
        """
        Returns elapsed game time
        """
        return GameTime._platform_timestamp

    @staticmethod
    def get_frame():
        """
        Returns elapsed game time
        """
        return GameTime._last_frame


class SimulationTime(py_trees.behaviour.Behaviour):
    def __init__(self, name="SimulationStartTime"):
        """
        Setup starttime
        """
        super(SimulationTime, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))

    def initialise(self):
        py_trees.blackboard.Blackboard().set('start_time',GameTime.get_time())
    
    def update(self):

        # print('update')
        return py_trees.common.Status.SUCCESS


class SimulationTimeCondition(py_trees.behaviour.Behaviour):

    """
    This class contains an atomic simulation time condition behavior. 
    It uses the CARLA game time, not the system time which is used by
    the py_trees timer.

    Returns, if the provided rule was successfully evaluated
    """

    def __init__(self, timeout, comparison_operator=operator.gt, name="SimulationTimeCondition"):
        """
        Setup timeout
        """
        super(SimulationTimeCondition, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._timeout_value = timeout
        self._start_time = 0.0
        self._comparison_operator = comparison_operator

        self.history = history('simulationtimerecord',self.name)

    def initialise(self):
        """
        Set start_time to current GameTime
        """
        self._start_time = GameTime.get_time()

        self.history.createHistory()

        self.logger.debug("%s.initialise()" % (self.__class__.__name__))

    def update(self):
        """
        Get current game time, and compare it to the timeout value
        Upon successfully comparison using the provided comparison_operator,
        the status changes to SUCCESS
        """

        elapsed_time = GameTime.get_time() - self._start_time

        self.history.setHistory(elapsed_time)

        if not self._comparison_operator(self.history.getHistory(), self._timeout_value):
            new_status = py_trees.common.Status.RUNNING
        else:
            new_status = py_trees.common.Status.SUCCESS
            
            self.history.deleteHistory()

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status
    
    
class Delay_SimulationTimeCondition(py_trees.behaviour.Behaviour):

    """
    This class contains an atomic simulation time condition behavior. 
    It uses the CARLA game time, not the system time which is used by
    the py_trees timer.

    Returns, if the provided rule was successfully evaluated
    """

    def __init__(self, timeout,condition_name, comparison_operator=operator.gt, name="SimulationTimeCondition"):
        """
        Setup timeout
        """
        super(Delay_SimulationTimeCondition, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._timeout_value = timeout
        self._start_time = 0.0
        self.condition_name = condition_name
        self._comparison_operator = comparison_operator

        self.history=history('delay',self.condition_name)

    def initialise(self):
        """
        Set start_time to current GameTime
        """
        self._start_time = GameTime.get_time()

        self.logger.debug("%s.initialise()" % (self.__class__.__name__))

        # if not 'delay' in py_trees.blackboard.Blackboard().dict():
        #     py_trees.blackboard.Blackboard().set('delay',{})

        # if not self.condition_name in py_trees.blackboard.Blackboard().get('delay'):
        #     py_trees.blackboard.Blackboard().get('delay')[self.condition_name] = [0,0]

        # py_trees.blackboard.Blackboard().get('delay')[self.condition_name][0] = py_trees.blackboard.Blackboard().get('delay')[self.condition_name][0]+py_trees.blackboard.Blackboard().get('delay')[self.condition_name][1]
        # py_trees.blackboard.Blackboard().get('delay')[self.condition_name][1] = 0 

        self.history.createHistory()

    def update(self):
        """
        Get current game time, and compare it to the timeout value
        Upon successfully comparison using the provided comparison_operator,
        the status changes to SUCCESS
        """

        


        elapsed_time = GameTime.get_time() - self._start_time

        self.history.setHistory(elapsed_time)

        if not self._comparison_operator(self.history.getHistory(), self._timeout_value):

            new_status = py_trees.common.Status.RUNNING
        else:

            new_status = py_trees.common.Status.SUCCESS



            self.history.deleteHistory()


        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status


class TimeOut(SimulationTimeCondition):

    """
    This class contains an atomic timeout behavior.
    It uses the CARLA game time, not the system time which is used by
    the py_trees timer.
    """

    def __init__(self, timeout, name="TimeOut"):
        """
        Setup timeout
        """
        super(TimeOut, self).__init__(timeout, name=name)
        self.timeout = False

    def update(self):
        """
        Upon reaching the timeout value the status changes to SUCCESS
        """

        new_status = super(TimeOut, self).update()

        if new_status == py_trees.common.Status.SUCCESS:
            self.timeout = True

        return new_status

class Delay(Delay_SimulationTimeCondition):

    """
    This class contains an atomic timeout behavior.
    It uses the CARLA game time, not the system time which is used by
    the py_trees timer.
    """

    def __init__(self, timeout,condition_name,name="TimeOut"):
        """
        Setup timeout
        """
        super(Delay, self).__init__(timeout,condition_name=condition_name,name=name)
        self.timeout = False

    def update(self):
        """
        Upon reaching the timeout value the status changes to SUCCESS
        """

        new_status = super(Delay, self).update()

        if new_status == py_trees.common.Status.SUCCESS:
            self.timeout = True

        return new_status
    
