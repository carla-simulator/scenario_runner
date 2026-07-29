#!/usr/bin/env python

# Copyright (c) 2018-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides all atomic scenario behaviors required to realize
complex, realistic scenarios such as "follow a leading vehicle", "lane change",
etc.

The atomic behaviors are implemented with py_trees.
"""

from __future__ import print_function

import copy
import math
import operator
import os
import time
import subprocess
from bisect import bisect_right

import numpy as np
from numpy import random
import py_trees
from py_trees.blackboard import Blackboard
import networkx

from shapely import Polygon

import carla
from agents.navigation.basic_agent import BasicAgent
from agents.navigation.constant_velocity_agent import ConstantVelocityAgent
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.local_planner import RoadOption, LocalPlanner
from agents.tools.misc import is_within_distance, get_speed

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.carla_data_provider import calculate_velocity
from srunner.scenariomanager.actorcontrols.actor_control import ActorControl
from srunner.scenariomanager.timer import GameTime
from srunner.tools.scenario_helper import detect_lane_obstacle
from srunner.tools.scenario_helper import generate_target_waypoint_list_multilane


import srunner.tools as sr_tools

EPSILON = 0.001


def calculate_distance(location, other_location, global_planner=None):
    """
    Method to calculate the distance between to locations

    Note: It uses the direct distance between the current location and the
          target location to estimate the time to arrival.
          To be accurate, it would have to use the distance along the
          (shortest) route between the two locations.
    """
    if global_planner:
        distance = 0

        # Get the route
        route = global_planner.trace_route(location, other_location)

        # Get the distance of the route
        for i in range(1, len(route)):
            curr_loc = route[i][0].transform.location
            prev_loc = route[i - 1][0].transform.location

            distance += curr_loc.distance(prev_loc)

        return distance

    return location.distance(other_location)


def get_actor_control(actor):
    """
    Method to return the type of control to the actor.
    """
    control = actor.get_control()
    actor_type = actor.type_id.split('.')[0]
    if not isinstance(actor, carla.Walker):
        control.steering = 0
    else:
        control.speed = 0

    return control, actor_type


class AtomicBehavior(py_trees.behaviour.Behaviour):

    """
    Base class for all atomic behaviors used to setup a scenario

    *All behaviors should use this class as parent*

    Important parameters:
    - name: Name of the atomic behavior
    """

    def __init__(self, name, actor=None):
        """
        Default init. Has to be called via super from derived class
        """
        super().__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self.name = name
        self._actor = actor
        self._actor_name = None
        if actor is not None and hasattr(actor, "attributes"):
            self._actor_name = actor.attributes.get("role_name")

    def _resolve_actor(self):
        # AddEntityAction actors may be represented by a role-name-only
        # placeholder while the tree is built. Resolve the real CARLA actor
        # once the AddActor atomic has spawned it.
        if self._actor is not None and hasattr(self._actor, "id"):
            return self._actor
        if self._actor_name is not None:
            self._actor = CarlaDataProvider.get_actor_by_name(self._actor_name)
        return self._actor

    def setup(self, unused_timeout=15):
        """
        Default setup
        """
        self.logger.debug("%s.setup()" % (self.__class__.__name__))
        return True

    def initialise(self):
        """
        Initialise setup terminates WaypointFollowers
        Check whether WF for this actor is running and terminate all active WFs
        """
        self._resolve_actor()
        if self._actor is not None and hasattr(self._actor, "id"):
            try:
                check_attr = operator.attrgetter("running_WF_actor_{}".format(self._actor.id))
                terminate_wf = copy.copy(check_attr(py_trees.blackboard.Blackboard()))
                py_trees.blackboard.Blackboard().set(
                    "terminate_WF_actor_{}".format(self._actor.id), terminate_wf, overwrite=True)
            except AttributeError:
                # It is ok to continue, if the Blackboard variable does not exist
                pass
        self.logger.debug("%s.initialise()" % (self.__class__.__name__))

    def terminate(self, new_status):
        """
        Default terminate. Can be extended in derived class
        """
        self.logger.debug("%s.terminate()[%s->%s]" % (self.__class__.__name__, self.status, new_status))


class RunScript(AtomicBehavior):

    """
    This is an atomic behavior to start execution of an additional script.

    Args:
        script (str): String containing the interpreter, scriptpath and arguments
            Example: "python /path/to/script.py --arg1"
        base_path (str): String containing the base path of for the script

    Attributes:
        _script (str): String containing the interpreter, scriptpath and arguments
            Example: "python /path/to/script.py --arg1"
        _base_path (str): String containing the base path of for the script
            Example: "/path/to/"

    Note:
        This is intended for the use with OpenSCENARIO. Be aware of security side effects.
    """

    def __init__(self, script, base_path=None, name="RunScript"):
        """
        Setup parameters
        """
        super().__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._script = script
        self._base_path = base_path

    def update(self):
        """
        Start script
        """
        path = None
        script_components = self._script.split(' ')
        if len(script_components) > 1:
            path = script_components[1]

        if not os.path.isfile(path):
            path = os.path.join(self._base_path, path)
        if not os.path.isfile(path):
            new_status = py_trees.common.Status.FAILURE
            print("Script file does not exists {}".format(path))
        else:
            subprocess.Popen(self._script, shell=True, cwd=self._base_path)  # pylint: disable=consider-using-with
            new_status = py_trees.common.Status.SUCCESS

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))
        return new_status


class ChangeParameter(AtomicBehavior):
    """
    This is an atomic behavior to change the osc parameter value.

    Args:
        parameter_ref (str): parameter name
        value (any): ParameterRef or number
    """

    def __init__(self, parameter_ref, value, rule=None, name="ChangeParameter"):
        super().__init__(name)
        self.logger.debug("%s.__init__()" % self.__class__.__name__)
        self._parameter_ref = parameter_ref
        self._rule = rule
        self._value = value

    def update(self):
        """
        update value of global osc parameter.
        """
        old_value = CarlaDataProvider.get_osc_global_param_value(self._parameter_ref)

        if self._rule == '+':
            new_value = self._value + float(old_value)
        elif self._rule == '*':
            new_value = self._value * float(old_value)
        else:
            new_value = self._value

        CarlaDataProvider.update_osc_global_params({self._parameter_ref: new_value})
        new_status = py_trees.common.Status.SUCCESS

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))
        return new_status


class ChangeWeather(AtomicBehavior):

    """
    Atomic to write a new weather configuration into the blackboard.
    Used in combination with OSCWeatherBehavior() to have a continuous weather simulation.

    The behavior immediately terminates with SUCCESS after updating the blackboard.

    Args:
        weather (srunner.scenariomanager.weather_sim.Weather): New weather settings.
        name (string): Name of the behavior.
            Defaults to 'UpdateWeather'.

    Attributes:
        _weather (srunner.scenariomanager.weather_sim.Weather): Weather settings.
    """

    def __init__(self, weather, name="ChangeWeather"):
        """
        Setup parameters
        """
        super().__init__(name)
        self._weather = weather

    def update(self):
        """
        Write weather into blackboard and exit with success

        returns:
            py_trees.common.Status.SUCCESS
        """
        py_trees.blackboard.Blackboard().set("CarlaWeather", self._weather, overwrite=True)
        return py_trees.common.Status.SUCCESS


class ChangeRoadFriction(AtomicBehavior):

    """
    Atomic to update the road friction in CARLA.

    The behavior immediately terminates with SUCCESS after updating the friction.

    Args:
        friction (float): New friction coefficient.
        name (string): Name of the behavior.
            Defaults to 'UpdateRoadFriction'.

    Attributes:
        _friction (float): Friction coefficient.
    """

    def __init__(self, friction, name="ChangeRoadFriction"):
        """
        Setup parameters
        """
        super().__init__(name)
        self._friction = friction

    def update(self):
        """
        Update road friction. Spawns new friction blueprint and removes the old one, if existing.

        returns:
            py_trees.common.Status.SUCCESS
        """

        for actor in CarlaDataProvider.get_all_actors().filter('static.trigger.friction'):
            actor.destroy()

        friction_bp = CarlaDataProvider.get_world().get_blueprint_library().find('static.trigger.friction')
        extent = carla.Location(1000000.0, 1000000.0, 1000000.0)
        friction_bp.set_attribute('friction', str(self._friction))
        friction_bp.set_attribute('extent_x', str(extent.x))
        friction_bp.set_attribute('extent_y', str(extent.y))
        friction_bp.set_attribute('extent_z', str(extent.z))

        # Spawn Trigger Friction
        transform = carla.Transform()
        transform.location = carla.Location(-10000.0, -10000.0, 0.0)
        CarlaDataProvider.get_world().spawn_actor(friction_bp, transform)

        return py_trees.common.Status.SUCCESS


class ChangeActorControl(AtomicBehavior):

    """
    Atomic to change the longitudinal/lateral control logic for an actor.
    The (actor, controller) pair is stored inside the Blackboard.

    The behavior immediately terminates with SUCCESS after the controller.

    Args:
        actor (carla.Actor): Actor that should be controlled by the controller.
        control_py_module (string): Name of the python module containing the implementation
            of the controller.
        args (dictionary): Additional arguments for the controller.
        scenario_file_path (string): Additional path to controller implementation.
        name (string): Name of the behavior.
            Defaults to 'ChangeActorControl'.

    Attributes:
        _actor_control (ActorControl): Instance of the actor control.
    """

    def __init__(self, actor, control_py_module, args, scenario_file_path=None, name="ChangeActorControl"):
        """
        Setup actor controller.
        """
        super().__init__(name, actor)

        self._control_py_module = control_py_module
        self._args = args
        self._scenario_file_path = scenario_file_path
        self._actor_control = None
        if self._actor is not None and hasattr(self._actor, "id"):
            self._actor_control = ActorControl(self._actor, control_py_module=self._control_py_module,
                                               args=self._args, scenario_file_path=self._scenario_file_path)

    def update(self):
        """
        Write (actor, controler) pair to Blackboard, or update the controller
        if actor already exists as a key.

        returns:
            py_trees.common.Status.SUCCESS
        """

        actor_dict = {}

        try:
            check_actors = operator.attrgetter("ActorsWithController")
            actor_dict = check_actors(py_trees.blackboard.Blackboard())
        except AttributeError:
            pass

        self._resolve_actor()
        if self._actor is None or not hasattr(self._actor, "id"):
            return py_trees.common.Status.RUNNING

        # Controller construction needs the concrete CARLA actor type, so it
        # is delayed for actors that are spawned by AddEntityAction.
        if self._actor_control is None:
            self._actor_control = ActorControl(self._actor, control_py_module=self._control_py_module,
                                               args=self._args, scenario_file_path=self._scenario_file_path)

        if actor_dict:
            if self._actor.id in actor_dict:
                actor_dict[self._actor.id].reset()

        actor_dict[self._actor.id] = self._actor_control
        py_trees.blackboard.Blackboard().set("ActorsWithController", actor_dict, overwrite=True)

        return py_trees.common.Status.SUCCESS


class UpdateAllActorControls(AtomicBehavior):

    """
    Atomic to update (run one control loop step) all actor controls.

    The behavior is always in RUNNING state.

    Args:
        name (string): Name of the behavior.
            Defaults to 'UpdateAllActorControls'.
    """

    def __init__(self, name="UpdateAllActorControls"):
        """
        Constructor
        """
        super().__init__(name)

    def update(self):
        """
        Execute one control loop step for all actor controls.

        returns:
            py_trees.common.Status.RUNNING
        """

        actor_dict = {}

        try:
            check_actors = operator.attrgetter("ActorsWithController")
            actor_dict = check_actors(py_trees.blackboard.Blackboard())
        except AttributeError:
            pass

        for actor_id in actor_dict:
            actor_dict[actor_id].run_step()

        return py_trees.common.Status.RUNNING


class ChangeActorTargetSpeed(AtomicBehavior):

    """
    Atomic to change the target speed for an actor controller.

    The behavior is in RUNNING state until the distance/duration
    conditions are satisfied, or if a second ChangeActorTargetSpeed atomic
    for the same actor is triggered.

    Args:
        actor (carla.Actor): Controlled actor.
        target_speed (float): New target speed [m/s].
        init_speed (boolean): Flag to indicate if the speed is the initial actor speed.
            Defaults to False.
        duration (float): Duration of the maneuver [s].
            Defaults to None.
        distance (float): Distance of the maneuver [m].
            Defaults to None.
        relative_actor (carla.Actor): If the target speed setting should be relative to another actor.
            Defaults to None.
        value (float): Offset, if the target speed setting should be relative to another actor.
            Defaults to None.
        value_type (string): Either 'Delta' or 'Factor' influencing how the offset to the reference actors
            velocity is applied. Defaults to None.
        continuous (boolean): If True, the atomic remains in RUNNING, independent of duration or distance.
            Defaults to False.
        name (string): Name of the behavior.
            Defaults to 'ChangeActorTargetSpeed'.

    Attributes:
        _target_speed (float): New target speed [m/s].
        _init_speed (boolean): Flag to indicate if the speed is the initial actor speed.
            Defaults to False.
        _start_time (float): Start time of the atomic [s].
            Defaults to None.
        _start_location (carla.Location): Start location of the atomic.
            Defaults to None.
        _duration (float): Duration of the maneuver [s].
            Defaults to None.
        _distance (float): Distance of the maneuver [m].
            Defaults to None.
        _relative_actor (carla.Actor): If the target speed setting should be relative to another actor.
            Defaults to None.
        _value (float): Offset, if the target speed setting should be relative to another actor.
            Defaults to None.
        _value_type (string): Either 'Delta' or 'Factor' influencing how the offset to the reference actors
            velocity is applied. Defaults to None.
        _continuous (boolean): If True, the atomic remains in RUNNING, independent of duration or distance.
            Defaults to False.
    """

    def __init__(self, actor, target_speed, init_speed=False,
                 duration=None, distance=None, relative_actor=None,
                 value=None, value_type=None, continuous=False, name="ChangeActorTargetSpeed"):
        """
        Setup parameters
        """
        super().__init__(name, actor)

        self._target_speed = target_speed
        self._init_speed = init_speed

        self._start_time = None
        self._start_location = None

        self._relative_actor = relative_actor
        self._value = value
        self._value_type = value_type
        self._continuous = continuous
        self._duration = duration
        self._distance = distance

    def initialise(self):
        """
        Set initial parameters such as _start_time and _start_location,
        and get (actor, controller) pair from Blackboard.

        May throw if actor is not available as key for the ActorsWithController
        dictionary from Blackboard.
        """
        actor_dict = {}

        try:
            check_actors = operator.attrgetter("ActorsWithController")
            actor_dict = check_actors(py_trees.blackboard.Blackboard())
        except AttributeError:
            pass

        self._resolve_actor()
        if self._actor is None or not hasattr(self._actor, "id"):
            raise RuntimeError("Actor not found in CarlaDataProvider")

        if not actor_dict or self._actor.id not in actor_dict:
            raise RuntimeError("Actor not found in ActorsWithController BlackBoard")

        self._start_time = GameTime.get_time()
        self._start_location = CarlaDataProvider.get_location(self._actor)

        if self._relative_actor:
            relative_velocity = CarlaDataProvider.get_velocity(self._relative_actor)

            # get target velocity
            if self._value_type == 'delta':
                self._target_speed = relative_velocity + self._value
            elif self._value_type == 'factor':
                self._target_speed = relative_velocity * self._value
            else:
                print('self._value_type must be delta or factor')

        actor_dict[self._actor.id].update_target_speed(self._target_speed, start_time=self._start_time)

        if self._init_speed:
            actor_dict[self._actor.id].set_init_speed()

        super().initialise()

    def update(self):
        """
        Check the actor's current state and update target speed, if it is relative to another actor.

        returns:
            py_trees.common.Status.SUCCESS, if the duration or distance driven exceeded limits
                                            if another ChangeActorTargetSpeed atomic for the same actor was triggered.
            py_trees.common.Status.FAILURE, if the actor is not found in ActorsWithController Blackboard dictionary.
            py_trees.common.Status.FAILURE, else.
        """
        try:
            check_actors = operator.attrgetter("ActorsWithController")
            actor_dict = check_actors(py_trees.blackboard.Blackboard())
        except AttributeError:
            pass

        self._resolve_actor()
        if self._actor is None or not hasattr(self._actor, "id"):
            return py_trees.common.Status.FAILURE

        if not actor_dict or self._actor.id not in actor_dict:
            return py_trees.common.Status.FAILURE

        if actor_dict[self._actor.id].get_last_longitudinal_command() != self._start_time:
            return py_trees.common.Status.SUCCESS

        new_status = py_trees.common.Status.RUNNING

        if self._relative_actor:
            relative_velocity = CarlaDataProvider.get_velocity(self._relative_actor)

            # get target velocity
            if self._value_type == 'delta':
                actor_dict[self._actor.id].update_target_speed(relative_velocity + self._value)
            elif self._value_type == 'factor':
                actor_dict[self._actor.id].update_target_speed(relative_velocity * self._value)
            else:
                print('self._value_type must be delta or factor')

        # check duration and driven_distance
        if not self._continuous:
            if (self._duration is not None) and (GameTime.get_time() - self._start_time > self._duration):
                new_status = py_trees.common.Status.SUCCESS

            driven_distance = CarlaDataProvider.get_location(self._actor).distance(self._start_location)
            if (self._distance is not None) and (driven_distance > self._distance):
                new_status = py_trees.common.Status.SUCCESS

        if self._distance is None and self._duration is None:
            new_status = py_trees.common.Status.SUCCESS

        return new_status


class SyncArrivalOSC(AtomicBehavior):

    """
    Atomic to make two actors arrive at their corresponding places at the same time

    The behavior is in RUNNING state until the "main" actor has rezached its destination

    Args:
        actor (carla.Actor): Controlled actor.
        master_actor (carla.Actor): Reference actor to sync up to.
        actor_target (carla.Transform): Endpoint of the actor after the behavior finishes.
        master_target (carla.Transform): Endpoint of the master_actor after the behavior finishes.
        final_speed (float): Speed of the actor after the behavior ends.
        relative_to_master (boolean): Whether or not the final speed is relative to master_actor.
            Defaults to False.
        relative_type (string): Type of relative speed. Either 'delta' or 'factor'.
            Defaults to ''.
        name (string): Name of the behavior.
            Defaults to 'SyncArrivalOSC'.
    """

    DISTANCE_THRESHOLD = 1

    def __init__(self, actor, master_actor, actor_target, master_target, final_speed,
                 relative_to_master=False, relative_type='', name="SyncArrivalOSC"):
        """
        Setup required parameters
        """
        super().__init__(name, actor)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))

        self._actor = actor
        self._actor_target = actor_target
        self._master_actor = master_actor
        self._master_target = master_target

        self._final_speed = final_speed
        self._final_speed_set = False
        self._relative_to_master = relative_to_master
        self._relative_type = relative_type

        self._start_time = None

    def initialise(self):
        """
        Set initial parameters and get (actor, controller) pair from Blackboard.

        May throw if actor is not available as key for the ActorsWithController
        dictionary from Blackboard.
        """
        actor_dict = {}

        try:
            check_actors = operator.attrgetter("ActorsWithController")
            actor_dict = check_actors(py_trees.blackboard.Blackboard())
        except AttributeError:
            pass

        self._resolve_actor()
        if self._actor is None or not hasattr(self._actor, "id"):
            raise RuntimeError("Actor not found in CarlaDataProvider")

        if not actor_dict or self._actor.id not in actor_dict:
            raise RuntimeError("Actor not found in ActorsWithController BlackBoard")

        self._start_time = GameTime.get_time()

        # Get the distance of the actor to its endpoint
        distance = calculate_distance(
            CarlaDataProvider.get_location(self._actor), self._actor_target.location)

        # Get the time to arrival of the reference to its endpoint
        distance_reference = calculate_distance(
            CarlaDataProvider.get_location(self._master_actor), self._master_target.location)

        velocity_reference = CarlaDataProvider.get_velocity(self._master_actor)
        if velocity_reference > 0:
            time_reference = distance_reference / velocity_reference
        else:
            time_reference = float('inf')

        # Get the required velocity of the actor
        desired_velocity = distance / time_reference
        actor_dict[self._actor.id].update_target_speed(desired_velocity, start_time=self._start_time)

    def update(self):
        """
        Dynamic control update for actor velocity to ensure that both actors reach their target
        positions at the same time.
        """

        try:
            check_actors = operator.attrgetter("ActorsWithController")
            actor_dict = check_actors(py_trees.blackboard.Blackboard())
        except AttributeError:
            pass

        self._resolve_actor()
        if self._actor is None or not hasattr(self._actor, "id"):
            return py_trees.common.Status.FAILURE

        if not actor_dict or self._actor.id not in actor_dict:
            return py_trees.common.Status.FAILURE

        if actor_dict[self._actor.id].get_last_longitudinal_command() != self._start_time:
            return py_trees.common.Status.SUCCESS

        new_status = py_trees.common.Status.RUNNING

        # Get the distance of the actor to its endpoint
        distance = calculate_distance(
            CarlaDataProvider.get_location(self._actor), self._actor_target.location)

        if distance < self.DISTANCE_THRESHOLD:
            return py_trees.common.Status.SUCCESS  # Behaviour ends when the actor reaches its endpoint

        # Get the time to arrival of the reference to its endpoint
        distance_reference = calculate_distance(
            CarlaDataProvider.get_location(self._master_actor), self._master_target.location)

        velocity_reference = CarlaDataProvider.get_velocity(self._master_actor)
        if velocity_reference > 0:
            time_reference = distance_reference / velocity_reference
        else:
            time_reference = float('inf')

        # Get the required velocity of the actor
        desired_velocity = distance / time_reference
        actor_dict[self._actor.id].update_target_speed(desired_velocity)

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))
        return new_status

    def terminate(self, new_status):
        """
        On termination of this behavior, set the target speed to its desired one.
        This function is called several times, so the use of self._final_speed_set
        is needed to avoid interfering with other running behaviors
        """
        if not self._final_speed_set:
            try:
                check_actors = operator.attrgetter("ActorsWithController")
                actor_dict = check_actors(py_trees.blackboard.Blackboard())
            except AttributeError:
                pass

            if actor_dict and self._actor.id in actor_dict:

                if self._relative_to_master:
                    master_speed = CarlaDataProvider.get_velocity(self._master_actor)
                    if self._relative_type == 'delta':
                        final_speed = master_speed + self._final_speed
                    elif self._relative_type == 'factor':
                        final_speed = master_speed * self._final_speed
                    else:
                        print("'relative_type' must be delta or factor")
                        self._final_speed_set = True
                        super().terminate(new_status)
                        return
                else:
                    final_speed = self._final_speed

                actor_dict[self._actor.id].update_target_speed(final_speed)

            self._final_speed_set = True

        super().terminate(new_status)


class ChangeActorWaypoints(AtomicBehavior):

    """
    Atomic to change the waypoints for an actor controller.

    The behavior is in RUNNING state until the last waypoint is reached, or if a
    second waypoint related atomic for the same actor is triggered. These are:
    - ChangeActorWaypoints
    - ChangeActorLateralMotion
    - ChangeActorLaneOffset

    Different modes are possible to change waypoints: replay (rts) and advanced replay (arts). These can be specified in OpenSCENARIO parameters. Example (to be included in <Trajectory> within a <FollowTrajectoryAction>):
    <ParameterDeclarations>
        <ParameterDeclaration name="arts" parameterType="string" value="True" />  # mode "arts" chosen"
        <ParameterDeclaration name="check_for_prioritization_rule" parameterType="string" value="left_before_right" />  # rules arts vehicles should follow
        <ParameterDeclaration name="check_for_prioritization_rule" parameterType="string" value="is_behind" />  # rules arts vehicles should follow
    </ParameterDeclarations>
    A general logic of arts can be found her: https://www.vvm-projekt.de/securedl/sdl-eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpYXQiOjE3NDc3MTQ2NTAsImV4cCI6MTc0NzgwNDY1MCwidXNlciI6MCwiZ3JvdXBzIjpbMCwtMV0sImZpbGUiOiJmaWxlYWRtaW4vdXNlcl91cGxvYWQvUGFwZXJzL0RlbGl2ZXJhYmxlMTMtU2NlbmFyaW8tYmFzZWRfTW9kZWxfb2ZfdGhlX09ERF90aHJvd_d6a6870608afd5e8/Deliverable13-Scenario-based_Model_of_the_ODD_through_Scenario_Databases.pdf

    Args:
        actor (carla.Actor): Controlled actor.
        waypoints (List of (OSC position, OSC route option)): List of (Position, Route Option) as OpenScenario elements.
            position will be converted to Carla transforms, considering the corresponding
            route option (e.g. shortest, fastest)
        name (string): Name of the behavior.
            Defaults to 'ChangeActorWaypoints'.

    Attributes:
        _waypoints (List of (OSC position, OSC route option)): List of (Position, Route Option) as OpenScenario elements
        _start_time (float): Start time of the atomic [s].
            Defaults to None.

    '''Note: When using routing options such as fastest or shortest, it is advisable to run
             in synchronous mode
    """

    def __init__(self, actor, waypoints, times=None, name="ChangeActorWaypoints", is_osc1=True, additional_parameters=None):
        """
        Setup parameters
        """
        super().__init__(name, actor)

        self._waypoints = waypoints
        self._start_time = None
        self._times = times
        self._is_osc1 = is_osc1
        self._rts_actor_control = None

        if self._times is not None and len(self._waypoints) != len(self._times):
            raise ValueError("Both 'waypoints' and 'times' must have the same length")

        # additions for Replay to Sim (RtS) or Advanced Replay to Sim (ARtS)
        self.arts = False
        self.rts = False
        if additional_parameters and "rts-mode" in additional_parameters.keys():
            if additional_parameters["rts-mode"] == "rts":
                self.rts = True
            elif additional_parameters["rts-mode"] == "arts":
                self.arts = True
                self._waypoint_transforms = [sr_tools.openscenario_parser.OpenScenarioParser.convert_position_to_transform(wp[0]) for wp in waypoints]
                self.moving_object_ids = []
                self.loop_time = time.time()

                if "check_for_road_user" in additional_parameters: 
                    self.prioritized_objects = additional_parameters["check_for_road_user"]
                else:
                    self.prioritized_objects = []
                if "check_for_prioritization_rule" in additional_parameters:
                    self.check_for_prioritization_rules = additional_parameters["check_for_prioritization_rule"]
                else:
                    self.check_for_prioritization_rules = []
                self.arts_config = self._get_arts_config(additional_parameters)

    def _get_arts_config(self, args):
        """
        create config for arts to prevent unwanted accidents
        """
        # set default config 
        config = {
            "catchup_velocity_percentage": 20.0,
            "threshold_thw": 1.0,
            "threshold_ttc": 1.5,
            "threshold_dhw": 1.0
        }
        
        # automated casting and rewriting values from config if necessary - overwriting values from args
        for arg_key in list(args.keys()):
            if "config" in arg_key:
                config_name = arg_key.split("_")[1]
                actual_type = None
                if config_name in self.config.keys():
                    actual_type = type(config[config_name])
                config[config_name] = args[arg_key]
                try:
                    if actual_type == int:
                        config[config_name] = int(args[arg_key])
                    elif actual_type == float: 
                        config[config_name] = float(args[arg_key])
                    elif actual_type == bool:
                        if args[arg_key] == "True":
                            config[config_name] = True
                        else:
                            config[config_name] = False
                except ValueError:
                    print("ERROR: Invalid entry casting config for key '" + str(arg_key) + "' in approaching control. Use default instead.")
        return config            

    def initialise(self):
        """
        Set _start_time and get (actor, controller) pair from Blackboard.

        Set waypoint list for actor controller.

        May throw if actor is not available as key for the ActorsWithController
        dictionary from Blackboard.
        """
        actor_dict = {}

        try:
            check_actors = operator.attrgetter("ActorsWithController")
            actor_dict = check_actors(py_trees.blackboard.Blackboard())
        except AttributeError:
            pass

        self._resolve_actor()
        if self._actor is None or not hasattr(self._actor, "id"):
            raise RuntimeError("Actor not found in CarlaDataProvider")

        if not actor_dict or self._actor.id not in actor_dict:
            raise RuntimeError("Actor not found in ActorsWithController BlackBoard")

        self._start_time = GameTime.get_time()

        if self.rts:
            # RtS owns pose and velocity exclusively. Keep the command marker
            # for behavior arbitration, but do not give the replay trajectory
            # to the normal actor controller.
            self._rts_actor_control = actor_dict[self._actor.id]
            self._rts_actor_control.update_waypoints(
                [], times=None, start_time=self._start_time)
            self._rts_actor_control.suspend_control(self)

            # CARLA keeps the last applied control command active. Neutralize it
            # once so no stale throttle, brake, steering, or walker command can
            # influence the actor while RtS owns its state.
            if isinstance(self._actor, carla.Vehicle):
                vehicle_control = self._actor.get_control()
                vehicle_control.throttle = 0.0
                vehicle_control.brake = 0.0
                vehicle_control.steer = 0.0
                vehicle_control.hand_brake = False
                self._actor.apply_control(vehicle_control)
            elif isinstance(self._actor, carla.Walker):
                walker_control = self._actor.get_control()
                walker_control.speed = 0.0
                self._actor.apply_control(walker_control)

            super().initialise()
            return

        if self._is_osc1:
            # Transforming OSC waypoints to Carla waypoints
            carla_route_elements = []
            for (osc_point, routing_option) in self._waypoints:
                carla_transforms = sr_tools.openscenario_parser.OpenScenarioParser.convert_position_to_transform(osc_point)
                carla_route_elements.append((carla_transforms, routing_option))
        else:
            carla_route_elements = []
            # mmap = CarlaDataProvider.get_map()
            for (point, routing_option) in self._waypoints:
                wp_transf = carla.Transform(location=carla.Location(point[0],point[1],point[2]))
                # carla_transforms = [mmap.get_waypoint(wp_transf.location)]
                carla_route_elements.append((wp_transf, routing_option))


        # Obtain final route, considering the routing option
        # At the moment everything besides "shortest" will use the CARLA GlobalPlanner
        grp = CarlaDataProvider.get_global_route_planner()
        route = []
        for i, element in enumerate(carla_route_elements):
            if element[1] == "shortest":
                route.append(element[0])
            else:
                if i == 0:
                    mmap = CarlaDataProvider.get_map()
                    ego_location = CarlaDataProvider.get_location(self._actor)
                    ego_waypoint = mmap.get_waypoint(ego_location)
                    try:
                        ego_next_wp = ego_waypoint.next(1)[0]
                    except IndexError:
                        ego_next_wp = ego_waypoint
                    waypoint = ego_next_wp.transform.location
                else:
                    waypoint = carla_route_elements[i - 1][0].location
                waypoint_next = element[0].location
                try:
                    interpolated_trace = grp.trace_route(waypoint, waypoint_next)
                except networkx.NetworkXNoPath:
                    print("WARNING: No route from {} to {} - Using direct path instead".format(waypoint, waypoint_next))
                    route.append(element[0])
                    continue
                for wp_tuple in interpolated_trace:
                    # The router sometimes produces points that go backward, or are almost identical
                    # We have to filter these, to avoid problems
                    if route and wp_tuple[0].transform.location.distance(route[-1].location) > 1.0:
                        new_heading_vec = wp_tuple[0].transform.location - route[-1].location
                        new_heading = np.arctan2(new_heading_vec.y, new_heading_vec.x)
                        if len(route) > 1:
                            last_heading_vec = route[-1].location - route[-2].location
                        else:
                            last_heading_vec = route[-1].location - ego_next_wp.transform.location
                        last_heading = np.arctan2(last_heading_vec.y, last_heading_vec.x)

                        heading_delta = math.fabs(new_heading - last_heading)
                        if math.fabs(heading_delta) < 0.5 or math.fabs(heading_delta) > 5.5:
                            route.append(wp_tuple[0].transform)
                    elif not route:
                        route.append(wp_tuple[0].transform)

        actor_dict[self._actor.id].update_waypoints(
            route, times=self._times, start_time=self._start_time)
        control_instance = actor_dict[self._actor.id].control_instance
        if carla_route_elements and hasattr(control_instance, "update_final_route_goal"):
            control_instance.update_final_route_goal(carla_route_elements[-1][0])

        super().initialise()

    def update(self):
        """
        Check the actor's state along the waypoint route.

        returns:
            py_trees.common.Status.SUCCESS, if the final waypoint was reached, or
                                            if another ChangeActorWaypoints atomic for the same actor was triggered.
            py_trees.common.Status.FAILURE, if the actor is not found in ActorsWithController Blackboard dictionary.
            py_trees.common.Status.FAILURE, else.
        """
        try:
            check_actors = operator.attrgetter("ActorsWithController")
            actor_dict = check_actors(py_trees.blackboard.Blackboard())
        except AttributeError:
            pass

        if not actor_dict or self._actor.id not in actor_dict:
            return py_trees.common.Status.FAILURE

        actor = actor_dict[self._actor.id]

        if actor.get_last_waypoint_command() != self._start_time:
            return py_trees.common.Status.SUCCESS

        # RtS completion is defined by the trajectory timestamps, not by the
        # road-projected LocalPlanner route.
        if not self.rts and actor.check_reached_waypoint_goal():
            return py_trees.common.Status.SUCCESS

        # additions for RtS or ARtS
        if self.rts or self.arts:
            if self._times is not None:
                current_relative_time = GameTime.get_time() - self._start_time
                current_waypoint_idx = bisect_right(self._times, current_relative_time)
                if current_waypoint_idx >= len(self._times):
                    if self.rts:
                        final_waypoint = self._waypoints[-1]
                        final_transform = sr_tools.openscenario_parser.OpenScenarioParser.convert_position_to_transform(
                            final_waypoint[0])
                        self._apply_rts_state(
                            actor,
                            final_transform,
                            carla.Vector3D(0.0, 0.0, 0.0),
                            0.0)
                    return py_trees.common.Status.SUCCESS
                try:
                    # check first if actor is available or already deleted - if deleted, no speed can be set anymore and no waypoints are needed
                    self._actor.get_velocity()
                except:
                    return py_trees.common.Status.RUNNING
                if self.arts:
                    self._update_speed_arts(actor=actor, current_waypoint_idx=current_waypoint_idx, current_relative_time=current_relative_time)
                else:
                    self._update_speed_rts(actor=actor, current_waypoint_idx=current_waypoint_idx, current_relative_time=current_relative_time)
                
        return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        """Release exclusive RtS control when the trajectory ends or is interrupted."""
        if self.rts and self._rts_actor_control is not None:
            self._rts_actor_control.resume_control(self)
            self._rts_actor_control = None
        super().terminate(new_status)

    def _update_speed_rts(self, actor, current_waypoint_idx, current_relative_time, teleporting=True, switch_following_method_at_time=math.inf, lookahead=10):
        """
        Update the velocity of the actor based on the distance to the target waypoint.
        If target waypoint is already passed, actor decelerate until next waypoint is reached.
        Check if waypoint is passed is done comparing velocities and target directions which should be similar for consecutive waypoints.
        
        different opportunities due to inaccuracies internally in Carla processing:
        teleporting = True: time-interpolated replay enforced by setting actor pose. Warning: no smooth trajectory!
        teleporting = False & current time occurence of road user < switch following method at time: setting velocity, teleport rotation, but not position
        teleporting = False & current time occurence of road user >= switch following method at time: try to follow route with carla internal speed controller - quite inaccurate
        
        lookahead: indices of trajectories which should be used as a lookahead to smooth velocity profile
        """
        # get actual and target location
        target_waypoint = self._waypoints[current_waypoint_idx]
        target_transform = sr_tools.openscenario_parser.OpenScenarioParser.convert_position_to_transform(target_waypoint[0])
        target_location = target_transform.location

        # Accurate replay: interpolate the pose between the surrounding
        # trajectory samples and report the matching segment velocity to CARLA.
        if teleporting:
            if current_waypoint_idx == 0:
                interpolated_transform = target_transform
                velocity_vector = carla.Vector3D(0.0, 0.0, 0.0)
            else:
                prior_idx = current_waypoint_idx - 1
                prior_waypoint = self._waypoints[prior_idx]
                prior_transform = sr_tools.openscenario_parser.OpenScenarioParser.convert_position_to_transform(
                    prior_waypoint[0])
                prior_location = prior_transform.location
                segment_time = self._times[current_waypoint_idx] - self._times[prior_idx]

                if segment_time <= 0.0:
                    raise ValueError("RtS trajectory timestamps must be strictly increasing")

                scale = (current_relative_time - self._times[prior_idx]) / segment_time
                scale = min(max(scale, 0.0), 1.0)

                def interpolate(before, after):
                    return before + (after - before) * scale

                interpolated_location = carla.Location(
                    x=interpolate(prior_location.x, target_location.x),
                    y=interpolate(prior_location.y, target_location.y),
                    z=interpolate(prior_location.z, target_location.z))
                interpolated_rotation = carla.Rotation(
                    pitch=interpolate(prior_transform.rotation.pitch, target_transform.rotation.pitch),
                    yaw=interpolate(prior_transform.rotation.yaw, target_transform.rotation.yaw),
                    roll=interpolate(prior_transform.rotation.roll, target_transform.rotation.roll))
                interpolated_transform = carla.Transform(interpolated_location, interpolated_rotation)
                velocity_vector = carla.Vector3D(
                    (target_location.x - prior_location.x) / segment_time,
                    (target_location.y - prior_location.y) / segment_time,
                    (target_location.z - prior_location.z) / segment_time)

            target_speed = math.sqrt(
                velocity_vector.x**2 + velocity_vector.y**2 + velocity_vector.z**2)
            self._apply_rts_state(
                actor, interpolated_transform, velocity_vector, target_speed)
            return

        actor_location = CarlaDataProvider.get_location(self._actor)
        
        # get further needed waypoints
        offset_idx = lookahead
        lookahead_idx = min(len(self._waypoints)-1, current_waypoint_idx+offset_idx)
        prior_waypoint = None if current_waypoint_idx == 0 else self._waypoints[current_waypoint_idx - 1]
        prior_transform = None if prior_waypoint == None else sr_tools.openscenario_parser.OpenScenarioParser.convert_position_to_transform(prior_waypoint[0])
        prior_location = None if prior_transform == None else prior_transform.location
        
        waypoint_ahead = self._waypoints[lookahead_idx]
        transform_ahead = sr_tools.openscenario_parser.OpenScenarioParser.convert_position_to_transform(waypoint_ahead[0])
        location_ahead = transform_ahead.location
        
        # calculate scalar speed according to lookahead
        remaining_dist = self._direct_distance(actor_location, location_ahead)
        remaining_time = self._times[lookahead_idx] - current_relative_time
        target_speed = remaining_dist / max(remaining_time, 0.001)  # just using a small number to avoid division by zero
    
        # set speed
        if current_relative_time < switch_following_method_at_time:
            # set speed at correct velocity without controller (not so smooth), but with lookahead
            direction = carla.Vector3D(location_ahead.x-actor_location.x, location_ahead.y-actor_location.y, location_ahead.z-actor_location.z)
            vector_length = math.sqrt(direction.x**2 + direction.y**2 + direction.z**2)
            normalized_direction = carla.Vector3D(direction.x/vector_length, direction.y/vector_length, direction.z/vector_length)
            velocity_vector = carla.Vector3D(normalized_direction.x*target_speed, normalized_direction.y*target_speed, normalized_direction.z*target_speed)
            
            # check if vehicle is on map. if not, leave velocity at 0
            if math.sqrt(velocity_vector.x**2 + velocity_vector.y**2 + velocity_vector.z**2) > 100:
                # case if road user is not at envelope, but somewhere outside (teleport needed, so way to long)
                actor.update_target_speed(0)
                return
            self._actor.set_target_velocity(velocity_vector)
            
            # set slip = 0 (otherwise drifts occur)
            self._control_direction(velocity_vector)
        else:
            # set target speed, according to physics/ controller smoothly. 
            # Caution: this seems to be really inaccurate and not usefull for densed sampled trajectories
        
            # check if waypoint has already been passed (velocity and direction should lead in similar direction)
            actual_velocity_vector = np.array([self._actor.get_velocity().x, self._actor.get_velocity().y, self._actor.get_velocity().z])
            if np.dot(np.array([target_location.x-actor_location.x, target_location.y-actor_location.y, target_location.z-actor_location.z]), actual_velocity_vector) >= 0:
                target_speed = target_speed
            else:
                target_speed = 0
        
            # give new speed to road user controller
            actor.update_target_speed(target_speed)

    def _apply_rts_state(self, actor, transform, velocity_vector, target_speed):
        """
        Apply an RtS pose and velocity through the actor-specific CARLA API.

        Walker trajectory transforms use the actor origin, whereas
        ApplyWalkerState expects a foot/navmesh position and adds half the capsule
        height. Apply the recorded transform and walker control in one batch to
        preserve the trajectory reference point. Vehicles accept separate
        transform and target-velocity updates. The regular actor controller
        remains suspended while RtS is active.
        """
        applied_speed = target_speed

        if isinstance(self._actor, carla.Walker):
            client = CarlaDataProvider.get_client()
            if client is None:
                raise RuntimeError("CARLA client is not available for RtS walker state")

            applied_speed = math.hypot(velocity_vector.x, velocity_vector.y)
            walker_control = carla.WalkerControl()
            walker_control.speed = applied_speed
            walker_control.jump = False

            if applied_speed > 0.0:
                walker_control.direction = carla.Vector3D(
                    velocity_vector.x / applied_speed,
                    velocity_vector.y / applied_speed,
                    0.0)
            else:
                forward_vector = transform.get_forward_vector()
                forward_length = math.hypot(forward_vector.x, forward_vector.y)
                if forward_length > 0.0:
                    walker_control.direction = carla.Vector3D(
                        forward_vector.x / forward_length,
                        forward_vector.y / forward_length,
                        0.0)

            client.apply_batch_sync([
                carla.command.ApplyTransform(self._actor, transform),
                carla.command.ApplyWalkerControl(self._actor, walker_control)
            ], False)
        else:
            self._actor.set_transform(transform)
            self._actor.set_target_velocity(velocity_vector)

        actor.update_target_speed(applied_speed)
            
    def _update_speed_arts(self, actor, current_waypoint_idx, current_relative_time, lookahead=10):
        from enum import IntEnum
        class ARtSMode(IntEnum):
            """
            different modes for road user for arts
            """
            Wait = 1
            Catchup = 2
            FollowTrajectory = 3 
            
        # get actual and target location
        target_waypoint = self._waypoints[current_waypoint_idx]
        target_transform = sr_tools.openscenario_parser.OpenScenarioParser.convert_position_to_transform(target_waypoint[0])
        actor_location = CarlaDataProvider.get_location(self._actor)
            
        current_waypoint_idx_guess = self._get_closest_waypoint_idx(actor_location, [trans.location for trans in self._waypoint_transforms])  # check for actual closest waypoint depending on actual location of actor
        length_offset_idx = self._direct_distance(actor_location, self._waypoint_transforms[current_waypoint_idx].location)  # calculate the distance how far it is off
        allowed_offset = 0.5  # m
        last_waypoint_past = (length_offset_idx < allowed_offset) or (current_waypoint_idx_guess > current_waypoint_idx) or (current_waypoint_idx < 10) # check if the regular waypoint is already past (or close to be past)
        #current_waypoint_idx = current_waypoint_idx_guess  # reset to actual point (not predicted)
        
        # get further needed waypoints
        offset_idx = lookahead
        lookahead_idx = min(len(self._waypoints)-1, current_waypoint_idx+offset_idx)
        
        waypoint_ahead = self._waypoints[lookahead_idx]
        transform_ahead = sr_tools.openscenario_parser.OpenScenarioParser.convert_position_to_transform(waypoint_ahead[0])
        location_ahead = transform_ahead.location
        
        # check whether thresholds are fullfilled
        ttc = math.inf
        thw = math.inf
        dhw = math.inf
        
        # only check for road users which are prioritized to reduce runtime
        all_actors = CarlaDataProvider.get_actors()
        actors_to_consider = []
        for actor_ent in all_actors:
            if actor_ent[1].attributes["role_name"] in self.prioritized_objects:
                actors_to_consider.append(actor_ent)
                        
        # calculate metrics only if road user is not standing still
        # CAUTION: no direction of ttc calculated or rules investigated - comes excusively from prioritized objects by now
        if self._calculate_velocity(self._actor.get_velocity()) > 0.2:  # to check whether road user stand still
            ttc, _ = self._calc_advanced_ttx_metric(metric_type="ttc", current_idx=current_waypoint_idx_guess, actors_to_consider=actors_to_consider, time_horizon=self.arts_config["threshold_ttc"], discretization=0.1)
            thw, _ = self._calc_advanced_ttx_metric(metric_type="thw", current_idx=current_waypoint_idx_guess, actors_to_consider=actors_to_consider, time_horizon=self.arts_config["threshold_thw"], discretization=0.1)
        dhw, _ = self._calc_advanced_ttx_metric(metric_type="dhw", current_idx=current_waypoint_idx_guess, actors_to_consider=actors_to_consider, time_horizon=self.arts_config["threshold_dhw"], discretization=0.1)
        
        # decide mode based on metrices and whether it is at correct waypoint        
        if ttc > self.arts_config["threshold_ttc"] and thw > self.arts_config["threshold_thw"] and dhw > self.arts_config["threshold_dhw"]:
            if last_waypoint_past:
                arts_mode = ARtSMode.FollowTrajectory
            else:
                arts_mode = ARtSMode.Catchup
        else:
            arts_mode = ARtSMode.Wait
        
        # calculate scalar speed according to lookahead
        remaining_dist = self._direct_distance(actor_location, location_ahead)
        remaining_time = self._times[lookahead_idx] - current_relative_time
        target_speed = remaining_dist / max(remaining_time, 0.001)  # just to avoid division by zero
    
        # set speed at correct velocity without controller (not so smooth), but with lookahead
        direction = carla.Vector3D(location_ahead.x-actor_location.x, location_ahead.y-actor_location.y, location_ahead.z-actor_location.z)
        vector_length = math.sqrt(direction.x**2 + direction.y**2 + direction.z**2)
        normalized_direction = carla.Vector3D(direction.x/vector_length, direction.y/vector_length, direction.z/vector_length)
        
        if arts_mode == ARtSMode.Catchup:
            target_speed = target_speed * (1.0 + (self.arts_config["catchup_velocity_percentage"]/100.0))
        elif arts_mode == ARtSMode.Wait:
            target_speed = 0.0
            
        velocity_vector = carla.Vector3D(normalized_direction.x*target_speed, normalized_direction.y*target_speed, normalized_direction.z*target_speed)
        
        # check if vehicle is on map. if not, leave velocity at 0 - if it is not at map, vehlocity would be too large to be feasible
        if math.sqrt(velocity_vector.x**2 + velocity_vector.y**2 + velocity_vector.z**2) > 200:
            # case if road user is not at envelope, but somewhere outside (teleport needed, so way to long)
            actor.update_target_speed(0)
            return
        
        self._actor.set_target_velocity(velocity_vector)
        self._control_direction(velocity_vector)
        
        return

    def _control_direction(self, velocity_vector):
        # set slip = 0 (otherwise drifts occur - only for moving road users)
        if math.sqrt(velocity_vector.x**2 + velocity_vector.y**2 + velocity_vector.z**2) > 1.0:  # check whether road user stand still
            transform = self._actor.get_transform()
            transform.rotation.yaw = np.arctan2(velocity_vector.y, velocity_vector.x)/math.pi*180 % 360
            self._actor.set_transform(transform) 
        return

    def _direct_distance(self, location_1, location_2):
        return math.sqrt((location_1.x-location_2.x)**2 + (location_1.y-location_2.y)**2 + (location_1.z-location_2.z)**2)

    def _calculate_velocity(self, velocity):
        return math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

    def _get_closest_waypoint_idx(self, actor_location, waypoints):
        """
        get closest waypoint to vehicle
        """
        actual_dist = math.inf
        for index, wp in enumerate(waypoints):
            cand_dist = self._direct_distance(actor_location, wp)
            if cand_dist > actual_dist:
                return index-1
            else:
                actual_dist = cand_dist
        return len(waypoints)-1

    def _calc_advanced_ttx_metric(self, metric_type, current_idx, discretization=0.05, time_horizon=5.0, actors_to_consider=None):
        """
        calculation of a given ttx metic
        available: thw, ttc, dhw
        calculation for intersections (based on ego path and linear extrapolation of object movement)
        """
        # check if metric type available
        if metric_type not in ["thw", "ttc", "dhw"]:
            print("ERROR: No such metric type '"+metric_type+"' available.")            
            return None, None
        
        ttx = math.inf
        considered_time_horizon_in_seconds = time_horizon
        
        ru_name = None
        
        # ego movement prediction based on polyline and actual velocity
        # for dhw set fictional value so that it can be calculated as it would drive uniformly and transformed to a distance afterwards
        dtw_pseudo_velocity = 5.0
        if metric_type == "dhw":
            ego_velocity_abs = dtw_pseudo_velocity
        else:
            ego_velocity_abs = self._calculate_velocity(self._actor.get_velocity())
        if current_idx > len(self._waypoint_transforms)-1:
            return ttx # then there is to less information to predict
        predicted_ego_state = self._predict_ego_state(ego_velocity_abs, current_idx, considered_time_horizon_in_seconds, discretization)
        
        if actors_to_consider is None:
            actors_to_consider = CarlaDataProvider.get_actors()
        
        for object_ru_item in actors_to_consider:
            ttx_to_ru = math.inf
            
            # get actual ru state
            object_ru = object_ru_item[1]
            object_ru_id = object_ru_item[0]
            object_transform = object_ru.get_transform()
            object_velocity = object_ru.get_velocity()
            object_velocity_abs = self._calculate_velocity(object_velocity)
            
            # font check against ego road user
            if object_ru == self._actor:
                continue
            
            # filter potentially irrelevant rus (always standing as observed) 
            # WARNING: this include objects always parking although they may be on street. Simplification done for calculation efficiency 
            if object_velocity_abs < 0.01 and object_ru_id not in self.moving_object_ids:
                continue
            
            if object_ru_id not in self.moving_object_ids:
                self.moving_object_ids.append(object_ru_id)
            
            if metric_type == "ttc":
                project_object_state = True
            elif metric_type == "thw" or "dhw":
                project_object_state = False
            
            # extrapolate state x seconds (according to velocity vector - without taking infrastructure into account)
            for index, timestep in enumerate(np.arange(0, considered_time_horizon_in_seconds, discretization)):
                if len(predicted_ego_state) > index:  # check if prediction is available or cannot be made (e.g. because of significant extrapolation)
                    distance, can_be_reached = self._check_projected_distance(predicted_ego_state[index], ego_vel_abs=ego_velocity_abs, object_ru=object_ru, object_transform=object_transform, object_velocity=object_velocity, object_abs_velocity=object_velocity_abs, timestep=timestep, max_time=considered_time_horizon_in_seconds, project_object=project_object_state)
                    if not can_be_reached:
                        break
                    if distance == 0:
                        ttx_to_ru = timestep
                        break
                
            # check whether it is the smallest 
            if ttx_to_ru < ttx:
                ru_name = object_ru.attributes["role_name"]
            ttx = min(ttx, ttx_to_ru)
            
            # back transformation in meter for dhw
            if metric_type == "dhw":
                ttx = dtw_pseudo_velocity * ttx
            
        return ttx, ru_name
    
    def _check_projected_distance(self, predicted_ego_state, ego_vel_abs, object_ru, object_transform, 
                                  object_velocity, object_abs_velocity, timestep, max_time, project_object=True):
        approx_distance = math.inf
        can_be_reached = True
        
        object_rotation = object_transform.rotation
        object_location = object_transform.location
        
        if project_object:
            length = timestep
        else:
            length = 0.0
        
        predict_object_location = carla.Vector3D(x=object_location.x + length * object_velocity.x, 
                                                 y=object_location.y + length * object_velocity.y, 
                                                 z=object_location.z + length * object_velocity.z)
        
        # check first approximated distance whether it make sense to include a more accurate calculation
        approx_distance = self._direct_distance(predict_object_location, predicted_ego_state["location"])
        
        # check if can be reached (really rough estimation - can be improved if necessary)
        if (object_abs_velocity + ego_vel_abs) * (max_time-timestep) > approx_distance:
            can_be_reached = True
        else:
            can_be_reached = False
        
        approx_bb_ego = max(2, predicted_ego_state["bb"].length / 4) # divided by 4 because it is circumfence and only half of length and width is needed
        approx_bb_object = max(2, object_ru.bounding_box.extent.x + object_ru.bounding_box.extent.y)
        if approx_distance < (approx_bb_ego + approx_bb_object):
            rect_1 = self._get_bb_shapely(predict_object_location, object_rotation, object_ru.bounding_box.extent.x, object_ru.bounding_box.extent.y)
            if rect_1.intersects(predicted_ego_state["bb"]):
                approx_distance = 0
                
        # check rules whether it should be counted or not (only relevant, if collision occurs)
        if approx_distance == 0:
            rules_okay = True
            if "left_before_right" in self.check_for_prioritization_rules:
                if not self._rule_right_before_left(predicted_ego_state["rotation"], object_rotation):
                    rules_okay = False
            if "is_behind" in self.check_for_prioritization_rules:
                if not self._rule_behind(predicted_ego_state["location"], predicted_ego_state["rotation"], 
                                                    predict_object_location, object_rotation):
                    rules_okay = False
            # if rules are not fulfilled, has not to be considered
            if not rules_okay:
                approx_distance += 0.1  # just increase to dont have 0 here (which would mean we have a valid collision we use afterwards)
                can_be_reached = False
        
        return approx_distance, can_be_reached
    
    def _get_bb_shapely(self, location, rotation, length, width):
        """
        get shapely bounding box of element with dimensions, location and rotation
        """
        # check if length and width is feasible (is done since carla bounding boxes for bicyclist have width and length 0):
        if length == 0:
            length = 1.5
        if width == 0:
            width = 0.3
        
        # Get the corners of the bounding box
        corners = [
            carla.Location(x=length, y=width),
            carla.Location(x=-length, y=width),
            carla.Location(x=-length, y=-width),
            carla.Location(x=length, y=-width),
        ]

        # Rotate the corners according to the actor's rotation
        rad_yaw = np.deg2rad(rotation.yaw)
        rotated_corners = [
            carla.Location(
                x=c.x * np.cos(rad_yaw) - c.y * np.sin(rad_yaw),
                y=c.x * np.sin(rad_yaw) + c.y * np.cos(rad_yaw)
            ) + location
            for c in corners
        ]

        return Polygon([[rotated_corners[0].x, rotated_corners[0].y],
                        [rotated_corners[1].x, rotated_corners[1].y],
                        [rotated_corners[2].x, rotated_corners[2].y],
                        [rotated_corners[3].x, rotated_corners[3].y]
            
        ])
        
    def _predict_ego_state(self, ego_velocity_abs, current_idx, considered_time_horizon_in_seconds, discretization, offset=0.0):
        """
        predict ego state since there may be curves
        """
        predicted_ego_state = []
        
        last_wp_idx = current_idx
        abs_length = 0
        
        # discretization of timesteps to assign the correct locatino for each of those
        for _, timestep in enumerate(np.arange(0.0, considered_time_horizon_in_seconds, discretization)):
            
            # distance to cover (ego drives in a certain timestep according to velocity)
            extrapolated_distance = ego_velocity_abs * timestep
            extrapolation_successful = False
            for index_wp_go_through, upcomming_wp in enumerate(self._waypoint_transforms[last_wp_idx+1:]):
                upcomming_wp = upcomming_wp.location
                last_wp_idx_cand = last_wp_idx+index_wp_go_through
                last_wp = self._waypoint_transforms[last_wp_idx_cand].location
                d_length = self._direct_distance(upcomming_wp, last_wp)
                
                if abs_length + d_length <= extrapolated_distance and not index_wp_go_through == (len(self._waypoint_transforms[last_wp_idx+1:])-1):
                    # distance not reached -> go to next wp; exception: last wp reached -> extrapolate
                    abs_length += d_length
                else:
                    # calculate values
                    # location in this region
                    scale = (extrapolated_distance-abs_length) / d_length
                    dx = upcomming_wp.x-last_wp.x
                    dy = upcomming_wp.y-last_wp.y
                    location = carla.Vector3D(x=last_wp.x + scale * dx, 
                                            y=last_wp.y + scale * dy)
                    rotation = carla.Rotation(yaw=180/math.pi*np.arctan2(dy, dx))
                    
                    # save new last waypoint
                    last_wp_idx = last_wp_idx_cand
                    
                    extrapolation_successful = True
                    break

            if extrapolation_successful:
                predicted_ego_state.append({"bb": self._get_bb_shapely(location, rotation, 
                                                                    length=self._actor.bounding_box.extent.x+offset, 
                                                                    width=self._actor.bounding_box.extent.y+offset), 
                                            "location": location, "rotation": rotation})
        return predicted_ego_state
        
    def _rule_right_before_left(self, ego_rotation, object_rotation):
        """
        check if colliding vehicle comes from left (then, no prioritization)
        only this is checked and then it is set to True (no real checking of right before left rule)
        """
        diff_angle = self._difference_angle(object_rotation.yaw, ego_rotation.yaw)
        
        # angle to check if comming from right side +-45 degree
        if 45 < diff_angle < 135:
            return False
        return True
    
    def _rule_behind(self, ego_location, ego_rotation, object_location, object_rotation, allowed_difference_angle=20):
        """
        check if the vehicle is behind or next to, but not in front.
        only return "False" if vehicle is in front of the other
        """
        diff_angle = self._difference_angle(ego_rotation.yaw, object_rotation.yaw)
        
        # check if following
        if diff_angle < allowed_difference_angle or diff_angle > (360-allowed_difference_angle):
            # check if ego is in front of other vehicle
            offset = np.array([(ego_location.x - object_location.x), (ego_location.y - object_location.y)])
            ego_yaw_rad = ego_rotation.yaw/180*math.pi
            direction_ego = np.array([np.cos(ego_yaw_rad), np.sin(ego_yaw_rad)])
            if np.dot(offset, direction_ego) > 0:
                return False
        return True
    
    def _difference_angle(self, angle1, angle2):
        """
        returns difference angle (counter clockwise in degrees)
        """
        return (angle2-angle1) % 360


class ChangeActorWaypointsToReachPosition(ChangeActorWaypoints):

    """
    Atomic to change the waypoints for an actor controller in order to reach
    a given position.

    The behavior is in RUNNING state until the last waypoint is reached, or if a
    second waypoint related atomic for the same actor is triggered. These are:
    - ChangeActorWaypoints
    - ChangeActorWaypointsToReachPosition
    - ChangeActorLateralMotion

    Args:
        actor (carla.Actor): Controlled actor.
        position (carla.Transform): CARLA transform to be reached by the actor.
        name (string): Name of the behavior.
            Defaults to 'ChangeActorWaypointsToReachPosition'.

    Attributes:
        _waypoints (List of carla.Transform): List of waypoints (CARLA transforms).
        _end_transform (carla.Transform): Final position (CARLA transform).
        _start_time (float): Start time of the atomic [s].
            Defaults to None.
        _grp (GlobalPlanner): global planner instance of the town
    """

    def __init__(self, actor, position, name="ChangeActorWaypointsToReachPosition"):
        """
        Setup parameters
        """
        super().__init__(actor, [])

        self._end_transform = position

        self._grp = GlobalRoutePlanner(CarlaDataProvider.get_world().get_map(), 2.0)
        self._grp.setup()

    def initialise(self):
        """
        Set _start_time and get (actor, controller) pair from Blackboard.

        Generate a waypoint list (route) which representes the route. Set
        this waypoint list for the actor controller.

        May throw if actor is not available as key for the ActorsWithController
        dictionary from Blackboard.
        """

        # get start position
        position_actor = CarlaDataProvider.get_location(self._actor)

        # calculate plan with global_route_planner function
        plan = self._grp.trace_route(position_actor, self._end_transform.location)

        for elem in plan:
            self._waypoints.append(elem[0].transform)

        super().initialise()


class ChangeActorLateralMotion(AtomicBehavior):

    """
    Atomic to change the waypoints for an actor controller.

    The behavior is in RUNNING state until the last waypoint is reached, or if a
    second waypoint related atomic for the same actor is triggered. These are:
    - ChangeActorWaypoints
    - ChangeActorLateralMotion
    - ChangeActorLaneOffset

    If an impossible lane change is asked for (due to the lack of lateral lanes,
    next waypoints, continuous line, etc) the atomic will return a plan with the
    waypoints until such impossibility is found.

    Args:
        actor (carla.Actor): Controlled actor.
        direction (string): Lane change direction ('left' or 'right').
            Defaults to 'left'.
        distance_lane_change (float): Distance of the lance change [meters].
            Defaults to 25.
        distance_other_lane (float): Driven distance after the lange change [meters].
            Defaults to 100.
        name (string): Name of the behavior.
            Defaults to 'ChangeActorLateralMotion'.

    Attributes:
        _waypoints (List of carla.Transform): List of waypoints representing the lane change (CARLA transforms).
        _direction (string): Lane change direction ('left' or 'right').
        _distance_same_lane (float): Distance on the same lane before the lane change starts [meters]
            Constant to 5.
        _distance_other_lane (float): Max. distance on the target lane after the lance change [meters]
            Constant to 100.
        _distance_lane_change (float): Max. total distance of the lane change [meters].
        _pos_before_lane_change: carla.Location of the actor before the lane change.
            Defaults to None.
        _target_lane_id (int): Id of the target lane
            Defaults to None.
        _start_time (float): Start time of the atomic [s].
            Defaults to None.
    """

    def __init__(self, actor, direction='left', distance_lane_change=25, distance_other_lane=100,
                 lane_changes=1, name="ChangeActorLateralMotion"):
        """
        Setup parameters
        """
        super().__init__(name, actor)

        self._waypoints = []
        self._direction = direction
        self._distance_same_lane = 5
        self._distance_other_lane = distance_other_lane
        self._distance_lane_change = distance_lane_change
        self._lane_changes = lane_changes
        self._pos_before_lane_change = None
        self._target_lane_id = None
        self._plan = None

        self._start_time = None

    def initialise(self):
        """
        Set _start_time and get (actor, controller) pair from Blackboard.

        Generate a waypoint list (route) which representes the lane change. Set
        this waypoint list for the actor controller.

        May throw if actor is not available as key for the ActorsWithController
        dictionary from Blackboard.
        """
        actor_dict = {}

        try:
            check_actors = operator.attrgetter("ActorsWithController")
            actor_dict = check_actors(py_trees.blackboard.Blackboard())
        except AttributeError:
            pass

        if not actor_dict or self._actor.id not in actor_dict:
            raise RuntimeError("Actor not found in ActorsWithController BlackBoard")

        self._start_time = GameTime.get_time()

        # get start position
        position_actor = CarlaDataProvider.get_map().get_waypoint(CarlaDataProvider.get_location(self._actor))

        # calculate plan with scenario_helper function
        self._plan, self._target_lane_id = generate_target_waypoint_list_multilane(
            position_actor, self._direction, self._distance_same_lane,
            self._distance_other_lane, self._distance_lane_change, check=False, lane_changes=self._lane_changes)

        if self._plan:
            for elem in self._plan:
                self._waypoints.append(elem[0].transform)

        actor_dict[self._actor.id].update_waypoints(self._waypoints, start_time=self._start_time)

        super().initialise()

    def update(self):
        """
        Check the actor's current state and if the lane change was completed

        returns:
            py_trees.common.Status.SUCCESS, if lane change was successful, or
                                            if another ChangeActorLateralMotion atomic for the same actor was triggered.
            py_trees.common.Status.FAILURE, if the actor is not found in ActorsWithController Blackboard dictionary.
            py_trees.common.Status.FAILURE, else.
        """

        try:
            check_actors = operator.attrgetter("ActorsWithController")
            actor_dict = check_actors(py_trees.blackboard.Blackboard())
        except AttributeError:
            pass

        if not actor_dict or self._actor.id not in actor_dict:
            return py_trees.common.Status.FAILURE

        if not self._plan:
            print("{} couldn't perform the expected lane change".format(self._actor))
            return py_trees.common.Status.FAILURE

        if actor_dict[self._actor.id].get_last_waypoint_command() != self._start_time:
            return py_trees.common.Status.SUCCESS

        new_status = py_trees.common.Status.RUNNING

        current_position_actor = CarlaDataProvider.get_map().get_waypoint(self._actor.get_location())
        current_lane_id = current_position_actor.lane_id

        if current_lane_id == self._target_lane_id:
            # driving on new lane
            distance = current_position_actor.transform.location.distance(self._pos_before_lane_change)

            if distance > self._distance_other_lane:
                # long enough distance on new lane --> SUCCESS
                new_status = py_trees.common.Status.SUCCESS

                new_waypoints = []
                map_wp = current_position_actor
                while len(new_waypoints) < 200:
                    map_wps = map_wp.next(2.0)
                    if map_wps:
                        new_waypoints.append(map_wps[0].transform)
                        map_wp = map_wps[0]
                    else:
                        break

                actor_dict[self._actor.id].update_waypoints(new_waypoints, start_time=self._start_time)

        else:
            self._pos_before_lane_change = current_position_actor.transform.location

        return new_status


class ChangeActorLaneOffset(AtomicBehavior):

    """
    OpenSCENARIO atomic.
    Atomic to change the offset of the controller.

    The behavior is in RUNNING state until the offset os reached (if 'continuous' is set to False)
    or forever (if 'continuous' is True). This behavior will automatically stop if a second waypoint
    related atomic for the same actor is triggered. These are:
    - ChangeActorWaypoints
    - ChangeActorLateralMotion
    - ChangeActorLaneOffset

    Args:
        actor (carla.Actor): Controlled actor.
        offset (float): Float determined the distance to the center of the lane. Positive distance imply a
            displacement to the right, while negative displacements are to the left.
        relative_actor (carla.Actor): The actor from which the offset is taken from. Defaults to None
        continuous (bool): If True, the behaviour never ends. If False, the behaviour ends when the lane
            offset is reached. Defaults to True.

    Attributes:
        _offset (float): lane offset.
        _relative_actor (carla.Actor): relative actor.
        _continuous (bool): stored the value of the 'continuous' argument.
        _start_time (float): Start time of the atomic [s].
            Defaults to None.
        _overwritten (bool): flag to check whether or not this behavior was overwritten by another. Helps
            to avoid the missinteraction between two ChangeActorLaneOffsets.
        _current_target_offset (float): stores the value of the offset when dealing with relative distances
        _map (carla.Map): instance of the CARLA map.
    """

    OFFSET_THRESHOLD = 0.1

    def __init__(self, actor, offset, relative_actor=None, continuous=True, name="ChangeActorWaypoints"):
        """
        Setup parameters
        """
        super().__init__(name, actor)

        self._offset = offset
        self._relative_actor = relative_actor
        self._continuous = continuous
        self._start_time = None
        self._current_target_offset = 0

        self._overwritten = False
        self._map = CarlaDataProvider.get_map()

    def initialise(self):
        """
        Set _start_time and get (actor, controller) pair from Blackboard.

        Set offset for actor controller.

        May throw if actor is not available as key for the ActorsWithController
        dictionary from Blackboard.
        """
        actor_dict = {}

        try:
            check_actors = operator.attrgetter("ActorsWithController")
            actor_dict = check_actors(py_trees.blackboard.Blackboard())
        except AttributeError:
            pass

        if not actor_dict or self._actor.id not in actor_dict:
            raise RuntimeError("Actor not found in ActorsWithController BlackBoard")

        self._start_time = GameTime.get_time()

        actor_dict[self._actor.id].update_offset(self._offset, start_time=self._start_time)

        super().initialise()

    def update(self):
        """
        Check the actor's state along the waypoint route.

        returns:
            py_trees.common.Status.SUCCESS, if the lane offset was reached (and 'continuous' was False), or
                                            if another waypoint atomic for the same actor was triggered
            py_trees.common.Status.FAILURE, if the actor is not found in ActorsWithController Blackboard dictionary.
            py_trees.common.Status.RUNNING, else.
        """
        try:
            check_actors = operator.attrgetter("ActorsWithController")
            actor_dict = check_actors(py_trees.blackboard.Blackboard())
        except AttributeError:
            pass

        if not actor_dict or self._actor.id not in actor_dict:
            return py_trees.common.Status.FAILURE

        if actor_dict[self._actor.id].get_last_lane_offset_command() != self._start_time:
            # Differentiate between lane offset and other lateral commands
            self._overwritten = True
            return py_trees.common.Status.SUCCESS

        if actor_dict[self._actor.id].get_last_waypoint_command() != self._start_time:
            return py_trees.common.Status.SUCCESS

        if self._relative_actor:
            # Calculate new offset
            relative_actor_loc = CarlaDataProvider.get_location(self._relative_actor)
            relative_center_wp = self._map.get_waypoint(relative_actor_loc)

            # Value
            relative_center_loc = relative_center_wp.transform.location
            relative_actor_offset = relative_actor_loc.distance(relative_center_loc)

            # Sign
            f_vec = relative_center_wp.transform.get_forward_vector()
            d_vec = relative_actor_loc - relative_center_loc
            cross = f_vec.x * d_vec.y - f_vec.y * d_vec.x

            if cross < 0:
                relative_actor_offset *= -1.0

            self._current_target_offset = relative_actor_offset + self._offset
            # Set the new offset
            actor_dict[self._actor.id].update_offset(self._current_target_offset)

        if not self._continuous:
            # Calculate new offset
            actor_loc = CarlaDataProvider.get_location(self._actor)
            center_wp = self._map.get_waypoint(actor_loc)

            # Value
            center_loc = center_wp.transform.location
            actor_offset = actor_loc.distance(center_loc)

            # Sign
            f_vec = center_wp.transform.get_forward_vector()
            d_vec = actor_loc - center_loc
            cross = f_vec.x * d_vec.y - f_vec.y * d_vec.x

            if cross < 0:
                actor_offset *= -1.0

            # Check if the offset has been reached
            if abs(actor_offset - self._current_target_offset) < self.OFFSET_THRESHOLD:
                return py_trees.common.Status.SUCCESS

        # TODO: As their is no way to check the distance to a specific lane, both checks will fail if the
        # actors are outside its 'route lane' or at an intersection

        new_status = py_trees.common.Status.RUNNING

        return new_status

    def terminate(self, new_status):
        """
        On termination of this behavior, the offset is set back to zero
        """

        if not self._overwritten:
            try:
                check_actors = operator.attrgetter("ActorsWithController")
                actor_dict = check_actors(py_trees.blackboard.Blackboard())
            except AttributeError:
                pass

            if actor_dict and self._actor.id in actor_dict:
                actor_dict[self._actor.id].update_offset(0)

            self._overwritten = True

        super().terminate(new_status)


class ChangeLateralDistance(AtomicBehavior):
    """
    OpenSCENARIO atomic.
    Atomic to change the offset of the controller.

    The behavior is in RUNNING state until the offset os reached (if 'continuous' is set to False)
    or forever (if 'continuous' is True). This behavior will automatically stop if a second waypoint
    related atomic for the same actor is triggered. These are:
    - ChangeActorWaypoints
    - ChangeActorLateralMotion
    - ChangeActorLaneOffset

    Args:
        actor (carla.Actor): Controlled actor.
        offset (float): Float determined the distance to the center of the lane. Positive distance imply a
            displacement to the right, while negative displacements are to the left.
        relative_actor (carla.Actor): The actor from which the offset is taken from. Defaults to None
        continuous (bool): If True, the behaviour never ends. If False, the behaviour ends when the lane
            offset is reached. Defaults to True.

    Attributes:
        _offset (float): lane offset.
        _relative_actor (carla.Actor): relative actor.
        _continuous (bool): stored the value of the 'continuous' argument.
        _start_time (float): Start time of the atomic [s].
            Defaults to None.
        _overwritten (bool): flag to check whether or not this behavior was overwritten by another. Helps
            to avoid the missinteraction between two ChangeActorLaneOffsets.
        _current_target_offset (float): stores the value of the offset when dealing with relative distances
        _map (carla.Map): instance of the CARLA map.
    """

    OFFSET_THRESHOLD = 0.3

    def __init__(self, actor, offset, relative_actor=None, freespace=False,
                 continuous=True, name="ChangeActorWaypoints", event_name=None):
        """
        Setup parameters
        """
        super().__init__(name, actor)

        self._offset = offset
        self._relative_actor = relative_actor
        self._continuous = continuous
        self._freespace = freespace
        self._start_time = None
        self._current_target_offset = 0
        self._overwritten = False
        self._map = CarlaDataProvider.get_map()
        if freespace:
            if self._offset > 0:
                self._offset += self._relative_actor.bounding_box.extent.y + self._actor.bounding_box.extent.y
            else:
                self._offset -= self._relative_actor.bounding_box.extent.y + self._actor.bounding_box.extent.y
        self._actor_name = actor.attributes.get("role_name")
        self._event_name = event_name
        if relative_actor:
            self._ref_actor_name = relative_actor.attributes.get("role_name")

    def initialise(self):
        """
        Set _start_time and get (actor, controller) pair from Blackboard.

        Set offset for actor controller.

        May throw if actor is not available as key for the ActorsWithController
        dictionary from Blackboard.
        """
        actor_dict = {}

        try:
            check_actors = operator.attrgetter("ActorsWithController")
            actor_dict = check_actors(py_trees.blackboard.Blackboard())
        except AttributeError:
            pass

        if not actor_dict or self._actor.id not in actor_dict:
            raise RuntimeError("Actor not found in ActorsWithController BlackBoard")

        self._start_time = GameTime.get_time()

        actor_dict[self._actor.id].update_offset(self._offset, start_time=self._start_time)

        super().initialise()

    def update(self):
        """
        Check the actor's state along the waypoint route.

        returns:
            py_trees.common.Status.SUCCESS, if the lane offset was reached (and 'continuous' was False), or
                                            if another waypoint atomic for the same actor was triggered
            py_trees.common.Status.FAILURE, if the actor is not found in ActorsWithController Blackboard dictionary.
            py_trees.common.Status.RUNNING, else.
        """
        try:
            check_actors = operator.attrgetter("ActorsWithController")
            actor_dict = check_actors(py_trees.blackboard.Blackboard())
        except AttributeError:
            pass

        if not actor_dict or self._actor.id not in actor_dict:
            return py_trees.common.Status.FAILURE

        if actor_dict[self._actor.id].get_last_lane_offset_command() != self._start_time:
            # Differentiate between lane offset and other lateral commands
            self._overwritten = True
            # last_lane_offset_command_time = actor_dict[self._actor.id].get_last_lane_offset_command()
            return py_trees.common.Status.SUCCESS

        if actor_dict[self._actor.id].get_last_waypoint_command() != self._start_time:
            # last_waypoint_command_time = actor_dict[self._actor.id].get_last_waypoint_command()
            return py_trees.common.Status.SUCCESS

        if self._relative_actor:
            # Calculate new offset
            relative_actor_loc = CarlaDataProvider.get_location(self._relative_actor)
            relative_center_wp = self._map.get_waypoint(relative_actor_loc)

            # Value
            relative_center_loc = relative_center_wp.transform.location
            relative_actor_offset = relative_actor_loc.distance(relative_center_loc)

            # Sign
            f_vec = relative_center_wp.transform.get_forward_vector()
            d_vec = relative_actor_loc - relative_center_loc
            cross = f_vec.x * d_vec.y - f_vec.y * d_vec.x

            if cross < 0:
                relative_actor_offset *= -1.0

            self._current_target_offset = relative_actor_offset + self._offset
            # Set the new offset
            actor_dict[self._actor.id].update_offset(self._current_target_offset)
        if not self._continuous:
            # Calculate new offset
            actor_loc = CarlaDataProvider.get_location(self._actor)
            center_wp = self._map.get_waypoint(actor_loc)

            # Value
            center_loc = center_wp.transform.location
            actor_offset = actor_loc.distance(center_loc)

            # Sign
            f_vec = center_wp.transform.get_forward_vector()
            d_vec = actor_loc - center_loc
            cross = f_vec.x * d_vec.y - f_vec.y * d_vec.x

            if cross < 0:
                actor_offset *= -1.0
            # Check if the offset has been reached
            if abs(actor_offset - self._current_target_offset) < self.OFFSET_THRESHOLD:
                # reach_offset = abs(actor_offset - self._current_target_offset)
                return py_trees.common.Status.SUCCESS

        # TODO: As their is no way to check the distance to a specific lane, both checks will fail if the
        # actors are outside its 'route lane' or at an intersection

        new_status = py_trees.common.Status.RUNNING
        return new_status

    def terminate(self, new_status):
        """
        On termination of this behavior, the offset is set back to zero
        """

        if not self._overwritten:
            try:
                check_actors = operator.attrgetter("ActorsWithController")
                actor_dict = check_actors(py_trees.blackboard.Blackboard())
            except AttributeError:
                pass

            if actor_dict and self._actor.id in actor_dict:
                actor_dict[self._actor.id].update_offset(0)

            self._overwritten = True

        super().terminate(new_status)


class ActorTransformSetterToOSCPosition(AtomicBehavior):

    """
    OpenSCENARIO atomic
    This class contains an atomic behavior to set the transform of an OpenSCENARIO actor.

    Important parameters:
    - actor: CARLA actor to execute the behavior
    - osc_position: OpenSCENARIO position
    - physics [optional]: If physics is true, the actor physics will be reactivated upon success

    The behavior terminates when actor is set to the new actor transform (closer than 1 meter)

    NOTE:
    It is very important to ensure that the actor location is spawned to the new transform because of the
    appearence of a rare runtime processing error. WaypointFollower with LocalPlanner,
    might fail if new_status is set to success before the actor is really positioned at the new transform.
    Therefore: calculate_distance(actor, transform) < 1 meter
    """

    def __init__(self, actor, osc_position, physics=True, name="ActorTransformSetterToOSCPosition"):
        """
        Setup parameters
        """
        super().__init__(name, actor)
        self._osc_position = osc_position
        self._physics = physics
        self._osc_transform = None

    def initialise(self):

        super().initialise()

        if self._actor.is_alive:
            self._actor.set_target_velocity(carla.Vector3D(0, 0, 0))
            self._actor.set_target_angular_velocity(carla.Vector3D(0, 0, 0))

    def update(self):
        """
        Transform actor
        """
        new_status = py_trees.common.Status.RUNNING

        # calculate transform with method in openscenario_parser.py
        self._osc_transform = sr_tools.openscenario_parser.OpenScenarioParser.convert_position_to_transform(
            self._osc_position)
        self._actor.set_transform(self._osc_transform)

        if not self._actor.is_alive:
            new_status = py_trees.common.Status.FAILURE

        if calculate_distance(self._actor.get_location(), self._osc_transform.location) < 1.0:
            if self._physics:
                self._actor.set_simulate_physics(enabled=True)
            new_status = py_trees.common.Status.SUCCESS

        return new_status


class AccelerateToVelocity(AtomicBehavior):

    """
    This class contains an atomic acceleration behavior. The controlled
    traffic participant will accelerate with _throttle_value_ until reaching
    a given _target_velocity_

    Important parameters:
    - actor: CARLA actor to execute the behavior
    - throttle_value: The amount of throttle used to accelerate in [0,1]
    - target_velocity: The target velocity the actor should reach in m/s

    The behavior will terminate, if the actor's velocity is at least target_velocity
    """

    def __init__(self, actor, throttle_value, target_velocity, name="Acceleration"):
        """
        Setup parameters including acceleration value (via throttle_value)
        and target velocity
        """
        super().__init__(name, actor)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._control, self._type = get_actor_control(actor)
        self._throttle_value = throttle_value
        self._target_velocity = target_velocity

    def initialise(self):
        # In case of walkers, we have to extract the current heading
        if self._type == 'walker':
            self._control.speed = self._target_velocity
            self._control.direction = CarlaDataProvider.get_transform(self._actor).get_forward_vector()

        super().initialise()

    def update(self):
        """
        Set throttle to throttle_value, as long as velocity is < target_velocity
        """
        new_status = py_trees.common.Status.RUNNING

        if self._type == 'vehicle':
            if CarlaDataProvider.get_velocity(self._actor) < self._target_velocity:
                self._control.throttle = self._throttle_value
            else:
                new_status = py_trees.common.Status.SUCCESS
                self._control.throttle = 0

        self._actor.apply_control(self._control)
        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status


class UniformAcceleration(AtomicBehavior):

    """
    This class contains an atomic acceleration behavior. The controlled
    traffic participant will accelerate with _throttle_value_ until reaching
    a given acceleration

    Important parameters:
    - actor: CARLA actor to execute the behavior
    - acceleration: Change in speed per unit time
    - target_velocity: The target velocity the actor should reach in m/s
    - start velocity: The start velocity the actor when start accelerate
    - start_time: The start time the actor when start accelerate

    The behavior will terminate, if the actor's velocity is at least target_velocity
    """
    OFFSET_THRESHOLD = 0.1

    def __init__(self, actor, start_velocity, target_velocity, acceleration, start_time, name="Acceleration"):
        """
        Setup parameters including acceleration value (via throttle_value),
        start_velocity, target velocity and duration
        """
        super().__init__(name, actor)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._control, self._type = get_actor_control(actor)
        self._start_velocity = start_velocity
        self._acceleration = acceleration
        self._start_time = start_time
        self._target_velocity = target_velocity
        print(f"actor_type:{self._type}, start_speed:{self._start_velocity}, "
              f"acceleration:{self._acceleration},target_speed:{self._target_velocity},start_time:{self._start_time}")

    def initialise(self):
        # In case of walkers, we have to extract the current heading
        if self._type == 'walker':
            self._control.speed = self._start_velocity
            self._control.direction = CarlaDataProvider.get_transform(self._actor).get_forward_vector()

        super().initialise()

    def update(self):
        """
        Set throttle to control acceleration to a fixed value , as long as velocity is < target_velocity
        """
        new_status = py_trees.common.Status.RUNNING

        time_now = GameTime.get_time()
        time_variation = time_now - self._start_time
        speed_variation = CarlaDataProvider.get_velocity(self._actor) - self._start_velocity
        if self._type == 'vehicle':
            curr_speed = CarlaDataProvider.get_velocity(self._actor)
            if abs(self._target_velocity - curr_speed) < self.OFFSET_THRESHOLD:
                self._control.throttle = 0
                self._control.brake = 0
                new_status = py_trees.common.Status.SUCCESS
                print(f"time_variation:{time_variation},speed_variation:{speed_variation},"
                      f" current_speed:{CarlaDataProvider.get_velocity(self._actor)}")
            if speed_variation / time_variation < self._acceleration:
                self._control.throttle = 1
                self._control.brake = 0
            else:
                self._control.throttle = 0
                self._control.brake = 1

        self._actor.apply_control(self._control)
        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status


class ChangeTargetSpeed(AtomicBehavior):

    """
    Important parameters:
    - actor: CARLA actor to execute the behavior
    - target_velocity: The target velocity the actor should reach in m/s

    The behavior will terminate, if the actor's velocity is at least target_velocity
    """
    OFFSET_THRESHOLD = 0.7

    def __init__(self, actor, target_velocity, name="ChangeTargetSpeed"):
        """
        Setup parameters including acceleration value (via throttle_value)
        and target velocity
        """
        super().__init__(name, actor)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._control, self._type = get_actor_control(actor)
        self._target_velocity = target_velocity

    def initialise(self):
        # In case of walkers, we have to extract the current heading
        if self._type == 'walker':
            self._control.speed = self._target_velocity
            self._control.direction = CarlaDataProvider.get_transform(self._actor).get_forward_vector()

        super().initialise()

    def update(self):
        """
        Set throttle to throttle_value, as long as velocity is < target_velocity
        """
        new_status = py_trees.common.Status.RUNNING

        if self._type == 'vehicle':
            # curr_speed = CarlaDataProvider.get_velocity(self._actor)
            curr_speed = calculate_velocity(self._actor)*3.6
            if abs(self._target_velocity - curr_speed) < self.OFFSET_THRESHOLD:
                self._control.throttle = 0
                self._control.brake = 0
                new_status = py_trees.common.Status.SUCCESS
                print(f'finish change speed!! current speed={curr_speed} km/h')
            else:
                if curr_speed < self._target_velocity:
                    self._control.throttle = 1
                    self._control.brake = 0
                    print(f'current speed={curr_speed} km/h, target speed={self._target_velocity} km/h, accelerate!!! ')
                else:
                    self._control.throttle = 0
                    self._control.brake = 1
                    print('decelerate!!!')

        self._actor.apply_control(self._control)
        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status


class DecelerateToVelocity(AtomicBehavior):

    """
    This class contains an atomic acceleration behavior. The controlled
    traffic participant will accelerate with _throttle_value_ until reaching
    a given _target_velocity_

    Important parameters:
    - actor: CARLA actor to execute the behavior
    - throttle_value: The amount of throttle used to accelerate in [0,1]
    - target_velocity: The target velocity the actor should reach in m/s

    The behavior will terminate, if the actor's velocity is at least target_velocity
    """

    def __init__(self, actor, brake_value, target_velocity, name="Deceleration"):
        """
        Setup parameters including acceleration value (via throttle_value)
        and target velocity
        """
        super().__init__(name, actor)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._control, self._type = get_actor_control(actor)
        self._brake_value = brake_value
        self._target_velocity = target_velocity

    def initialise(self):
        # In case of walkers, we have to extract the current heading
        if self._type == 'walker':
            self._control.speed = self._target_velocity
            self._control.direction = CarlaDataProvider.get_transform(self._actor).get_forward_vector()

        super().initialise()

    def update(self):
        """
        Set throttle to throttle_value, as long as velocity is < target_velocity
        """
        new_status = py_trees.common.Status.RUNNING

        if self._type == 'vehicle':
            speed = CarlaDataProvider.get_velocity(self._actor)
            if speed > self._target_velocity:
                self._control.brake = self._brake_value
            else:
                print(speed)
                print(self._target_velocity)
                new_status = py_trees.common.Status.SUCCESS
                self._control.brake = 0

        self._actor.apply_control(self._control)
        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status


class AccelerateToCatchUp(AtomicBehavior):

    """
    This class contains an atomic acceleration behavior.
    The car will accelerate until it is faster than another car, in order to catch up distance.
    This behaviour is especially useful before a lane change (e.g. LaneChange atom).

    Important parameters:
    - actor: CARLA actor to execute the behaviour
    - other_actor: Reference CARLA actor, actor you want to catch up to
    - throttle_value: acceleration value between 0.0 and 1.0
    - delta_velocity: speed up to the velocity of other actor plus delta_velocity
    - trigger_distance: distance between the actors
    - max_distance: driven distance to catch up has to be smaller than max_distance

    The behaviour will terminate succesful, when the two actors are in trigger_distance.
    If max_distance is driven by the actor before actors are in trigger_distance,
    then the behaviour ends with a failure.
    """

    def __init__(self, actor, other_actor, throttle_value=1, delta_velocity=10, trigger_distance=5,
                 max_distance=500, name="AccelerateToCatchUp"):
        """
        Setup parameters
        The target_speet is calculated on the fly.
        """
        super().__init__(name, actor)

        self._other_actor = other_actor
        self._throttle_value = throttle_value
        self._delta_velocity = delta_velocity  # 1m/s=3.6km/h
        self._trigger_distance = trigger_distance
        self._max_distance = max_distance

        self._control, self._type = get_actor_control(actor)

        self._initial_actor_pos = None

    def initialise(self):

        # get initial actor position
        self._initial_actor_pos = CarlaDataProvider.get_location(self._actor)
        super().initialise()

    def update(self):

        # get actor speed
        actor_speed = CarlaDataProvider.get_velocity(self._actor)
        target_speed = CarlaDataProvider.get_velocity(self._other_actor) + self._delta_velocity

        # distance between actors
        distance = CarlaDataProvider.get_location(self._actor).distance(
            CarlaDataProvider.get_location(self._other_actor))

        # driven distance of actor
        driven_distance = CarlaDataProvider.get_location(self._actor).distance(self._initial_actor_pos)

        if actor_speed < target_speed:
            # set throttle to throttle_value to accelerate
            self._control.throttle = self._throttle_value

        if actor_speed >= target_speed:
            # keep velocity until the actors are in trigger distance
            self._control.throttle = 0

        self._actor.apply_control(self._control)

        # new status:
        if distance <= self._trigger_distance:
            new_status = py_trees.common.Status.SUCCESS

        elif driven_distance > self._max_distance:
            new_status = py_trees.common.Status.FAILURE
        else:
            new_status = py_trees.common.Status.RUNNING

        return new_status


class KeepVelocity(AtomicBehavior):

    """
    This class contains an atomic behavior to keep the provided velocity.
    The controlled traffic participant will accelerate as fast as possible
    until reaching a given _target_velocity_, which is then maintained for
    as long as this behavior is active.

    Important parameters:
    - actor: CARLA actor to execute the behavior
    - target_velocity: The target velocity the actor should reach
    - forced_speed: Whether or not to forcefully set the actors speed. This will ony be active until a collision happens
    - duration[optional]: Duration in seconds of this behavior
    - distance[optional]: Maximum distance in meters covered by the actor during this behavior

    A termination can be enforced by providing distance or duration values.
    Alternatively, a parallel termination behavior has to be used.
    """

    def __init__(self, actor, target_velocity, force_speed=False,
                 duration=float("inf"), distance=float("inf"), name="KeepVelocity"):
        """
        Setup parameters including acceleration value (via throttle_value)
        and target velocity
        """
        super().__init__(name, actor)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._target_velocity = target_velocity

        self._control, self._type = get_actor_control(actor)
        self._world = CarlaDataProvider.get_world()
        self._map = CarlaDataProvider.get_map()
        self._waypoint = self._map.get_waypoint(self._actor.get_location())

        self._forced_speed = force_speed
        self._duration = duration
        self._target_distance = distance
        self._distance = 0
        self._start_time = 0
        self._location = None

    def initialise(self):
        self._location = CarlaDataProvider.get_location(self._actor)
        self._start_time = GameTime.get_time()

        # In case of walkers, we have to extract the current heading
        if self._type == 'walker':
            self._control.speed = self._target_velocity
            self._control.direction = CarlaDataProvider.get_transform(self._actor).get_forward_vector()
        elif self._type == 'vehicle':
            self._control.hand_brake = False
        self._actor.apply_control(self._control)

        super().initialise()

    def update(self):
        """
        As long as the stop condition (duration or distance) is not violated, set a new vehicle control
        For vehicles: set throttle to throttle_value, as long as velocity is < target_velocity
        For walkers: simply apply the given self._control
        """
        new_status = py_trees.common.Status.RUNNING

        if self._type == 'vehicle':
            if not self._forced_speed:
                if CarlaDataProvider.get_velocity(self._actor) < self._target_velocity:
                    self._control.throttle = 1.0
                else:
                    self._control.throttle = 0.0
                self._actor.apply_control(self._control)
            else:
                yaw = CarlaDataProvider.get_transform(self._actor).rotation.yaw * (math.pi / 180)
                self._actor.set_target_velocity(carla.Vector3D(
                    math.cos(yaw) * self._target_velocity, math.sin(yaw) * self._target_velocity, 0))

                # Add a throttle. Useless speed-wise, but makes the bicycle riders pedal.
                self._actor.apply_control(carla.VehicleControl(throttle=1.0))

        new_location = CarlaDataProvider.get_location(self._actor)
        self._distance += calculate_distance(self._location, new_location)
        self._location = new_location

        if self._distance > self._target_distance:
            new_status = py_trees.common.Status.SUCCESS

        if GameTime.get_time() - self._start_time > self._duration:
            new_status = py_trees.common.Status.SUCCESS

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status

    def terminate(self, new_status):
        """
        On termination of this behavior, the throttle should be set back to 0.,
        to avoid further acceleration.
        """
        try:
            if self._type == 'vehicle':
                self._control.throttle = 0.0
            elif self._type == 'walker':
                self._control.speed = 0.0
            if self._actor is not None and self._actor.is_alive:
                self._actor.apply_control(self._control)
        except RuntimeError:
            pass
        super().terminate(new_status)


class ChangeAutoPilot(AtomicBehavior):

    """
    This class contains an atomic behavior to disable/enable the use of the autopilot.

    Important parameters:
    - actor: CARLA actor to execute the behavior
    - activate: True (=enable autopilot) or False (=disable autopilot)
    - lane_change: Traffic Manager parameter. True (=enable lane changes) or False (=disable lane changes)
    - distance_between_vehicles: Traffic Manager parameter
    - max_speed: Traffic Manager parameter. Max speed of the actor. This will only work for road segments
                 with the same speed limit as the first one

    The behavior terminates after changing the autopilot state
    """

    def __init__(self, actor, activate, parameters=None, name="ChangeAutoPilot"):
        """
        Setup parameters
        """
        super().__init__(name, actor)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._activate = activate
        self._tm = CarlaDataProvider.get_client().get_trafficmanager(
            CarlaDataProvider.get_traffic_manager_port())
        self._parameters = parameters

    def update(self):
        """
        De/activate autopilot
        """
        self._actor.set_autopilot(self._activate, CarlaDataProvider.get_traffic_manager_port())

        if self._parameters is not None:
            if "auto_lane_change" in self._parameters:
                lane_change = self._parameters["auto_lane_change"]
                self._tm.auto_lane_change(self._actor, lane_change)

            if "max_speed" in self._parameters:
                max_speed = self._parameters["max_speed"]
                max_road_speed = self._actor.get_speed_limit()
                if max_road_speed is not None:
                    percentage = (max_road_speed - max_speed) / max_road_speed * 100.0
                    self._tm.vehicle_percentage_speed_difference(self._actor, percentage)
                else:
                    print("ChangeAutopilot: Unable to find the vehicle's speed limit")

            if "distance_between_vehicles" in self._parameters:
                dist_vehicles = self._parameters["distance_between_vehicles"]
                self._tm.distance_to_leading_vehicle(self._actor, dist_vehicles)

            if "force_lane_change" in self._parameters:
                force_lane_change = self._parameters["force_lane_change"]
                self._tm.force_lane_change(self._actor, force_lane_change)

            if "ignore_vehicles_percentage" in self._parameters:
                ignore_vehicles = self._parameters["ignore_vehicles_percentage"]
                self._tm.ignore_vehicles_percentage(self._actor, ignore_vehicles)

        new_status = py_trees.common.Status.SUCCESS

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))
        return new_status


class StopVehicle(AtomicBehavior):

    """
    This class contains an atomic stopping behavior. The controlled traffic
    participant will decelerate with _bake_value_ until reaching a full stop.

    Important parameters:
    - actor: CARLA actor to execute the behavior
    - brake_value: Brake value in [0,1] applied

    The behavior terminates when the actor stopped moving
    """

    def __init__(self, actor, brake_value, name="Stopping"):
        """
        Setup _actor and maximum braking value
        """
        super().__init__(name, actor)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._control, self._type = get_actor_control(actor)
        if self._type == 'walker':
            self._control.speed = 0
        self._brake_value = brake_value

    def update(self):
        """
        Set brake to brake_value until reaching full stop
        """
        new_status = py_trees.common.Status.RUNNING

        if self._type == 'vehicle':
            if CarlaDataProvider.get_velocity(self._actor) > EPSILON:
                self._control.brake = self._brake_value
            else:
                new_status = py_trees.common.Status.SUCCESS
                self._control.brake = 0
        else:
            new_status = py_trees.common.Status.SUCCESS

        self._actor.apply_control(self._control)

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status


class SyncArrival(AtomicBehavior):

    """
    This class contains an atomic behavior to
    set velocity of actor so that it reaches location at the same time as
    actor_reference. The behavior assumes that the two actors are moving
    towards location in a straight line.
    Important parameters:
    - actor: CARLA actor to execute the behavior
    - actor_reference: Reference actor with which arrival is synchronized
    - target_location: CARLA location where the actors should "meet"
    - gain[optional]: Coefficient for actor's throttle and break controls
    Note: In parallel to this behavior a termination behavior has to be used
          to keep continue synchronization for a certain duration, or for a
          certain distance, etc.
    """

    def __init__(self, actor, actor_reference, target_location, gain=1, name="SyncArrival"):
        """
        Setup required parameters
        """
        super().__init__(name, actor)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._control = carla.VehicleControl()
        self._actor_reference = actor_reference
        self._target_location = target_location
        self._gain = gain

        self._control.steering = 0

    def update(self):
        """
        Dynamic control update for actor velocity
        """
        new_status = py_trees.common.Status.RUNNING

        distance_reference = calculate_distance(CarlaDataProvider.get_location(self._actor_reference),
                                                self._target_location)
        distance = calculate_distance(CarlaDataProvider.get_location(self._actor),
                                      self._target_location)

        velocity_reference = CarlaDataProvider.get_velocity(self._actor_reference)
        time_reference = float('inf')
        if velocity_reference > 0:
            time_reference = distance_reference / velocity_reference

        velocity_current = CarlaDataProvider.get_velocity(self._actor)
        time_current = float('inf')
        if velocity_current > 0:
            time_current = distance / velocity_current

        control_value = (self._gain) * (time_current - time_reference)

        if control_value > 0:
            self._control.throttle = min([control_value, 1])
            self._control.brake = 0
        else:
            self._control.throttle = 0
            self._control.brake = min([abs(control_value), 1])

        self._actor.apply_control(self._control)
        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))
        return new_status

    def terminate(self, new_status):
        """
        On termination of this behavior, the throttle should be set back to 0.,
        to avoid further acceleration.
        """
        if self._actor is not None and self._actor.is_alive:
            self._control.throttle = 0.0
            self._control.brake = 0.0
            self._actor.apply_control(self._control)
        super().terminate(new_status)


class SyncArrivalWithAgent(AtomicBehavior):

    """
    Atomic to make two actors arrive at their corresponding places at the same time.
    This uses a controller and presuposes that the actor can reach its destination by following the lane.

    The behavior is in RUNNING state until the "main" actor has reached its destination.

    Args:
        actor (carla.Actor): Controlled actor.
        reference_actor (carla.Actor): Reference actor to sync up to.
        actor_target (carla.Transform): Endpoint of the actor after the behavior finishes.
        reference_target (carla.Transform): Endpoint of the reference_actor after the behavior finishes.
        delay (float): Time difference between the actors synchronization.
        end_dist (float): Minimum distance from the target to finish the behavior.
        name (string): Name of the behavior.
            Defaults to 'SyncArrivalWithAgent'.
    """

    def __init__(self, actor, reference_actor, actor_target, reference_target, end_dist=1,
                 name="SyncArrivalWithAgent"):
        """
        Setup required parameters
        """
        super().__init__(name, actor)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))

        self._actor = actor
        self._actor_target = actor_target
        self._reference_actor = reference_actor
        self._reference_target = reference_target
        self._end_dist = end_dist
        self._agent = None

    def initialise(self):
        """Initialises the agent"""
        self._agent = ConstantVelocityAgent(
            self._actor,
            map_inst=CarlaDataProvider.get_map(),
            grp_inst=CarlaDataProvider.get_global_route_planner())

    def update(self):
        """
        Dynamic control update for actor velocity to ensure that both actors reach their target
        positions at the same time.
        """
        new_status = py_trees.common.Status.RUNNING

        # Get the distance of the actor to its endpoint
        distance = calculate_distance(
            CarlaDataProvider.get_location(self._actor), self._actor_target.location)

        # Check if the reference actor has passed its target
        if distance < self._end_dist:
            ref_dir = self._reference_target.get_forward_vector()
            ref_veh = self._reference_target.location - self._reference_actor.get_location()
            if ref_veh.dot(ref_dir) > 0:
                return py_trees.common.Status.SUCCESS

        # Get the time to arrival of the reference to its endpoint
        distance_reference = calculate_distance(
            CarlaDataProvider.get_location(self._reference_actor), self._reference_target.location)

        velocity_reference = CarlaDataProvider.get_velocity(self._reference_actor)
        if velocity_reference > 0:
            time_reference = distance_reference / velocity_reference
        else:
            time_reference = float('inf')

        # Get the required velocity of the actor
        desired_velocity = distance / time_reference

        self._agent.set_target_speed(3.6 * desired_velocity)
        self._actor.apply_control(self._agent.run_step())

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))
        return new_status

    def terminate(self, new_status):
        """Destroy the collision sensor of the agent"""
        if self._agent:
            self._agent.destroy_sensor()
        return super().terminate(new_status)


class CutIn(AtomicBehavior):

    """
    Atomic to make an actor lane change using a Python API agent, cutting in front of another one

    The behavior creates a lane change path and is in RUNNING state until the "main" actor has finsihes it.

    Args:
        actor (carla.Actor): Controlled actor.
        reference_actor (carla.Actor): Reference actor to cut in.
        direction (string): Side from which the cut in happens. Either 'left' or 'right'.
        speed_perc (float): Percentage of the reference actor speed on which the cut in is performed.
        same_lane_time (float): Amount of time spent at the same lane before cutting in.
        other_lane_time (float): Amount of time spent at the other lane after cutting in.
        change_time (float): Amount of time spent changing into the other
        name (string): Name of the behavior.
            Defaults to 'CutIn'.
    """

    def __init__(self, actor, reference_actor, direction, speed_perc=100,
                 same_lane_time=0, other_lane_time=0, change_time=2,
                 name="CutIn"):
        """
        Setup required parameters
        """
        super().__init__(name, actor)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))

        self._reference_actor = reference_actor
        self._direction = direction
        self._speed_perc = speed_perc
        self._same_lane_time = same_lane_time
        self._other_lane_time = other_lane_time
        self._change_time = change_time

        self._map = CarlaDataProvider.get_map()
        self._grp = CarlaDataProvider.get_global_route_planner()
        self._agent = None

    def initialise(self):
        """Initialises the agent"""
        speed = CarlaDataProvider.get_velocity(self._reference_actor)
        self._agent = BasicAgent(
            self._actor,
            3.6 * speed * self._speed_perc / 100,
            map_inst=CarlaDataProvider.get_map(),
            grp_inst=CarlaDataProvider.get_global_route_planner())
        self._agent.lane_change(self._direction, self._same_lane_time, self._other_lane_time, self._change_time)

    def update(self):
        """
        Dynamic control update for actor velocity to ensure that both actors reach their target
        positions at the same time.
        """
        new_status = py_trees.common.Status.RUNNING
        if self._agent.done():
            return py_trees.common.Status.SUCCESS

        self._actor.apply_control(self._agent.run_step())

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))
        return new_status


class AddNoiseToVehicle(AtomicBehavior):

    """
    This class contains an atomic jitter behavior.
    To add noise to steer as well as throttle of the vehicle.

    Important parameters:
    - actor: CARLA actor to execute the behavior
    - steer_value: Applied steering noise in [0,1]
    - throttle_value: Applied throttle noise in [0,1]

    The behavior terminates after setting the new actor controls
    """

    def __init__(self, actor, steer_value, throttle_value, name="Jittering"):
        """
        Setup actor , maximum steer value and throttle value
        """
        super().__init__(name, actor)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._control = carla.VehicleControl()
        self._steer_value = steer_value
        self._throttle_value = throttle_value

    def update(self):
        """
        Set steer to steer_value and throttle to throttle_value until reaching full stop
        """
        self._control = self._actor.get_control()
        self._control.steer = self._steer_value
        self._control.throttle = self._throttle_value
        new_status = py_trees.common.Status.SUCCESS

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))
        self._actor.apply_control(self._control)

        return new_status


class AddNoiseToRouteEgo(AtomicBehavior):

    """
    This class contains an atomic jitter behavior.
    To add noise to steer as well as throttle of the vehicle.

    Important parameters:
    - actor: CARLA actor to execute the behavior
    - steer_value: Applied steering noise in [0,1]
    - throttle_value: Applied throttle noise in [0,1]

    The behavior terminates after setting the new actor controls
    """

    def __init__(self, actor, throttle_mean, throttle_std, steer_mean, steer_std, name="AddNoiseToVehicle"):
        """
        Setup actor , maximum steer value and throttle value
        """
        super().__init__(name, actor)
        self._throttle_mean = throttle_mean
        self._throttle_std = throttle_std
        self._steer_mean = steer_mean
        self._steer_std = steer_std

        self._rng = CarlaDataProvider.get_random_seed()

    def update(self):
        """
        Set steer to steer_value and throttle to throttle_value until reaching full stop
        """
        new_status = py_trees.common.Status.RUNNING

        control = py_trees.blackboard.Blackboard().get("AV_control")
        if not control:
            print("WARNING: Couldn't add noise to the ego because the control couldn't be found")
            return new_status

        throttle_noise = random.normal(self._throttle_mean, self._throttle_std)
        control.throttle = max(-1, min(1, control.throttle + throttle_noise))

        steer_noise = random.normal(self._steer_mean, self._steer_std)
        control.steer = max(0, min(1, control.steer + steer_noise))

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))
        self._actor.apply_control(control)

        return new_status


class ChangeNoiseParameters(AtomicBehavior):

    """
    This class contains an atomic jitter behavior.
    To add noise to steer as well as throttle of the vehicle.

    This behavior should be used in conjuction with AddNoiseToVehicle

    The behavior terminates after one iteration
    """

    def __init__(self, new_steer_noise, new_throttle_noise,
                 noise_mean, noise_std, dynamic_mean_for_steer, dynamic_mean_for_throttle, name="ChangeJittering"):
        """
        Setup actor , maximum steer value and throttle value
        """
        super().__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._new_steer_noise = new_steer_noise
        self._new_throttle_noise = new_throttle_noise
        self._noise_mean = noise_mean
        self._noise_std = noise_std
        self._dynamic_mean_for_steer = dynamic_mean_for_steer
        self._dynamic_mean_for_throttle = dynamic_mean_for_throttle

        self._noise_to_apply = abs(random.normal(self._noise_mean, self._noise_std))

    def update(self):
        """
        Change the noise parameters from the structure copy that it receives.
        """

        self._new_steer_noise[0] = min(0, -(self._noise_to_apply - self._dynamic_mean_for_steer))
        self._new_throttle_noise[0] = min(self._noise_to_apply + self._dynamic_mean_for_throttle, 1)

        new_status = py_trees.common.Status.SUCCESS
        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))
        return new_status


class BasicAgentBehavior(AtomicBehavior):
    """
    This class contains an atomic behavior, which uses the
    basic_agent from CARLA to control the actor until
    reaching a target location.
    Important parameters:
    - actor: CARLA actor to execute the behavior
    - target_location: Is the desired target location (carla.location),
                       the actor should move to
    The behavior terminates after reaching the target_location (within 2 meters)
    """

    def __init__(
        self,
        actor,
        target_location=None,
        plan=None,
        target_speed=20,
        opt_dict=None,
        name="BasicAgentBehavior",
    ):
        """
        Setup actor and maximum steer value
        """
        super().__init__(name, actor)
        self._map = CarlaDataProvider.get_map()
        self._target_location = target_location
        self._target_speed = target_speed
        self._plan = plan

        self._opt_dict = opt_dict if opt_dict else {}
        self._control = carla.VehicleControl()
        self._agent = None

        if self._target_location and self._plan:
            raise ValueError("Choose either a destination or a plan, but not both")

    def initialise(self):
        """Initialises the agent"""
        self._agent = BasicAgent(self._actor, self._target_speed, opt_dict=self._opt_dict,
            map_inst=CarlaDataProvider.get_map(), grp_inst=CarlaDataProvider.get_global_route_planner())
        if self._plan:
            self._agent.set_global_plan(self._plan)
        elif self._target_location:
            init_wp = self._map.get_waypoint(CarlaDataProvider.get_location(self._actor))
            end_wp = self._map.get_waypoint(self._target_location)
            self._plan = self._agent.trace_route(init_wp, end_wp)
            self._agent.set_global_plan(self._plan)

    def update(self):
        new_status = py_trees.common.Status.RUNNING

        if self._agent.done():
            new_status = py_trees.common.Status.SUCCESS
        self._control = self._agent.run_step()
        self._actor.apply_control(self._control)

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))
        return new_status

    def terminate(self, new_status):
        """Resets the control"""
        self._control.throttle = 0.0
        self._control.brake = 0.0
        self._actor.apply_control(self._control)
        super().terminate(new_status)


class ConstantVelocityAgentBehavior(AtomicBehavior):

    """
    This class contains an atomic behavior, which uses the
    constant_velocity_agent from CARLA to control the actor until
    reaching a target location.
    Important parameters:
    - actor: CARLA actor to execute the behavior
    - target_location: Is the desired target location (carla.location),
                       the actor should move to
    - plan: List of [carla.Waypoint, RoadOption] to pass to the controller
    - target_speed: Desired speed of the actor
    The behavior terminates after reaching the target_location (within 2 meters)
    """

    def __init__(self, actor, target_location, target_speed=None,
                 opt_dict=None, name="ConstantVelocityAgentBehavior"):
        """
        Set up actor and local planner
        """
        super().__init__(name, actor)
        self._target_speed = target_speed
        self._map = CarlaDataProvider.get_map()
        self._target_location = target_location
        self._opt_dict = opt_dict if opt_dict else {}
        self._control = carla.VehicleControl()
        self._agent = None
        self._plan = None

        self._map = CarlaDataProvider.get_map()
        self._grp = CarlaDataProvider.get_global_route_planner()

    def initialise(self):
        """Initialises the agent"""
        self._agent = ConstantVelocityAgent(
            self._actor, self._target_speed * 3.6, opt_dict=self._opt_dict,
            map_inst=CarlaDataProvider.get_map(), grp_inst=CarlaDataProvider.get_global_route_planner())
        self._plan = self._agent.trace_route(
            self._map.get_waypoint(CarlaDataProvider.get_location(self._actor)),
            self._map.get_waypoint(self._target_location))
        self._agent.set_global_plan(self._plan)

    def update(self):
        """Moves the actor and waits for it to finish the plan"""
        new_status = py_trees.common.Status.RUNNING

        if self._agent.done():
            new_status = py_trees.common.Status.SUCCESS

        self._control = self._agent.run_step()
        self._actor.apply_control(self._control)

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status

    def terminate(self, new_status):
        """Resets the control"""
        self._control.throttle = 0.0
        self._control.brake = 0.0
        self._actor.apply_control(self._control)
        if self._agent:
            self._agent.destroy_sensor()
        super().terminate(new_status)

class AdaptiveConstantVelocityAgentBehavior(AtomicBehavior):

    """
    This class contains an atomic behavior, which uses the
    constant_velocity_agent from CARLA to control the actor until
    reaching a target location.
    Important parameters:
    - actor: CARLA actor to execute the behavior.
    - reference_actor: Reference CARLA actor to get target speed.
    - speed_increment: Float value (m/s). 
                       How much the actor will be faster then the reference_actor.
    - target_location: Is the desired target location (carla.location),
                       the actor should move to. 
                       If it's None, the actor will follow the lane and never stop.
    - plan: List of [carla.Waypoint, RoadOption] to pass to the controller.
    The behavior terminates after reaching the target_location (within 2 meters)
    """

    def __init__(self, actor, reference_actor, target_location=None, speed_increment=10,
                 opt_dict=None, name="AdaptiveConstantVelocityAgentBehavior"):
        """
        Set up actor and local planner
        """
        super().__init__(name, actor)
        self._speed_increment = speed_increment
        self._reference_actor = reference_actor
        self._target_location = target_location
        self._opt_dict = opt_dict if opt_dict else {}
        self._control = carla.VehicleControl()
        self._agent = None
        self._plan = None

        self._map = CarlaDataProvider.get_map()
        self._grp = CarlaDataProvider.get_global_route_planner()

    def initialise(self):
        """Initialises the agent"""
        # Get target speed
        target_speed = get_speed(self._reference_actor) + self._speed_increment * 3.6

        self._agent = ConstantVelocityAgent(self._actor, target_speed, opt_dict=self._opt_dict,
                                            map_inst=self._map, grp_inst=self._grp)

        if self._target_location is not None:
            self._plan = self._agent.trace_route(
                self._map.get_waypoint(CarlaDataProvider.get_location(self._actor)),
                self._map.get_waypoint(self._target_location))
            self._agent.set_global_plan(self._plan)

    def update(self):
        """Moves the actor and waits for it to finish the plan"""
        new_status = py_trees.common.Status.RUNNING
        target_speed = get_speed(self._reference_actor) + self._speed_increment * 3.6
        self._agent.set_target_speed(target_speed)

        if self._agent.done():
            new_status = py_trees.common.Status.SUCCESS

        self._control = self._agent.run_step()
        self._actor.apply_control(self._control)

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status

    def terminate(self, new_status):
        """Resets the control"""
        self._control.throttle = 0.0
        self._control.brake = 0.0
        self._actor.apply_control(self._control)
        if self._agent:
            self._agent.destroy_sensor()
        super().terminate(new_status)

class Idle(AtomicBehavior):

    """
    This class contains an idle behavior scenario

    Important parameters:
    - duration[optional]: Duration in seconds of this behavior

    A termination can be enforced by providing a duration value.
    Alternatively, a parallel termination behavior has to be used.
    """

    def __init__(self, duration=float("inf"), name="Idle"):
        """
        Setup actor
        """
        super().__init__(name)
        self._duration = duration
        self._start_time = 0
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))

    def initialise(self):
        """
        Set start time
        """
        self._start_time = GameTime.get_time()
        super().initialise()

    def update(self):
        """
        Keep running until termination condition is satisfied
        """
        new_status = py_trees.common.Status.RUNNING

        if GameTime.get_time() - self._start_time > self._duration:
            new_status = py_trees.common.Status.SUCCESS

        return new_status

class WaitForever(AtomicBehavior):

    """
    This class contains a behavior that just waits forever.
    Useful to stop some behavior sequences from stopping unwated parts of the behavior tree

    Alternatively, a parallel termination behavior has to be used to stop it.
    """

    def __init__(self, name="WaitForever"):
        """
        Setup actor
        """
        super().__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))

    def update(self):
        """
        wait forever
        """
        return py_trees.common.Status.RUNNING


class WaypointFollower(AtomicBehavior):

    """
    This is an atomic behavior to follow waypoints while maintaining a given speed.
    If no plan is provided, the actor will follow its foward waypoints indefinetely.
    Otherwise, the behavior will end with SUCCESS upon reaching the end of the plan.
    If no target velocity is provided, the actor continues with its current velocity.

    Args:
        actor (carla.Actor):  CARLA actor to execute the behavior.
        target_speed (float, optional): Desired speed of the actor in m/s. Defaults to None.
        plan ([carla.Location] or [(carla.Waypoint, carla.agent.navigation.local_planner)], optional):
            Waypoint plan the actor should follow. Defaults to None.
        blackboard_queue_name (str, optional):
            Blackboard variable name, if additional actors should be created on-the-fly. Defaults to None.
        avoid_collision (bool, optional):
            Enable/Disable(=default) collision avoidance for vehicles/bikes. Defaults to False.
        name (str, optional): Name of the behavior. Defaults to "FollowWaypoints".

    Attributes:
        actor (carla.Actor):  CARLA actor to execute the behavior.
        name (str, optional): Name of the behavior.
        _target_speed (float, optional): Desired speed of the actor in m/s. Defaults to None.
        _plan ([carla.Location] or [(carla.Waypoint, carla.agent.navigation.local_planner)]):
            Waypoint plan the actor should follow. Defaults to None.
        _blackboard_queue_name (str):
            Blackboard variable name, if additional actors should be created on-the-fly. Defaults to None.
        _avoid_collision (bool): Enable/Disable(=default) collision avoidance for vehicles/bikes. Defaults to False.
        _actor_dict: Dictonary of all actors, and their corresponding plans (e.g. {actor: plan}).
        _local_planner_dict: Dictonary of all actors, and their corresponding local planners.
            Either "Walker" for pedestrians, or a carla.agent.navigation.LocalPlanner for other actors.
        _args_lateral_dict: Parameters for the PID of the used carla.agent.navigation.LocalPlanner.
        _unique_id: Unique ID of the behavior based on timestamp in nanoseconds.

    Note:
        OpenScenario:
        The WaypointFollower atomic must be called with an individual name if multiple consecutive WFs.
        Blackboard variables with lists are used for consecutive WaypointFollower behaviors.
        Termination of active WaypointFollowers in initialise of AtomicBehavior because any
        following behavior must terminate the WaypointFollower.
    """

    def __init__(self, actor, target_speed=None, plan=None, blackboard_queue_name=None,
                 avoid_collision=False, name="FollowWaypoints"):
        """
        Set up actor and local planner
        """
        super().__init__(name, actor)
        self._actor_dict = {}
        self._actor_dict[actor] = None
        self._target_speed = target_speed
        self._local_planner_dict = {}
        self._local_planner_dict[actor] = None
        self._plan = plan
        self._blackboard_queue_name = blackboard_queue_name
        if blackboard_queue_name is not None:
            self._queue = Blackboard().get(blackboard_queue_name)
        self._args_lateral_dict = {'K_P': 1.0, 'K_D': 0.01, 'K_I': 0.0, 'dt': 0.05}
        self._avoid_collision = avoid_collision
        self._start_time = None
        self._unique_id = 0

    def initialise(self):
        """
        Delayed one-time initialization

        Checks if another WaypointFollower behavior is already running for this actor.
        If this is the case, a termination signal is sent to the running behavior.
        """
        super().initialise()
        self._start_time = GameTime.get_time()
        self._unique_id = int(round(time.time() * 1e9))
        try:
            # check whether WF for this actor is already running and add new WF to running_WF list
            check_attr = operator.attrgetter("running_WF_actor_{}".format(self._actor.id))
            running = check_attr(py_trees.blackboard.Blackboard())
            active_wf = copy.copy(running)
            active_wf.append(self._unique_id)
            py_trees.blackboard.Blackboard().set(
                "running_WF_actor_{}".format(self._actor.id), active_wf, overwrite=True)
        except AttributeError:
            # no WF is active for this actor
            py_trees.blackboard.Blackboard().set("terminate_WF_actor_{}".format(self._actor.id), [], overwrite=True)
            py_trees.blackboard.Blackboard().set(
                "running_WF_actor_{}".format(self._actor.id), [self._unique_id], overwrite=True)

        for actor in self._actor_dict:
            self._apply_local_planner(actor)
        return True

    def _apply_local_planner(self, actor):
        """
        Convert the plan into locations for walkers (pedestrians), or to a waypoint list for other actors.
        For non-walkers, activate the carla.agent.navigation.LocalPlanner module.
        """
        if self._target_speed is None:
            self._target_speed = CarlaDataProvider.get_velocity(actor)
        else:
            self._target_speed = self._target_speed

        if isinstance(actor, carla.Walker):
            self._local_planner_dict[actor] = "Walker"
            if self._plan is not None:
                if isinstance(self._plan[0], carla.Location):
                    self._actor_dict[actor] = self._plan
                else:
                    self._actor_dict[actor] = [element[0].transform.location for element in self._plan]
        else:
            local_planner = LocalPlanner(  # pylint: disable=undefined-variable
                actor, opt_dict={
                    'target_speed': self._target_speed * 3.6,
                    'lateral_control_dict': self._args_lateral_dict,
                    'max_throttle': 1.0})

            if self._plan is not None:
                if isinstance(self._plan[0], carla.Location):
                    plan = []
                    for location in self._plan:
                        waypoint = CarlaDataProvider.get_map().get_waypoint(location,
                                                                            project_to_road=True,
                                                                            lane_type=carla.LaneType.Any)
                        plan.append((waypoint, RoadOption.LANEFOLLOW))
                    local_planner.set_global_plan(plan)
                else:
                    local_planner.set_global_plan(self._plan)

            self._local_planner_dict[actor] = local_planner
            self._actor_dict[actor] = self._plan

    def update(self):
        """
        Compute next control step for the given waypoint plan, obtain and apply control to actor
        """
        new_status = py_trees.common.Status.RUNNING

        check_term = operator.attrgetter("terminate_WF_actor_{}".format(self._actor.id))
        terminate_wf = check_term(py_trees.blackboard.Blackboard())

        check_run = operator.attrgetter("running_WF_actor_{}".format(self._actor.id))
        active_wf = check_run(py_trees.blackboard.Blackboard())

        # Termination of WF if the WFs unique_id is listed in terminate_wf
        # only one WF should be active, therefore all previous WF have to be terminated
        if self._unique_id in terminate_wf:
            terminate_wf.remove(self._unique_id)
            if self._unique_id in active_wf:
                active_wf.remove(self._unique_id)

            py_trees.blackboard.Blackboard().set(
                "terminate_WF_actor_{}".format(self._actor.id), terminate_wf, overwrite=True)
            py_trees.blackboard.Blackboard().set(
                "running_WF_actor_{}".format(self._actor.id), active_wf, overwrite=True)
            new_status = py_trees.common.Status.SUCCESS
            return new_status

        if self._blackboard_queue_name is not None:
            while not self._queue.empty():
                actor = self._queue.get()
                if actor is not None and actor not in self._actor_dict:
                    self._apply_local_planner(actor)

        success = True
        for actor in self._local_planner_dict:
            local_planner = self._local_planner_dict[actor] if actor else None
            if actor is not None and actor.is_alive and local_planner is not None:
                # Check if the actor is a vehicle/bike
                if not isinstance(actor, carla.Walker):
                    control = local_planner.run_step(debug=False)
                    if self._avoid_collision and detect_lane_obstacle(actor):
                        control.throttle = 0.0
                        control.brake = 1.0
                    actor.apply_control(control)
                    # Check if the actor reached the end of the plan
                    # @TODO replace access to private _waypoints_queue with public getter
                    if local_planner._waypoints_queue:  # pylint: disable=protected-access
                        success = False
                # If the actor is a pedestrian, we have to use the WalkerAIController
                # The walker is sent to the next waypoint in its plan
                else:
                    actor_location = CarlaDataProvider.get_location(actor)
                    success = False
                    if self._actor_dict[actor]:
                        location = self._actor_dict[actor][0]
                        direction = location - actor_location
                        direction_norm = math.sqrt(direction.x**2 + direction.y**2)
                        control = actor.get_control()
                        control.speed = self._target_speed
                        control.direction = direction / direction_norm
                        actor.apply_control(control)
                        if direction_norm < 1.0:
                            self._actor_dict[actor] = self._actor_dict[actor][1:]
                            if self._actor_dict[actor] is None:
                                success = True
                    else:
                        control = actor.get_control()
                        control.speed = self._target_speed
                        control.direction = CarlaDataProvider.get_transform(actor).rotation.get_forward_vector()
                        actor.apply_control(control)

        if success:
            new_status = py_trees.common.Status.SUCCESS

        return new_status

    def terminate(self, new_status):
        """
        On termination of this behavior,
        the controls should be set back to 0.
        """
        for actor in self._local_planner_dict:
            if actor is not None and actor.is_alive:
                control, _ = get_actor_control(actor)
                actor.apply_control(control)
                local_planner = self._local_planner_dict[actor]
                if local_planner is not None and local_planner != "Walker":
                    local_planner.reset_vehicle()
                    local_planner = None

        self._local_planner_dict = {}
        self._actor_dict = {}
        super().terminate(new_status)


class LaneChange(WaypointFollower):

    """
    This class inherits from the class WaypointFollower.

    This class contains an atomic lane change behavior to a parallel lane.
    The vehicle follows a waypoint plan to the other lane, which is calculated in the initialise method.
    This waypoint plan is calculated with a scenario helper function.

    If an impossible lane change is asked for (due to the lack of lateral lanes,
    next waypoints, continuous line, etc) the atomic will return a plan with the
    waypoints until such impossibility is found.

    Important parameters:
    - actor: CARLA actor to execute the behavior
    - speed: speed of the actor for the lane change, in m/s
    - direction: 'right' or 'left', depending on which lane to change
    - distance_same_lane: straight distance before lane change, in m
    - distance_other_lane: straight distance after lane change, in m
    - distance_lane_change: straight distance for the lane change itself, in m

    The total distance driven is greater than the sum of distance_same_lane and distance_other_lane.
    It results from the lane change distance plus the distance_same_lane plus distance_other_lane.
    The lane change distance is set to 25m (straight), the driven distance is slightly greater.

    A parallel termination behavior has to be used.
    """

    def __init__(self, actor, speed=10, direction='left', distance_same_lane=5, distance_other_lane=100,
                 distance_lane_change=25, lane_changes=1, name='LaneChange'):

        self._direction = direction
        self._distance_same_lane = distance_same_lane
        self._distance_other_lane = distance_other_lane
        self._distance_lane_change = distance_lane_change
        self._lane_changes = lane_changes

        self._target_lane_id = None
        self._distance_new_lane = 0
        self._pos_before_lane_change = None
        self._plan = None

        super().__init__(actor, target_speed=speed, name=name)

    def initialise(self):

        # get start position
        position_actor = CarlaDataProvider.get_map().get_waypoint(self._actor.get_location())

        # calculate plan with scenario_helper function
        self._plan, self._target_lane_id = generate_target_waypoint_list_multilane(
            position_actor, self._direction, self._distance_same_lane,
            self._distance_other_lane, self._distance_lane_change, check=True, lane_changes=self._lane_changes)
        super().initialise()

    def update(self):

        if not self._plan:
            print("{} couldn't perform the expected lane change".format(self._actor))
            return py_trees.common.Status.FAILURE

        status = super().update()

        current_position_actor = CarlaDataProvider.get_map().get_waypoint(self._actor.get_location())
        current_lane_id = current_position_actor.lane_id

        if current_lane_id == self._target_lane_id:
            # driving on new lane
            distance = current_position_actor.transform.location.distance(self._pos_before_lane_change)

            if distance > self._distance_other_lane:
                # long enough distance on new lane --> SUCCESS
                status = py_trees.common.Status.SUCCESS
        else:
            self._pos_before_lane_change = current_position_actor.transform.location

        return status


class SetInitSpeed(AtomicBehavior):

    """
    This class contains an atomic behavior to set the init_speed of an actor,
    succeding immeditely after initializing
    """

    def __init__(self, actor, init_speed=10, name='SetInitSpeed'):

        self._init_speed = init_speed
        self._terminate = None
        self._actor = actor

        super().__init__(name, actor)

    def initialise(self):
        """
        Initialize it's speed
        """

        transform = self._actor.get_transform()
        yaw = transform.rotation.yaw * (math.pi / 180)

        vx = math.cos(yaw) * self._init_speed
        vy = math.sin(yaw) * self._init_speed
        self._actor.set_target_velocity(carla.Vector3D(vx, vy, 0))

    def update(self):
        """
        Nothing to update, end the behavior
        """

        return py_trees.common.Status.SUCCESS


class HandBrakeVehicle(AtomicBehavior):

    """
    This class contains an atomic hand brake behavior.
    To set the hand brake value of the vehicle.

    Important parameters:
    - vehicle: CARLA actor to execute the behavior
    - hand_brake_value to be applied in [0,1]

    The behavior terminates after setting the hand brake value
    """

    def __init__(self, vehicle, hand_brake_value, name="Braking"):
        """
        Setup vehicle control and brake value
        """
        super().__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._vehicle = vehicle
        self._control, self._type = get_actor_control(vehicle)
        self._hand_brake_value = hand_brake_value

    def update(self):
        """
        Set handbrake
        """
        new_status = py_trees.common.Status.SUCCESS
        if self._type == 'vehicle':
            self._control.hand_brake = self._hand_brake_value
            self._vehicle.apply_control(self._control)
        else:
            self._hand_brake_value = None
            self.logger.debug("%s.update()[%s->%s]" %
                              (self.__class__.__name__, self.status, new_status))
            self._vehicle.apply_control(self._control)

        return new_status


class ActorDestroy(AtomicBehavior):

    """
    This class contains an actor destroy behavior.
    Given an actor this behavior will delete it.

    Important parameters:
    - actor: CARLA actor to be deleted

    The behavior terminates after removing the actor
    """

    def __init__(self, actor, actor_name=None, name="ActorDestroy"):
        """
        Setup actor
        """
        super().__init__(name, actor)
        self._actor_name = actor_name or self._actor_name
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))

    def update(self):
        new_status = py_trees.common.Status.RUNNING
        self._resolve_actor()
        if self._actor:
            CarlaDataProvider.remove_actor_by_id(self._actor.id)
            self._actor = None
            new_status = py_trees.common.Status.SUCCESS

        return new_status

class ActorTransformSetter(AtomicBehavior):

    """
    This class contains an atomic behavior to set the transform
    of an actor.

    Important parameters:
    - actor: CARLA actor to execute the behavior
    - transform: New target transform (position + orientation) of the actor
    - physics [optional]: Change the physics of the actors to true / false. To not change the physics, use None.

    The behavior terminates when actor is set to the new actor transform (closer than 1 meter)

    NOTE:
    It is very important to ensure that the actor location is spawned to the new transform because of the
    appearence of a rare runtime processing error. WaypointFollower with LocalPlanner,
    might fail if new_status is set to success before the actor is really positioned at the new transform.
    Therefore: calculate_distance(actor, transform) < 1 meter
    """

    def __init__(self, actor, transform, physics=True, name="ActorTransformSetter"):
        """
        Init
        """
        super().__init__(name, actor)
        self._transform = transform
        self._physics = physics
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))

    def initialise(self):
        if self._actor.is_alive:
            self._actor.set_target_velocity(carla.Vector3D(0, 0, 0))
            self._actor.set_target_angular_velocity(carla.Vector3D(0, 0, 0))
            self._actor.set_transform(self._transform)
        super().initialise()

    def update(self):
        """
        Transform actor
        """
        new_status = py_trees.common.Status.RUNNING

        if not self._actor.is_alive:
            new_status = py_trees.common.Status.FAILURE

        if calculate_distance(self._actor.get_location(), self._transform.location) < 1.0:
            if self._physics is not None:
                self._actor.set_simulate_physics(self._physics)
            new_status = py_trees.common.Status.SUCCESS

        return new_status


class BatchActorTransformSetter(AtomicBehavior):

    """
    This class contains an atomic behavior to set the transform
    of an actor.

    Important parameters:
    - actor_transform_list: list [carla.Actor, carla.Transform]
    - physics [optional]: Change the physics of the actors to true / false. To not change the physics, use None.

    The behavior terminates immediately
    """

    def __init__(self, actor_transform_list, physics=True, name="BatchActorTransformSetter"):
        """
        Init
        """
        super().__init__(name)
        self._actor_transform_list = actor_transform_list
        self._physics = physics
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))

    def update(self):
        """
        Transform actor
        """

        for actor, transform in self._actor_transform_list:
            actor.set_target_velocity(carla.Vector3D(0, 0, 0))
            actor.set_target_angular_velocity(carla.Vector3D(0, 0, 0))
            actor.set_transform(transform)
            if self._physics is not None:
                actor.set_simulate_physics(self._physics)

        return py_trees.common.Status.SUCCESS


class TrafficLightStateSetter(AtomicBehavior):

    """
    This class contains an atomic behavior to set the state of a given traffic light

    Args:
        actor (carla.TrafficLight): ID of the traffic light that shall be changed
        state (carla.TrafficLightState): New target state

    The behavior terminates after trying to set the new state
    """

    def __init__(self, actor, state, name="TrafficLightStateSetter"):
        """
        Init
        """
        super().__init__(name)

        self._actor = actor if "traffic_light" in actor.type_id else None
        self._state = state
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))

    def update(self):
        """
        Change the state of the traffic light
        """
        if self._actor is None:
            return py_trees.common.Status.FAILURE

        new_status = py_trees.common.Status.RUNNING
        if self._actor.is_alive:
            self._actor.set_state(self._state)
            new_status = py_trees.common.Status.SUCCESS
        else:
            # For some reason the actor is gone...
            new_status = py_trees.common.Status.FAILURE

        return new_status


class TrafficLightControllerSetter(AtomicBehavior):
    """
    This class contains an atomic behavior to set the phase of a given traffic light controller

    Args:
        actor (carla.TrafficLight): ID of the traffic light controller that shall be changed
        state (carla.TrafficLightState): New target state

    """

    def __init__(self, traffic_signal_id, state, duration, delay=None, ref_id=None,
                 name="TrafficLightControllerSetter"):
        """
        Init
        """
        super().__init__(name)
        self.actor_id = traffic_signal_id
        self._actor = None
        self._start_time = None
        self.duration_time = None
        self.timeout = float(duration)
        self.delay = float(delay) if delay else None
        self.ref_tl_id = ref_id
        self._state = state
        self._previous_traffic_light_info = {}
        self.logger.debug("%s.__init__()" % self.__class__.__name__)

    def initialise(self):
        self._start_time = GameTime.get_time()
        self._actor = CarlaDataProvider.get_world().get_traffic_light_from_opendrive_id(self.actor_id)
        if self._actor is None:
            return py_trees.common.Status.FAILURE

        if self.ref_tl_id is not None and self.delay is not None:
            elapsed_time = self._actor.get_elapsed_time()
            self.duration_time = self.delay + self.timeout + elapsed_time
        elif self.ref_tl_id is None and self.delay is None:
            self.duration_time = self.timeout
        else:
            return py_trees.common.Status.FAILURE
        self._previous_traffic_light_info[self._actor] = {
            'state': self._actor.get_state(),
            'green_time': self._actor.get_green_time(),
            'red_time': self._actor.get_red_time(),
            'yellow_time': self._actor.get_yellow_time()
        }
        self._actor.set_state(self._state)
        self._actor.set_green_time(self.duration_time)
        return None

    def update(self):
        """Waits until the adequate time has passed"""

        new_status = py_trees.common.Status.RUNNING

        if self._actor.is_alive:
            if GameTime.get_time() - self._start_time > self.duration_time:
                new_status = py_trees.common.Status.SUCCESS
                return new_status
            else:
                return new_status
        else:
            # For some reason the actor is gone...
            return py_trees.common.Status.FAILURE

    def terminate(self, new_status):
        """Reset all traffic lights back to their previous states"""
        if self._previous_traffic_light_info:
            self._actor.set_state(self._previous_traffic_light_info[self._actor]['state'])
            self._actor.set_green_time(self._previous_traffic_light_info[self._actor]['green_time'])
            self._actor.set_red_time(self._previous_traffic_light_info[self._actor]['red_time'])
            self._actor.set_yellow_time(self._previous_traffic_light_info[self._actor]['yellow_time'])

        super().terminate(new_status)


class ActorSource(AtomicBehavior):

    """
    Implementation for a behavior that will indefinitely create actors
    at a given transform if no other actor exists in a given radius
    from the transform.

    Important parameters:
    - actor_type_list: Type of CARLA actors to be spawned
    - transform: Spawn location
    - threshold: Min available free distance between other actors and the spawn location
    - blackboard_queue_name: Name of the blackboard used to control this behavior
    - actor_limit [optional]: Maximum number of actors to be spawned (default=7)

    A parallel termination behavior has to be used.
    """

    def __init__(self, actor_type_list, transform, threshold, blackboard_queue_name,
                 actor_limit=7, name="ActorSource"):
        """
        Setup class members
        """
        super().__init__(name)
        self._world = CarlaDataProvider.get_world()
        self._actor_types = actor_type_list
        self._spawn_point = transform
        self._threshold = threshold
        self._queue = Blackboard().get(blackboard_queue_name)
        self._actor_limit = actor_limit
        self._last_blocking_actor = None

    def update(self):
        new_status = py_trees.common.Status.RUNNING
        if self._actor_limit > 0:
            world_actors = CarlaDataProvider.get_all_actors()
            spawn_point_blocked = False
            if (self._last_blocking_actor and
                    self._spawn_point.location.distance(self._last_blocking_actor.get_location()) < self._threshold):
                spawn_point_blocked = True

            if not spawn_point_blocked:
                for actor in world_actors:  # pylint: disable=not-an-iterable
                    if self._spawn_point.location.distance(actor.get_location()) < self._threshold:
                        spawn_point_blocked = True
                        self._last_blocking_actor = actor
                        break

            if not spawn_point_blocked:
                try:
                    new_actor = CarlaDataProvider.request_new_actor(
                        random.choice(self._actor_types), self._spawn_point)
                    self._actor_limit -= 1
                    self._queue.put(new_actor)
                except:                             # pylint: disable=bare-except
                    print("ActorSource unable to spawn actor")
        return new_status


class ActorSink(AtomicBehavior):

    """
    Implementation for a behavior that will indefinitely destroy actors
    that wander near a given location within a specified threshold.

    Important parameters:
    - actor_type_list: Type of CARLA actors to be spawned
    - sink_location: Location (carla.location) at which actors will be deleted
    - threshold: Distance around sink_location in which actors will be deleted

    A parallel termination behavior has to be used.
    """

    def __init__(self, sink_location, threshold, name="ActorSink"):
        """
        Setup class members
        """
        super().__init__(name)
        self._sink_location = sink_location
        self._threshold = threshold

    def update(self):
        new_status = py_trees.common.Status.RUNNING
        CarlaDataProvider.remove_actors_in_surrounding(self._sink_location, self._threshold)
        return new_status


class ActorFlow(AtomicBehavior):
    """
    Behavior that indefinitely creates actors at a location,
    controls them until another location, and then destroys them.
    Therefore, a parallel termination behavior has to be used.

    Important parameters:
    - source_transform (carla.Transform): Transform at which actors will be spawned
    - sink_location (carla.Location): Location at which actors will be deleted
    - spawn_distance: Distance between spawned actors
    - sink_distance: Actors closer to the sink than this distance will be deleted
    - actors_speed: Speed of the actors part of the flow [m/s]
    - initial_actors: Populates all the flow trajectory at the start
    """

    def __init__(self, source_wp, sink_wp, spawn_dist_interval, sink_dist=2,
                 actor_speed=20 / 3.6, initial_actors=False, initial_junction=False, name="ActorFlow"):
        """
        Setup class members
        """
        super().__init__(name)
        self._rng = CarlaDataProvider.get_random_seed()
        self._world = CarlaDataProvider.get_world()
        self._tm = CarlaDataProvider.get_client().get_trafficmanager(CarlaDataProvider.get_traffic_manager_port())

        self._collision_bp = self._world.get_blueprint_library().find('sensor.other.collision')
        self._is_constant_velocity_active = True

        self._source_wp = source_wp
        self._sink_wp = sink_wp

        self._sink_location = self._sink_wp.transform.location
        self._source_transform = self._source_wp.transform
        self._source_location = self._source_transform.location

        self._sink_dist = sink_dist
        self._speed = actor_speed
        self._initial_actors = initial_actors
        self._initial_junction = initial_junction

        self._min_spawn_dist = spawn_dist_interval[0]
        self._max_spawn_dist = spawn_dist_interval[1]
        self._spawn_dist = self._rng.uniform(self._min_spawn_dist, self._max_spawn_dist)

        self._attribute_filter = {'base_type': 'car', 'has_lights': True, 'special_type': ''}

        self._actor_list = []
        self._collision_sensor_list = []

        self._terminated = False

    def initialise(self):
        if self._initial_actors:
            grp = CarlaDataProvider.get_global_route_planner()
            plan = grp.trace_route(self._source_location, self._sink_location)

            ref_loc = plan[0][0].transform.location
            for wp, _ in plan:
                if wp.is_junction and not self._initial_junction:
                    continue  # Spawning at junctions might break the path, so don't
                if wp.transform.location.distance(ref_loc) < self._spawn_dist:
                    continue
                self._spawn_actor(wp.transform)
                ref_loc = wp.transform.location
                self._spawn_dist = self._rng.uniform(self._min_spawn_dist, self._max_spawn_dist)

    def _spawn_actor(self, transform):
        actor = CarlaDataProvider.request_new_actor(
            'vehicle.*', transform, rolename='scenario',
            attribute_filter=self._attribute_filter, tick=False
        )
        if actor is None:
            return py_trees.common.Status.RUNNING

        actor.set_autopilot(True, CarlaDataProvider.get_traffic_manager_port())
        self._tm.set_path(actor, [self._sink_location])
        self._tm.auto_lane_change(actor, False)
        self._tm.set_desired_speed(actor, 3.6 * self._speed)
        self._tm.update_vehicle_lights(actor, True)

        self._spawn_dist = self._rng.uniform(self._min_spawn_dist, self._max_spawn_dist)

        sensor = None
        if self._is_constant_velocity_active:
            self._tm.ignore_vehicles_percentage(actor, 100)
            actor.enable_constant_velocity(carla.Vector3D(self._speed, 0, 0))  # For when physics are active

            sensor = self._world.spawn_actor(self._collision_bp, carla.Transform(), attach_to=actor)
            sensor.listen(lambda _: self.stop_constant_velocity())

        self._tm.ignore_lights_percentage(actor, 100)
        self._tm.ignore_signs_percentage(actor, 100)
        self._collision_sensor_list.append(sensor)
        self._actor_list.append(actor)
        return None

    def update(self):
        """Controls the created actors and creaes / removes other when needed"""
        # Control the vehicles, removing them when needed
        for actor, sensor in zip(list(self._actor_list), list(self._collision_sensor_list)):
            location = CarlaDataProvider.get_location(actor)
            if not location:
                continue
            sink_distance = self._sink_location.distance(location)
            if sink_distance < self._sink_dist:
                if sensor is not None:
                    sensor.stop()
                    sensor.destroy()
                self._collision_sensor_list.remove(sensor)
                CarlaDataProvider.remove_actor_by_id(actor.id)
                self._actor_list.remove(actor)

        # Spawn new actors if needed
        if len(self._actor_list) == 0:
            distance = self._spawn_dist + 1
        else:
            actor_location = CarlaDataProvider.get_location(self._actor_list[-1])
            distance = self._source_location.distance(actor_location) if actor_location else 0

        if distance > self._spawn_dist:
            self._spawn_actor(self._source_transform)

        return py_trees.common.Status.RUNNING

    def stop_constant_velocity(self):
        """Stops the constant velocity behavior"""
        self._is_constant_velocity_active = False
        for actor in self._actor_list:
            actor.disable_constant_velocity()
            self._tm.ignore_vehicles_percentage(actor, 0)

    def terminate(self, new_status):
        """
        Default terminate. Can be extended in derived class
        """
        if self._terminated:
            return

        self._terminated = True

        for sensor in self._collision_sensor_list:
            if sensor is None:
                continue
            try:
                sensor.stop()
                sensor.destroy()
            except RuntimeError:
                pass  # Actor was already destroyed

        for actor in self._actor_list:
            # TODO: Actors spawned in the same frame as the behavior termination won't be removed.
            # Patched by removing its movement
            actor.disable_constant_velocity()
            actor.set_autopilot(False, CarlaDataProvider.get_traffic_manager_port())
            actor.set_target_velocity(carla.Vector3D(0,0,0))
            actor.set_target_angular_velocity(carla.Vector3D(0,0,0))
            try:
                CarlaDataProvider.remove_actor_by_id(actor.id)
            except RuntimeError:
                pass  # Actor was already destroyed


class OppositeActorFlow(AtomicBehavior):
    """
    Similar to ActorFlow, but this is meant as an actor flow in the opposite direction.
    As such, some configurations are different and for clarity, another behavior has been created

    Important parameters:
    - source_wp (carla.Waypoint): Waypoint at which actors will be spawned
    - sink_wp (carla.Waypoint): Waypoint at which actors will be deleted
    - spawn_dist_interval: Distance interval between spawned actors
    - sink_dist: Actors closer to the sink than this distance will be deleted
    - actors_speed: Speed of the actors part of the flow [m/s]
    - offset: offset from the center lane of the actors
    """

    def __init__(self, reference_wp, reference_actor, spawn_dist_interval,
                 time_distance=1.5, base_distance=30, sink_dist=2, name="OppositeActorFlow"):
        """
        Setup class members
        """
        super().__init__(name)
        self._rng = CarlaDataProvider.get_random_seed()
        self._world = CarlaDataProvider.get_world()
        self._tm = CarlaDataProvider.get_client().get_trafficmanager(CarlaDataProvider.get_traffic_manager_port())

        self._reference_wp = reference_wp
        self._reference_actor = reference_actor
        self._time_distance = time_distance
        self._base_distance = base_distance
        self._min_spawn_dist = spawn_dist_interval[0]
        self._max_spawn_dist = spawn_dist_interval[1]
        self._spawn_dist = self._rng.uniform(self._min_spawn_dist, self._max_spawn_dist)

        self._sink_dist = sink_dist

        self._attribute_filter = {'base_type': 'car', 'has_lights': True, 'special_type': ''}

        # Opposite direction needs earlier vehicle detection
        self._opt_dict = {'base_vehicle_threshold': 10, 'detection_speed_ratio': 1.6}

        self._actor_list = []
        self._grp = CarlaDataProvider.get_global_route_planner()
        self._map = CarlaDataProvider.get_map()

        self._terminated = False

        # initialise attributes
        self._speed = None # Km / h
        self._flow_distance = -1.0

        self._sink_wp = None
        self._source_wp = None

        self._source_transform = None
        self._source_location = None
        self._sink_location = None

        self._route = None

    def _move_waypoint_forward(self, wp, distance):
        """Moves forward a certain distance, stopping at junctions"""
        dist = 0
        next_wp = wp
        while dist < distance:
            next_wps = next_wp.next(1)
            if next_wps[0].is_junction:
                break
            next_wp = next_wps[0]
            dist += 1

        return next_wp

    def _move_waypoint_backwards(self, wp, distance):
        """Moves backwards a certain distance, stopping at junctions"""
        dist = 0
        prev_wp = wp
        while dist < distance:
            prev_wps = prev_wp.previous(1)
            if prev_wps[0].is_junction:
                break
            prev_wp = prev_wps[0]
            dist += 1

        return prev_wp

    def initialise(self):
        """Get the actor flow source and sink, depending on the reference actor speed"""
        self._speed = self._reference_actor.get_speed_limit() # Km / h
        self._flow_distance = self._time_distance * self._speed + self._base_distance

        self._sink_wp = self._move_waypoint_forward(self._reference_wp, self._flow_distance)
        self._source_wp = self._move_waypoint_backwards(self._reference_wp, self._flow_distance)

        self._source_transform = self._source_wp.transform
        self._source_location = self._source_transform.location
        self._sink_location = self._sink_wp.transform.location

        self._route = self._grp.trace_route(self._source_location, self._sink_location)

        return super().initialise()

    def _spawn_actor(self):
        actor = CarlaDataProvider.request_new_actor(
            'vehicle.*', self._source_transform, rolename='scenario',
            attribute_filter=self._attribute_filter, tick=False
        )
        if actor is None:
            return py_trees.common.Status.RUNNING

        controller = BasicAgent(actor, self._speed, self._opt_dict, self._map, self._grp)
        controller.set_global_plan(self._route)
        self._actor_list.append([actor, controller])

        self._spawn_dist = self._rng.uniform(self._min_spawn_dist, self._max_spawn_dist)
        return None

    def update(self):
        """Controls the created actors and creates / removes other when needed"""
        # Control the vehicles, removing them when needed
        for actor_data in list(self._actor_list):
            actor, controller = actor_data
            location = CarlaDataProvider.get_location(actor)
            if not location:
                continue
            sink_distance = self._sink_location.distance(location)
            if sink_distance < self._sink_dist:
                CarlaDataProvider.remove_actor_by_id(actor.id)
                self._actor_list.remove(actor_data)
            else:
                actor.apply_control(controller.run_step())

        # Spawn new actors if needed
        if len(self._actor_list) == 0:
            distance = self._spawn_dist + 1
        else:
            actor_location = CarlaDataProvider.get_location(self._actor_list[-1][0])
            distance = self._source_location.distance(actor_location) if actor_location else 0
        if distance > self._spawn_dist:
            self._spawn_actor()

        return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        """
        Default terminate. Can be extended in derived class
        """
        if self._terminated:
            return

        self._terminated = True

        for actor, _ in self._actor_list:
            # TODO: Actors spawned in the same frame as the behavior termination won't be removed.
            # Patched by removing its movement
            actor.disable_constant_velocity()
            actor.set_autopilot(False, CarlaDataProvider.get_traffic_manager_port())
            actor.set_target_velocity(carla.Vector3D(0,0,0))
            actor.set_target_angular_velocity(carla.Vector3D(0,0,0))
            try:
                CarlaDataProvider.remove_actor_by_id(actor.id)
            except RuntimeError:
                pass  # Actor was already destroyed


class InvadingActorFlow(AtomicBehavior):
    """
    Similar to ActorFlow, but this is meant as an actor flow in the opposite direction that invades the lane.
    As such, some configurations are different and for clarity, another behavior has been created

    Important parameters:
    - source_wp (carla.Waypoint): Waypoint at which actors will be spawned
    - sink_wp (carla.Waypoint): Waypoint at which actors will be deleted
    - spawn_dist_interval: Distance interval between spawned actors
    - sink_dist: Actors closer to the sink than this distance will be deleted
    - actors_speed: Speed of the actors part of the flow [m/s]
    - offset: offset from the center lane of the actors
    """

    def __init__(self, source_wp, sink_wp, reference_actor, spawn_dist,
                 sink_dist=2, offset=0, name="OppositeActorFlow"):
        """
        Setup class members
        """
        super().__init__(name)
        self._world = CarlaDataProvider.get_world()
        self._tm = CarlaDataProvider.get_client().get_trafficmanager(CarlaDataProvider.get_traffic_manager_port())

        self._reference_actor = reference_actor

        self._source_wp  = source_wp
        self._source_transform = self._source_wp.transform
        self._source_location = self._source_transform.location

        self._sink_wp = sink_wp
        self._sink_location = self._sink_wp.transform.location

        self._spawn_dist = spawn_dist

        self._sink_dist = sink_dist

        self._attribute_filter = {'base_type': 'car', 'has_lights': True, 'special_type': ''}

        self._actor_list = []

        # Opposite direction needs earlier vehicle detection
        self._opt_dict = {'base_vehicle_threshold': 10, 'detection_speed_ratio': 2, 'distance_ratio': 0.2}
        self._opt_dict['offset'] = offset

        self._grp = CarlaDataProvider.get_global_route_planner()
        self._map = CarlaDataProvider.get_map()

        self._terminated = False
        # Initialise attributes
        self._speed = None
        self._route = None

    def initialise(self):
        """Get the actor flow source and sink, depending on the reference actor speed"""
        self._speed = self._reference_actor.get_speed_limit()  # Km / h
        self._route = self._grp.trace_route(self._source_location, self._sink_location)
        return super().initialise()

    def _spawn_actor(self):
        actor = CarlaDataProvider.request_new_actor(
            'vehicle.*', self._source_transform, rolename='scenario',
            attribute_filter=self._attribute_filter, tick=False
        )
        if actor is None:
            return py_trees.common.Status.RUNNING

        controller = BasicAgent(actor, self._speed, self._opt_dict, self._map, self._grp)
        controller.set_global_plan(self._route)
        self._actor_list.append([actor, controller])
        return None

    def update(self):
        """Controls the created actors and creates / removes other when needed"""
        # Control the vehicles, removing them when needed
        for actor_data in list(self._actor_list):
            actor, controller = actor_data
            location = CarlaDataProvider.get_location(actor)
            if not location:
                continue
            sink_distance = self._sink_location.distance(location)
            if sink_distance < self._sink_dist:
                CarlaDataProvider.remove_actor_by_id(actor.id)
                self._actor_list.remove(actor_data)
            else:
                actor.apply_control(controller.run_step())

        # Spawn new actors if needed
        if len(self._actor_list) == 0:
            distance = self._spawn_dist + 1
        else:
            actor_location = CarlaDataProvider.get_location(self._actor_list[-1][0])
            distance = self._source_location.distance(actor_location) if actor_location else 0
        if distance > self._spawn_dist:
            self._spawn_actor()

        return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        """
        Default terminate. Can be extended in derived class
        """
        if self._terminated:
            return

        self._terminated = True

        for actor, _ in self._actor_list:
            # TODO: Actors spawned in the same frame as the behavior termination won't be removed.
            # Patched by removing its movement
            actor.disable_constant_velocity()
            actor.set_autopilot(False, CarlaDataProvider.get_traffic_manager_port())
            actor.set_target_velocity(carla.Vector3D(0,0,0))
            actor.set_target_angular_velocity(carla.Vector3D(0,0,0))
            try:
                CarlaDataProvider.remove_actor_by_id(actor.id)
            except RuntimeError:
                pass  # Actor was already destroyed


class BicycleFlow(AtomicBehavior):
    """
    Behavior that indefinitely creates bicycles at a location,
    controls them until another location, and then destroys them.
    Therefore, a parallel termination behavior has to be used.

    Important parameters:
    - plan (list(carla.Waypoint)): plan used by the bicycles.
    - spawn_distance_interval (list(float, float)): Distance between spawned actors
    - sink_distance (float): Actors at this distance from the sink will be deleted
    - actors_speed (float): Speed of the actors part of the flow [m/s]
    - initial_actors (bool): Boolean to initialy populate all the flow with bicycles
    """

    def __init__(self, plan, spawn_dist_interval, sink_dist=2,
                 actor_speed=20 / 3.6, initial_actors=False, name="BicycleFlow"):
        """
        Setup class members
        """
        super().__init__(name)
        self._rng = CarlaDataProvider.get_random_seed()

        self._plan = plan
        self._sink_dist = sink_dist
        self._speed = actor_speed

        self._source_transform = self._plan[0][0].transform
        self._source_location = self._source_transform.location
        self._sink_location = self._plan[-1][0].transform.location

        self._min_spawn_dist = spawn_dist_interval[0]
        self._max_spawn_dist = spawn_dist_interval[1]
        self._spawn_dist = self._rng.uniform(self._min_spawn_dist, self._max_spawn_dist)

        self._initial_actors = initial_actors

        self._opt_dict = {"ignore_traffic_lights": True, "ignore_vehicles": True}

        self._actor_data = []
        self._grp = CarlaDataProvider.get_global_route_planner()

        self._terminated = False

    def initialise(self):
        if self._initial_actors:
            ref_loc = self._plan[0][0].transform.location
            for wp, _ in self._plan:
                if wp.is_junction:
                    continue  # Spawning at junctions might break the path, so don't
                if wp.transform.location.distance(ref_loc) < self._spawn_dist:
                    continue
                self._spawn_actor(wp.transform)
                ref_loc = wp.transform.location
                self._spawn_dist = self._rng.uniform(self._min_spawn_dist, self._max_spawn_dist)

    def _spawn_actor(self, transform):
        """Spawn the actor"""
        # Initial actors don't want all the plan. Remove the points behind them
        plan = self._plan
        actor_loc = transform.location
        while len(plan) > 0:
            wp, _ = plan[0]
            loc = wp.transform.location
            actor_heading = transform.get_forward_vector()
            actor_wp_vec = loc - actor_loc
            if actor_heading.dot(actor_wp_vec) < 0 or loc.distance(actor_loc) < 10:
                plan.pop(0)
            else:
                break

        if not plan:
            return

        actor = CarlaDataProvider.request_new_actor(
            'vehicle.*', transform, rolename='scenario no lights',
            attribute_filter={'base_type': 'bicycle'}, tick=False
        )
        if actor is None:
            return

        controller = BasicAgent(actor, 3.6 * self._speed, opt_dict=self._opt_dict,
            map_inst=CarlaDataProvider.get_map(), grp_inst=CarlaDataProvider.get_global_route_planner())
        controller.set_global_plan(plan)

        initial_vec = plan[0][0].transform.get_forward_vector()
        actor.set_target_velocity(self._speed * initial_vec)
        actor.apply_control(carla.VehicleControl(throttle=1, gear=1, manual_gear_shift=True))

        self._actor_data.append([actor, controller])
        self._spawn_dist = self._rng.uniform(self._min_spawn_dist, self._max_spawn_dist)

    def update(self):
        """Controls the created actors and creaes / removes other when needed"""
        # Control the vehicles, removing them when needed
        for actor_data in list(self._actor_data):
            actor, controller = actor_data
            location = CarlaDataProvider.get_location(actor)
            if not location:
                continue
            sink_distance = self._sink_location.distance(location)
            if sink_distance < self._sink_dist:
                CarlaDataProvider.remove_actor_by_id(actor.id)
                self._actor_data.remove(actor_data)
            else:
                actor.apply_control(controller.run_step())

        # Spawn new actors if needed
        if len(self._actor_data) == 0:
            distance = self._spawn_dist + 1
        else:
            actor_location = CarlaDataProvider.get_location(self._actor_data[-1][0])
            if actor_location is None:
                distance = 0
            else:
                distance = self._source_location.distance(actor_location)

        if distance > self._spawn_dist:
            self._spawn_actor(self._source_transform)

        return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        """
        Default terminate. Can be extended in derived class
        """
        if self._terminated:
            return

        self._terminated = True

        for actor, _ in self._actor_data:
            # TODO: Actors spawned in the same frame as the behavior termination won't be removed.
            # Patched by removing its movement
            actor.disable_constant_velocity()
            actor.set_autopilot(False, CarlaDataProvider.get_traffic_manager_port())
            actor.set_target_velocity(carla.Vector3D(0,0,0))
            actor.set_target_angular_velocity(carla.Vector3D(0,0,0))
            try:
                CarlaDataProvider.remove_actor_by_id(actor.id)
            except RuntimeError:
                pass  # Actor was already destroyed


class OpenVehicleDoor(AtomicBehavior):

    """
    Implementation for a behavior that will open the door of a vehicle,
    then close it after a while.

    Important parameters:
    - actor: Type of CARLA actors to be spawned
    - vehicle_door: The specific door that will be opened
    - duration: Duration of the open door
    """

    def __init__(self, actor, vehicle_door, name="OpenVehicleDoor"):
        """
        Setup class members
        """
        super().__init__(name, actor)
        self._vehicle_door = vehicle_door
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))

    def initialise(self):
        """
        Set start time
        """
        self._actor.open_door(self._vehicle_door)
        super().initialise()

    def update(self):
        """
        Keep running until termination condition is satisfied
        """
        new_status = py_trees.common.Status.SUCCESS
        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))
        return new_status


class TrafficLightFreezer(AtomicBehavior):
    """
    Behavior that freezes a group of traffic lights for a specific amount of time,
    returning them to their original state after ending it

    Important parameters:
    - traffic_lights_dict dict{carla.TrafficLight: carla.TrafficLightState}
    - timeout: Amount of time the traffic lights are frozen
    """

    def __init__(self, traffic_lights_dict, duration=10000, name="TrafficLightFreezer"):
        """Setup class members"""
        super().__init__(name)
        self._traffic_lights_dict = traffic_lights_dict
        self._duration = duration
        self._previous_traffic_light_info = {}
        self._start_time = None

    def initialise(self):
        """
        Sets the traffic lights to the desired states and remembers their previous one.
        These should technically be frozen, but that freezes all tls in the town
        """
        self._start_time = GameTime.get_time()
        for tl in self._traffic_lights_dict:
            elapsed_time = tl.get_elapsed_time()
            self._previous_traffic_light_info[tl] = {
                'state': tl.get_state(),
                'green_time': tl.get_green_time(),
                'red_time': tl.get_red_time(),
                'yellow_time': tl.get_yellow_time()
            }
            tl.set_state(self._traffic_lights_dict[tl])
            tl.set_green_time(self._duration + elapsed_time)
            tl.set_red_time(self._duration + elapsed_time)
            tl.set_yellow_time(self._duration + elapsed_time)

    def update(self):
        """Waits until the adequate time has passed"""
        if GameTime.get_time() - self._start_time > self._duration:
            return py_trees.common.Status.SUCCESS
        else:
            return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        """Reset all traffic lights back to their previous states"""
        if self._previous_traffic_light_info:
            for tl in self._traffic_lights_dict:
                tl.set_state(self._previous_traffic_light_info[tl]['state'])
                tl.set_green_time(self._previous_traffic_light_info[tl]['green_time'])
                tl.set_red_time(self._previous_traffic_light_info[tl]['red_time'])
                tl.set_yellow_time(self._previous_traffic_light_info[tl]['yellow_time'])


class StartRecorder(AtomicBehavior):

    """
    Atomic that starts the CARLA recorder. Only one can be active
    at a time, and if this isn't the case, the recorder will
    automatically stop the previous one.

    Args:
        recorder_name (str): name of the file to write the recorded data.
            Remember that a simple name will save the recording in
            'CarlaUE4/Saved/'. Otherwise, if some folder appears in the name,
            it will be considered an absolute path.
        name (str): name of the behavior
    """

    def __init__(self, recorder_name, name="StartRecorder"):
        """
        Setup class members
        """
        super().__init__(name)
        self._client = CarlaDataProvider.get_client()
        self._recorder_name = recorder_name

    def update(self):
        self._client.start_recorder(self._recorder_name)
        return py_trees.common.Status.SUCCESS


class StopRecorder(AtomicBehavior):

    """
    Atomic that stops the CARLA recorder.

    Args:
        name (str): name of the behavior
    """

    def __init__(self, name="StopRecorder"):
        """
        Setup class members
        """
        super().__init__(name)
        self._client = CarlaDataProvider.get_client()

    def update(self):
        self._client.stop_recorder()
        return py_trees.common.Status.SUCCESS


class TrafficLightManipulator(AtomicBehavior):

    """
    Atomic behavior that manipulates traffic lights around the ego_vehicle to trigger scenarios 7 to 10.
    This is done by setting 2 of the traffic light at the intersection to green (with some complex precomputation
    to set everything up).

    Important parameters:
    - ego_vehicle: CARLA actor that controls this behavior
    - subtype: string that gathers information of the route and scenario number
      (check SUBTYPE_CONFIG_TRANSLATION below)
    """

    RED = carla.TrafficLightState.Red
    YELLOW = carla.TrafficLightState.Yellow
    GREEN = carla.TrafficLightState.Green

    # Time constants
    RED_TIME = 1.5  # Minimum time the ego vehicle waits in red (seconds)
    YELLOW_TIME = 2  # Time spent at yellow state (seconds)
    RESET_TIME = 6  # Time waited before resetting all the junction (seconds)

    # Experimental values
    TRIGGER_DISTANCE = 10  # Distance that makes all vehicles in the lane enter the junction (meters)
    DIST_TO_WAITING_TIME = 0.04  # Used to wait longer at larger intersections (s/m)

    INT_CONF_OPP1 = {'ego': RED, 'ref': RED, 'left': RED, 'right': RED, 'opposite': GREEN}
    INT_CONF_OPP2 = {'ego': GREEN, 'ref': GREEN, 'left': RED, 'right': RED, 'opposite': GREEN}
    INT_CONF_LFT1 = {'ego': RED, 'ref': RED, 'left': GREEN, 'right': RED, 'opposite': RED}
    INT_CONF_LFT2 = {'ego': GREEN, 'ref': GREEN, 'left': GREEN, 'right': RED, 'opposite': RED}
    INT_CONF_RGT1 = {'ego': RED, 'ref': RED, 'left': RED, 'right': GREEN, 'opposite': RED}
    INT_CONF_RGT2 = {'ego': GREEN, 'ref': GREEN, 'left': RED, 'right': GREEN, 'opposite': RED}

    INT_CONF_REF1 = {'ego': GREEN, 'ref': GREEN, 'left': RED, 'right': RED, 'opposite': RED}
    INT_CONF_REF2 = {'ego': YELLOW, 'ref': YELLOW, 'left': RED, 'right': RED, 'opposite': RED}

    # Depending on the scenario, IN ORDER OF IMPORTANCE, the traffic light changed
    # The list has to contain only items of the INT_CONF
    SUBTYPE_CONFIG_TRANSLATION = {
        'S7left': ['left', 'opposite', 'right'],
        'S7right': ['left', 'opposite'],
        'S7opposite': ['right', 'left', 'opposite'],
        'S8left': ['opposite'],
        'S9right': ['left', 'opposite']
    }

    CONFIG_TLM_TRANSLATION = {
        'left': [INT_CONF_LFT1, INT_CONF_LFT2],
        'right': [INT_CONF_RGT1, INT_CONF_RGT2],
        'opposite': [INT_CONF_OPP1, INT_CONF_OPP2]
    }

    def __init__(self, ego_vehicle, subtype, debug=False, name="TrafficLightManipulator"):
        super().__init__(name)
        self.ego_vehicle = ego_vehicle
        self.subtype = subtype
        self.current_step = 1
        self.debug = debug

        self.traffic_light = None
        self.annotations = None
        self.configuration = None
        self.prev_junction_state = None
        self.junction_location = None
        self.seconds_waited = 0
        self.prev_time = None
        self.max_trigger_distance = None
        self.waiting_time = None
        self.inside_junction = False

        self.logger.debug("%s.__init__()" % (self.__class__.__name__))

    def update(self):

        new_status = py_trees.common.Status.RUNNING

        # 1) Set up the parameters
        if self.current_step == 1:

            # Traffic light affecting the ego vehicle
            self.traffic_light = CarlaDataProvider.get_next_traffic_light(self.ego_vehicle, use_cached_location=False)
            if not self.traffic_light:
                # nothing else to do in this iteration...
                return new_status

            # "Topology" of the intersection
            self.annotations = CarlaDataProvider.annotate_trafficlight_in_group(self.traffic_light)

            # Which traffic light will be modified (apart from the ego lane)
            self.configuration = self.get_traffic_light_configuration(self.subtype, self.annotations)
            if self.configuration is None:
                self.current_step = 0  # End the behavior
                return new_status

            # Modify the intersection. Store the previous state
            self.prev_junction_state = self.set_intersection_state(self.INT_CONF_REF1)

            self.current_step += 1
            if self.debug:
                print("--- All set up")

        # 2) Modify the ego lane to yellow when closeby
        elif self.current_step == 2:

            ego_location = CarlaDataProvider.get_location(self.ego_vehicle)

            if self.junction_location is None:
                ego_waypoint = CarlaDataProvider.get_map().get_waypoint(ego_location)
                junction_waypoint = ego_waypoint.next(0.5)[0]
                while not junction_waypoint.is_junction:
                    next_wp = junction_waypoint.next(0.5)[0]
                    junction_waypoint = next_wp
                self.junction_location = junction_waypoint.transform.location

            distance = ego_location.distance(self.junction_location)

            # Failure check
            if self.max_trigger_distance is None:
                self.max_trigger_distance = distance + 1
            if distance > self.max_trigger_distance:
                self.current_step = 0

            elif distance < self.TRIGGER_DISTANCE:
                _ = self.set_intersection_state(self.INT_CONF_REF2)
                self.current_step += 1

            if self.debug:
                print("--- Distance until traffic light changes: {}".format(distance))

        # 3) Modify the ego lane to red and the chosen one to green after several seconds
        elif self.current_step == 3:

            if self.passed_enough_time(self.YELLOW_TIME):
                _ = self.set_intersection_state(self.CONFIG_TLM_TRANSLATION[self.configuration][0])

                self.current_step += 1

        # 4) Wait a bit to let vehicles enter the intersection, then set the ego lane to green
        elif self.current_step == 4:

            # Get the time in red, dependent on the intersection dimensions
            if self.waiting_time is None:
                self.waiting_time = self.get_waiting_time(self.annotations, self.configuration)

            if self.passed_enough_time(self.waiting_time):
                _ = self.set_intersection_state(self.CONFIG_TLM_TRANSLATION[self.configuration][1])

                self.current_step += 1

        # 5) Wait for the end of the intersection
        elif self.current_step == 5:
            # the traffic light has been manipulated, wait until the vehicle finsihes the intersection
            ego_location = CarlaDataProvider.get_location(self.ego_vehicle)
            ego_waypoint = CarlaDataProvider.get_map().get_waypoint(ego_location)

            if not self.inside_junction:
                if ego_waypoint.is_junction:
                    # Wait for the ego_vehicle to enter a junction
                    self.inside_junction = True
                else:
                    if self.debug:
                        print("--- Waiting to ENTER a junction")

            else:
                if ego_waypoint.is_junction:
                    if self.debug:
                        print("--- Waiting to EXIT a junction")
                else:
                    # And to leave it
                    self.inside_junction = False
                    self.current_step += 1

        # 6) At the end (or if something failed), reset to the previous state
        else:
            if self.prev_junction_state:
                CarlaDataProvider.reset_lights(self.prev_junction_state)
                if self.debug:
                    print("--- Returning the intersection to its previous state")

            self.variable_cleanup()
            new_status = py_trees.common.Status.SUCCESS

        return new_status

    def passed_enough_time(self, time_limit):
        """
        Returns true or false depending on the time that has passed from the
        first time this function was called
        """
        # Start the timer
        if self.prev_time is None:
            self.prev_time = GameTime.get_time()

        timestamp = GameTime.get_time()
        self.seconds_waited += (timestamp - self.prev_time)
        self.prev_time = timestamp

        if self.debug:
            print("--- Waited seconds: {}".format(self.seconds_waited))

        if self.seconds_waited >= time_limit:
            self.seconds_waited = 0
            self.prev_time = None

            return True
        return False

    def set_intersection_state(self, choice):
        """
        Changes the intersection to the desired state
        """
        prev_state = CarlaDataProvider.update_light_states(
            self.traffic_light,
            self.annotations,
            choice,
            freeze=True)

        return prev_state

    def get_waiting_time(self, annotation, direction):
        """
        Calculates the time the ego traffic light will remain red
        to let vehicles enter the junction
        """

        tl = annotation[direction][0]
        ego_tl = annotation["ref"][0]

        tl_location = CarlaDataProvider.get_trafficlight_trigger_location(tl)
        ego_tl_location = CarlaDataProvider.get_trafficlight_trigger_location(ego_tl)

        distance = ego_tl_location.distance(tl_location)

        return self.RED_TIME + distance * self.DIST_TO_WAITING_TIME

    def get_traffic_light_configuration(self, subtype, annotations):
        """
        Checks the list of possible altered traffic lights and gets
        the first one that exists in the intersection

        Important parameters:
        - subtype: Subtype of the scenario
        - annotations: list of the traffic light of the junction, with their direction (right, left...)
        """
        configuration = None

        if subtype in self.SUBTYPE_CONFIG_TRANSLATION:
            possible_configurations = self.SUBTYPE_CONFIG_TRANSLATION[self.subtype]
            while possible_configurations:
                # Chose the first one and delete it
                configuration = possible_configurations[0]
                possible_configurations = possible_configurations[1:]
                if configuration in annotations:
                    if annotations[configuration]:
                        # Found a valid configuration
                        break
                    else:
                        # The traffic light doesn't exist, get another one
                        configuration = None
                else:
                    if self.debug:
                        print("This configuration name is wrong")
                    configuration = None

            if configuration is None and self.debug:
                print("This subtype has no traffic light available")
        else:
            if self.debug:
                print("This subtype is unknown")

        return configuration

    def variable_cleanup(self):
        """
        Resets all variables to the intial state
        """
        self.current_step = 1
        self.traffic_light = None
        self.annotations = None
        self.configuration = None
        self.prev_junction_state = None
        self.junction_location = None
        self.max_trigger_distance = None
        self.waiting_time = None
        self.inside_junction = False


class ScenarioTriggerer(AtomicBehavior):

    """
    Handles the triggering of the scenarios that are part of a route.

    Initializes a list of blackboard variables to False, and only sets them to True when
    the ego vehicle is very close to the scenarios
    """

    WINDOWS_SIZE = 5

    def __init__(self, actor, route, blackboard_list, distance, debug=False, name="ScenarioTriggerer"):
        """
        Setup class members
        """
        super().__init__(name)
        self._world = CarlaDataProvider.get_world()
        self._map = CarlaDataProvider.get_map()
        self._debug = debug

        self._actor = actor
        self._route = route
        self._distance = distance
        self._blackboard_list = blackboard_list
        self._triggered_scenarios = []  # List of already done scenarios

        self._current_index = 0
        self._route_length = len(self._route)
        self._waypoints, _ = zip(*self._route)

    def add_blackboard(self, blackboard):
        """
        Adds new blackboards to the list. Used by the runtime initialization of scenarios
        """
        self._blackboard_list.append(blackboard)

    def update(self):
        new_status = py_trees.common.Status.RUNNING

        location = CarlaDataProvider.get_location(self._actor)
        if location is None:
            return new_status

        lower_bound = self._current_index
        upper_bound = min(self._current_index + self.WINDOWS_SIZE + 1, self._route_length)

        shortest_distance = float('inf')
        closest_index = -1

        for index in range(lower_bound, upper_bound):
            ref_waypoint = self._waypoints[index]
            ref_location = ref_waypoint.location

            dist_to_route = ref_location.distance(location)
            if dist_to_route <= shortest_distance:
                closest_index = index
                shortest_distance = dist_to_route

        if closest_index == -1 or shortest_distance == float('inf'):
            return new_status

        # Update the ego position at the route
        self._current_index = closest_index

        route_location = self._waypoints[closest_index].location

        # Check which scenarios can be triggered
        blackboard = py_trees.blackboard.Blackboard()
        for black_var_name, scen_location, scen_name in self._blackboard_list:

            # Close enough
            scen_distance = route_location.distance(scen_location)
            condition1 = bool(scen_distance < self._distance)

            # Not being currently done
            value = blackboard.get(black_var_name)
            condition2 = bool(not value)

            # Already done, if needed
            condition3 = bool(black_var_name not in self._triggered_scenarios)

            if condition1 and condition2 and condition3:
                _ = blackboard.set(black_var_name, True)
                self._triggered_scenarios.append(black_var_name)

                CarlaDataProvider.set_latest_scenario(scen_name)

                if self._debug:
                    self._world.debug.draw_point(
                        scen_location + carla.Location(z=4),
                        size=0.5,
                        life_time=0.5,
                        color=carla.Color(255, 255, 0)
                    )
                    self._world.debug.draw_string(
                        scen_location + carla.Location(z=5),
                        str(black_var_name),
                        False,
                        color=carla.Color(0, 0, 0),
                        life_time=1000
                    )

        return new_status


class KeepLongitudinalGap(AtomicBehavior):
    """
    This class contains an atomic behavior to maintain a set gap with leading/adjacent vehicle.

    Important parameters:
    - actor: CARLA actor to execute the behavior
    - reference_actor: Reference actor the distance shall be kept to.
    - distance: target gap between the two actors in meters
    - distance_type: Specifies how distance should be calculated between the two actors

    The behavior terminates after overwritten by other events / when target distance is reached(if continues).
    """

    def __init__(self, actor, reference_actor, gap, gap_type="distance", max_speed=None, continues=False,
                 freespace=False, name="AutoKeepDistance"):
        """
        Setup parameters
        """
        super().__init__(name, actor)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._reference_actor = reference_actor
        self._gap = gap
        self._gap_type = gap_type
        self._continues = continues
        self._freespace = freespace
        self._global_rp = None
        max_speed_limit = 100
        self.max_speed = max_speed_limit if max_speed is None else float(max_speed)
        if freespace and self._gap_type == "distance":
            self._gap += self._reference_actor.bounding_box.extent.x + self._actor.bounding_box.extent.x

        self._start_time = None

    def initialise(self):
        actor_dict = {}

        try:
            check_actors = operator.attrgetter("ActorsWithController")
            actor_dict = check_actors(py_trees.blackboard.Blackboard())
        except AttributeError:
            pass

        if not actor_dict or self._actor.id not in actor_dict:
            raise RuntimeError("Actor not found in ActorsWithController BlackBoard")

        self._start_time = GameTime.get_time()
        actor_dict[self._actor.id].update_target_speed(self.max_speed, start_time=self._start_time)

        self._global_rp = CarlaDataProvider.get_global_route_planner()

        super().initialise()

    def update(self):
        """
        keeps track of gap and update the controller accordingly
        """
        try:
            check_actors = operator.attrgetter("ActorsWithController")
            actor_dict = check_actors(py_trees.blackboard.Blackboard())
        except AttributeError:
            pass

        if not actor_dict or self._actor.id not in actor_dict:
            return py_trees.common.Status.FAILURE

        if actor_dict[self._actor.id].get_last_longitudinal_command() != self._start_time:
            return py_trees.common.Status.SUCCESS

        new_status = py_trees.common.Status.RUNNING

        actor_velocity = CarlaDataProvider.get_velocity(self._actor)
        reference_velocity = CarlaDataProvider.get_velocity(self._reference_actor)

        gap = sr_tools.scenario_helper.get_distance_between_actors(self._actor, self._reference_actor,
                                                                   distance_type="longitudinal",
                                                                   freespace=self._freespace,
                                                                   global_planner=self._global_rp)
        actor_transform = CarlaDataProvider.get_transform(self._actor)
        ref_actor_transform = CarlaDataProvider.get_transform(self._reference_actor)
        if is_within_distance(ref_actor_transform, actor_transform, float('inf'), [0, 90]) and \
                operator.le(gap, self._gap):
            try:
                factor = abs(actor_velocity - reference_velocity)/actor_velocity
                if actor_velocity > reference_velocity:
                    actor_velocity = actor_velocity - (factor*actor_velocity)
                elif actor_velocity < reference_velocity and operator.gt(gap, self._gap):
                    actor_velocity = actor_velocity + (factor*actor_velocity)
            except ZeroDivisionError:
                pass
            actor_dict[self._actor.id].update_target_speed(actor_velocity)

            if not self._continues:
                if operator.le(gap, self._gap):
                    new_status = py_trees.common.Status.SUCCESS
        else:
            actor_dict[self._actor.id].update_target_speed(self.max_speed)

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))
        return new_status


class AddActor(AtomicBehavior):
    """
    Implementation for a behavior that will create a actor
    at a given transform if no other actor exists in a given radius
    from the transform.

    Important parameters:
    - actor_type: Type of CARLA actors to be spawned
    - transform: Spawn location
    A parallel termination behavior has to be used.
    """

    def __init__(self, actor, actor_type, transform, init_velocity=carla.Vector3D(), color=None,
                 actor_name="scenario", actor_category="car", name="SpawnActor"):
        """
        Setup class members
        """
        super().__init__(name, actor)
        self._actor_type = actor_type
        self._actor = actor
        self._spawn_point = transform
        self._color = color
        self._init_velocity = init_velocity
        self._actor_name = actor_name
        self._actor_category = actor_category

    def update(self):
        new_status = py_trees.common.Status.RUNNING
        try:
            new_actor = CarlaDataProvider.request_new_actor(
                self._actor_type,
                self._spawn_point,
                rolename=self._actor_name,
                color=self._color,
                actor_category=self._actor_category)
            if new_actor:
                self._actor = new_actor
                self._actor.set_target_velocity(self._init_velocity)
                self._ensure_default_actor_control(new_actor)
                new_status = py_trees.common.Status.SUCCESS
        except RuntimeError:
            print("ActorSource unable to spawn actor")
            return py_trees.common.Status.FAILURE
        return new_status

    @staticmethod
    def _ensure_default_actor_control(actor):
        # Some scenarios assign route/speed immediately after AddEntityAction.
        # Register a default controller so those follow-up actions have a
        # blackboard entry even without an explicit ControllerAction.
        actor_dict = {}
        try:
            check_actors = operator.attrgetter("ActorsWithController")
            actor_dict = check_actors(py_trees.blackboard.Blackboard())
        except AttributeError:
            pass

        if actor.id not in actor_dict:
            actor_dict[actor.id] = ActorControl(actor, control_py_module=None, args={}, scenario_file_path=None)
            py_trees.blackboard.Blackboard().set("ActorsWithController", actor_dict, overwrite=True)

class SwitchWrongDirectionTest(AtomicBehavior):

    """
    Atomic that switch the OutsideRouteLanesTest criterion.

    Args:
        active (bool): True: activated; False: deactivated
        name (str): name of the behavior
    """

    def __init__(self, active, name="SwitchWrongDirectionTest"):
        """
        Setup class members
        """
        self._active = active
        super().__init__(name)

    def update(self):
        py_trees.blackboard.Blackboard().set("AC_SwitchWrongDirectionTest", self._active, overwrite=True)
        return py_trees.common.Status.SUCCESS


class SwitchMinSpeedCriteria(AtomicBehavior):
    """
    Atomic that switch the SwitchMinSpeedCriteria criterion.

    Args:
        active (bool): True: activated; False: deactivated
        name (str): name of the behavior
    """


    def __init__(self, active, name="ChangeMinSpeed"):
        """
        Setup parameters
        """
        super().__init__(name)
        self._active = active
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))

    def update(self):
        """
        keeps track of gap and update the controller accordingly
        """
        new_status = py_trees.common.Status.SUCCESS
        py_trees.blackboard.Blackboard().set("SwitchMinSpeedCriteria", self._active, overwrite=True)
        return new_status


class WalkerFlow(AtomicBehavior):
    """
    Behavior that indefinitely creates walkers at a location,
    controls them until another location, and then destroys them.
    Therefore, a parallel termination behavior has to be used.

    There can be more than one target location.

    Important parameters:
    - source_location (carla.Location): Location at which actors will be spawned
    - sink_locations (list(carla.Location)): Locations at which actors will be deleted
    - sink_locations_prob (list(float)): The probability of each sink_location
    - spawn_dist_interval (list(float)): Distance between spawned actors
    - random_seed : Optional. The seed of numpy's random
    - sink_distance: Actors closer to the sink than this distance will be deleted. 
                     Probably due to the navigation module rerouting the walkers, a sink distance of 2 is reasonable.
    """

    def __init__(
        self,
        source_location,
        sink_locations,
        sink_locations_prob,
        spawn_dist_interval,
        random_seed=None,
        sink_dist=2,
        name="WalkerFlow",
    ):
        """
        Setup class members
        """
        super().__init__(name)

        if random_seed is not None:
            self._rng = random.RandomState(random_seed)
        else:
            self._rng = CarlaDataProvider.get_random_seed()
        self._world = CarlaDataProvider.get_world()

        self._controller_bp = self._world.get_blueprint_library().find('controller.ai.walker')

        self._source_location = source_location

        self._sink_locations = sink_locations
        self._sink_locations_prob = sink_locations_prob
        self._sink_dist = sink_dist

        self._min_spawn_dist = spawn_dist_interval[0]
        self._max_spawn_dist = spawn_dist_interval[1]
        self._spawn_dist = self._rng.uniform(self._min_spawn_dist, self._max_spawn_dist)

        self._batch_size_list = [1,2,3]

        self._walkers = []

    def update(self):
        """Controls the created actors and creates / removes other when needed"""
        # Remove walkers when needed
        for item in self._walkers:
            walker, controller, sink_location = item
            loc = CarlaDataProvider.get_location(walker)
            if loc.distance(sink_location) < self._sink_dist:
                self._destroy_walker(walker, controller)
                self._walkers.remove(item)

        # Spawn new walkers
        if len(self._walkers) == 0:
            distance = self._spawn_dist + 1
        else:
            actor_location = CarlaDataProvider.get_location(self._walkers[-1][0])
            distance = self._source_location.distance(actor_location)

        if distance > self._spawn_dist:
            # spawn new walkers
            walker_amount = self._rng.choice(self._batch_size_list)
            for i in range(walker_amount):
                spawn_tran = carla.Transform(self._source_location)
                spawn_tran.location.y -= i
                walker = CarlaDataProvider.request_new_actor(
                    'walker.*', spawn_tran, rolename='scenario'
                )
                if walker is None:
                    continue
                # Use ai.walker to controll walkers
                controller = self._world.try_spawn_actor(self._controller_bp, carla.Transform(), walker)
                sink_location = self._rng.choice(a = self._sink_locations, p = self._sink_locations_prob)
                controller.start()
                controller.go_to_location(sink_location)
                # Add to walkers list
                self._walkers.append((walker, controller, sink_location))

            self._spawn_dist = self._rng.uniform(self._min_spawn_dist, self._max_spawn_dist)

        return py_trees.common.Status.RUNNING

    def _destroy_walker(self, walker, controller):
        controller.stop()
        controller.destroy()
        CarlaDataProvider.remove_actor_by_id(walker.id)

    def terminate(self, new_status):
        """
        Default terminate. Can be extended in derived class
        """
        for walker, controller, _ in self._walkers:
            try:
                self._destroy_walker(walker, controller)
            except RuntimeError:
                pass  # Actor was already destroyed

class AIWalkerBehavior(AtomicBehavior):
    """
    Behavior that creates a walker controlled by AI Walker controller.
    The walker go from source location to sink location.
    A parallel termination behavior has to be used.

    Important parameters:
    - source_location (carla.Location): Location at which the actor will be spawned
    - sink_location (carla.Location): Location at which the actor will be deleted
    """

    def __init__(self, source_location, sink_location,
                 name="AIWalkerBehavior"):
        """
        Setup class members
        """
        super().__init__(name)

        self._world = CarlaDataProvider.get_world()
        self._controller_bp = self._world.get_blueprint_library().find('controller.ai.walker')

        self._source_location = source_location

        self._sink_location = sink_location
        self._sink_dist = 2

        self._walker = None
        self._controller = None

    def initialise(self):
        """
        Spawn the walker at source location.
        Setup the AI controller.

        May throw RuntimeError if the walker can not be
        spawned at given location.
        """
        spawn_tran = carla.Transform(self._source_location)
        self._walker = CarlaDataProvider.request_new_actor(
            'walker.*', spawn_tran, rolename='scenario'
        )
        if self._walker is None:
            raise RuntimeError("Couldn't spawn the walker")
        # Use ai.walker to controll the walker
        self._controller = self._world.try_spawn_actor(
            self._controller_bp, carla.Transform(), self._walker)
        self._controller.start()
        self._controller.go_to_location(self._sink_location)

        super().initialise()

    def update(self):
        """Controls the created walker"""
        # Remove walkers when needed
        if self._walker is not None:
            loc = CarlaDataProvider.get_location(self._walker)
            # At the very beginning of the scenario, the get_location may return None
            if loc is not None:
                if loc.distance(self._sink_location) < self._sink_dist:
                    self.terminate(py_trees.common.Status.SUCCESS)

        return py_trees.common.Status.RUNNING

    def _destroy_walker(self, walker, controller):
        if controller:
            controller.stop()
            controller.destroy()
        if walker:
            CarlaDataProvider.remove_actor_by_id(walker.id)

    def terminate(self, new_status):
        """
        Default terminate. Can be extended in derived class
        """
        try:
            self._destroy_walker(self._walker, self._controller)
        except RuntimeError:
            pass  # Actor was already destroyed


class ScenarioTimeout(AtomicBehavior):

    """
    This class is an idle behavior that waits for a set amount of time
    before stopping.

    It is meant to be used with the `ScenarioTimeoutTest` to be used at scenarios
    that block the ego's route (such as adding obstacles) so that if the ego is
    incapable of surpassing them, it isn't blocked forever. Instead,
    the scenario will timeout, but it will be penalized by the `ScenarioTimeoutTest`

    Parameters:
    - duration: Duration in seconds of this behavior
    """

    def __init__(self, duration, scenario_name, name="ScenarioTimeout"):
        """
        Setup actor
        """
        super().__init__(name)
        self._duration = duration
        self._scenario_name = scenario_name
        self._start_time = 0
        self._scenario_timeout = False
        self._terminated = False
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))

    def initialise(self):
        """
        Set start time
        """
        self._start_time = GameTime.get_time()
        py_trees.blackboard.Blackboard().set("AC_SwitchActorBlockedTest", False, overwrite=True)
        super().initialise()

    def update(self):
        """
        Keep running until termination condition is satisfied
        """
        new_status = py_trees.common.Status.RUNNING

        if GameTime.get_time() - self._start_time > self._duration:
            self._scenario_timeout = True
            new_status = py_trees.common.Status.SUCCESS

        return new_status

    def terminate(self, new_status):
        """
        Modifies the blackboard to tell the `ScenarioTimeoutTest` if the timeout was triggered
        """
        if not self._terminated:  # py_trees calls the terminate several times for some reason.
            py_trees.blackboard.Blackboard().set(
                f"ScenarioTimeout_{self._scenario_name}",
                self._scenario_timeout,
                overwrite=True,
            )
            py_trees.blackboard.Blackboard().set(
                "AC_SwitchActorBlockedTest", True, overwrite=True
            )
            self._terminated = True
        super().terminate(new_status)


class MovePedestrianWithEgo(AtomicBehavior):
    """This class is an atomic behavior that moves a pedestrian with the ego vehicle."""

    def __init__(self, reference_actor, actor, distance, displacement=0, name="TrackActor"):
        """
        Setup actor
        """
        super().__init__(name)
        self._actor = actor
        self._reference_actor = reference_actor
        self._distance = distance
        self._displacement = displacement

        added_location = carla.Location(x=self._displacement, z=-self._distance)
        self._actor.set_location(self._reference_actor.get_location() + added_location)

        self._start_time = 0
        self._teleport_time = 5

        self.logger.debug("%s.__init__()" % (self.__class__.__name__))

    def initialise(self):
        """
        Set start time
        """
        self._start_time = GameTime.get_time()
        added_location = carla.Location(x=self._displacement, z=-self._distance)
        self._actor.set_location(self._reference_actor.get_location() + added_location)
        super().initialise()

    def update(self):
        """
        Keep running until termination condition is satisfied
        """
        new_status = py_trees.common.Status.RUNNING

        if GameTime.get_time() - self._start_time > self._teleport_time:
            added_location = carla.Location(x=self._displacement, z=-self._distance)
            self._actor.set_location(self._reference_actor.get_location() + added_location)
            self._start_time = GameTime.get_time()
        return new_status
