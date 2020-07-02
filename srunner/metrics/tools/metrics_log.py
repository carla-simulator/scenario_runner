#!/usr/bin/env python

# Copyright (c) 2020 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Support class of the MetricsManager to query the information available
to the metrics.

It also provides a series of functions to help the user querry
specific information
"""

import fnmatch
from srunner.metrics.tools.metrics_parser import MetricsParser

class MetricsLog(object):
    """
    Utility class to query the log.
    """

    def __init__(self, recorder):
        """
        Initializes the log class and parses it to extract the dictionaries
        """
        # Parse the information
        self._simulation, self._actors, self._frames = MetricsParser.parse_recorder_info(recorder)

    ###############################################
    # Functions used to get info about the actors #
    ###############################################
    def get_ego_vehicle_id(self):
        """
        Returns the id of the ego vehicle
        """
        return self.get_actor_ids_with_role_name("hero")[0]

    def get_actor_ids_with_role_name(self, role_name):
        """
        Returns a list of actor ids that match the given role_name

        Args:
            role_name (str): string with the desired role_name to filter the actors
        """
        actor_list = []

        for actor_id in self._actors:
            actor = self._actors[actor_id]
            if "role_name" in actor and actor["role_name"] == role_name:
                actor_list.append(actor_id)

        return actor_list

    def get_actor_ids_with_type_id(self, type_id):
        """
        Returns a list of actor ids that match the given type_id, matching fnmatch standard

        Args:
            type_id (str): string with the desired type id to filter the actors
        """
        actor_list = []

        for actor_id in self._actors:
            actor = self._actors[actor_id]
            if "type_id" in actor and fnmatch.fnmatch(actor["type_id"], type_id):
                actor_list.append(actor_id)

        return actor_list

    def get_actor_attributes(self, actor_id):
        """
        Returns a dictionary with all the attributes of an actor

        Args:
            actor_id (int): ID of the actor
        """
        if actor_id in self._actors:
            return self._actors[actor_id]

        return None

    def get_actor_alive_frames(self, actor_id):
        """
        Returns a tuple with the first and last frame an actor was alive.

        Args:
            actor_id (int): Id of the actor from which the information will be returned
        """

        if actor_id in self._actors:

            actor_info = self._actors[actor_id]
            first_frame = actor_info["created"]
            if "destroyed" in actor_info:
                last_frame = actor_info["destroyed"] - 1
            else:
                last_frame = self.get_total_frame_count()

            return first_frame, last_frame

        return None, None

    ##########################################
    # Functions used to get the actor states #
    ##########################################
    def _get_actor_state(self, actor_id, state, frame):
        """
        Given an actor id, returns the specific variable of that actor at a given frame.
        Returns None if the actor_id or the state are missing

        Args:
            actor_id (int): Id of the actor to be checked
            frame: (int): frame number of the simulation
            attribute (str): name of the actor's attribute to be returned
        """
        frame_state = self._frames[frame - 1]["actors"]

        # Check if the actor exists
        if actor_id in frame_state:

            
            # Check if the state exists
            if state not in frame_state[actor_id]:
                # print("HI")
                return None

            state_info = frame_state[actor_id][state]
            return state_info

        return None

    def _get_all_actor_states(self, actor_id, state, ini_frame=None, end_frame=None):
        """
        Given an actor id, returns a list of the specific variable of that actor during
        a frame interval. Some elements might be None

        By default, ini_frame and end_frame are the start and end of the simulation, respectively.

        Args:
            actor_id (int): ID of the actor
            attribute: name of the actor's attribute to be returned
            ini_frame (int): First frame checked. By default, 0
            end_frame (int): Last frame checked. By default, max number of frames.
        """
        if ini_frame is None:
            ini_frame = 0
        if end_frame is None:
            end_frame = self.get_total_frame_count()

        state_list = []

        for frame_number in range(ini_frame, end_frame + 1):
            state_info = self._get_actor_state(actor_id, state, frame_number)
            state_list.append(state_info)

        return state_list

    def _get_states_at_frame(self, frame, state, actor_list=None):
        """
        Returns a dictionary {int - carla.Transform} with the actor ID and transform
        at a given frame of all the actors at actor_list. Some states might be None.

        By default, all actors will be considered
        """
        states = {}
        actor_info = self._frames[frame]["actors"]

        for actor_id in actor_info:
            if not actor_list or actor_id in actor_list:
                _state = self._get_actor_state(actor_id, state, frame)
                states.update({actor_id: _state})

        return states

    # Transforms
    def get_transform(self, actor_id, frame):
        """
        Returns the transform of the actor at a specific frame.
        """
        return self._get_actor_state(actor_id, "transform", frame)

    def get_all_transforms(self, actor_id, ini_frame=None, end_frame=None):
        """
        Returns a list with all the transforms of the actor at the frame interval.
        """
        return self._get_all_actor_states(actor_id, "transform", ini_frame, end_frame)

    def get_transforms_at_frame(self, frame, actor_list=None):
        """
        Returns a dictionary {int - carla.Transform} with the actor ID and transform
        at a given frame of all the actors at actor_list.
        """
        return self._get_states_at_frame(frame, "transform", actor_list)

    # Velocities
    def get_velocity(self, actor_id, frame):
        """
        Returns the velocity of the actor at a specific frame.
        """
        return self._get_actor_state(actor_id, "velocity", frame)

    def get_all_velocities(self, actor_id, ini_frame=None, end_frame=None):
        """
        Returns a list with all the velocities of the actor at the frame interval.
        """
        return self._get_all_actor_states(actor_id, "velocity", ini_frame, end_frame)

    def get_velocities_at_frame(self, frame, actor_list=None):
        """
        Returns a dictionary {int - carla.Vector3D} with the actor ID and velocity
        at a given frame of all the actors at actor_list.
        """
        return self._get_states_at_frame(frame, "velocity", actor_list)

    # Controls
    def get_vehicle_control(self, vehicle_id, frame):
        """
        Returns the control of the vehicle at a specific frame.
        """
        return self._get_actor_state(vehicle_id, "control", frame)

    def get_walker_speed(self, walker_id, frame):
        """
        Returns the speed of the walker at a specific frame.
        """
        return self._get_actor_state(walker_id, "speed", frame)

    # Traffic lights
    def get_traffic_light_state(self, tl_id, frame):
        """
        Returns the state of the traffic light at a specific frame.
        """
        return self._get_actor_state(tl_id, "state", frame)

    def is_traffic_light_frozen(self, actor_id, frame):
        """
        Returns whether or not the traffic light is frozen at a specific frame.
        """
        return self._get_actor_state(actor_id, "frozen", frame)

    def get_traffic_light_elapsed_time(self, actor_id, frame):
        """
        Returns the elapsed time of the traffic light at a specific frame.
        """
        return self._get_actor_state(actor_id, "elapsed_time", frame)

    ########################################################
    # Functions used to get general info of the simulation #
    ########################################################
    def get_collisions(self, actor_id):
        """
        Returns a {frame_number - other_ID} dictionary containing the
        frames at which the actor collided and the id of the other actor

        Args:
            actor_id (int): ID of the actor
        """
        collisions = self._general_info["collisions"]

        if actor_id in collisions:
            return collisions[actor_id]

        return None

    def get_total_frame_count(self):
        """
        Returns an int with the total amount of frames the simulation lasted
        """

        return self._general_info["total_frames"]

