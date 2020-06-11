import carla


class MetricsLog(object):
    """
    Utility class to query the metrics log.
    
    The information of the log should be accesed through the functions,
    but the dictionaries are public in case the users wants to use them.

    It consists of three attributes:
        - states ([dict{frame ID: info}]): Each dictionary contains
            information of a specific frame.
        - actors (dict{actor ID: actor info}): Dictionary with information
            about the actors at the simulation
        - criteria (dict{criteria name: criteria info}): Dictionary with all
            the criterias and its attributes.

    The states dictionary has the following information:
        - transform: (vehicles and walkers)

        - velocity: (vehicles)
        - control: (vehicles)

        - speed: (walkers)

        - state: (traffic lights)
        - frozen: (traffic lights)
        - elapsed_time: (traffic lights)

    The actors dictionary has the following information:
        - type_id: 
        - carla.ActorAttribute: this hugely vary and some examples are:
            · role_name
            · number_of_wheels:
            · is_invincible:
            · color:
    """

    def __init__(self, recorder, criteria):
        """
        Initializes the log class and parses it to extract the disctionaries
        """
        self._actors = recorder[0]
        self._states = recorder[1]
        self._criteria = criteria

    def get_actor_state(self, actor_id, frame, attribute):
        """
        Searched the states dictionary at a specific for the attribute of
        actor_id and returns the results. Returns None if the dictionary
        doesn't have the actor_id or if the attribute is missing.

        Args:
            actor_id (str): Id of the actor to be checked
            frame: (int): frame number of the dictionary
            attribute (str): name of the actor's attribute to be returned
        """
        frame_state = self._states[frame]
        if actor_id in frame_state:

            if attribute not in frame_state[actor_id]:
                print("WARNING: Can't find {} for actor with ID {}".format(attribute, actor_id))
                return None
        
            attribute_info = frame_state[actor_id][attribute]
            return attribute_info
        return None
    
    def get_all_actor_states(self, actor_id, attribute, ini_frame=None, end_frame=None):
        """
        Searches the states dictionary for all the attributes during the interval between
        ini_frame and end_frame. Returns a list with the attributes at each frame. Some of these
        might be None, if the attribute hasn't been found.

        Args:
            actor_id (srt): ID of the actor
            attribute: name of the actor's attribute to be returned
            ini_frame (int): First frame checked. By default, 0 
            end_frame (int): Last frame checked. By default, max number of frames.
        """
        if ini_frame is None:
            ini_frame = 0
        if end_frame is None:
            end_frame = len(self._states) - 1

        attributes_list = []

        for frame_number in range(ini_frame, end_frame + 1):

            attribute_info = self.get_actor_state(actor_id, frame_number, attribute)
            attributes_list.append(attribute_info)

        return attributes_list

    def get_ego_vehicle_id(self):
        """
        Returns a string with the id of the ego vehicle.
        """
        return get_actor_id_with_role_name("hero")

    def get_actor_id_with_role_name(self, rolename):
        """
        Returns a string with the actor's id of the one with a specific rolename
        Useful for identifying special actors such as the ego vehicle
        """
        for actor_id in self._actors:

            actor_info = self._actors[actor_id]
            if "role_name" in actor_info and actor_info["role_name"] == rolename:
                return actor_id

        return None
        
    def get_actor_id_with_type_id(self, type_id):
        """
        Returns a string with the actor's id of the one with a specific type_id
        """
        for actor_id in self._actors:

            actor_info = self._actors[actor_id]
            if "type_id" in actor_info and actor_info["type_id"] == rolename:
                return actor_id

        return None
    
    def get_actor_attributes(self, actor_id):
        """
        Returns an int with the actor id of the ego vehicle. This is done by checking the "hero" rolename
        """
        if actor_id in self._actors:
            return self._actors[actor_id]

        return None

    def get_actor_alive_frames(self, actor_id):
        """
        Returns a tuple with the first and last frame an actor was alive.
        """

        if actor_id in self._actors:

            actor_info = self._actors[actor_id]
            first_frame = actor_info ["created"]
            if "destroyed" in actor_info:
                last_frame = actor_info ["destroyed"]
            else:
                last_frame = self.get_total_frame_count()

            return first_frame, last_frame
        
        return None, None

    def get_criteria(self, name):
        """
        Returns an int with the actor id of the ego vehicle. This is done by checking the "hero" rolename
        """
        if name in self._criteria:
            return self._criteria[name]

        return None

    def get_total_frame_count(self):
        """
        Returns an int with the actor id of the ego vehicle. This is done by checking the "hero" rolename
        """

        return len(self._states)
