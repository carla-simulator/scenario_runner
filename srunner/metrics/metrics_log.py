import carla


class MetricsLog(object):
    """
    Utility class to query the metrics log.
    
    The information of the log should be accesed through the functions,
    but the dictionaries are public in case the users wants to use them.

    It consits of three attributes:
        - states ([dict{frame ID: info}]): Each dictionary contains
            information of a specific frame.
        - actors (dict{actor ID: actor info}): Dictionary with information
            about the actors at the simulation
        - criteria (dict{criteria name: criteria info}): Dictionary with all
            the criterias and its attributes.
    """

    def __init__(self, recorder, criteria):
        """
        Initializes the log class and parses it to extract the disctionaries
        """
        self._actors = recorder[0]
        self._states = recorder[1]
        self._criteria = criteria

    def get_states_attribute(self):
        """
        returns the _states attribute
        """
        return self._states

    def get_actors_attribute(self):
        """
        returns the _actors attribute
        """
        return self._actors

    def get_criteria_attribute(self):
        """
        returns the _criteria attribute
        """
        return self._criteria

    def get_actor_transform(self, actor_id, frame):
        """
        Returns a carla.Transform with an actor's transform at a given frame,
        or None if the actor wasn't alive at that frame

        Args:
            actor_id (srt): ID of the actor
            frame (int): Frame checked
        """

        frame_state = self._states[frame]
        if actor_id in frame_state:

            transform = frame_state[actor_id]["transform"]
            return transform
        return None

    def get_actor_transforms(self, actor_id, ini_frame=None, end_frame=None):
        """
        Returns a carla.Transform list with all the actor's transform of a given actor
        throughout the simulation.

        Args:
            actor_id (srt): ID of the actor
            ini_frame (int): First frame checked. By default, 0 
            end_frame (int): Last frame checked. By default, max number of frames.
        """
        if ini_frame is None:
            ini_frame = 0
        if end_frame is None:
            end_frame = len(self._states) - 1

        transforms = []

        for frame_number in range(ini_frame, end_frame + 1):

            transform = self.get_actor_transform(actor_id, frame_number)
            transforms.append(transform)

        return transforms

    def get_actor_location(self, actor_id, frame):
        """
        Returns a carla.Location with an actor's location at a given frame,
        or None if the actor wasn't alive at that frame

        Args:
            actor_id (srt): ID of the actor
            frame (int): Frame checked
        """
        transform = self.get_actor_transform(actor_id, frame)
        if transform:
            return transform.location
        return None

    def get_actor_locations(self, actor_id, ini_frame=None, end_frame=None):
        """
        Returns a carla.Location list with all the actor's location of a given actor
        throughout the simulation.

        Args:
            actor_id (srt): ID of the actor
            ini_frame (int): First frame checked. By default, 0 
            end_frame (int): Last frame checked. By default, max number of frames.
        """
        if ini_frame is None:
            ini_frame = 0
        if end_frame is None:
            end_frame = len(self._states) - 1

        locations = []

        for frame_number in range(ini_frame, end_frame + 1):

            location = self.get_actor_location(actor_id, frame_number)
            locations.append(location)

        return locations

    def get_vehicle_control(self, actor_id, frame):
        """
        Returns a carla.VehicleControl with an actor's control at a given frame

        Args:
            actor_id (srt): ID of the actor
            frame (int): Frame checked
        """
        frame_state = self._states[frame]
        if actor_id in frame_state:
    
            if "control" not in frame_state[actor_id]:
                print("WARNING: Can't find actor's {} control".format(actor_id))
                return None

            control = frame_state[actor_id]["control"]
            return control
        return None

    def get_vehicle_controls(self, actor_id, ini_frame=None, end_frame=None):
        """
        Returns a carla.VehicleControl list with all the vehicle's control of a given vehicle
        throughout the simulation.

        Args:
            actor_id (srt): ID of the actor
            ini_frame (int): First frame checked. By default, 0 
            end_frame (int): Last frame checked. By default, max number of frames.
        """
        if ini_frame is None:
            ini_frame = 0
        if end_frame is None:
            end_frame = len(self._states) - 1

        controls = []

        for frame_number in range(ini_frame, end_frame + 1):

            control = self.get_vehicle_control(actor_id, frame_number)
            controls.append(control)

        return controls

    def get_walker_speed(self, actor_id, frame):
        """
        Returns a float with a walkers's speed at a given frame

        Args:
            actor_id (srt): ID of the actor
            frame (int): Frame checked
        """

        frame_state = self._states[frame]
        if actor_id in frame_state:

            if "speed" not in frame_state[actor_id]:
                print("WARNING: Can't find actor's {} speed".format(actor_id))
                return None

            speed = frame_state[actor_id]["speed"]
            return speed
        return None

    def get_walker_speeds(self, actor_id, ini_frame=None, end_frame=None):
        """
        Returns a list of floats with all the speeds of a given walker
        throughout the simulation.

        Args:
            actor_id (srt): ID of the actor
            ini_frame (int): First frame checked. By default, 0 
            end_frame (int): Last frame checked. By default, max number of frames.
        """
        if ini_frame is None:
            ini_frame = 0
        if end_frame is None:
            end_frame = len(self._states) - 1

        speeds = []

        for frame_number in range(ini_frame, end_frame + 1):

            speed = self.get_walker_speed(actor_id, frame_number)
            speeds.append(speed)

        return speeds

    def get_vehicle_velocity(self, actor_id, frame):
        """
        Returns a carla.Vector3D with a vehicle's speed at a given frame

        Args:
            actor_id (srt): ID of the actor
            frame (int): Frame checked
        """
        frame_state = self._states[frame]
        if actor_id in frame_state:

            if "velocity" not in frame_state[actor_id]:
                print("WARNING: Can't find actor's {} velocity".format(actor_id))
                return None

            velocity = frame_state[actor_id]["velocity"]
            return velocity
        return None

    def get_vehicle_velocities(self, actor_id, ini_frame=None, end_frame=None):
        """
        Returns a list of carla.Vector3D with all the velocities of a given vehicle
        throughout the simulation.

        Args:
            actor_id (srt): ID of the actor
            ini_frame (int): First frame checked. By default, 0 
            end_frame (int): Last frame checked. By default, max number of frames.
        """
        if ini_frame is None:
            ini_frame = 0
        if end_frame is None:
            end_frame = len(self._states) - 1

        velocities = []

        for frame_number in range(ini_frame, end_frame + 1):

            velocity = self.get_vehicle_velocity(actor_id, frame_number)
            velocities.append(velocity)

        return velocities

    def get_traffic_light_state(self, actor_id, frame):
        """
        Returns a carla.TrafficLightState with a traffic lights's state at a given frame

        Args:
            actor_id (srt): ID of the actor
            frame (int): Frame checked
        """
        frame_state = self._states[frame]
        if actor_id in frame_state:

            if "state" not in frame_state[actor_id]:
                print("WARNING: Can't find actor's {} state".format(actor_id))
                return None

            velocity = frame_state[actor_id]["state"]
            return velocity
        return None

    def get_traffic_light_states(self, actor_id, ini_frame=None, end_frame=None):
        """
        Returns a list of carla.TrafficLightState with all the states of a given traffic light
        throughout the simulation.

        Args:
            actor_id (srt): ID of the actor
            ini_frame (int): First frame checked. By default, 0 
            end_frame (int): Last frame checked. By default, max number of frames.
        """
        if ini_frame is None:
            ini_frame = 0
        if end_frame is None:
            end_frame = len(self._states) - 1

        states = []

        for frame_number in range(ini_frame, end_frame + 1):

            state = self.get_traffic_light_state(actor_id, frame_number)
            velocities.append(states)

        return velocities

    def is_traffic_light_frozen(self, actor_id, frame):
        """
        Returns a bool checking wether or not the traffic light is frozen

        Args:
            actor_id (srt): ID of the actor
            frame (int): Frame checked
        """
        frame_state = self._states[frame]
        if actor_id in frame_state:

            if "frozen" not in frame_state[actor_id]:
                print("WARNING: Can't find actor's {} frozen attribute".format(actor_id))
                return None

            frozen = frame_state[actor_id]["state"]
            return frozen
        return None

    def get_traffic_light_elapsed_time(self, actor_id, frame):
        """
        Returns a float with a traffic light's elapsed time at a given frame

        Args:
            actor_id (srt): ID of the actor
            frame (int): Frame checked
        """
        frame_state = self._states[frame]
        if actor_id in frame_state:

            if "elapsed_time" not in frame_state[actor_id]:
                print("WARNING: Can't find actor's {} elapsed_time".format(actor_id))
                return None

            elapsed_time = frame_state[actor_id]["elapsed_time"]
            return frozen
        return None

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
    
    def get_actor_attributes(self, actor_id):
        """
        Returns an int with the actor id of the ego vehicle. This is done by checking the "hero" rolename
        """
        if actor_id in self._actors:
            return self._actors[actor_id]

        return None

    def get_ego_vehicle_id(self):
        """
        Returns a string with the id of the ego vehicle.
        """
        return get_actor_id_with_role_name("hero")

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

        return self._criteria[name]

    def get_total_frame_count(self):
        """
        Returns an int with the actor id of the ego vehicle. This is done by checking the "hero" rolename
        """

        return len(self._states)
