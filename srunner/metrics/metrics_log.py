import carla
import os





class MetricsLog(object):
    """
    Utility class to query the metrics log.
    
    The information of the log should be accesed through the functions,
    but the dictionaries are public in case the users wants to use them.
    
    If doing so, take into account that some information are dictionaries
    instead of "carla classes":
        - transforms
        - vehicle control
        - traffic light states
        - vehicle velocities
    """
    
    number_to_state = {
        "0": carla.TrafficLightState.Red,
        "1": carla.TrafficLightState.Yellow,
        "2": carla.TrafficLightState.Green,
        "3": carla.TrafficLightState.Off,
        "4": carla.TrafficLightState.Unknown,
    }

    def __init__(self, recorder, criteria):
        """
        Initializes the log class and parses it to extract the disctionaries
        """
        self.actors = recorder[0]
        self.states = recorder[1]
        self.criteria = criteria

    def get_actor_location(self, actor_id, frame):
        """
        Returns a carla.Location with an actor's location at a given frame
        """
        actor_id = str(actor_id)

        frame_state = self.states[frame]
        if actor_id in frame_state:
            transform = frame_state[actor_id]["transform"]
            location = carla.Location(
                transform["x"],
                transform["y"],
                transform["z"],
            )

            return location

        return None

    # def get_all_actor_locations(self, actor_id):
    #     """
    #     Returns a carla.Location list with all the actor's location of a given actor
    #     througout the entire simulation
    #     """
    #     actor_id = str(actor_id)
    #     locations = []

    #     for i, frame in enumerate(self.states):

    #         if actor_id in frame:
    #             transform = frame[actor_id]["transform"]
    #             location = carla.Location(
    #                 transform["x"],
    #                 transform["y"],
    #                 transform["z"],
    #             )
    #             locations.append({i: location})

    #     return locations

    def get_actor_transform(self, actor_id, frame):
        """
        Returns a carla.Transform with an actor's transform at a given frame
        """
        transform = self.states[frame][actor_id]["transform"]
        location = carla.Location(
            transform["x"],
            transform["y"],
            transform["z"],
        )
        rotation = carla.Rotation(
            transform["pitch"],
            transform["yaw"],
            transform["roll"],
        )

        return carla.Transform(location,rotation)

    # def get_all_actor_transforms(self, actor_id):
    #     """
    #     Returns a carla.Location with an actor's location at a given frame
    #     """
    #     transforms = []

    #     for frame in self.states:

    #         transform = self.get_actor_transform(actor_id, frame)
    #         transforms.append([frame, transform])

    #     return transforms

    def get_vehicle_control(self, actor_id, frame):
        """
        Returns a carla.VehicleControl with an actor's control at a given frame
        """
        control_info = self.states[frame][actor_id]["control"]
        control = carla.VehicleControl(
            control_info["throttle"],
            control_info["steer"],
            control_info["brake"],
            control_info["hand_brake"],
            control_info["reverse"],
            False,
            control_info["gear"],
        )

        return location

    def get_walker_velocity(self, actor_id, frame):
        """
        Returns a float with a walker's speed at a given frame
        """
        return self.states[frame][actor_id]["control"]["speed"]

    def get_vehicle_speed(self, actor_id, frame):
        """
        Returns a float with an actor's speed at a given frame
        """

        velocity_info = self.states[frame][actor_id]["velocity"]
        velocity = carla.Vector3D(
            velocity_info["x"],
            velocity_info["y"],
            velocity_info["z"]
        )
        return velocity

    def get_actor_id_with_role_name(self, rolename):
        """
        Returns an int with the actor id of the ego vehicle. This is done by checking the "hero" rolename
        """
        for actor_id in self.actors:
            if "role_name" in self.actors[actor_id] and self.actors[actor_id]["role_name"] == rolename:
                return actor_id

    def get_criteria(self, name):
        """
        Returns an int with the actor id of the ego vehicle. This is done by checking the "hero" rolename
        """

        return self.criteria[name]

    def get_simulation_frame_count(self):
        """
        Returns an int with the actor id of the ego vehicle. This is done by checking the "hero" rolename
        """

        return len(self.states)
