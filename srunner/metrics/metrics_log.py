import carla
import os


def parse_actor(info):

    actor = {
        "type_id": info[2],
        "location": {
            "x": float(info[5][1:-1]) / 100,
            "y": float(info[6][:-1]) / 100,
            "z": float(info[7][:-1]) / 100,
        }
    }
    
    return actor

def parse_transform(info):

    transform = {
        "x": float(info[3][1:-1]) / 100,
        "y": float(info[4][:-1]) / 100,
        "z": float(info[5][:-1]) / 100,
        "roll": float(info[7][1:-1]),
        "pitch": float(info[8][:-1]),
        "yaw": float(info[9][:-1]),
    }

    return transform

def parse_control(info):

    control = {
        "throttle": float(info[6]),          # throttle
        "steer": float(info[4]),             # steer
        "brake": float(info[8]),             # brake
        "hand_brake": bool(int(info[10])),    # hand_brake
        "reverse": int(info[12]) < 0,         # reverse
        "gear": int(info[12]),                # gear
    }
    return control

def parse_traffic_light(info):

    traffic_light = {
        "state": info[3],
        "frozen": bool(int(info[5])),
        "elapsed_time": float(info[7]),
    }

    return traffic_light

def parse_velocity(transform, prev_transform, frame_time, prev_frame_time):

    if prev_transform is None:
        velocity = {
        "x": 0,
        "y": 0,
        "z": 0
        }
    else:
        delta_time = frame_time - prev_frame_time
        velocity = {
            "x": (transform["x"] - prev_transform["x"]) / delta_time,
            "y": (transform["y"] - prev_transform["y"]) / delta_time,
            "z": (transform["z"] - prev_transform["z"]) / delta_time
        }

    return velocity


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

    def __init__(self, log_string, criteria):
        """
        Initializes the log class and parses it to extract the disctionaries
        """
        self.parse_log(log_string)
        self.criteria = criteria

    def parse_log(self, log_string):
        """
        Parsing the recorder into readable information.
        
            - self.actors: a dictionary of ID's with all the information
                related to the actors of the simulation.
            - self.states: a dictionary of frame dictionaries with the
                information of the simulation at that frame
        """
        self.actors = {}
        self.states = []

        # Divide it into frames
        log_list = log_string.split("Frame")
        log_list = log_list[1:-2]

        for frame in log_list:

            # Variable to store all the information about the frame
            frame_state = {}

            # Split the frame into lines
            frame_list = frame.split("\n")

            # Get the frame information
            frame_info = frame_list[0].split(" ")
            frame_number = int(frame_info[1])
            frame_time = float(frame_info[3])

            frame_state["elapsed_time"] =  frame_time

            # Loop through all the other rows.
            i = 1
            frame_row = frame_list[i]

            while frame_row.startswith(' Create') or frame_row.startswith('  '):

                if frame_row.startswith(' Create'):
                    # Get the elements of the row
                    frame_row = frame_row[1:]
                    elements = frame_row.split(" ")

                    # Save them to the dictionary
                    actor_id = int(elements[1][:-1])
                    actor = parse_actor(elements)
                    self.actors[actor_id] = actor
                    self.actors[actor_id]["created"] = frame_number
                else:
                    # Get the elements of the row
                    frame_row = frame_row[2:]
                    attributes = frame_row.split(' = ')

                    # Save them to the dictionary
                    self.actors[actor_id][attributes[0]] = attributes[1]

                # Advance one row
                i += 1
                frame_row = frame_list[i]

            while frame_row.startswith(' Destroy'):

                # Get the elements of the row
                frame_row = frame_row[1:]
                elements = frame_row.split(" ")

                # Save them to the dictionary
                actor_id = int(elements[1])
                self.actors[actor_id]["destroyed"] = frame_number

                # Advance one row
                i += 1
                frame_row = frame_list[i]

            while frame_row.startswith(' Parenting'):
                i += 1
                frame_row = frame_list[i]

            while frame_row.startswith(' Positions') or frame_row.startswith('  '):

                if frame_row.startswith('  '):
                    # Get the elements of the row
                    frame_row = frame_row[2:]
                    elements = frame_row.split(" ")

                    # Save them to the dictionary
                    transform_id = elements[1]
                    transform = parse_transform(elements)
                    frame_state[transform_id] = {"transform": transform}

                    # Get the velocity
                    prev_frame = frame_number - 1
                    if prev_frame > 0:
                        prev_transform = self.states[-1][transform_id]["transform"]
                        prev_frame_time = self.states[-1]["elapsed_time"]
                    else:
                        prev_transform = None
                        prev_frame_time = None
                    velocity = parse_velocity(transform, prev_transform, frame_time, prev_frame_time)
                    frame_state[transform_id]["velocity"] = velocity

                # Advance one row
                i += 1
                frame_row = frame_list[i]

            while frame_row.startswith(' State traffic lights') or frame_row.startswith('  '):

                if frame_row.startswith('  '):
                    # Get the elements of the row
                    frame_row = frame_row[2:]
                    elements = frame_row.split(" ")

                    # Save them to the dictionary
                    traffic_light_id = elements[1]
                    traffic_light = parse_traffic_light(elements)
                    frame_state[traffic_light_id] = traffic_light

                # Advance one row
                i += 1
                frame_row = frame_list[i]

            while frame_row.startswith(' Vehicle animations') or frame_row.startswith('  '):

                if frame_row.startswith('  '):
                    # Get the elements of the row
                    frame_row = frame_row[2:]
                    elements = frame_row.split(" ")

                    # Save them to the dictionary
                    control_id = elements[2][:-1]
                    control = parse_control(elements)
                    frame_state[control_id]["control"] = control

                # Advance one row
                i += 1
                frame_row = frame_list[i]

            while frame_row.startswith(' Walker animations') or frame_row.startswith('  '):

                if frame_row.startswith('  '):
                    # Get the elements of the row
                    frame_row = frame_row[2:]
                    elements = frame_row.split(" ")

                    # Save them to the dictionary
                    walker_id = elements[2][:-1]
                    frame_state[walker_id]["speed"] = elements[4]

                # Advance one row
                i += 1
                frame_row = frame_list[i]

            self.states.append(frame_state)

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
