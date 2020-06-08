import carla
import os
import pprint
import json


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


class Metrics(object):
    """
    Support class for the users to easily create their own metrics.
    The user should only modify the create_metrics function, returning
    "something in particular"
    """

    def __init__(self, log_location, criterias, open_drive=None):
        """
        Initialization of the metrics class.

        Args:
            log_location (str): name of the log file
        """
        self.pp = pprint.PrettyPrinter(indent=4)

        recorder_info = client.show_recorder_file_info(log_location, True)
        log = Log(recorder_info, criterias, open_drive)

        metrics = self.create_metrics(log)

        self.write_to_terminal(metrics)

        self.write_to_json(metrics)

    def create_metrics(self, metrics):
        """
        Implementation of the metrics
        """
        ego_id = metrics.get_ego_vehicle_id()
        metrics.get_vehicle_speed(ego_id)

        return metrics

    def write_to_terminal(self, metrics):
        """
        Print the metrics table through the terminal
        """
        # self.pp.pprint(metrics.states)
        # self.pp.pprint(metrics.actors)


    def write_to_json(self, metrics):
        """
        Writes the metrics into a json file
        """

        with open('data.json', 'w') as fp:
            json.dump(metrics.states, fp, sort_keys=True, indent=4)
            json.dump(metrics.actors, fp, sort_keys=True, indent=4)


class Log(object):
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

    def __init__(self, location, criterias, open_drive):
        """
        Initializes the log class and parses it to extract the disctionaries
        """
        self._location = location
        log_string = client.show_recorder_file_info(location, True)
        self.parse_log(log_string)
        self.criterias = criterias
        self.open_drive = open_drive

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
        pass

    def get_actor_transform(self, actor_id, frame):
        """
        Returns a carla.Transform with an actor's transform at a given frame
        """
        pass

    def get_vehicle_control(self, actor_id, frame):
        """
        Returns a carla.VehicleControl with an actor's control at a given frame
        """
        pass

    def get_walker_velocity(self, actor_id, frame):
        """
        Returns a float with a walker's speed at a given frame
        """
        pass

    def get_vehicle_speed(self, actor_id):
        """
        Returns a float with an actor's speed at a given frame
        """
        for frame in self.states:
            velocity = frame[str(actor_id)]["velocity"]
            print("x: {} -- y: {} -- z: {}".format(velocity["x"], velocity["y"], velocity["z"]))

        pass

    def get_ego_vehicle_id(self):
        """
        Returns an int with an actor's speed at a given frame
        """
        for actor_id in self.actors:
            if "role_name" in self.actors[actor_id] and self.actors[actor_id]["role_name"] == "hero":
                return actor_id

        return None



client = carla.Client('127.0.0.1', 2000)

location = "{}/{}.log".format(os.getenv('SCENARIO_RUNNER_ROOT', "./"), 'Frozen')

log = Metrics(location, None)