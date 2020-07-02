#!/usr/bin/env python

# Copyright (c) 2020 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Support class of the MetricsManager to parse the information of
the CARLA recorder into a readable dictionary
"""

import pprint
import carla

def parse_actor(info):
    """
    Returns a dictionary with the basic actor information

    Args:
        info (list): list corresponding to a row of the recorder
    """

    actor = {
        "type_id": info[2],
        "location": carla.Location(
            float(info[5][1:-1]) / 100,
            float(info[6][:-1]) / 100,
            float(info[7][:-1]) / 100
        )
    }

    return actor

def parse_traffic_actor(info, world):
    """
    Returns an instante of a carla.TrafficSign / carla.TrafficLight

    Args:
        info (list): list corresponding to a row of the recorder
    """

    type_id = info[2]
    location = carla.Location(
        float(info[5][1:-1]) / 100,
        float(info[6][:-1]) / 100,
        float(info[7][:-1]) / 100
    )

    traffic_actors = world.get_actors().filter("traffic.*")

    for actor in traffic_actors:
        actor_location = actor.get_transform().location
        distance = actor_location.distance(location)

        if distance < 0.1:  # Can't use "equal" due to str-float conversion errors
            actor_dict = {
                "type_id": type_id,
                "actor": actor
            }
            return actor_dict

    return None

def parse_transform(info):
    """
    Parses a list into a carla.Transform

    Args:
        info (list): list corresponding to a row of the recorder
    """
    transform = carla.Transform(
        carla.Location(
            float(info[3][1:-1]) / 100,
            float(info[4][:-1]) / 100,
            float(info[5][:-1]) / 100,
        ),
        carla.Rotation(
            float(info[8][:-1]),   # pitch
            float(info[9][:-1]),   # yaw
            float(info[7][1:-1])   # roll
        )
    )

    return transform

def parse_control(info):
    """
    Parses a list into a carla.VehicleControl

    Args:
        info (list): list corresponding to a row of the recorder
    """
    control = carla.VehicleControl(
        float(info[6]),         # throttle
        float(info[4]),         # steer
        float(info[8]),         # brake
        bool(int(info[10])),    # hand_brake
        int(info[12]) < 0,      # reverse
        int(info[12]),          # gear
    )

    return control

def parse_traffic_light(info):
    """
    Parses a list into a dictionary with all the traffic light's information

    Args:
        info (list): list corresponding to a row of the recorder
    """
    number_to_state = {
        "0": carla.TrafficLightState.Red,
        "1": carla.TrafficLightState.Yellow,
        "2": carla.TrafficLightState.Green,
        "3": carla.TrafficLightState.Off,
        "4": carla.TrafficLightState.Unknown,
    }

    traffic_light = {
        "state": number_to_state[info[3]],
        "frozen": bool(int(info[5])),
        "elapsed_time": float(info[7]),
    }

    return traffic_light

def parse_velocity(transform, prev_transform, frame_time, prev_frame_time):
    """
    Parses a list into a dictionary with all the traffic light's information

    Args:
        info (list): list corresponding to a row of the recorder
    """
    if transform is None or prev_transform is None :
        velocity = carla.Vector3D(0, 0, 0)
    else:
        delta_time = frame_time - prev_frame_time
        location = transform.location
        prev_location = prev_transform.location

        velocity = carla.Vector3D(
            (location.x - prev_location.x) / delta_time,
            (location.y - prev_location.y) / delta_time,
            (location.z - prev_location.z) / delta_time
        )

    return velocity


class MetricsParser(object):
    """
    Support class to the MetricsManager to parse the CARLA recorder
    into readable information
    """

    @staticmethod
    def parse_recorder_info(recorder_info):
        """
        Parses recorder_info into readable information.

        Args:
            recorder_info (str): string taken from the
                client.show_recorder_file_info() function.
        """
        pp = pprint.PrettyPrinter(indent=4)

        frame_info = []

        actors_info = {}
        collisions = []

        # Divide it into frames and ignore the first and last part
        recorder_list = recorder_info.split("Frame")
        recorder_list = recorder_list[1:-1]

        for frame in recorder_list:

            # Divide the frame in lines
            frame_list = frame.split("\n")

            # Get the general frame information
            frame_info = frame_list[0].split(" ")
            frame_number = int(frame_info[1])
            frame_time = float(frame_info[3])

            # Variable to store all the information about the frame
            frames_info.append({
                "elapsed_time": frame_time,
                "delta_time": None,
                "platform_time": None
            })

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
                    actor_type = elements[2]

                    if "spectator" in actor_type or "static.prop" == actor_type:
                        # Ignore the spectator and static actors that we haven't spawned
                        pass
                    else:
                        if "traffic" in actor_type:
                            if "traffic_light" in actor_type:
                                actor = FakeTrafficLight(actor_id, actor_type, frame_number)
                            else:
                                actor = FakeTrafficSign(actor_id, actor_type, frame_number)
                        elif "vehicle" in actor_type:
                            actor = FakeVehicle(actor_id, actor_type, frame_number)
                        elif "walker" in actor_type:
                            actor = FakeWalker(actor_id, actor_type, frame_number)
                        else:
                            actor = FakeActor(actor_id, actor_type, frame_number)

                        actors_info.update({actor_id: actor})

                else:
                    # Get the elements of the row
                    frame_row = frame_row[2:]
                    attributes = frame_row.split(' = ')

                    # Save them to the dictionary
                    actors_info[actor_id].add_attribute(attributes[0], attributes[1])

                # Advance one row
                i += 1
                frame_row = frame_list[i]

            while frame_row.startswith(' Destroy'):

                # Get the elements of the row
                frame_row = frame_row[1:]
                elements = frame_row.split(" ")

                # Save them to the dictionary
                actor_id = int(elements[1])
                actors_info[actor_id].alive_frames[-1] = frame_number

                # Advance one row
                i += 1
                frame_row = frame_list[i]

            # TODO: Collision
            while frame_row.startswith(' Collision'):
                # Get the elements of the row
                frame_row = frame_row[1:]
                elements = frame_row.split(" ")
                collision_id = int(elements[4])
                other_id = int(elements[4])
                frame_state["collisions"].update({collision_id: other_id})

                # Advance one row
                i += 1
                frame_row = frame_list[i]

            while frame_row.startswith(' Parenting'):

                # Get the elements of the row
                frame_row = frame_row[1:]
                elements = frame_row.split(" ")

                # Save them to the dictionary
                actor_id = int(elements[1])
                parent_id = int(elements[3])
                actors_info[actor_id].parent = parent_id

                # Advance one row
                i += 1
                frame_row = frame_list[i]

            # print(frame_number)
            while frame_row.startswith(' Positions') or frame_row.startswith('  '):

                if frame_row.startswith('  '):
                    # Get the elements of the row
                    frame_row = frame_row[2:]
                    elements = frame_row.split(" ")

                    # Save them to the dictionary
                    transform_id = int(elements[1])
                    transform = parse_transform(elements)
                    actors_info[transform_id].add_transform(transform)

                    #TODO: get velocity
                    # actors_info[transform_id].add_velocity(velocity)
                    #TODO: get acceleration
                    # actors_info[transform_id].add_acceleration(acceleration)

                # Advance one row
                i += 1
                frame_row = frame_list[i]

            while frame_row.startswith(' State traffic lights') or frame_row.startswith('  '):

                if frame_row.startswith('  '):
                    # Get the elements of the row
                    frame_row = frame_row[2:]
                    elements = frame_row.split(" ")

                    # Save them to the dictionary
                    traffic_light_id = int(elements[1])
                    traffic_light = parse_traffic_light(elements)
                    actors_info[traffic_light_id].add_states(traffic_light)

                # Advance one row
                i += 1
                frame_row = frame_list[i]

            while frame_row.startswith(' Vehicle animations') or frame_row.startswith('  '):

                if frame_row.startswith('  '):
                    # Get the elements of the row
                    frame_row = frame_row[2:]
                    elements = frame_row.split(" ")

                    # Save them to the dictionary
                    control_id = int(elements[2][:-1])
                    control = parse_control(elements)
                    actors_info[control_id].add_control(control)

                # Advance one row
                i += 1
                frame_row = frame_list[i]

            while frame_row.startswith(' Walker animations') or frame_row.startswith('  '):

                if frame_row.startswith('  '):
                    # Get the elements of the row
                    frame_row = frame_row[2:]
                    elements = frame_row.split(" ")

                    # Save them to the dictionary
                    walker_id = int(elements[2][:-1])
                    actors_info[walker_id].add_control(elements[4])

                # Advance one row
                i += 1
                frame_row = frame_list[i]

        actors_list = [actors_info[x] for x in actors_info]

        return [frames_info, actors_list]


class FakeActor(object):
    """
    This is a "carla.Actor" instance with all the information about a specific actor
    during the simulation
    """

    def __init__(self, actor_id, type_id, frame):

        self.attributes = {}
        self.id = actor_id
        self.type_id = type_id
        self.parent = None
        self.alive_frames = [frame, None]
        self._transforms = []
        self._velocity = []
        self._angular_velocity = []
        self._acceleration = []
        self._impulse = []
        self._angular_impulse = []

    def add_attribute(self, name, value):

        self.attributes.update({name: value})

    def add_transform(self, value):

        self._transforms.append(value)

    def add_velocity(self, value):

        self._velocity.append(value)

    def add_acceleration(self, value):

        self._acceleration.append(value)

    def get_transform_at_frame(self, frame):

        if frame < self.alive_frames[0]:
            return None
        elif self.alive_frames[1] and self.alive_frames[1] > frame:
            return None
        else:
            return self._transforms[frame - self.alive_frames[0]]


class FakeVehicle(FakeActor):

    def __init__(self, actor_id, type_id, frame):

        super(FakeVehicle, self).__init__(actor_id, type_id, frame)
        self.bounding_box = None
        self._controls = []
        self._light_states = []

    def add_control(self, value):

        self._controls.append(value)


class FakeTrafficSign(FakeActor):

    def __init__(self, actor_id, type_id, frame):

        super(FakeTrafficSign, self).__init__(actor_id, type_id, frame)
        self.trigger_volume = None

    def add_trigger_volume(self, value):
        pass
        # TODO: add the trigger volume

class FakeTrafficLight(FakeTrafficSign):

    def __init__(self, actor_id, type_id, frame):

        super(FakeTrafficLight, self).__init__(actor_id, type_id, frame)
        self._states = []
        self._frozens = []
        self._elapsed_times = []

    def add_states(self, value):

        self._states.append(value["state"])
        self._frozens.append(value["frozen"])
        self._elapsed_times.append(value["elapsed_time"])


class FakeWalker(FakeActor):

    def __init__(self, actor_id, type_id, frame):

        super(FakeTrafficLight, self).__init__(actor_id, type_id, frame)
        self._controls = []

    def add_control(self):

        self._controls.append(value)