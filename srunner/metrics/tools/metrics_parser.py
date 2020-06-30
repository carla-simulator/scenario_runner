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
    def get_map_name(recorder_info):
        """
        Returns the name of the map the simualtion took place in

        Args:
            recorder_info (str): log created by the recorder
        """

        header = recorder_info.split("\n")
        sim_map = header[1][5:]

        return sim_map


    @staticmethod
    def parse_recorder_info(recorder_info, world):
        """
        Parses recorder_info into readable information.

        Args:
            recorder_info (str): string taken from the
                client.show_recorder_file_info() function.
        """

        actors_info = {}
        simulation_info = []

        # Divide it into frames
        recorder_list = recorder_info.split("Frame")
        header = recorder_list[0].split("\n")
        sim_map = header[1][5:]
        sim_date = header[2][6:]

        annex = recorder_list[-1].split("\n")
        sim_frames = int(annex[0][3:])
        sim_duration = float(annex[1][10:-8])

        simulation_info.append({
            "map": sim_map,
            "date:": sim_date,
            "total_frames": sim_frames,
            "duration": sim_duration
        })
        recorder_list = recorder_list[1:-1]

        for frame in recorder_list:

            # Divide the frame in lines
            frame_list = frame.split("\n")

            # Get the general frame information
            frame_info = frame_list[0].split(" ")
            frame_number = int(frame_info[1])
            frame_time = float(frame_info[3])

            # Variable to store all the information about the frame
            frame_state = {
                "elapsed_time": frame_time,
                "collisions": {},
                "actors": {}
            }

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
                    if "traffic" in elements[2]:  # Gets their instance
                        actor = parse_traffic_actor(elements, world)
                    else:
                        actor = parse_actor(elements)
                    actors_info.update({actor_id: actor})
                    actors_info[actor_id].update({"created": frame_number})
                else:
                    # Get the elements of the row
                    frame_row = frame_row[2:]
                    attributes = frame_row.split(' = ')

                    # Save them to the dictionary
                    actors_info[actor_id].update({attributes[0]: attributes[1]})

                # Advance one row
                i += 1
                frame_row = frame_list[i]

            while frame_row.startswith(' Destroy'):

                # Get the elements of the row
                frame_row = frame_row[1:]
                elements = frame_row.split(" ")

                # Save them to the dictionary
                actor_id = int(elements[1])
                actors_info[actor_id].update({"destroyed": frame_number})

                # Advance one row
                i += 1
                frame_row = frame_list[i]

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
                    frame_state["actors"].update({transform_id: {"transform": transform}})

                    # Get the velocity
                    prev_frame = frame_number - 1
                    if prev_frame > 0:
                        prev_transform = simulation_info[prev_frame]["actors"][transform_id]["transform"]
                        prev_frame_time = simulation_info[prev_frame]["elapsed_time"]
                    else:
                        prev_transform = None
                        prev_frame_time = None
                    velocity = parse_velocity(transform, prev_transform, frame_time, prev_frame_time)
                    frame_state["actors"][transform_id].update({"velocity": velocity})

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
                    frame_state["actors"][traffic_light_id] = traffic_light

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
                    frame_state["actors"][control_id]["control"] = control

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
                    frame_state["actors"][walker_id]["speed"] = elements[4]

                # Advance one row
                i += 1
                frame_row = frame_list[i]

            simulation_info.append(frame_state)

        return [actors_info, simulation_info]
