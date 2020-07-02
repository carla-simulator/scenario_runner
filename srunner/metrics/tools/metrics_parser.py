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
    Class used to parse the CARLA recorder into readable information
    """

    @staticmethod
    def parse_recorder_info(recorder_info):
        """
        Parses the recorder into readable information.

        Args:
            recorder_info (str): string given by the recorder
        """

        # Divide it into frames
        recorder_list = recorder_info.split("Frame")

        # Get general information
        header = recorder_list[0].split("\n")
        sim_map = header[1][5:]
        sim_date = header[2][6:]

        annex = recorder_list[-1].split("\n")
        sim_frames = int(annex[0][3:])
        sim_duration = float(annex[1][10:-8])

        recorder_list = recorder_list[1:-1]

        simulation_info = {
            "map": sim_map,
            "date:": sim_date,
            "total_frames": sim_frames,
            "duration": sim_duration,
            "collisions": {}
        }

        actors_info = {}
        frames_info = []

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
                "actors": {}
            }

            # Loop through all the other rows.
            i = 1
            frame_row = frame_list[i]

            while frame_row.startswith(' Create') or frame_row.startswith('  '):

                if frame_row.startswith(' Create'):
                    frame_row = frame_row[1:]
                    elements = frame_row.split(" ")

                    actor_id = int(elements[1][:-1])
                    actor_type = elements[2]

                    if "spectator" not in actor_type and "static.prop" != actor_type:
                        # Ignore the spectator and the static elements not spawned by the user
                        actor = parse_actor(elements)
                        actors_info.update({actor_id: actor})
                        actors_info[actor_id].update({"created": frame_number})
                else:
                    frame_row = frame_row[2:]
                    attributes = frame_row.split(' = ')

                    actors_info[actor_id].update({attributes[0]: attributes[1]})

                i += 1
                frame_row = frame_list[i]

            while frame_row.startswith(' Destroy'):

                frame_row = frame_row[1:]
                elements = frame_row.split(" ")

                actor_id = int(elements[1])
                actors_info[actor_id].update({"destroyed": frame_number})

                i += 1
                frame_row = frame_list[i]

            while frame_row.startswith(' Collision'):

                frame_row = frame_row[1:]
                elements = frame_row.split(" ")

                collision_actor_id = int(elements[4])
                collision_other_id = int(elements[-1])

                if collision_actor_id in simulation_info["collisions"]:
                    # Add it to the collisions list
                    simulation_info["collisions"][collision_actor_id].update({frame_number: collision_other_id})
                else:
                    # Create the collisions list (list of dictionaries)
                    simulation_info["collisions"].update({collision_actor_id: {frame_number: collision_other_id}})

                i += 1
                frame_row = frame_list[i]

            while frame_row.startswith(' Parenting'):

                frame_row = frame_row[1:]
                elements = frame_row.split(" ")

                actor_id = int(elements[1])
                parent_id = int(elements[3])
                actors_info[actor_id].update({"parent": parent_id})

                i += 1
                frame_row = frame_list[i]

            while frame_row.startswith(' Positions') or frame_row.startswith('  '):

                if frame_row.startswith('  '):
                    frame_row = frame_row[2:]
                    elements = frame_row.split(" ")

                    transform_id = int(elements[1])
                    transform = parse_transform(elements)
                    frame_state["actors"].update({transform_id: {"transform": transform}})

                    prev_frame = frame_number - 1
                    if prev_frame > 1:
                        prev_transform = frames_info[prev_frame - 1]["actors"][transform_id]["transform"]
                        prev_frame_time = frames_info[prev_frame - 1]["elapsed_time"]
                    else:
                        prev_transform = None
                        prev_frame_time = None
                    velocity = parse_velocity(transform, prev_transform, frame_time, prev_frame_time)
                    frame_state["actors"][transform_id].update({"velocity": velocity})

                i += 1
                frame_row = frame_list[i]

            while frame_row.startswith(' State traffic lights') or frame_row.startswith('  '):

                if frame_row.startswith('  '):
                    frame_row = frame_row[2:]
                    elements = frame_row.split(" ")

                    traffic_light_id = int(elements[1])
                    traffic_light = parse_traffic_light(elements)
                    frame_state["actors"][traffic_light_id] = traffic_light

                i += 1
                frame_row = frame_list[i]

            while frame_row.startswith(' Vehicle animations') or frame_row.startswith('  '):

                if frame_row.startswith('  '):
                    frame_row = frame_row[2:]
                    elements = frame_row.split(" ")

                    control_id = int(elements[2][:-1])
                    control = parse_control(elements)
                    frame_state["actors"][control_id]["control"] = control

                i += 1
                frame_row = frame_list[i]

            while frame_row.startswith(' Walker animations') or frame_row.startswith('  '):

                if frame_row.startswith('  '):
                    frame_row = frame_row[2:]
                    elements = frame_row.split(" ")

                    walker_id = int(elements[2][:-1])
                    frame_state["actors"][walker_id]["speed"] = elements[4]

                i += 1
                frame_row = frame_list[i]

            frames_info.append(frame_state)

        return simulation_info, actors_info, frames_info
