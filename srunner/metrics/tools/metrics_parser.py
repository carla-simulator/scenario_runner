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

def parse_vehicle_lights(info):
    """
    Parses a list into a carla.VehicleLightState

    Args:
        info (list): list corresponding to a row of the recorder
    """
    srt_to_vlight = {
        "Position": carla.VehicleLightState.Position,
        "Low Beam": carla.VehicleLightState.LowBeam,
        "Hight Beam": carla.VehicleLightState.HighBeam,
        "Brake": carla.VehicleLightState.Brake,
        "Right Blinker": carla.VehicleLightState.RightBlinker,
        "Left Blinker": carla.VehicleLightState.LeftBlinker,
        "Reverse": carla.VehicleLightState.Reverse,
        "Fog": carla.VehicleLightState.Fog,
        "Interior": carla.VehicleLightState.Interior,
        "Special1": carla.VehicleLightState.Special1,
        "Special2": carla.VehicleLightState.Special2,
    }

    first_light_list = info[0].split(" ")[5:]
    first_light_str = " ".join(first_light_list)
    lights = [carla.VehicleLightState(srt_to_vlight[first_light_str])]

    for i in range (1, len(info)):
        lights.append(srt_to_vlight[info[i]])

    return lights

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

def parse_velocity(info):
    """
    Parses a list into a carla.Vector3D with the velocity

    Args:
        info (list): list corresponding to a row of the recorder
    """
    velocity = carla.Vector3D(
        float(info[5][1:-1]),
        float(info[6][:-1]),
        float(info[7][:-1])
    )

    return velocity

def parse_angular_velocity(info):
    """
    Parses a list into a carla.Vector3D with the angular velocity

    Args:
        info (list): list corresponding to a row of the recorder
    """
    velocity = carla.Vector3D(
        float(info[10][1:-1]),
        float(info[11][:-1]),
        float(info[12][:-1])
    )

    return velocity


def parse_scene_lights(info):
    """
    Parses a list into a carla.VehicleLightState

    Args:
        info (list): list corresponding to a row of the recorder
    """
    str_to_bool = {
        "enabled": True,
        "disabled": False
    }

    red = int(float(info[-3][1:]) * 255)
    green = int(float(info[-2]) * 255)
    blue = int(float(info[-1][:-1]) * 255)

    scene_light = {
        "enabled": str_to_bool[info[3][:-1]],
        "intensity": int(info[5][:-1]),
        "color": carla.Color(red, green, blue)
    }
    return scene_light


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

            try:
                prev_frame = frames_info[frame_number - 2]
                prev_time = prev_frame["frame"]["elapsed_time"]
                delta_time = round(frame_time - prev_time, 6)
            except IndexError:
                delta_time = 0

            # Variable to store all the information about the frame
            frame_state = {
                "frame": {
                    "elapsed_time": frame_time,
                    "delta_time": delta_time,
                    "platform_time": None
                },
                "actors": {},
                "scene_lights": {}
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

                actor_id = int(elements[4])
                other_id = int(elements[-1])

                if actor_id in simulation_info["collisions"]:
                    # Add it to the collisions list
                    simulation_info["collisions"][actor_id].update({frame_number: other_id})
                else:
                    # Create the collisions list (list of dictionaries)
                    simulation_info["collisions"].update({actor_id: {frame_number: other_id}})

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

                    actor_id = int(elements[1])
                    transform = parse_transform(elements)
                    frame_state["actors"].update({actor_id: {"transform": transform}})

                i += 1
                frame_row = frame_list[i]

            while frame_row.startswith(' State traffic lights') or frame_row.startswith('  '):

                if frame_row.startswith('  '):
                    frame_row = frame_row[2:]
                    elements = frame_row.split(" ")

                    actor_id = int(elements[1])
                    traffic_light = parse_traffic_light(elements)
                    frame_state["actors"].update({actor_id: traffic_light})

                i += 1
                frame_row = frame_list[i]

            while frame_row.startswith(' Vehicle animations') or frame_row.startswith('  '):

                if frame_row.startswith('  '):
                    frame_row = frame_row[2:]
                    elements = frame_row.split(" ")

                    actor_id = int(elements[2][:-1])
                    control = parse_control(elements)
                    frame_state["actors"][actor_id].update({"control": control})

                i += 1
                frame_row = frame_list[i]

            while frame_row.startswith(' Walker animations') or frame_row.startswith('  '):

                if frame_row.startswith('  '):
                    frame_row = frame_row[2:]
                    elements = frame_row.split(" ")

                    actor_id = int(elements[2][:-1])
                    frame_state["actors"][actor_id].update({"speed": elements[4]})

                i += 1
                frame_row = frame_list[i]

            while frame_row.startswith(' Vehicle light animations') or frame_row.startswith('  '):

                if frame_row.startswith('  '):
                    frame_row = frame_row[2:]
                    elements = frame_row.split(" ")

                    actor_id = int(elements[2][:-1])
                    if elements[3] == "no":
                        lights = [carla.VehicleLightState.NONE]
                    else:
                        elements_2 = frame_row.split(", ")
                        lights = parse_vehicle_lights(elements_2)
                    frame_state["actors"][actor_id].update({"lights": lights})


                i += 1
                frame_row = frame_list[i]

            while frame_row.startswith(' Scene light changes') or frame_row.startswith('  '):

                if frame_row.startswith('  '):
                    frame_row = frame_row[2:]
                    elements = frame_row.split(" ")

                    light_id = int(elements[2][:-1])
                    scene_light = parse_scene_lights(elements)
                    frame_state["scene_lights"].update({light_id: scene_light})

                i += 1
                frame_row = frame_list[i]

            while frame_row.startswith(' Dynamic actors') or frame_row.startswith('  '):

                if frame_row.startswith('  '):
                    frame_row = frame_row[2:]
                    elements = frame_row.split(" ")

                    velocity_id = int(elements[2][:-1])
                    velocity = parse_velocity(elements)
                    frame_state["actors"][velocity_id].update({"velocity": velocity})
                    angular_v = parse_angular_velocity(elements)
                    frame_state["actors"][velocity_id].update({"angular_velocity": angular_v})

                    if delta_time == 0:
                        acceleration = carla.Vector3D(0, 0, 0)
                    else:
                        prev_velocity = frame_state["actors"][velocity_id]["velocity"]
                        acceleration = (velocity - prev_velocity) / delta_time

                    frame_state["actors"][velocity_id].update({"acceleration": acceleration})

                i += 1
                frame_row = frame_list[i]

            while frame_row.startswith(' Actor bounding boxes') or frame_row.startswith('  '):

                i += 1
                frame_row = frame_list[i]

            while frame_row.startswith(' Current platform time'):

                frame_row = frame_row[1:]
                elements = frame_row.split(" ")

                platform_time = float(elements[-1])
                frame_state["frame"]["platform_time"] = platform_time

                i += 1
                frame_row = frame_list[i]

            frames_info.append(frame_state)

        return simulation_info, actors_info, frames_info
