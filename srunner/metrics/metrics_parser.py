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


class MetricsParser(object):

    @staticmethod
    def parse_recorder_info(recorder_file):
        """
        Parsing the recorder into readable information.
        
            - self.actors: a dictionary of ID's with all the information
                related to the actors of the simulation.
            - self.states: a dictionary of frame dictionaries with the
                information of the simulation at that frame
        """
        actors = {}
        states = []

        # Divide it into frames
        recorder_list = recorder_file.split("Frame")
        recorder_list = recorder_list[1:-2]

        for frame in recorder_list:

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
                    actors[actor_id] = actor
                    actors[actor_id]["created"] = frame_number
                else:
                    # Get the elements of the row
                    frame_row = frame_row[2:]
                    attributes = frame_row.split(' = ')

                    # Save them to the dictionary
                    actors[actor_id][attributes[0]] = attributes[1]

                # Advance one row
                i += 1
                frame_row = frame_list[i]

            while frame_row.startswith(' Destroy'):

                # Get the elements of the row
                frame_row = frame_row[1:]
                elements = frame_row.split(" ")

                # Save them to the dictionary
                actor_id = int(elements[1])
                actors[actor_id]["destroyed"] = frame_number

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
                        prev_transform = states[-1][transform_id]["transform"]
                        prev_frame_time = states[-1]["elapsed_time"]
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

            states.append(frame_state)
        
        return [actors, states]
