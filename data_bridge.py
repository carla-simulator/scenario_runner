"""
Filename: data_bridge.py
Create Contact between Carla and Lawbreaker
"""
import json
import math
import numpy as np
import shapely.geometry
import carla
from srunner.tools.osc2_helper import OSC2Helper
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

class DataBridge(object):
    """
    Main class of the databridge module.
    """
    def __init__(self, world):
        """
        Initialization of the databridge. This creates the instance, needed to parse
        the information from the Carla World Server and parses the trace log argument into lawbreaker readable information

        Usage:
        data_bridge = DataBridge(self.world)
        data_bridge.update_ego_vehicle_start(self.ego_vehicles[0])
        data_bridge.update_trace()
        data_bridge.end_trace()
        del data_bridge

        Results: trace.json at WorkSpacedir
        """
        self._world = world
        self._map = self._world.get_map()
        self.ego_vehicle = None
        self.trace_list = []
        self._list_traffic_lights = []
        self._last_red_light = None
        
        print("start write trace")
        
        map_name = self._map.name
        self.trace = {'ScenarioName': "scenario0",
                    "MapVariable": "",
                    "map": map_name,
                    "time": {
                        "hour": self.get_hour(),
                        "minute": 0
                    }}

        rain = self._world.get_weather().precipitation
        sunny = self._world.get_weather().cloudiness
        wetness = self._world.get_weather().wetness
        fog = self._world.get_weather().fog_density
        
        self.trace['weather'] = {"rain": rain,
                            "sunny": sunny,
                            "wetness": wetness,
                            "fog": fog}
    
    def update_ego_vehicle_start(self, ego: carla.Vehicle) -> None:
        """
        when scenario loaded, parses init info from ego_vehicle
        """
        self.ego_vehicle = ego
        ego_wp = self._map.get_waypoint(ego.get_location())
    
        self.trace['ego'] = {"ID": "ego_vehicle",
                            "name": ego.type_id,
                            "groundTruthPerception": True,
                            "color": ego.attributes['color'],
                            "start": {
                                "lane_position": {
                                    "lane": "lane_" + str(ego_wp.road_id),
                                    "offset": ego_wp.s,
                                    "roadID": None
                                },
                                "heading": {
                                    "ref_lane_point": {
                                        "lane": "lane_" + str(ego_wp.road_id),
                                        "offset": ego_wp.s,
                                        "roadID": None
                                    },
                                    "ref_angle": 0.0
                                },
                                "speed": 0.0
                            }
                        }

        self.trace["pedestrianList"] = []
        self.trace["obstacleList"] = []
        self.trace["AgentNames"] = []
        self.trace["groundTruthPerception"] = True
        self.trace["testFailures"] = []
        self.trace["testResult"] = "PASS"
        self.trace["minEgoObsDist"] = 100
        self.trace["destinationReached"] = True

    def end_trace(self) -> None:
        """
        update result information, when all scenario end
        """
        ego_wp = self._map.get_waypoint(self.ego_vehicle.get_location()) 
        self.trace['ego']['destination'] =  {
                                "lane_position": {
                                    "lane": "lane_" + str(ego_wp.road_id),
                                    "offset": ego_wp.s,
                                    "roadID": None
                                },
                                "heading": {
                                    "ref_lane_point": {
                                        "lane": "lane_" + str(ego_wp.road_id),
                                        "offset": ego_wp.s,
                                        "roadID": None
                                    },
                                    "ref_angle": 0.0
                                },
                                "speed": 0.0
                            }
        
        self.trace['trace'] = self.trace_list
        
        with open('trace.json', 'w', encoding='utf-8') as fw:
            json.dump(self.trace, fw, sort_keys=False, indent=4)
            
    def update_trace(self) -> None:
        """
        Parses information per every Carla world tick
        """
        timestamp = self._world.get_snapshot().timestamp.elapsed_seconds * 100
        
        frame_dict = {}
        frame_dict['timestamp'] = int(timestamp)
        
        ego_transform = self.ego_vehicle.get_transform()
        qx, qy, qz, qw = OSC2Helper.euler_orientation(ego_transform.rotation)
        ego_velocity = self.ego_vehicle.get_velocity()
        ego_speed = self.convert_velocity_to_speed(ego_velocity)
        ego_angular_velocity = self.ego_vehicle.get_angular_velocity()
        ego_linear_acceleration = self.ego_vehicle.get_acceleration()
        ego_control = self.ego_vehicle.get_control()
        
        is_ego_engine_on = False
        if float(ego_control.throttle) >0:
            is_ego_engine_on = True
        
        # 获取当前帧的waypoint
        ego_wp = self._map.get_waypoint(ego_transform.location)   
        
        if self.ego_vehicle is not None:
            light_state = self.ego_vehicle.get_light_state()
        else:
            light_state = None
                
        upper_limit = 1000
        if self.ego_vehicle.get_speed_limit() is not None:
            upper_limit = self.ego_vehicle.get_speed_limit()
            
        junction_ahead = self.get_junction_ahead(ego_transform)
        
        plan = CarlaDataProvider.get_local_planner()
        
        i = 2
        turn = 0

        if plan is not None:

            wpt = None

            if len(plan._waypoints_queue)>2:
                wpt = plan._waypoints_queue[i][0]

            if wpt is not None:
            
                while wpt.road_id == ego_wp.road_id and self.is_within_distance(wpt.transform, ego_wp.transform, 5, [-180, 180]) and len(plan._waypoints_queue)>=i+2:
                    i+= 2
                    wpt = plan._waypoints_queue[i][0]
        
                direction = self.compute_connection(ego_transform, wpt)
                
                if direction == "LEFT":
                    turn = 1
                elif direction == "RIGHT":
                    turn = 2
                else:
                    pass


        frame_dict['ego'] = {
            "pose": {
            "position": {
                "x": ego_transform.location.x,
                "y": ego_transform.location.y,
                "z": ego_transform.location.z
            },
            "orientation": {
                "qx": qx,
                "qy": qy,
                "qz": qz,
                "qw": qw
            },
            "linearVelocity": {
                "x": ego_velocity.x,
                "y": ego_velocity.y,
                "z": ego_velocity.z
            },
            "linearAcceleration": {
                "x": ego_linear_acceleration.x,
                "y": ego_linear_acceleration.y,
                "z": ego_linear_acceleration.z
            },
            "angularVelocity": {
                "x": ego_angular_velocity.x,
                "y": ego_angular_velocity.y,
                "z": ego_angular_velocity.z
            },
            "heading": 0,
            "linearAccelerationVrf": {
                "x": 0.0,
                "y": 0.0,
                "z": 0.0
            },
            "angularVelocityVrf": {
                "x": 0.0,
                "y": 0.0,
                "z": 0.0
            },
            "eulerAngles": {
                "x": 0.00,
                "y": 0.00,
                "z": 0.00
            }
            },
            "size": {
            "length": 4.7,
            "width": 2.06
            },
            "Chasis": {
            "lowBeamOn": self.get_lowbeam(light_state),
            "highBeamOn": self.get_highbeam(light_state),
            "turnSignal": self.get_turnsignal(light_state),
            "speed": ego_speed,
            "hornOn": False,
            "engineOn": is_ego_engine_on,
            "gear": ego_control.gear,
            "brake": ego_control.brake,
            "day": 1,
            "hours": self.get_hour(),
            "minutes": 0,
            "seconds": 0,
            "error_code": 0
            },
            "currentLane": {
            "currentLaneId": "lane_" + str(ego_wp.road_id),
            "turn": turn,
            "number": int(ego_wp.lane_id)
            },
            "crosswalkAhead": 200,
            "junctionAhead": junction_ahead,
            "stopSignAhead": 200,
            "stoplineAhead": junction_ahead,
            "planning_of_turn": turn,
            "isTrafficJam": False,
            "isOverTaking": False,
            "isLaneChanging": False,
            "isTurningAround": False,
            "PriorityNPCAhead": False,
            "PriorityPedsAhead": False,
            "upperLimit" : float(upper_limit)
        }
        
        
        frame_dict['truth'] = {
            "obsList": [],
            "NearestNPC": None,
            "minDistToEgo": 200,
            "nearestGtObs": None,
            "NPCAhead": None,
            "PedAhead": None,
            "NPCOpposite": None,
            "npcClassification": {
            "NextToEgo": [],
            "OntheDifferentRoad": [],
            "IntheJunction": [],
            "EgoInjunction_Lane": [],
            "EgoInjunction_junction": []
            }
        }
        
        frame_dict.update(self.get_affected_traffic_light())
        
        self.trace_list.append(frame_dict)
        
    def is_within_distance(self, target_transform, reference_transform, max_distance, angle_interval=None):
        """
        Check if a location is both within a certain distance from a reference object.
        By using 'angle_interval', the angle between the location and reference transform
        will also be tkaen into account, being 0 a location in front and 180, one behind.

        :param target_transform: location of the target object
        :param reference_transform: location of the reference object
        :param max_distance: maximum allowed distance
        :param angle_interval: only locations between [min, max] angles will be considered. This isn't checked by default.
        :return: boolean
        """
        target_vector = np.array([
            target_transform.location.x - reference_transform.location.x,
            target_transform.location.y - reference_transform.location.y
        ])
        norm_target = np.linalg.norm(target_vector)

        # If the vector is too short, we can simply stop here
        if norm_target < 0.001:
            return True

        # Further than the max distance
        if norm_target > max_distance:
            return False

        # We don't care about the angle, nothing else to check
        if not angle_interval:
            return True

        min_angle = angle_interval[0]
        max_angle = angle_interval[1]

        fwd = reference_transform.get_forward_vector()
        forward_vector = np.array([fwd.x, fwd.y])
        angle = math.degrees(math.acos(np.clip(np.dot(forward_vector, target_vector) / norm_target, -1., 1.)))

        return min_angle < angle < max_angle
    
    def get_trafficlight_trigger_location(self, traffic_light):
        """
        Calculates the yaw of the waypoint that represents the trigger volume of the traffic light
        """
        def rotate_point(point, radians):
            """
            rotate a given point by a given angle
            """
            rotated_x = math.cos(radians) * point.x - math.sin(radians) * point.y
            rotated_y = math.sin(radians) * point.x - math.cos(radians) * point.y

            return carla.Vector3D(rotated_x, rotated_y, point.z)

        base_transform = traffic_light.get_transform()
        base_rot = base_transform.rotation.yaw
        area_loc = base_transform.transform(traffic_light.trigger_volume.location)
        area_ext = traffic_light.trigger_volume.extent

        point = rotate_point(carla.Vector3D(0, 0, area_ext.z), math.radians(base_rot))
        point_location = area_loc + carla.Location(x=point.x, y=point.y)

        return carla.Location(point_location.x, point_location.y, point_location.z)
    
    def get_junction_distance(self, junction, ref_waypoint):
        """
        Calculates the yaw of the waypoint that represents the trigger volume of the traffic light
        """
        def rotate_point(point, radians):
            """
            rotate a given point by a given angle
            """
            rotated_x = math.cos(radians) * point.x - math.sin(radians) * point.y
            rotated_y = math.sin(radians) * point.x - math.cos(radians) * point.y

            return carla.Vector3D(rotated_x, rotated_y, point.z)

        base_rot = junction.bounding_box.rotation.yaw
        area_ext = junction.bounding_box.extent

        point = rotate_point(carla.Vector3D(0, 0, area_ext.z), math.radians(base_rot))
        point_location = junction.bounding_box.location + carla.Location(x=point.x, y=point.y)
        
        return self.get_distance_between_points(carla.Transform(carla.Location(point_location.x, point_location.y, point_location.z)), ref_waypoint.transform)
    
    def convert_velocity_to_speed(self, velocity):
        
        x = velocity.x
        y = velocity.y
        z = velocity.z

        return math.sqrt(x*x+y*y+z*z)
    
    def get_distance_between_points(self, first_wp, second_wp):
        """
        calculate distance between two waypoint
        """
        x1 = first_wp.location.x
        x2 = second_wp.location.x
        y1 = first_wp.location.y
        y2 = second_wp.location.y
        z1 = first_wp.location.z
        z2 = second_wp.location.z
        
        return pow(pow(x1-x2, 2) + pow(y1-y2, 2) + pow(z1-z2, 2), 0.5)
    
    def get_junction_ahead(self, ego_transform):
        """
        Parses distance between ego_vehicle and junction
        """
        ego_wp = self._map.get_waypoint(ego_transform.location)
        
        # 以当前waypoint为起点，往后找waypoint
        waypoint_list = ego_wp.next_until_lane_end(distance = 200)
        
        # 如果list中的waypoint是在junction中，计算当前waypoint和junction bounding_box距离
        junction_ahead = 200
        if ego_wp.is_junction is not True:
            if waypoint_list is not None:
                # 需要检查下一条road是不是在junction中，来判断当前路尽头是不是处于Junction
                if waypoint_list[-1].next(1) is not None:
                    for wp in waypoint_list[-1].next(1):
                        if self.is_within_distance(waypoint_list[-1].transform, ego_wp.transform, 5, [-90, 90]) and wp.is_junction:
                            junction_ahead = self.get_distance_between_points(ego_wp.transform, waypoint_list[-1].transform) - self.ego_vehicle.bounding_box.extent.x
                            if junction_ahead < 0:
                                junction_ahead = 0.1
                    
                            break
            else:
                print("waypoint_list is None")
                
        return junction_ahead
    
    def get_highbeam(self, light_state) -> bool:
        """
        get ego_vehicle highbeam state
        """        
        if light_state is None:
            return False
        if light_state == carla.VehicleLightState.HighBeam:
            return True
        return False
    
    def get_lowbeam(self, light_state) -> bool:
        """
        get ego_vehicle lowbeam state
        """            
        if light_state is None:
            return False
        if light_state == carla.VehicleLightState.LowBeam:
            return True
        return False
    
    def get_turnsignal(self, light_state) -> bool:
        """
        get ego_vehicle Blinker state
        """
        if light_state is None:
            return False
        if light_state == carla.VehicleLightState.RightBlinker or light_state == carla.VehicleLightState.LeftBlinker:
            return True
        return False
    
    def get_affected_traffic_light(self):
        
        frame_dict = {}
        traffic_light_list = self._world.get_actors().filter("*traffic_light*")

        for traffic_light in traffic_light_list:
            center, waypoints = self.get_traffic_light_waypoints(traffic_light)
            self._list_traffic_lights.append((traffic_light, center, waypoints))

        
        ego_transform = self.ego_vehicle.get_transform()
        location = ego_transform.location

        veh_extent = self.ego_vehicle.bounding_box.extent.x

        tail_close_pt = self.rotate_point(carla.Vector3D(-0.8 * veh_extent, 0.0, location.z), ego_transform.rotation.yaw)
        tail_close_pt = location + carla.Location(tail_close_pt)

        tail_far_pt = self.rotate_point(carla.Vector3D(-veh_extent - 1, 0.0, location.z), ego_transform.rotation.yaw)
        tail_far_pt = location + carla.Location(tail_far_pt)

        for traffic_light, center, waypoints in self._list_traffic_lights:

            center_loc = carla.Location(center)

            if center_loc.distance(location) > 5:
                continue

            for wp in waypoints:

                tail_wp = self._map.get_waypoint(tail_far_pt)

                # Calculate the dot product (Might be unscaled, as only its sign is important)
                ve_dir = self.ego_vehicle.get_transform().get_forward_vector()
                wp_dir = wp.transform.get_forward_vector()
                dot_ve_wp = ve_dir.x * wp_dir.x + ve_dir.y * wp_dir.y + ve_dir.z * wp_dir.z

                # Check the lane until all the "tail" has passed
                if tail_wp.road_id == wp.road_id and tail_wp.lane_id == wp.lane_id and dot_ve_wp > 0:
                    # This light is red and is affecting our lane
                    yaw_wp = wp.transform.rotation.yaw
                    lane_width = wp.lane_width
                    location_wp = wp.transform.location
                    
                    lft_lane_wp = self.rotate_point(carla.Vector3D(0.4 * lane_width, 0.0, location_wp.z), yaw_wp + 90)
                    lft_lane_wp = location_wp + carla.Location(lft_lane_wp)
                    rgt_lane_wp = self.rotate_point(carla.Vector3D(0.4 * lane_width, 0.0, location_wp.z), yaw_wp - 90)
                    rgt_lane_wp = location_wp + carla.Location(rgt_lane_wp)
                    
                    self._last_red_light = traffic_light

                    # Is the vehicle traversing the stop line?
                    if self.is_vehicle_crossing_line((tail_close_pt, tail_far_pt), (lft_lane_wp, rgt_lane_wp)):
                        self._last_red_light = None
                        
                    if self._last_red_light is not None:
                        
                        color = 0
                        if self._last_red_light.state == carla.TrafficLightState.Red:
                            color = 1
                        elif self._last_red_light.state == carla.TrafficLightState.Yellow:
                            color = 2
                        elif self._last_red_light.state == carla.TrafficLightState.Green:
                            color = 3
 
                        frame_dict["traffic_lights"] = {
                        "containLights": True,
                        "trafficLightList": [
                        {
                            "color": color,
                            "id": "signal_" + str(traffic_light.id),
                            "blink": False
                        }
                        ],
                        "trafficLightStopLine": self.get_distance_between_points(wp.transform, tail_wp.transform)
                        }
                        
                    
        if "traffic_lights" in frame_dict:
            pass
        else:
            frame_dict["traffic_lights"] = {
                "containLights": False
            }
                
        return frame_dict
    
    def get_traffic_light_ahead(self) -> dict:
        """
        Parses distance between ego_vehicle and affected trafficlight
        """
        traffic_light_list = self._world.get_actors().filter("*traffic_light*")
        ego_wp = self._map.get_waypoint(self.ego_vehicle.get_transform().location)
        
        base_tlight_threshold = 5.0
        
        frame_dict = {}
        
        for traffic_light in traffic_light_list:
            object_location = self.get_trafficlight_trigger_location(traffic_light)
            object_waypoint = self._map.get_waypoint(object_location)

            if object_waypoint.road_id != ego_wp.road_id:
                continue

            ve_dir = ego_wp.transform.get_forward_vector()
            wp_dir = object_waypoint.transform.get_forward_vector()
            dot_ve_wp = ve_dir.x * wp_dir.x + ve_dir.y * wp_dir.y + ve_dir.z * wp_dir.z

            if dot_ve_wp < 0:
                continue

            if traffic_light.get_state() != carla.TrafficLightState.Red:
                continue

            if self.is_within_distance(object_waypoint.transform, ego_wp.transform, base_tlight_threshold, [0, 90]):
                
                traffic_light_stop_line = self.get_distance_between_points(ego_wp.transform, object_waypoint.transform) - self.ego_vehicle.bounding_box.extent.x
                traffic_light_stop_line = max(traffic_light_stop_line, 0)
                
                frame_dict["traffic_lights"] = {
                    "containLights": True,
                    "trafficLightList": [
                    {
                        "color": 1,
                        "id": "signal_" + str(traffic_light.id),
                        "blink": False
                    }
                    ],
                    "trafficLightStopLine": traffic_light_stop_line
                    }

        if "traffic_lights" in frame_dict:
            pass
        else:
            frame_dict["traffic_lights"] = {
                "containLights": False
            }
                
        return frame_dict
    
    def get_traffic_light_waypoints(self, traffic_light):
        """
        get area of a given traffic light
        """
        base_transform = traffic_light.get_transform()
        base_rot = base_transform.rotation.yaw
        area_loc = base_transform.transform(traffic_light.trigger_volume.location)

        # Discretize the trigger box into points
        area_ext = traffic_light.trigger_volume.extent
        x_values = np.arange(-0.9 * area_ext.x, 0.9 * area_ext.x, 1.0)  # 0.9 to avoid crossing to adjacent lanes

        area = []
        for x in x_values:
            point = self.rotate_point(carla.Vector3D(x, 0, area_ext.z), base_rot)
            point_location = area_loc + carla.Location(x=point.x, y=point.y)
            area.append(point_location)

        # Get the waypoints of these points, removing duplicates
        ini_wps = []
        for pt in area:
            wpx = self._map.get_waypoint(pt)
            # As x_values are arranged in order, only the last one has to be checked
            if not ini_wps or ini_wps[-1].road_id != wpx.road_id or ini_wps[-1].lane_id != wpx.lane_id:
                ini_wps.append(wpx)

        # Advance them until the intersection
        wps = []
        for wpx in ini_wps:
            while not wpx.is_intersection:
                next_wp = wpx.next(0.5)[0]
                if next_wp and not next_wp.is_intersection:
                    wpx = next_wp
                else:
                    break
            wps.append(wpx)

        return area_loc, wps

    def is_vehicle_crossing_line(self, seg1, seg2):
        """
        check if vehicle crosses a line segment
        """
        line1 = shapely.geometry.LineString([(seg1[0].x, seg1[0].y), (seg1[1].x, seg1[1].y)])
        line2 = shapely.geometry.LineString([(seg2[0].x, seg2[0].y), (seg2[1].x, seg2[1].y)])
        inter = line1.intersection(line2)

        return not inter.is_empty
    
    def rotate_point(self, point, angle):
        """
        rotate a given point by a given angle
        """
        x_ = math.cos(math.radians(angle)) * point.x - math.sin(math.radians(angle)) * point.y
        y_ = math.sin(math.radians(angle)) * point.x + math.cos(math.radians(angle)) * point.y
        return carla.Vector3D(x_, y_, point.z)
    
    def get_hour(self) -> int:
        """
        get hour information from sun_altitude_angle
        """
        return int((90 + self._world.get_weather().sun_altitude_angle) * 0.06666666)
    
    def compute_connection(self, current_waypoint, next_waypoint, threshold=35):
        """
        Compute the type of topological connection between an active waypoint (current_waypoint) and a target waypoint
        (next_waypoint).

        :param current_waypoint: active waypoint
        :param next_waypoint: target waypoint
        :return: the type of topological connection encoded as a RoadOption enum:
                RoadOption.STRAIGHT
                RoadOption.LEFT
                RoadOption.RIGHT
        """
        
        n = next_waypoint.transform.rotation.yaw
        n = n % 360.0

        c = current_waypoint.rotation.yaw
        c = c % 360.0

        diff_angle = (n - c) % 180.0
        if diff_angle < threshold or diff_angle > (180 - threshold):
            return "STRAIGHT"
        elif diff_angle > 90.0:
            return "LEFT"
        else:
            return "RIGHT"
        