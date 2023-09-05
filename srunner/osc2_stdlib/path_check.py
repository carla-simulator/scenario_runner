import carla
from carla import LandmarkType, Waypoint

from srunner.osc2_dm.physical_types import Physical
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.tools.osc2_helper import OSC2Helper


class OverJunctionCheck:
    def __init__(
        self,
        direction: Physical,
        distance_before: Physical = None,
        distance_in: Physical = None,
        distance_after: Physical = None,
    ) -> None:
        self.direction = direction
        self.distance_before = distance_before
        self.distance_in = distance_in
        self.distance_after = distance_after

        self.line1_start = None
        self.line1_end = None
        self.line2_start = None
        self.line2_end = None

        self.start_wps_in_junction = []

    def _check_distance_before(self, wp: Waypoint) -> bool:
        """
        Input: wp, the wp being checked
        Output: Whether wp is at distance_before before the intersection
        Side effect: If the check is successful, set line1_end to the first wp before the intersection.
        """
        if not wp:
            print("_check_distance_before wp is none")
            return False

        if not self.distance_before:
            print("no need to check distance before")
            return True

        distance_list = []
        if self.distance_before.is_single_value():
            distance_list.append(int(self.distance_before.num))
        else:
            start = self.distance_before.num.start
            end = self.distance_before.num.end
            for dis in range(start, end + 1):
                distance_list.append(dis)

        for dis in distance_list:
            curr_wp = wp
            for dis_iter in range(1, dis + 1):
                next_wps = wp.next(dis_iter)
                if len(next_wps) > 1:  # Denote an intersection
                    # To reach the intersection, save the starting wp for each direction of the intersection
                    self.start_wps_in_junction = next_wps
                    self.line1_end = curr_wp
                    return True
                else:
                    curr_wp = next_wps[0]

        return False

    def _check_distance_in(self, wp: Waypoint) -> bool:
        """
        Input: wp, the first waypoint in the intersection
        Output: the distance in the intersection is equal to the set distance
        Side effect: Check successful, set line2_start to the second wwp after the intersection.
        """
        if not wp:
            print("_check_distance_in wp is none")
            return False

        if not self.distance_in:
            print("no need to check distance in")
            return True

        if not wp.is_junction:
            print("start waypoint is not in junction")
            return False

        distance_list = []
        if self.distance_in.is_single_value():
            distance_list.append(int(self.distance_in.num))
        else:
            start = self.distance_before.num.start
            end = self.distance_before.num.end
            for dis in range(start, end + 1):
                distance_list.append(dis)

        for dis in distance_list:
            wps = []
            for dis_iter in range(1, dis + 1):
                wps.extend(wp.next(dis_iter))

            for p in wps:
                if not p.is_junction:
                    self.line2_start = p
                    return True
        return False

    def check_distance_in(self) -> bool:
        return any(map(self._check_distance_in, self.start_wps_in_junction))

    def _check_distance_after(self, wp: Waypoint) -> bool:
        """
        Input: wp, the first waypoint after the intersection
        Return: The distance after the intersection is equal to the set distance
        Side effect: Check successful, set line2_end to the waypoint at a distance of distance after the intersection.
        """
        if not wp:
            print("_check_distance_after wp is none")
            return False

        if not self.distance_after:
            print("no need to check distance after")
            return True

        distance_list = []
        if self.distance_after.is_single_value():
            distance_list.append(int(self.distance_after.num))
        else:
            start = self.distance_before.num.start
            end = self.distance_before.num.end
            for dis in range(start, end + 1):
                distance_list.append(dis)

        for dis in distance_list:
            wps = wp.next(dis)
            if len(wps) != 0:
                self.line2_end = wps[0]
                return True

        return False

    def _check_direction(self) -> bool:
        if not self.direction:
            print("over junction check must have direction parameter")
            return False

        line1_start_loc = self.line1_start.transform.location
        line1_end_loc = self.line1_end.transform.location
        line1 = [line1_start_loc.x, line1_start_loc.y, line1_end_loc.x, line1_end_loc.y]

        line2_start_loc = self.line2_start.transform.location
        line2_end_loc = self.line2_end.transform.location
        line2 = [line2_start_loc.x, line2_start_loc.y, line2_end_loc.x, line2_end_loc.y]

        angle = OSC2Helper.vector_angle(line1, line2)
        if angle < 0:
            angle = 360 + angle

        return self.direction.is_in_range(angle)

    def check(self, wp: Waypoint) -> bool:
        self.line1_start = wp
        return (
            self._check_distance_before(self.line1_start)
            and self.check_distance_in()
            and self._check_distance_after(self.line2_start)
            and self._check_direction()
        )


class OverLanesDecreaseCheck:
    def __init__(self, sp_more_lanes_path_length: Physical) -> None:
        self.distance = sp_more_lanes_path_length

    def check(self, wp: Waypoint) -> bool:
        """
        Input: wp, path points to be checked
        Output: Whether the number of lanes decreases within distance ahead of wp.
        """
        distance = int(self.distance.gen_single_value())
        lane_cnt = CarlaDataProvider.get_road_lane_cnt(wp)

        for dis in range(1, distance + 1):
            next_wps = wp.next(dis)

            if len(next_wps) == 1:
                next_wp = next_wps[0]
                if next_wp.is_junction:
                    return False
                next_lanes_cnt = CarlaDataProvider.get_road_lane_cnt(next_wp)
                if next_lanes_cnt < lane_cnt:
                    return True
            else:
                return False

        return False


class PathTrafficSign(CarlaDataProvider):
    _length = 100.0
    distance_step = 2.0
    wps = []

    @staticmethod
    def path_has_traffic_sign(waypoint: Waypoint, sign_type: str, length=None) -> bool:
        if length:
            PathTrafficSign._length = length
        temp_list = waypoint.next_until_lane_end(PathTrafficSign.distance_step)
        lane_length = len(temp_list) * PathTrafficSign.distance_step
        if lane_length >= PathTrafficSign._length:
            sign_type_list = waypoint.get_landmarks_of_type(
                lane_length, sign_type, stop_at_junction=False
            )
            if sign_type_list:
                PathTrafficSign.wps.append(waypoint.road_id)
        if PathTrafficSign.wps:
            return True

    @staticmethod
    def path_has_no_traffic_signs(
        waypoint: Waypoint, sign_type: list, length=None
    ) -> bool:
        lane_length_list = []
        if length:
            PathTrafficSign._length = length
        temp_list = waypoint.next_until_lane_end(PathTrafficSign.distance_step)
        lane_length = len(temp_list) * PathTrafficSign.distance_step
        if lane_length >= PathTrafficSign._length:
            lane_length_list.append(lane_length)
            for sign in sign_type:
                sign_type_list = waypoint.get_landmarks_of_type(
                    lane_length, sign, stop_at_junction=False
                )
                if sign_type_list:
                    return False
        return True


class PathDiffDest(object):
    def __init__(self):
        self.distance = 4
        self.path_length = 30.0
        self.current_length = float(0)

    def get_diff_dest_point(self, wp: Waypoint, length: float) -> bool:
        if length and length > self.path_length:
            self.path_length = length
        # Stay away from the road reference line
        right_most_pont = None
        temp_pont = wp
        while True:
            temp_pont = temp_pont.get_right_lane()
            if temp_pont is None:
                break
            right_most_pont = temp_pont

        if right_most_pont is None:
            return False
        while True:
            np = right_most_pont.next(self.distance)
            if not np:
                break
            if len(np) > 1:
                return True

            self.current_length += self.distance
            if self.current_length > self.path_length:
                return False

            right_most_pont = np[0]


class PathDiffOrigin(object):
    def __init__(self):
        self.distance = 3
        self.path_length = 30.0
        self.current_length = float(0)

    def get_diff_origin_point(self, wp):
        temp_wp = wp
        wp_id = wp.road_id
        path1_wps = []
        n = 0
        step_num = int(self.path_length / self.distance)
        # Find a road of the specified length that has three road ids within the specified length
        for i in range(step_num):
            p = temp_wp.next(self.distance)[0]
            if p.road_id != wp_id:
                n += 1
            wp_id = p.road_id
            path1_wps.append(p)
            temp_wp = p
        if n < 2:
            return False

        # Find all generative points on the map except the one you just made.
        # Each point generates a path of the specified length
        _map = CarlaDataProvider.get_map(CarlaDataProvider.world)
        wps = _map.get_spawn_points()
        all_path = []
        path_wps = []
        for pos in wps:
            pont = _map.get_waypoint(
                pos.location, project_to_road=True, lane_type=carla.LaneType.Driving
            )
            if pont.road_id == wp.road_id:
                continue
            for j in range(step_num):
                p = pont.next(self.distance)[0]
                path_wps.append(p)
                pont = p
            all_path.append(path_wps)

        # The points on each road are compared with all the points on the first road.
        # If there is a road with the same id, it is considered to intersect
        for path in all_path:
            for wpoint in path1_wps:
                for p in path:
                    if wpoint.road_id == p.road_id:
                        return True


class PathExplicit(object):
    def __init__(self, start, end, tolerance):
        self.start_point_parm = start
        self.end_point_parm = end
        self.tolerance = float(tolerance)

    def gen_path_by_point(self, wp):
        carla_map = CarlaDataProvider.get_map()
        start_road_id, start_lane_id, start_s = (
            self.start_point_parm[0],
            self.start_point_parm[1],
            self.start_point_parm[2],
        )
        start_point = carla_map.get_waypoint_xodr(
            int(start_road_id), int(start_lane_id), int(start_s)
        )

        end_road_id, end_lane_id, end_s = (
            self.end_point_parm[0],
            self.end_point_parm[1],
            self.end_point_parm[2],
        )
        end_point = carla_map.get_waypoint_xodr(
            int(end_road_id), int(end_lane_id), int(end_s)
        )

        if start_road_id != end_road_id or start_lane_id != end_lane_id:
            print(
                "Please check parameters, start road id and lane id must remain the same."
            )
            return False
        if start_s == end_s:
            print(
                "Please check parameters, the start and end positions must be different."
            )
            return False
        if end_point is None:
            print(
                "Please check parameters,the end point exceeds the length of the road."
            )
            return False

        if wp.road_id == int(start_road_id) and wp.lane_id == int(end_lane_id):
            if (start_point.s - self.tolerance) < wp.s < (end_point.s + self.tolerance):
                print("Find a point to generate a car on the road.")
                return True
        else:
            return False


class PathOverDiffLimitMarks(object):
    def __init__(self, first_speed, sec_speed):
        self.speed1 = float(first_speed)
        self.speed2 = float(sec_speed)
        self.length = 50.0

    def path_over_speed_limit_mark(self, wp, length):
        if length:
            self.length = length
        land_marks = wp.get_landmarks(self.length)
        print(land_marks)
        marks_value_list = []
        for i in land_marks:
            if i.type == LandmarkType.MaximumSpeed:
                marks_value_list.append(i.value)

        if len(marks_value_list) > 1:
            if (
                self.speed1 == marks_value_list[0]
                and self.speed2 == marks_value_list[1]
            ):
                print(
                    f"The car will cross the road section with speed limit signs of {self.speed1} and {self.speed2}"
                )
                return True


class PathCurve:
    def __init__(
        self, min_radius: Physical, max_radius: Physical, side: str = None
    ) -> None:
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.side = side

    def check(self, wp: Waypoint) -> bool:
        """
        Input: wp, the path point to be checked
        Output: Whether the curvature radius of the path where wp is located meets the requirements
        Implementation strategy: the radius of the circle formed by wp and the 3 points of 2 waypoints 2 meters ahead.
        """
        points = [[wp]]

        for i in range(1, 3):
            next_wps = wp.next(i)
            if len(next_wps) > 0:
                points.append(next_wps)
            else:
                return False

        for wp1 in points[0]:
            for wp2 in points[1]:
                for wp3 in points[2]:
                    point1 = [
                        wp1.transform.location.x,
                        wp1.transform.location.y,
                        wp1.transform.location.z,
                    ]
                    point2 = [
                        wp2.transform.location.x,
                        wp2.transform.location.y,
                        wp2.transform.location.z,
                    ]
                    point3 = [
                        wp3.transform.location.x,
                        wp3.transform.location.y,
                        wp3.transform.location.z,
                    ]
                    r = OSC2Helper.curve_radius(point1, point2, point3)
                    if r is None:
                        continue
                    loc = OSC2Helper.point_line_location(point1, point2, point3)
                    if (
                        self.min_radius.gen_single_value()
                        <= r
                        <= self.max_radius.gen_single_value()
                        and loc == self.side
                    ):
                        return True

        return False
