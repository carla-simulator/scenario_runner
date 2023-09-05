from typing import Set

import carla

import srunner.scenariomanager.carla_data_provider as carla_data
from srunner.osc2_dm.physical_types import Physical
from srunner.osc2_stdlib.path_check import (
    OverJunctionCheck,
    OverLanesDecreaseCheck,
    PathCurve,
    PathDiffDest,
    PathDiffOrigin,
    PathExplicit,
    PathOverDiffLimitMarks,
    PathTrafficSign,
)


class Path:
    map_name = None
    min_driving_lanes = None
    max_lanes = None
    _length = None
    sign_type = None
    sign_types = None
    over_junction_check = None
    over_lanes_decrease_check = None
    is_path_dest = None
    is_path_origin = None
    is_explicit = None
    over_different_marks = None
    curve: PathCurve = None

    @classmethod
    def set_map(cls, map_name: str) -> None:
        cls.map_name = map_name

    @classmethod
    def get_map(cls) -> str:
        return cls.map_name

    @classmethod
    def path_length(cls, length: str) -> None:
        cls._length = float(length)

    @classmethod
    def path_min_driving_lanes(cls, min_lanes: str) -> None:
        cls.min_driving_lanes = float(min_lanes)

    @classmethod
    def path_max_lanes(cls, max_lanes: str) -> None:
        cls.max_lanes = float(max_lanes)

    @classmethod
    def path_different_dest(cls):
        cls.is_dest = True

    @classmethod
    def path_different_origin(cls):
        cls.is_path_origin = True

    @classmethod
    def path_has_sign(cls, sign_type: str):
        if sign_type == "speed_limit":
            cls.sign_type = carla.LandmarkType.MaximumSpeed
        elif sign_type == "stop_sign":
            cls.sign_type = carla.LandmarkType.StopSign
        elif sign_type == "yield":
            cls.sign_type = carla.LandmarkType.YieldSign
        elif sign_type == "roundabout":
            cls.sign_type = carla.LandmarkType.Roundabout

    @classmethod
    def path_has_no_signs(cls):
        cls.sign_types = [
            carla.LandmarkType.MaximumSpeed,
            carla.LandmarkType.StopSign,
            carla.LandmarkType.YieldSign,
            carla.LandmarkType.Roundabout,
        ]

    @classmethod
    def path_over_junction(
        cls,
        direction: Physical,
        distance_before: Physical = None,
        distance_in: Physical = None,
        distance_after: Physical = None,
    ) -> None:
        # dir degree
        # #dis_before m
        # #dis_in m
        # #dis_after m

        cls.over_junction_check = OverJunctionCheck(
            direction, distance_before, distance_in, distance_after
        )

    @classmethod
    def path_over_lanes_decrease(cls, distance: Physical) -> None:
        cls.over_lanes_decrease_check = OverLanesDecreaseCheck(distance)

    @classmethod
    def path_explicit(cls, start_point, end_point, tolerance):
        start_point = start_point.split(",")
        end_point = end_point.split(",")
        cls.is_explicit = PathExplicit(start_point, end_point, tolerance)
        print(cls.is_explicit)

    @classmethod
    def path_over_speed_limit_change(cls, first_speed, sec_speed):
        cls.over_different_marks = PathOverDiffLimitMarks(first_speed, sec_speed)

    @classmethod
    def path_curve(cls, min_radius, max_radius, side):
        min_radius = Physical.from_str(min_radius)  # Minimum radius of curvature
        max_radius = Physical.from_str(max_radius)  # Maximum radius of curvature
        print(min_radius, max_radius, side)
        cls.curve = PathCurve(min_radius=min_radius, max_radius=max_radius, side=side)

    @classmethod
    def check(cls, pos) -> bool:
        _map = carla_data.CarlaDataProvider.get_map(carla_data.CarlaDataProvider.world)
        wp = _map.get_waypoint(
            pos.location, project_to_road=True, lane_type=carla.LaneType.Driving
        )
        # Remove the intersection

        road_lanes = carla_data.CarlaDataProvider.get_road_lanes(wp)
        lane_cnt = len(road_lanes)

        # Check whether the number of lanes is satisfied
        if cls.min_driving_lanes is not None and lane_cnt < cls.min_driving_lanes:
            return False
        if cls.max_lanes is not None and lane_cnt > cls.max_lanes:
            return False

        # Check whether the length of the test road meets the constraints
        if cls._length:
            len_ok = carla_data.CarlaDataProvider.check_road_length(wp, cls._length)
            if not len_ok:
                return False
        # Check if the test road is signposted
        if cls.sign_type:
            is_sign = PathTrafficSign.path_has_traffic_sign(
                wp, cls.sign_type, cls._length
            )
            if not is_sign:
                return False
        # Restricted test roads cannot be signposted
        if cls.sign_types:
            no_sign = PathTrafficSign.path_has_no_traffic_signs(
                wp, cls.sign_types, cls._length
            )
            if not no_sign:
                return False

        # Check for an intersection
        if cls.over_junction_check:
            over_junction = cls.over_junction_check.check(wp)
            if not over_junction:
                return False

        # Check for lane reduction
        if cls.over_lanes_decrease_check:
            check_pass = cls.over_lanes_decrease_check.check(wp)
            if not check_pass:
                return False

        if cls.is_path_dest:
            dest = PathDiffDest().get_diff_dest_point(wp, cls._length)
            if not dest:
                return False

        if cls.is_path_origin:
            origin = PathDiffOrigin().get_diff_origin_point(wp)
            if not origin:
                return False

        if cls.is_explicit:
            path_explicit = cls.is_explicit.gen_path_by_point(wp)
            if not path_explicit:
                return False

        if cls.over_different_marks:
            ret = cls.over_different_marks.path_over_speed_limit_mark(wp, cls._length)
            if not ret:
                return False

        if cls.curve:
            ret = cls.curve.check(wp)
            if not ret:
                return False
        return True
