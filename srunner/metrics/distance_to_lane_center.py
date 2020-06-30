#!/usr/bin/env python

# Copyright (c) 2020 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This metric calculates the distance between the ego vehicle and
the center of the lane, dumping it to a json file.

It is meant to serve as an example of how to use the map API
"""

import math
import json

from srunner.metrics.basic_metric import BasicMetric


class DistanceToLaneCenter(BasicMetric):
    """
    Metric class DistanceToLaneCenter
    """

    def _create_metrics(self, metrics_log):
        """
        Implementation of the metric.
        """

        ### Rough calculus of the distance to the center of the lane ###

        # Get their ID's
        hero_id = metrics_log.get_ego_vehicle_id()

        distances_list = []
        lane_width_list = []
        frames_list = []

        # Get the frames the hero actor was alive and its transforms
        start, end = metrics_log.get_actor_alive_frames(hero_id)
        hero_transform_list = metrics_log.get_all_actor_states(hero_id, "transform", start, end)

        # Get the projected distance vector to the center of the lane
        for i in range(start, end):

            # Frames start at 1! Indexes should be one less
            hero_location = hero_transform_list[i - 1].location

            hero_waypoint = self._map.get_waypoint(hero_location)
            lane_width = hero_waypoint.lane_width / 2  # Get the lane width

            perp_wp_vec = hero_waypoint.transform.get_right_vector()
            hero_wp_vec = hero_location - hero_waypoint.transform.location

            dot = perp_wp_vec.x * hero_wp_vec.x + perp_wp_vec.y * hero_wp_vec.y + perp_wp_vec.z * hero_wp_vec.z
            perp_wp = math.sqrt(
                perp_wp_vec.x * perp_wp_vec.x +
                perp_wp_vec.y * perp_wp_vec.y +
                perp_wp_vec.z * perp_wp_vec.z
            )

            project_vec = dot / (perp_wp * perp_wp) * perp_wp_vec
            project_dist = math.sqrt(
                project_vec.x * project_vec.x +
                project_vec.y * project_vec.y +
                project_vec.z * project_vec.z
            )

            # Differentiate between one side and another
            forw_wp_vec = hero_waypoint.transform.get_forward_vector()
            cross_z = forw_wp_vec.x * hero_wp_vec.y - forw_wp_vec.y * hero_wp_vec.x
            if cross_z < 0:
                project_dist = - project_dist

            lane_width_list.append(lane_width)

            distances_list.append(project_dist)
            frames_list.append(i)

        results = {'frames': frames_list, 'distance': distances_list}

        with open('srunner/metrics/data/DistanceToLaneCenter_Result.json', 'w') as fw:
            json.dump(results, fw, sort_keys=False, indent=4)
