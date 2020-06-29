#!/usr/bin/env python

# Copyright (c) 2020 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Follow Leading Vehicle metric:

This metric calculates the distance between the ego vehicle and
the other actor, dumping it to a json file.

It is meant to serve as an example of how to use the information from
the recorder
"""

import math
import json

from srunner.metrics.basic_metric import BasicMetric


class FollowLeadingVehicleMetrics(BasicMetric):
    """
    Class containing an example metric of the FollowLeadingVehicle scenario.
    """

    def _create_metrics(self, metrics_log):
        """
        Implementation of the metric. This is an example to show how to use the recorder,
        accessed via the metrics_log.
        """

        ##### Calculate distance between the two vehicles and plot it #####

        # Get their ID's
        hero_id = metrics_log.get_ego_vehicle_id()
        adve_id = metrics_log.get_actor_ids_with_role_name("scenario")[0]

        distances_list = []
        frames_list = []

        # Get the frames both actors were alive
        start_hero, end_hero = metrics_log.get_actor_alive_frames(hero_id)
        start_adve, end_adve = metrics_log.get_actor_alive_frames(adve_id)
        start = max(start_hero, start_adve)
        end = min(end_hero, end_adve)

        # Get the list of transforms
        hero_transform_list = metrics_log.get_all_actor_states(hero_id, "transform", start, end)
        adve_transform_list = metrics_log.get_all_actor_states(adve_id, "transform", start, end)

        # Get the distance between the two
        for i in range(start, end):

            hero_location = hero_transform_list[i - 1].location  # Frames start at 1! Indexes should be one less
            adve_location = adve_transform_list[i - 1].location  # Frames start at 1! Indexes should be one less

            # Filter some points for a better graph
            if adve_location.z < -10:
                continue

            distance_vec = hero_location - adve_location
            distance = math.sqrt(
                distance_vec.x * distance_vec.x +
                distance_vec.y * distance_vec.y +
                distance_vec.z * distance_vec.z
            )

            distances_list.append(distance)
            frames_list.append(i)

        results = {'frames': frames_list, 'distance': distances_list}

        with open('srunner/metrics/data/FollowLeadingVehicle_metric.json', 'w') as fw:
            json.dump(results, fw, sort_keys=False, indent=4)
