#!/usr/bin/env python

# Copyright (c) 2020 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This metric calculates the distance between the ego vehicle and
another actor, dumping it to a json file.

It is meant to serve as an example of how to use the information from
the recorder
"""

import math
import json

from srunner.metrics.basic_metric import BasicMetric


class DistanceBetweenVehicles(BasicMetric):
    """
    Metric class DistanceBetweenVehicles
    """

    def _create_metrics(self, town_map, log, criteria):
        """
        Implementation of the metric. This is an example to show how to use the recorder,
        accessed via the log.
        """

        ##### Calculate distance between the two vehicles and plot it #####

        # Get the ID of the two vehicles
        ego_id = log.get_ego_vehicle_id()
        adv_id = log.get_actor_ids_with_role_name("scenario")[0]  # Could have also used its type_id

        dist_list = []
        frames_list = []

        # Get the frames both actors were alive
        start_ego, end_ego = log.get_actor_alive_frames(ego_id)
        start_adv, end_adv = log.get_actor_alive_frames(adv_id)
        start = max(start_ego, start_adv)
        end = min(end_ego, end_adv)

        # Get the list of transforms
        ego_transform_list = log.get_all_transforms(ego_id, start, end)
        adv_transform_list = log.get_all_transforms(adv_id, start, end)

        # Get the distance between the two
        for i in range(start, end):

            ego_location = ego_transform_list[i - 1].location  # Frames start at 1!
            adv_location = adv_transform_list[i - 1].location  # Frames start at 1!

            # Filter some points for a better graph
            if adv_location.z < -10:
                continue

            dist_v = ego_location - adv_location
            dist = math.sqrt(dist_v.x * dist_v.x + dist_v.y * dist_v.y + dist_v.z * dist_v.z)

            dist_list.append(dist)
            frames_list.append(i)

        results = {'frames': frames_list, 'distance': dist_list}

        with open('srunner/metrics/data/DistanceBetweenVehicles_results.json', 'w') as fw:
            json.dump(results, fw, sort_keys=False, indent=4)
