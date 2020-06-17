import pprint
import json
import math
import matplotlib.pyplot as plt

from srunner.metrics.metrics_log import MetricsLog
from srunner.metrics.basic_metric import BasicMetric


class ControlLossMetric(BasicMetric):
    """
    Class containing a metric of the FollowLeadingVehicle scenario.

    This is just an example to show the user how one might be created, but
    many more can be achieved
    """

    def __init__(self, town_map, recorder, criteria=None):
        """
        Initialization of the metric class. Must always call the BasicMetric __init__

        Args:
            recorder (dict): dictionary with all the information
                of the simulation
            criteria (list): list of dictionaries with all the
                information regarding the criterias used 
        """
        self._map = town_map
        super(ControlLossMetric, self).__init__(town_map, recorder, criteria)

    def _create_metrics(self, metrics_log):
        """
        Implementation of the metric. Here the user is meant to freely calculate the
        wanted metrics
        
        Args:
            metrics_log (srunner.metrics.metrics_log.MetricsLog): class with all the 
                information passed through the MetricsManager. This information has been
                parsed and some functions have been added for easy access

        Here we have two metrics. The first one plots the distance between the two vehicles
        and the second one, prints the amount of collisions using the criteria
        """

        ### Rough calculus of the distance to the center of the lane ###

        # Get their ID's
        hero_id = metrics_log.get_actor_ids_with_role_name("hero")[0]

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

        # Plot the results
        plt.plot(frames_list, distances_list)
        # plt.plot(frames_list, lane_width_list)
        # plt.plot(frames_list, [-x for x in lane_width_list])
        plt.show()
