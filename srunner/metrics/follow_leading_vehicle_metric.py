import pprint
import json
import math
import matplotlib.pyplot as plt

from srunner.metrics.metrics_log import MetricsLog
from srunner.metrics.basic_metric import BasicMetric


class FollowLeadingVehicleMetrics(BasicMetric):
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

        super(FollowLeadingVehicleMetrics, self).__init__(town_map, recorder, criteria)

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

        ##### Example 1: Get the distance between two vehicles and plot it #####

        # Get their ID's
        hero_id = metrics_log.get_actor_ids_with_role_name("hero")[0]
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

        # Plot the results
        plt.plot(frames_list, distances_list)
        plt.show()

        ##### Example 2: Get the amount of collision using the CollisionTest criteria #####

        collision_criteria = metrics_log.get_criterion("CollisionTest")
        number_of_hits = collision_criteria["actual_value"]
        print("-- The ego vehicles has had a total of {} collisions --".format(number_of_hits))

        ##### Example 2.1: Or using the collisions utility of the recorder #####

        collisions = metrics_log.get_collisions(hero_id)
        number_of_hits = len(collisions)
        print("-- The ego vehicles has had a total of {} collisions --".format(number_of_hits))

        # These two might differ as the computation methods are different
