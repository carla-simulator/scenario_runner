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

    def __init__(self, recorder, criteria=None):
        """
        Initialization of the metric class. Must always call the BasicMetric __init__

        Args:
            recorder (dict): dictionary with all the information
                of the simulation
            criteria (list): list of dictionaries with all the
                information regarding the criterias used 
        """

        super(FollowLeadingVehicleMetrics, self).__init__(recorder, criteria)

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

        hero_id = metrics_log.get_actor_ids_with_role_name("hero")[0]
        adversary_id = metrics_log.get_actor_ids_with_role_name("scenario")[0]

        distances_list = []
        frames_list = []
        num_frames = metrics_log.get_total_frame_count()

        for frame in range(0, num_frames):

            hero_transform = metrics_log.get_actor_state(hero_id, frame, "transform")
            adversary_transform = metrics_log.get_actor_state(adversary_id, frame, "transform")

            if hero_transform and adversary_transform and adversary_transform.location.z > -10:
                distance_vec = hero_transform.location - adversary_transform.location
                distance = math.sqrt(
                    distance_vec.x * distance_vec.x + 
                    distance_vec.y * distance_vec.y + 
                    distance_vec.z * distance_vec.z
                )

                distances_list.append(distance)
                frames_list.append(frame)

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
