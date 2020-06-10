import pprint
import json
import math
import matplotlib.pyplot as plt

from srunner.metrics.metrics_log import MetricsLog
from srunner.metrics.basic_metric import BasicMetric


class ExampleMetrics(BasicMetric):
    """
    Example class containing a metric to show the user how one might be created.
    This example is meant to be used with the recorded information from a
    FollowLeadingVehicle scenario
    """

    def __init__(self, recorder, criteria):
        """
        Initialization of the metrics class.

        Args:
            recorder (dict): dictionary with all the information
                of the simulation
            criteria (list): list of dictionaries with all the
                information regarding the criterias used 
        """

        super(ExampleMetrics, self).__init__(recorder, criteria)

    def _create_metrics(self, metrics_log):
        """
        Implementation of the metric.
        
        Args:
            metrics_log (srunner.metrics.metrics_log.MetricsLog): class with all the 
                information passed through the MetricsManager. this information has been
                parsed and some function have been added for easy access

        Here we have two metrics. The first one plots the distance between the two vehicles
        and the second one, prints the amount of collisions using the criterias
        """

        ##### Example 1: Get the distance between two vehicles and plot it #####

        hero_id = metrics_log.get_actor_id_with_role_name("hero")
        adversary_id = metrics_log.get_actor_id_with_role_name("scenario")

        distances_list = []
        frames_list = []
        num_frames = metrics_log.get_total_frame_count()

        for frame in range(0, num_frames):

            hero_location = metrics_log.get_actor_location(hero_id, frame)
            adversary_location = metrics_log.get_actor_location(adversary_id, frame)

            if hero_location and adversary_location and adversary_location.z > -10:
                distance_vec = hero_location - adversary_location
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

        collision_criteria = metrics_log.get_criteria("CollisionTest")
        number_of_hits = collision_criteria["actual_value"]
        print("The ego vehicles has had a total of {} collisions".format(number_of_hits))
