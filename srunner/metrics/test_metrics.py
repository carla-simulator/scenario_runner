import pprint
import json
import math
import matplotlib.pyplot as plt

from srunner.metrics.metrics_log import MetricsLog
from srunner.metrics.basic_metric import BasicMetric


class TestMetrics(BasicMetric):
    """
    Support class for the users to easily create their own metrics.
    The user should only modify the create_metrics function, returning
    "something in particular"
    """
    def __init__(self, recorder, criteria):
        """
        Initializes the class. Here, the user can select which output
        option will be used. If activated:
            - self._metrics_to_plot: Creates a plot using the matplotlib library
            - self._metrics_to_json: Saves the metrics into a json file
            - self._metrics_to_terminal: Print the results through the terminal
            - self._metrics_to_file: Dumps the results into a file

        By default they are all false and more than one can be active at the same time. It is
        up to the user to return metrics that are compatible with these output methods
        """
        self._metrics_to_plot = False
        self._metrics_to_json = True
        self._metrics_to_terminal = True
        self._metrics_to_file = False

        super(TestMetrics, self).__init__(recorder, criteria)

    def _create_metrics(self, metrics_log):
        """
        Implementation of the metrics
        """
        pp = pprint.PrettyPrinter(indent = 4)

        ##### Example 1: Get the distance between two vehicles and plot it #####

        hero = metrics_log.get_actor_id_with_role_name("hero")
        adversary = metrics_log.get_actor_id_with_role_name("scenario")

        distances_list = []
        frames_list = []

        num_frames = metrics_log.get_simulation_frame_count()

        for frame in range(0, num_frames):

            hero_location = metrics_log.get_actor_location(hero, frame)
            adversary_location = metrics_log.get_actor_location(adversary, frame)

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

        return number_of_hits

    def _write_to_json(self, metrics):
        """
        Writes the metrics into a json file
        """
        self._file_name = "test.json"

        super(TestMetrics, self)._write_to_json(metrics)