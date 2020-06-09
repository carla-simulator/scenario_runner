import pprint
import json

from srunner.metrics.metrics_log import MetricsLog
from srunner.metrics.basic_metric import BasicMetric


class TestMetrics(BasicMetric):
    """
    Support class for the users to easily create their own metrics.
    The user should only modify the create_metrics function, returning
    "something in particular"
    """

    def _create_metrics(self, metrics_log):
        """
        Implementation of the metrics
        """
        hero = metrics_log.get_vehicle_id_with_role_name("hero")
        adversary = metrics_log.get_vehicle_id_with_role_name("scenario")

        hero_locations = metrics_log.get_all_actor_locations(hero)
        adversary_locations = metrics_log.get_all_actor_locations(adversary)

        collision_criteria = metrics_log.get_criteria("CollisionTest")

        print(hero_locations)
        print(adversary_locations)

        return collision_criteria["actual_value"]

    def _write_to_json(self, metrics):
        """
        Writes the metrics into a json file
        """
        self._file_name = "test.json"

        super(TestMetrics, self)._write_to_json(metrics)