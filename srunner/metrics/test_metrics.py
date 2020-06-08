import pprint
import json

from srunner.metrics.logging import Log
from srunner.metrics.basic_metric import BasicMetric


class TestMetrics(BasicMetric):
    """
    Support class for the users to easily create their own metrics.
    The user should only modify the create_metrics function, returning
    "something in particular"
    """

    def _create_metrics(self, metrics):
        """
        Implementation of the metrics
        """
        pp = pprint.PrettyPrinter(indent=4)

        # pp.pprint(metrics.actors)

        return metrics
