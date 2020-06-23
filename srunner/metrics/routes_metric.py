"""
RouteScenario metric:

This metric filters the useful information of the criteria (sucess / fail ...),
and dump it into a json file

It is meant to serve as an example of how to use the criteria
"""

import json

from srunner.metrics.basic_metric import BasicMetric


class RoutesMetric(BasicMetric):
    """
    Class containing a metric of the Route scenario.
    """

    def _create_metrics(self, metrics_log):
        """
        Implementation of the metric. This is an example to show how to use the criteria,
        accessed via the metrics_log.
        """

        ### Parse the criteria information, filtering only the useful information, and dump it into a json ###

        criteria = metrics_log.get_criteria()

        results = {}
        for criterion_name in criteria:
            criterion = criteria[criterion_name]
            results.update({criterion_name:
                {
                    "test_status": criterion["test_status"],
                    "actual_value": criterion["actual_value"],
                    "success_value": criterion["expected_value_success"]
                }
            }
        )

        with open('srunner/metrics/data/Routes_metric.json', 'w') as fw:
            json.dump(results, fw, sort_keys=False, indent=4)
