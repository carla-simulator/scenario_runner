from tabulate import tabulate

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

        ### Parsing of the criterias into a printable table ###

        output = "\n"
        output += " ======= Metrics of the scenario =======\n"
        output += "\n"

        # Get all the criterias
        criteria = metrics_log.get_criteria()
        list_criteria = [['Name', 'Result', 'Actual Value', 'Expected Value']]

        for criterion_name in criteria:
            criterion = criteria[criterion_name]
            name = criterion_name
            actual_value = criterion["actual_value"]
            status = criterion["test_status"]
            expected_value = criterion["expected_value_success"]

            list_criteria.extend([[name, status, actual_value, expected_value]])

        output += tabulate(list_criteria, tablefmt='fancy_grid')
        output += "\n"

        print(output)
