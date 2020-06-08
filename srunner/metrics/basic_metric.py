import pprint
import json

from srunner.metrics.logging import Log


class BasicMetric(object):
    """
    Support class for the users to easily create their own metrics.
    The user should only modify the create_metrics function, returning
    "something in particular"
    """

    def __init__(self, recorder_info, criteria_info, log_path):
        """
        Initialization of the metrics class.

        Args:
            log_location (str): name of the log file
        """
        self.log_path = log_path

        log = Log(recorder_info, criteria_info)

        metrics = self._create_metrics(log)

        # self._write_to_terminal(metrics)

        self._write_to_json(metrics)

    def _create_metrics(self, metrics):
        """
        Pure virtual function to setup the metrics by the user
        """
        raise NotImplementedError(
            "This function should be re-implemented by all metrics"
            "If this error becomes visible the class hierarchy is somehow broken")

    def _write_to_terminal(self, metrics):
        """
        Print the metrics table through the terminal
        """
        pass


    def _write_to_json(self, metrics):
        """
        Writes the metrics into a json file
        """
        file_name = "srunner/metrics/" + self.log_path.split("/")[-1][:-4] + "_metrics.json"

        with open(file_name, 'w') as fp:
            json.dump(metrics.states, fp, sort_keys=True, indent=4)
            json.dump(metrics.actors, fp, sort_keys=True, indent=4)
