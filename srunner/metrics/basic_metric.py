import pprint
import json

from srunner.metrics.metrics_log import MetricsLog


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
        self._log_path = log_path
        self._file_name = "srunner/metrics/" + self._log_path.split("/")[-1][:-4] + "_metrics.json"

        self._metrics_log = MetricsLog(recorder_info, criteria_info)

        metrics = self._create_metrics(self._metrics_log)

        # self._write_to_terminal(metrics)

        self._write_to_json(metrics)

    def _create_metrics(self, metrics_log):
        """
        Pure virtual function to setup the metrics by the user.
        
        Args:
            metrics_log (srunner.metrics.metrics_log.MetricsLog): has all the
                information regarding a specific simulation. This is taken from
                parsing the CARLA recorder into variables

        metrics_log consits of three attributes:
            - states: list of dictionaries. Each dictionary contains all the information
                of the simulation at a specific frame.
            - actors: dictionary of (ID - actor info) regarding the actors part of the simulation
            - criteria: dictionary with all the criterias and its attributes, read from a .json file.
                This file is automatically created by ScenarioRunner if the recorder is on
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

        with open(self._file_name, 'w') as fp:
            json.dump(metrics, fp, sort_keys=True, indent=4)
