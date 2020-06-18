from srunner.metrics.tools.metrics_log import MetricsLog

class BasicMetric(object):
    """
    Base class of all the metrics.
    """

    def __init__(self, town_map, recorder, criteria=None):
        """
        Initialization of the metric class. This calls the metrics log and creates the metrics

        Args:
            town_map (carla.Map): map of the simulation. Used to access the Waypoint API.
            recorder (dict): dictionary with all the information of the simulation
            criteria (list): list of dictionaries with all the criteria information
        """
        self._map = town_map

        # Instanciate the MetricsLog, used to querry the needed info
        metrics_log = MetricsLog(recorder, criteria)

        # Create the metrics of the simulation. This part is left to the user
        self._create_metrics(metrics_log)

    def _create_metrics(self, metrics_log):
        """
        Pure virtual function to setup the metrics by the user.

        Args:
            metrics_log (srunner.metrics.metrics_log.MetricsLog): class with all the
                information passed through the MetricsManager. this information has been
                parsed and some function have been added for easy access
        """
        raise NotImplementedError(
            "This function should be re-implemented by all metrics"
            "If this error becomes visible the class hierarchy is somehow broken")
