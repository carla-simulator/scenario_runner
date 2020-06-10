import pprint
import json

from srunner.metrics.metrics_log import MetricsLog


class BasicMetric(object):
    """
    Base class for all the user-made metrics. It instanciates a
    MetricsLog object with all the information of the simulation
    and creates the desired metric
    """

    def __init__(self, recorder, criteria=None):
        """
        Initialization of the metrics class.

        Args:
            recorder (dict): dictionary with all the information
                of the simulation
            criteria (list): list of dictionaries with all the
                information regarding the criterias used 
        """

        # Instanciate the MetricsLog, used to querry the needed info
        metrics_log = MetricsLog(recorder, criteria)

        # Create the metrics of the simulation. This part is left to the user
        metrics = self._create_metrics(metrics_log)

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
