import carla
import os
import glob
import sys
import argparse
import importlib
import inspect
import json
from argparse import RawTextHelpFormatter


class MetricsManager(object):
    """
    Main class of the metrics module. Handles the parsing and execution of
    the metrics.
    """

    def __init__(self, args):

        self._args = args
        self.client = carla.Client(self._args.host, self._args.port)
        self.client.set_timeout(self._args.timeout)

    def get_metric_class(self, metric_file):

        # Path of all scenario at "srunner/scenarios" folder + the path of the additional scenario argument

        # Get their module
        module_name = os.path.basename(metric_file).split('.')[0]
        sys.path.insert(0, os.path.dirname(metric_file))
        metric_module = importlib.import_module(module_name)

        # And their members of type class
        for member in inspect.getmembers(metric_module, inspect.isclass):
            # Get the first one with paren BasicMetrics
            member_parent = member[1].__bases__[0]
            if 'BasicMetric' in str(member_parent):
                return member[1]

        print("No class was found that was a child of BasicMetric ... Exiting")
        sys.exit(-1)

    def run(self):
        """
        Executes the metric class given by the arguments
        """
        # Read the metric argument to get the metric class
        metric_class = self.get_metric_class(self._args.metric)

        # Get the log information. Here to avoid passing the client instance
        log_file = "{}/{}".format(os.getenv('SCENARIO_RUNNER_ROOT', "./"), self._args.log)

        recorder_info = self.client.show_recorder_file_info(log_file, True)
        with open(self._args.criteria) as criteria_file:
            criteria_info = json.load(criteria_file)

        # Run the metric class
        metric = metric_class(recorder_info, criteria_info, self._args.log)


def main():
    """
    main function
    """
    description = ("Scenario Runner's metrics module. Evaluate the scenario's execution\n"
                   "Current version: ")

    parser = argparse.ArgumentParser(description=description,
                                    formatter_class=RawTextHelpFormatter)
    parser.add_argument('--host', default='127.0.0.1',
                        help='IP of the host server (default: localhost)')
    parser.add_argument('--port', default=2000,
                        help='TCP port to listen to (default: 2000)')
    parser.add_argument('--timeout', default=10.0,
                        help='Set the CARLA client timeout value in seconds')
    parser.add_argument('--metric', default="",
                        help='Path to the .py file definning the used criterias')
    parser.add_argument('--log', default="",
                        help='Path to the .log file')
    parser.add_argument('--criteria', default="",
                        help='path to the .json file with the criteria information')

    args = parser.parse_args()

    # log_location = "{}/{}.log".format(os.getenv('SCENARIO_RUNNER_ROOT', "./"), '/srunner/metrics/ControlLoss_1')
    metrics_manager = MetricsManager(args)
    metrics_manager.run()

if __name__ == "__main__":
    sys.exit(main())
