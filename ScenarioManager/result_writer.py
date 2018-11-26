#!/usr/bin/env python

#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module contains the result gatherer and write for CARLA scenarios.
It shall be used from the ScenarioManager only.
"""

import logging
import time


class ResultOutputProvider(object):

    """
    This module contains the result gatherer and write for CARLA scenarios.
    It shall be used from the ScenarioManager only.
    """

    def __init__(self, data, result, stdout=True, filename=None, junit=None):
        """
        Setup all parameters
        - data contains all scenario-related information
        - result is overall pass/fail info
        - stdout (True/False) is used to (de)activate terminal output
        - filename is used to (de)activate file output in tabular form
        - junit is used to (de)activate file output in junit form
        """
        self.data = data
        self.result = result
        self.stdout = stdout
        self.filename = filename
        self.junit = junit
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

        self.start_time = time.strftime('%Y-%m-%d %H:%M:%S',
                                        time.localtime(self.data.start_system_time))
        self.end_time = time.strftime('%Y-%m-%d %H:%M:%S',
                                      time.localtime(self.data.end_system_time))

    def write(self):
        """
        Public write function
        """
        if self.stdout:
            channel = logging.StreamHandler()
            self.logger.addHandler(channel)
        if self.filename is not None:
            filehandle = logging.FileHandler(self.filename)
            self.logger.addHandler(filehandle)
        if self.junit is not None:
            self.write_to_junit()

        if self.stdout or (self.filename is not None) or (self.junit is not None):
            self.write_to_logger()

    def write_to_logger(self):
        """
        Writing to logger automatically writes to all handlers in parallel,
        i.e. stdout and file are both captured with this function
        """
        self.logger.info("\n")
        self.logger.info("Scenario: %s --- Result: %s" %
                         (self.data.scenario_tree.name, self.result))
        self.logger.info("Start time: %s" % (self.start_time))
        self.logger.info("End time: %s" % (self.end_time))
        self.logger.info("Duration: System Time %5.2fs --- Game Time %5.2fs" %
                         (self.data.scenario_duration_system,
                          self.data.scenario_duration_game))

        self.logger.info("\n")
        self.logger.info(
            "           Criterion           |  Result  | Actual Value | Expected Value ")
        self.logger.info(
            "--------------------------------------------------------------------------")

        for criterion in self.data.scenario.test_criteria:
            self.logger.info("%30s | %8s | %12.2f | %12.2f " %
                             (criterion.name,
                              "SUCCESS" if criterion.test_status == "SUCCESS" else "FAILURE",
                              criterion.actual_value,
                              criterion.expected_value))

        # Handle timeout separately
        self.logger.info("%30s | %8s | %12.2f | %12.2f " %
                         ("Duration",
                          "SUCCESS" if self.data.scenario_duration_game < self.data.scenario.timeout else "FAILURE",
                          self.data.scenario_duration_game,
                          self.data.scenario.timeout))

        self.logger.info("\n")

    def write_to_junit(self):
        """
        Writing to Junit XML
        """
        test_count = 0
        failure_count = 0
        for criterion in self.data.scenario.test_criteria:
            test_count += 1
            if criterion.test_status != "SUCCESS":
                failure_count += 1

        junit_file = open(self.junit, "w")

        junit_file.write("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n")

        test_suites_string = ("<testsuites tests=\"%d\" failures=\"%d\" disabled=\"0\" "
                              "errors=\"0\" timestamp=\"%s\" time=\"%5.2f\" "
                              "name=\"Simulation\" package=\"Scenarios\">\n" %
                              (test_count,
                               failure_count,
                               self.start_time,
                               self.data.scenario_duration_system))
        junit_file.write(test_suites_string)

        test_suite_string = ("  <testsuite name=\"%s\" tests=\"%d\" failures=\"%d\" "
                             "disabled=\"0\" errors=\"0\" time=\"%5.2f\">\n" %
                             (self.data.scenario_tree.name,
                              test_count,
                              failure_count,
                              self.data.scenario_duration_system))
        junit_file.write(test_suite_string)

        for criterion in self.data.scenario.test_criteria:
            result_string = ("    <testcase name=\"{}\" status=\"run\" "
                             "time=\"0\" classname=\"Scenarios.{}\">\n".format(
                                 criterion.name, self.data.scenario_tree.name))
            if criterion.test_status == "FAILURE":
                result_string += "      <failure message=\"{}\"  type=\"\"><!\[CDATA\[\n".format(
                    criterion.name)
                result_string += "  Actual:   {}\n".format(
                    criterion.actual_value)
                result_string += "  Expected: {}\n".format(
                    criterion.expected_value)
                result_string += "\n"
                result_string += "  Exact Value: {} = {}\]\]></failure>\n".format(
                    criterion.name, criterion.actual_value)
            else:
                result_string += "  Exact Value: {} = {}\n".format(
                    criterion.name, criterion.actual_value)
            result_string += "    </testcase>\n"
            junit_file.write(result_string)

        # Handle timeout separately
        result_string = ("    <testcase name=\"{}\" status=\"run\" time=\"{}\" "
                         "classname=\"Scenarios.{}\">\n".format(
                             criterion.name,
                             self.data.scenario_duration_system,
                             "Duration"))
        if self.data.scenario_duration_game >= self.data.scenario.timeout:
            result_string += "      <failure message=\"{}\"  type=\"\"><!\[CDATA\[\n".format(
                "Duration")
            result_string += "  Actual:   {}\n".format(
                self.data.scenario_duration_game)
            result_string += "  Expected: {}\n".format(
                self.data.scenario.timeout)
            result_string += "\n"
            result_string += "  Exact Value: {} = {}\]\]></failure>\n".format(
                "Duration", self.data.scenario_duration_game)
        else:
            result_string += "  Exact Value: {} = {}\n".format(
                "Duration", self.data.scenario_duration_game)
        result_string += "    </testcase>\n"
        junit_file.write(result_string)

        junit_file.write("  </testsuite>\n")
        junit_file.write("</testsuites>\n")
        junit_file.close()
