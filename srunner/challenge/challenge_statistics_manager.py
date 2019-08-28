#!/usr/bin/env python

# Copyright (c) 2018-2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module contains a statistics manager for the CARLA AD challenge
"""

from __future__ import print_function

import os
import json

import py_trees

from srunner.scenariomanager.traffic_events import TrafficEventType

PENALTY_COLLISION_STATIC = 6
PENALTY_COLLISION_VEHICLE = 6
PENALTY_COLLISION_PEDESTRIAN = 9
PENALTY_TRAFFIC_LIGHT = 3
PENALTY_WRONG_WAY = 2
PENALTY_SIDEWALK_INVASION = 2
PENALTY_STOP = 2


class ChallengeStatisticsManager(object):

    """
    This is the statistics manager for the CARLA AD challenge
    It gathers data at runtime via the scenario evaluation criteria and
    provides the final results as json output.

    Note: The class is purely static
    """

    system_error = None
    error_message = ""
    n_routes = 1
    statistics_routes = []

    current_route_score = 0
    current_penalty = 0
    list_collisions = []
    list_red_lights = []
    list_wrong_way = []
    list_route_dev = []
    list_sidewalk_inv = []
    list_stop_inf = []

    @staticmethod
    def set_number_of_scenarios(number):
        """
        Set the total number of scenarios
        """
        ChallengeStatisticsManager.n_routes = number

    @staticmethod
    def next_scenario(scenario):
        """
        Update the scenario to the next executed scenario
        """
        ChallengeStatisticsManager.scenario = scenario
        ChallengeStatisticsManager.error_message = ""

    @staticmethod
    def record_fatal_error(error_message):
        """
        Record the statistics in case of a fatal error (All scores = 0)
        """
        result = "ERROR"
        score_composed = 0.0
        score_penalty = 0.0
        score_route = 0.0
        ChallengeStatisticsManager.system_error = True

        return_message = error_message
        return_message += "\n=================================="

        current_statistics = {'id': -1,
                              'score_composed': score_composed,
                              'score_route': score_route,
                              'score_penalty': score_penalty,
                              'result': result,
                              'help_text': return_message
                              }

        ChallengeStatisticsManager.statistics_routes.append(current_statistics)

    @staticmethod
    def set_error_message(message):
        """
        Set the error message to the provided message
        """
        ChallengeStatisticsManager.error_message = message

    @staticmethod
    def compute_current_statistics():
        """
        Compute the current statistics by evaluating all relevant scenario criteria
        """
        target_reached = False
        score_penalty = 0.0
        score_route = 0.0
        list_traffic_events = []

        ChallengeStatisticsManager.list_collisions = []
        ChallengeStatisticsManager.list_red_lights = []
        ChallengeStatisticsManager.list_wrong_way = []
        ChallengeStatisticsManager.list_route_dev = []
        ChallengeStatisticsManager.list_sidewalk_inv = []
        ChallengeStatisticsManager.list_stop_inf = []

        for node in ChallengeStatisticsManager.scenario.get_criteria():
            if node.list_traffic_events:
                list_traffic_events.extend(node.list_traffic_events)

        # analyze all traffic events
        for event in list_traffic_events:
            if event.get_type() == TrafficEventType.COLLISION_STATIC:
                score_penalty += PENALTY_COLLISION_STATIC
                msg = event.get_message()
                if msg:
                    ChallengeStatisticsManager.list_collisions.append(event.get_message())

            elif event.get_type() == TrafficEventType.COLLISION_VEHICLE:
                score_penalty += PENALTY_COLLISION_VEHICLE
                msg = event.get_message()
                if msg:
                    ChallengeStatisticsManager.list_collisions.append(event.get_message())

            elif event.get_type() == TrafficEventType.COLLISION_PEDESTRIAN:
                score_penalty += PENALTY_COLLISION_PEDESTRIAN
                msg = event.get_message()
                if msg:
                    ChallengeStatisticsManager.list_collisions.append(event.get_message())

            elif event.get_type() == TrafficEventType.TRAFFIC_LIGHT_INFRACTION:
                score_penalty += PENALTY_TRAFFIC_LIGHT
                msg = event.get_message()
                if msg:
                    ChallengeStatisticsManager.list_red_lights.append(event.get_message())

            elif event.get_type() == TrafficEventType.WRONG_WAY_INFRACTION:
                score_penalty += PENALTY_WRONG_WAY
                msg = event.get_message()
                if msg:
                    ChallengeStatisticsManager.list_wrong_way.append(event.get_message())

            elif event.get_type() == TrafficEventType.ROUTE_DEVIATION:
                msg = event.get_message()
                if msg:
                    ChallengeStatisticsManager.list_route_dev.append(event.get_message())

            elif event.get_type() == TrafficEventType.ON_SIDEWALK_INFRACTION:
                score_penalty += PENALTY_SIDEWALK_INVASION
                msg = event.get_message()
                if msg:
                    ChallengeStatisticsManager.list_sidewalk_inv.append(event.get_message())

            elif event.get_type() == TrafficEventType.STOP_INFRACTION:
                score_penalty += PENALTY_STOP
                msg = event.get_message()
                if msg:
                    ChallengeStatisticsManager.list_stop_inf.append(event.get_message())

            elif event.get_type() == TrafficEventType.ROUTE_COMPLETED:
                score_route = 100.0
                target_reached = True
            elif event.get_type() == TrafficEventType.ROUTE_COMPLETION:
                if not target_reached:
                    if event.get_dict():
                        score_route = event.get_dict()['route_completed']
                    else:
                        score_route = 0

        ChallengeStatisticsManager.current_route_score = score_route
        ChallengeStatisticsManager.current_penalty = score_penalty

        print("Current Score: {}/{}".format(score_route, score_penalty))

    @staticmethod
    def record_scenario_statistics():
        """
        Record the statistics of the current scenario (route)

        In case of scenario failure, the last data gathered by compute_current_statistics will be used.
        """
        failure = False
        result = "SUCCESS"
        score_composed = 0.0
        return_message = ""
        route_id = ChallengeStatisticsManager.scenario.name.split('_')[1]

        if ChallengeStatisticsManager.error_message == "":
            for criterion in ChallengeStatisticsManager.scenario.get_criteria():
                if criterion.status == py_trees.common.Status.FAILURE:
                    failure = True
                    result = "FAILURE"
                    break

            if ChallengeStatisticsManager.scenario.timeout and not failure:
                result = "TIMEOUT"

            ChallengeStatisticsManager.compute_current_statistics()
        else:
            result = "CRASH"
            return_message += "\n=================================="
            return_message += "\nCrash message: {}".format(ChallengeStatisticsManager.error_message)
            return_message += "\n=================================="

        score_composed = max(
            ChallengeStatisticsManager.current_route_score - ChallengeStatisticsManager.current_penalty, 0.0)

        return_message += "\n=================================="
        # pylint: disable=line-too-long
        return_message += "\n==[r{}:{}] [Score = {:.2f} : (route_score={}, infractions=-{})]".format(route_id, result,
                                                                                                     score_composed,
                                                                                                     ChallengeStatisticsManager.current_route_score,
                                                                                                     ChallengeStatisticsManager.current_penalty)
        # pylint: enable=line-too-long

        if ChallengeStatisticsManager.list_collisions:
            return_message += "\n===== Collisions:"
            for item in ChallengeStatisticsManager.list_collisions:
                return_message += "\n========== {}".format(item)

        if ChallengeStatisticsManager.list_red_lights:
            return_message += "\n===== Red lights:"
            for item in ChallengeStatisticsManager.list_red_lights:
                return_message += "\n========== {}".format(item)

        if ChallengeStatisticsManager.list_stop_inf:
            return_message += "\n===== STOP infractions:"
            for item in ChallengeStatisticsManager.list_stop_inf:
                return_message += "\n========== {}".format(item)

        if ChallengeStatisticsManager.list_wrong_way:
            return_message += "\n===== Wrong way:"
            for item in ChallengeStatisticsManager.list_wrong_way:
                return_message += "\n========== {}".format(item)

        if ChallengeStatisticsManager.list_sidewalk_inv:
            return_message += "\n===== Sidewalk invasions:"
            for item in ChallengeStatisticsManager.list_sidewalk_inv:
                return_message += "\n========== {}".format(item)

        if ChallengeStatisticsManager.list_route_dev:
            return_message += "\n===== Route deviation:"
            for item in ChallengeStatisticsManager.list_route_dev:
                return_message += "\n========== {}".format(item)

        return_message += "\n=================================="

        current_statistics = {'id': route_id,
                              'score_composed': score_composed,
                              'score_route': ChallengeStatisticsManager.current_route_score,
                              'score_penalty': ChallengeStatisticsManager.current_penalty,
                              'result': result,
                              'help_text': return_message
                              }

        ChallengeStatisticsManager.statistics_routes.append(current_statistics)

    @staticmethod
    def report_challenge_statistics(filename, debug):
        """
        Print and save the challenge statistics over all routes
        """
        score_composed = 0.0
        score_route = 0.0
        score_penalty = 0.0
        help_message = ""

        phase_codename = os.getenv('CHALLENGE_PHASE_CODENAME', 'dev_track_3')
        phase = phase_codename.split("_")[0]

        if ChallengeStatisticsManager.system_error:
            submission_status = 'FAILED'

            for stats in ChallengeStatisticsManager.statistics_routes:
                help_message += "{}\n\n".format(stats['help_text'])

        else:
            submission_status = 'FINISHED'

            for stats in ChallengeStatisticsManager.statistics_routes:
                score_composed += stats['score_composed'] / float(ChallengeStatisticsManager.n_routes)
                score_route += stats['score_route'] / float(ChallengeStatisticsManager.n_routes)
                score_penalty += stats['score_penalty'] / float(ChallengeStatisticsManager.n_routes)
                help_message += "{}\n\n".format(stats['help_text'])

            if debug:
                print(help_message)

        # create json structure
        json_data = {
            'submission_status': submission_status,
            'stderr': help_message if phase == 'dev' or phase == 'debug' else 'No metadata provided for this phase',
            'result': [
                {
                    'split': phase,
                    'accuracies': {
                        'avg. route points': score_route,
                        'infraction points': score_penalty,
                        'total avg.': score_composed
                    }
                }],
            'metadata': [
                {
                    'stderr': help_message,
                    'accuracies': {
                        'avg. route points': score_route,
                        'infraction points': score_penalty,
                        'total avg.': score_composed
                    }
                }
            ]
        }

        with open(filename, "w+") as fd:
            fd.write(json.dumps(json_data, indent=4))
