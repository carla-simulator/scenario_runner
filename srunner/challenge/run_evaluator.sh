#!/bin/bash

python3 ${ROOT_SCENARIO_RUNNER}/srunner/challenge/test_routes.py \
--scenarios=${ROOT_SCENARIO_RUNNER}/srunner/challenge/all_towns_traffic_scenarios.json \
--routes=${ROOT_SCENARIO_RUNNER}/srunner/challenge/routes_training.xml \
--agent=${ROOT_SCENARIO_RUNNER}/srunner/challenge/autoagents/HumanAgent.py
