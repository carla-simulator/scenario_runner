#!/bin/bash

# !Make sure you set $CHALLENGE_PHASE_CODENAME (e.g. dev_track_3)
# !Make sure you set $CHALLENGE_TIME_AVAILABLE (e.g. 10000)

python ${ROOT_SCENARIO_RUNNER}/srunner/challenge/challenge_evaluator_routes.py \
--scenarios=${ROOT_SCENARIO_RUNNER}/srunner/challenge/all_towns_traffic_scenarios1_3_4.json \
--routes=${ROOT_SCENARIO_RUNNER}/srunner/challenge/routes_training.xml \
--debug=0 \
--agent=${ROOT_SCENARIO_RUNNER}/srunner/challenge/autoagents/HumanAgent.py
