#!/bin/bash

# !Make sure you set $CHALLENGE_PHASE_CODENAME (e.g. dev_track_3)
# !Make sure you set $CHALLENGE_TIME_AVAILABLE (e.g. 10000)

python ${ROOT_SCENARIO_RUNNER}/srunner/challenge/challenge_evaluator_routes.py \
--scenarios=${ROOT_SCENARIO_RUNNER}/srunner/challenge/all_towns_traffic_scenarios1_3_4.json \
--debug=0 \
--agent=${TEAM_AGENT} \
--config=${TEAM_CONFIG}