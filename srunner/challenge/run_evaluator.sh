#!/bin/bash

python ${ROOT_SCENARIO_RUNNER}/srunner/challenge/challenge_evaluator_routes.py \
--scenarios=${ROOT_SCENARIO_RUNNER}/srunner/challenge/no_scenarios.json \
--routes=${ROOT_SCENARIO_RUNNER}/srunner/challenge/routes_training.xml \
--debug=1 \
--agent=${ROOT_SCENARIO_RUNNER}/srunner/challenge/autoagents/HumanAgent.py
