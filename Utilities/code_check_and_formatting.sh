#!/bin/bash

autopep8 scenario_runner.py --in-place --max-line-length=120
autopep8 ScenarioManager/*.py --in-place --max-line-length=120
autopep8 Scenarios/*.py --in-place --max-line-length=120

pylint --rcfile=.pylintrc --disable=I ScenarioManager/
pylint --rcfile=.pylintrc Scenarios/
pylint --rcfile=.pylintrc scenario_runner.py
