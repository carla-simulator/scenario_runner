#!/bin/bash

autopep8 scenario_runner.py --in-place --max-line-length=120
autopep8 srunner/scenariomanager/*.py --in-place --max-line-length=120
autopep8 srunner/scenarios/*.py --in-place --max-line-length=120
autopep8 srunner/challenge/*.py --in-place --max-line-length=120
autopep8 srunner/tools/*.py --in-place --max-line-length=120


pylint --rcfile=.pylintrc --disable=I srunner/scenariomanager/
pylint --rcfile=.pylintrc srunner/scenarios/
#pylint --rcfile=.pylintrc srunner/challenge/
pylint --rcfile=.pylintrc srunner/tools/
pylint --rcfile=.pylintrc scenario_runner.py
