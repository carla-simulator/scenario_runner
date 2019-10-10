#!/bin/bash

autopep8 scenario_runner.py --in-place --max-line-length=120
autopep8 srunner/scenariomanager/*.py --in-place --max-line-length=120
autopep8 srunner/scenariomanager/scenarioatomics/*.py --in-place --max-line-length=120
autopep8 srunner/scenarios/*.py --in-place --max-line-length=120
autopep8 srunner/challenge/*.py --in-place --max-line-length=120
autopep8 srunner/challenge/autoagents/*.py --in-place --max-line-length=120
autopep8 srunner/tools/*.py --in-place --max-line-length=120
autopep8 srunner/scenarioconfigs/*.py --in-place --max-line-length=120


pylint --rcfile=.pylintrc --disable=I --extension-pkg-whitelist=numpy srunner/scenariomanager/
pylint --rcfile=.pylintrc --extension-pkg-whitelist=numpy srunner/scenarios/
pylint --rcfile=.pylintrc --extension-pkg-whitelist=numpy srunner/challenge/autoagents/
pylint --rcfile=.pylintrc --extension-pkg-whitelist=numpy srunner/challenge/challenge_statistics_manager.py
pylint --rcfile=.pylintrc --extension-pkg-whitelist=numpy srunner/tools/
pylint --rcfile=.pylintrc --extension-pkg-whitelist=numpy srunner/scenarioconfigs/
pylint --rcfile=.pylintrc --extension-pkg-whitelist=numpy scenario_runner.py
