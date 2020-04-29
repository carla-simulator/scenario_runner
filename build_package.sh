#!/bin/bash
rm -rf build/ dist/ scenario_runner.egg-info/
python3 setup_py3.py sdist bdist_wheel

