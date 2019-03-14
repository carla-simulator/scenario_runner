#!/bin/bash

TEAM_AGENT=/workspace/team_code/YOUR_AGENT.py
TEAM_CONFIG=/workspace/team_code/YOUR_CONFIG_FILE

echo "export TEAM_AGENT=${TEAM_AGENT}" >> ~/.bashrc
echo "export TEAM_CONFIG=${TEAM_CONFIG}" >> ~/.bashrc
