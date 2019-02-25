#!/bin/bash
# Copyright (c) 2018-2019 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
#
# Setup the environment variables needed to find the CARLA server and PythonAPI
#

usage() {
	echo -e "Run: $0 [OPTIONS...] [OTHER ARGUMENTS]"
	echo ""
	echo "options:"
	echo -e "\t--carla-root PATH : Absolute path to the CARLA root folder "
}

if [ $# -lt 1 ]
then
	usage
	exit -1
fi

if [[ $1 == "-h" || $1 == "--help" ]];
then
	usage
	exit 0
elif [[ $1 == "--carla-root" && $# -gt 1 ]]
then
    CARLA_ROOT=$2
    CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh

    echo " " >> ${HOME}/.bashrc
    echo "export CARLA_ROOT=${CARLA_ROOT}" >> ${HOME}/.bashrc
    echo "export CARLA_SERVER=${CARLA_SERVER}" >> ${HOME}/.bashrc
    echo "export PYTHONPATH=\"${CARLA_ROOT}/PythonAPI/:`pwd`:${PYTHONPATH}\" " >> ${HOME}/.bashrc
    echo "export ROOT_SCENARIO_RUNNER=`pwd`" >> ${HOME}/.bashrc

    echo "== CARLA server setup successfully!"
    echo "== Remember to run: source ${HOME}/.bashrc"

    exit 0
else
    usage
    exit -1
fi


