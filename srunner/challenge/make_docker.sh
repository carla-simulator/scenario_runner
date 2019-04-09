#!/bin/bash

if [ -z "$CARLA_ROOT" ]
then
    echo "Error $CARLA_ROOT is empty. Set \$CARLA_ROOT as an environment variable first."
    exit 1
fi

if [ -z "$ROOT_SCENARIO_RUNNER" ]
then echo "Error $ROOT_SCENARIO_RUNNER is empty. Set \$ROOT_SCENARIO_RUNNER as an environment variable first."
    exit 1
fi

# copy required resources from the host
cp -fr ${CARLA_ROOT}/PythonAPI  .

mkdir .tmp
git clone -b carla_challenge --single-branch https://github.com/carla-simulator/scenario_runner.git .tmp


# build docker image
docker build --force-rm --build-arg HTTP_PROXY=${HTTP_PROXY} \
             --build-arg HTTPS_PROXY=${HTTPS_PROXY} \
             --build-arg http_proxy=${http_proxy} \
             -t carla-challenge-master -f ${ROOT_SCENARIO_RUNNER}/srunner/challenge/Dockerfile.master .

rm -fr .tmp