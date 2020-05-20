from ubuntu:18.04

# Install base libs
run apt-get update && apt-get install --no-install-recommends -y \
    libpng16-16 libtiff5 libjpeg8 build-essential wget git python3.6 python3.6-dev python3-pip 

# Install python requirements
run pip3 install --user setuptools wheel && pip3 install py_trees==0.8.3 networkx==2.2 pygame==1.9.6 \
    six==1.14.0 numpy==1.18.4 psutil shapely xmlschema ephem==3.7.6.0 \
&& mkdir -p /app/scenario_runner

# Install scenario_runner 
add . /app/scenario_runner

# setup environment :
# 
#   CARLA_HOST :    uri for carla package without trailing slash. 
#                   For example, "https://carla-releases.s3.eu-west-3.amazonaws.com/Linux".
#                   If this environment is not passed to docker build, the value
#		    is taken from CARLA_VER file inside the repository.
#
#   CARLA_RELEASE : Name of the package to be used. For example, "CARLA_0.9.9".
#                   If this environment is not passed to docker build, the value
#                   is taken from CARLA_VER file inside the repository.
# 
#
#  It's expected that $(CARLA_HOST)/$(CARLA_RELEASE).tar.gz is a downloadable resource.
#

env CARLA_HOST ""
env CARLA_RELEASE ""

# Extract and install python API and resources from CARLA

run export DEFAULT_CARLA_HOST=$(cat /app/scenario_runner/CARLA_VER|grep HOST|sed 's/HOST\s*=\s*//g') \
&&  export CARLA_HOST=${CARLA_HOST:-$DEFAULT_CARLA_HOST} \
&&  DEFAULT_CARLA_RELEASE=$(cat /app/scenario_runner/CARLA_VER|grep RELEASE|sed 's/RELEASE\s*=\s*//g') \
&&  export CARLA_RELEASE=${CARLA_RELEASE:-$DEFAULT_CARLA_RELEASE} \
&&  echo $CARLA_HOST/$CARLA_RELEASE.tar.gz \
&&  wget -qO- $CARLA_HOST/$CARLA_RELEASE.tar.gz | tar -xzv PythonAPI/carla -C / \
&&  mv /PythonAPI/carla /app/ \
&&  python3 -m easy_install --no-find-links --no-deps $(find /app/carla/ -iname "*py3.*.egg" )


# Setup working environment
workdir /app/scenario_runner
env PYTHONPATH "${PYTHONPATH}:/app/carla/agents:/app/carla"
entrypoint ["/bin/sh" ]

