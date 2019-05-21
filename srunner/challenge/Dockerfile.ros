ARG CARLA_VERSION=0.9.5
ARG CARLA_BUILD=''

ARG ROS_VERSION=kinetic-ros-base-xenial

FROM carlasim/carla:$CARLA_VERSION$CARLA_BUILD as carla

FROM ros:$ROS_VERSION

ARG CARLA_VERSION

RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-kinetic-tf \
        ros-kinetic-ackermann-msgs \
        ros-kinetic-derived-object-msgs \
        ros-kinetic-pcl-conversions \
        ros-kinetic-pcl-ros \
        ros-kinetic-cv-bridge \
        libjpeg8 \
        libpng16-16 \
        libtiff5 \
        python3.5 \
        python-setuptools \
        python3-setuptools \
        python-wheel \
        python3-wheel \
        python-pip \
        python3-pip && \
        rm -rf /var/lib/apt/lists/*

RUN pip install py_trees==0.8.3 shapely six numpy networkx==2.2 scipy && \
    pip3 install py_trees==0.8.3 shapely six numpy networkx==2.2 scipy

COPY --from=carla /home/carla/PythonAPI /workspace/CARLA/PythonAPI
COPY --from=carla /home/carla/HDMaps /workspace/CARLA/HDMaps

ENV CARLA_ROOT /workspace/CARLA

ENV ROOT_SCENARIO_RUNNER /workspace/scenario_runner

#assuming ROS kinetic using Python 2.7
ENV PYTHONPATH "${ROOT_SCENARIO_RUNNER}":"${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.5-py2.7-linux-x86_64.egg":"${CARLA_ROOT}/PythonAPI/carla":${PYTHONPATH}

RUN mkdir -p /workspace/results

COPY scenario_runner /workspace/scenario_runner
COPY team_code /workspace/team_code
RUN chmod +x /workspace/scenario_runner/srunner/challenge/run_evaluator.sh

RUN /bin/bash -c 'source /opt/ros/kinetic/setup.bash; cd /workspace/team_code/catkin_ws/src; catkin_init_workspace; cd ../; catkin_make -DCMAKE_BUILD_TYPE=Release'

WORKDIR /workspace

########################################################################################################################
########################################################################################################################
############                                BEGINNING OF USER COMMANDS                                      ############
########################################################################################################################
########################################################################################################################

COPY ros.bashrc /root/.bashrc

ENV TEAM_AGENT /workspace/team_code/YOUR_PYTHON_CODE.py
ENV TEAM_CONFIG /workspace/team_code/YOUR_CONFIG_FILE

########################################################################################################################
########################################################################################################################
############                                   END OF USER COMMANDS                                         ############
########################################################################################################################
########################################################################################################################

CMD ["/bin/bash"]
