ARG UBUNTU_VERSION="22.04"
ARG CARLA_REMOTE_PATH="https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.13.tar.gz"

############ PythonAPI ############
FROM "ubuntu:${UBUNTU_VERSION}" as artifacts

USER root
SHELL ["/bin/bash", "-c"]
ARG CARLA_REMOTE_PATH

# Install wget
RUN apt-get update && \
    apt-get install -y \
      wget \
    && rm -rf /var/lib/apt/lists/*

# Download and extract API
RUN wget -qO- "${CARLA_REMOTE_PATH}" | tar -xzv PythonAPI/carla

############ dependencies ############
ARG UBUNTU_VERSION
FROM "ubuntu:${UBUNTU_VERSION}" as dependencies

USER root
SHELL ["/bin/bash", "-c"]

ARG CARLA_API_PATH=/opt/carla/PythonAPI
ARG SCENARIO_RUNNER_PATH=/opt/carla/scenario_runner

# Install essentials
RUN apt-get update && \
    apt-get install -y \
      libpng16-16 \
      libtiff5 \
      libjpeg8 \
      build-essential \
      wget \
      git \
      libxerces-c-dev \
      python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 10

# Install scenario_runner
COPY . $SCENARIO_RUNNER_PATH
RUN pip3 install -r $SCENARIO_RUNNER_PATH/requirements.txt

# Copy over PythonAPI
COPY --from=artifacts PythonAPI ${CARLA_API_PATH}

# Recursively install PythonAPI requirements, keep version of first occurrence
RUN cat $(find ${CARLA_API_PATH} -type f -name "requirements.txt") > /tmp/requirements_raw.txt \
    && awk -F '==' '{print $1}' /tmp/requirements_raw.txt | awk '!visited[$1]++' > /tmp/requirements.txt \
    && pip3 install -r /tmp/requirements.txt

# Create script that adds API to pythonpath and make .bashrc source it
RUN echo "export PYTHONPATH=\$PYTHONPATH:$CARLA_API_PATH/carla/dist/$(ls $CARLA_API_PATH/carla/dist | grep py$(python --version | awk -F'[ .]' '{print $2"."$3}').)" >> /setup_carla_env.sh; \
    echo "export PYTHONPATH=\$PYTHONPATH:$CARLA_API_PATH/carla/agents" >> /setup_carla_env.sh; \
    echo "export PYTHONPATH=\$PYTHONPATH:$CARLA_API_PATH/carla" >> /setup_carla_env.sh; \
    echo "export CARLA_API_PATH=$CARLA_API_PATH" >> /setup_carla_env.sh; \
    echo "source /setup_carla_env.sh" >> ~/.bashrc

############ final ############
FROM dependencies as final

# Needed for (pygame based) scripts that have a GUI
ENV SDL_VIDEODRIVER=x11

USER root
SHELL ["/bin/bash", "-c"]

ARG WORKSPACE=${SCENARIO_RUNNER_PATH}
ENV WORKSPACE=${WORKSPACE}
WORKDIR ${WORKSPACE}

# Set entrypoint
COPY ./docker/entrypoint.sh /
ENTRYPOINT [ "/entrypoint.sh" ]

CMD [ "python3", "scenario_runner.py", "--help" ]
