FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG http_proxy

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         vim \
         ca-certificates \
         libjpeg-dev \
	     libpng16-16 \
	     libtiff5 \
         libpng-dev \
         python-dev \
         python3.5 \
         python3.5-dev \
         python-networkx \
         python-setuptools \
         python3-setuptools \
         python-pip \
         python3-pip && \
         pip install --upgrade pip && \
         pip3 install --upgrade pip && \
         rm -rf /var/lib/apt/lists/*

RUN pip2 install psutil py_trees==0.8.3 shapely six && \
         pip3 install py_trees==0.8.3  shapely six


WORKDIR /workspace
COPY PythonAPI /workspace/CARLA/PythonAPI
COPY HDMaps /workspace/CARLA/HDMaps
ENV CARLA_ROOT /workspace/CARLA

# installing conda
RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda clean -ya && \
     /opt/conda/bin/conda create -n python35 python=3.5 numpy networkx scipy six
ENV PATH "/workspace/CARLA/PythonAPI/carla/dist/carla-0.9.5-py3.5-linux-x86_64.egg":/opt/conda/envs/python35/bin:/opt/conda/envs/bin:$PATH

# adding CARLA egg to default python environment
RUN pip install --user setuptools py_trees==0.8.3 psutil shapely six

ENV ROOT_SCENARIO_RUNNER "/workspace/scenario_runner"
ENV PYTHONPATH "/workspace/CARLA/PythonAPI/carla/dist/carla-0.9.5-py3.5-linux-x86_64.egg":"${ROOT_SCENARIO_RUNNER}":"${CARLA_ROOT}/PythonAPI/carla":${PYTHONPATH}

COPY .tmp /workspace/scenario_runner
COPY team_code /workspace/team_code
RUN mkdir -p /workspace/results
RUN chmod +x /workspace/scenario_runner/srunner/challenge/run_evaluator.sh


########################################################################################################################
########################################################################################################################
############                                BEGINNING OF USER COMMANDS                                      ############
########################################################################################################################
########################################################################################################################

ENV TEAM_AGENT /workspace/team_code/YOUR_PYTHON_CODE.py
ENV TEAM_CONFIG /workspace/team_code/YOUR_CONFIG_FILE


########################################################################################################################
########################################################################################################################
############                                   END OF USER COMMANDS                                         ############
########################################################################################################################
########################################################################################################################

ENV HTTP_PROXY ""
ENV HTTPS_PROXY ""
ENV http_proxy ""
ENV https_proxy ""


CMD ["/bin/bash"]
