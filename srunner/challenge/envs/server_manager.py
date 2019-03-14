from enum import Enum
import fcntl
import logging
import os
import psutil
import random
import string
import subprocess
import time

class Track(Enum):
    """
    Track modality.
    """
    SENSORS = 1
    NO_RENDERING = 2

class ServerManager(object):
    def __init__(self, opt_dict):
        log_level = logging.INFO
        logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

        self._proc = None
        self._outs = None
        self._errs = None

    def reset(self, host="127.0.0.1", port=2000):
        raise NotImplementedError("This function is to be implemented")


    def wait_until_ready(self, wait=10.0):
        time.sleep(wait)


class ServerManagerBinary(ServerManager):
    def __init__(self, opt_dict):
        super(ServerManagerBinary, self).__init__(opt_dict)

        if 'CARLA_SERVER' in opt_dict:
            self._carla_server_binary = opt_dict['CARLA_SERVER']
        else:
            logging.error('CARLA_SERVER binary not provided!')


    def reset(self, host="127.0.0.1", port=2000):
        self._i = 0
        # first we check if there is need to clean up
        if self._proc is not None:
            logging.info('Stopping previous server [PID=%s]', self._proc.pid)
            self._proc.kill()
            self._outs, self._errs = self._proc.communicate()

        exec_command = "{} -carla-rpc-port={} -benchmark -fps=20 -quality-level=Epic >/dev/null".format(
            self._carla_server_binary, port)
        print(exec_command)
        self._proc = subprocess.Popen(exec_command, shell=True)

    def stop(self):
        parent = psutil.Process(self._proc.pid)
        for child in parent.children(recursive=True):
            child.kill()
        parent.kill()
        self._outs, self._errs = self._proc.communicate()

    def check_input(self):
        while True:
            _ = self._proc.stdout.readline()
            print(self._i)
            self._i += 1

class ServerManagerDocker(ServerManager):
    def __init__(self, opt_dict):
        super(ServerManagerDocker, self).__init__(opt_dict)

        if 'DOCKER_VERSION' in opt_dict:
            self._docker_string = '{}'.format(opt_dict['DOCKER_VERSION'])
        else:
            logging.error('Docker version not provided!')

        self._docker_id = ''


    def reset(self, host="127.0.0.1", port=2000):

        # first we check if there is need to clean up
        if self._proc is not None and self._docker_id is not '':
            logging.info('Stopping previous server [PID=%s]', self._proc.pid)
            self.stop()
            self._proc.kill()
            self._outs, self._errs = self._proc.communicate()

        self._docker_id = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(64))
        # temporary config file

        exec_command = "docker run --name {} -p {}-{}:{}-{} --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 " \
                       "carlasim/carla:{} /bin/bash CarlaUE4.sh > -world-port={} -benchmark -fps=20 /dev/null".format(
            self._docker_id, port, port+2, port, port+2, self._docker_string, port)

        print(exec_command)
        self._proc = subprocess.Popen(exec_command)

    def stop(self):
        exec_command = ['docker', 'kill', '{}'.format(self._docker_id)]
        self._proc = subprocess.Popen(exec_command)