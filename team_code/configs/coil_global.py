from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from ast import literal_eval
from coilutils import AttributeDict
import copy
import numpy as np
import os
import yaml

from configs.namer import generate_name



_g_conf = AttributeDict()


_g_conf.immutable(False)

"""#### GENERAL CONFIGURATION PARAMETERS ####"""
_g_conf.NUMBER_OF_LOADING_WORKERS = 12
_g_conf.FINISH_ON_VALIDATION_STALE = None


"""#### INPUT RELATED CONFIGURATION PARAMETERS ####"""
_g_conf.SENSORS = {'rgb': (3, 88, 200)}
_g_conf.MEASUREMENTS = {'float_data': (31)}
_g_conf.TARGETS = ['steer', 'throttle', 'brake']
_g_conf.INPUTS = ['speed_module']
_g_conf.INTENTIONS = []
_g_conf.BALANCE_DATA = True
_g_conf.STEERING_DIVISION = [0.05, 0.05, 0.1, 0.3, 0.3, 0.1, 0.05, 0.05]
_g_conf.PEDESTRIAN_PERCENTAGE = 0
_g_conf.SPEED_DIVISION = []
_g_conf.LABELS_DIVISION = [[0, 2, 5], [3], [4]]
_g_conf.BATCH_SIZE = 120
_g_conf.SPLIT = None
_g_conf.REMOVE = None
_g_conf.AUGMENTATION = None


_g_conf.DATA_USED = 'all' #  central, all, sides,
_g_conf.USE_NOISE_DATA = True
_g_conf.TRAIN_DATASET_NAME = '1HoursW1-3-6-8'  # We only set the dataset in configuration for training
_g_conf.LOG_SCALAR_WRITING_FREQUENCY = 2   # TODO NEEDS TO BE TESTED ON THE LOGGING FUNCTION ON  CREATE LOG
_g_conf.LOG_IMAGE_WRITING_FREQUENCY = 1000
_g_conf.EXPERIMENT_BATCH_NAME = "eccv"
_g_conf.EXPERIMENT_NAME = "default"
_g_conf.EXPERIMENT_GENERATED_NAME = None

# TODO: not necessarily the configuration need to know about this
_g_conf.PROCESS_NAME = "None"
_g_conf.NUMBER_ITERATIONS = 20000
_g_conf.SAVE_SCHEDULE = range(0, 2000, 200)
_g_conf.NUMBER_FRAMES_FUSION = 1
_g_conf.NUMBER_IMAGES_SEQUENCE = 1
_g_conf.SEQUENCE_STRIDE = 1
_g_conf.TEST_SCHEDULE = range(0, 2000, 200)
_g_conf.SPEED_FACTOR = 12.0
_g_conf.AUGMENT_LATERAL_STEERINGS = 6
_g_conf.NUMBER_OF_HOURS = 1
_g_conf.WEATHERS = [1, 3, 6, 8]
#### Starting the model by loading another
_g_conf.PRELOAD_MODEL_BATCH = None
_g_conf.PRELOAD_MODEL_ALIAS = None
_g_conf.PRELOAD_MODEL_CHECKPOINT = None


"""#### Network Related Parameters ####"""


_g_conf.MODEL_TYPE = 'coil_icra'
_g_conf.MODEL_CONFIGURATION = {}
_g_conf.PRE_TRAINED = False
_g_conf.MAGICAL_SEED = 42


_g_conf.LEARNING_RATE_DECAY_INTERVAL = 50000
_g_conf.LEARNING_RATE_DECAY_LEVEL = 0.5
_g_conf.LEARNING_RATE_THRESHOLD = 1000
_g_conf.LEARNING_RATE = 0.0002  # First
_g_conf.BRANCH_LOSS_WEIGHT = [0.95, 0.95, 0.95, 0.95, 0.05]
_g_conf.VARIABLE_WEIGHT = {'Steer': 0.5, 'Gas': 0.45, 'Brake': 0.05}
_g_conf.USED_LAYERS_ATT = []

_g_conf.LOSS_FUNCTION = 'L2'

"""#### Simulation Related Parameters ####"""

_g_conf.IMAGE_CUT = [115, 510]  # How you should cut the input image that is received from the server
_g_conf.USE_ORACLE = False
_g_conf.USE_FULL_ORACLE = False
_g_conf.AVOID_STOPPING = False


def merge_with_yaml(yaml_filename):
    """Load a yaml config file and merge it into the global config object"""
    global _g_conf
    with open(yaml_filename, 'r') as f:

        yaml_file = yaml.load(f)

        yaml_cfg = AttributeDict(yaml_file)


    _merge_a_into_b(yaml_cfg, _g_conf)

    path_parts = os.path.split(yaml_filename)
    _g_conf.EXPERIMENT_BATCH_NAME = os.path.split(path_parts[-2])[-1]
    _g_conf.EXPERIMENT_NAME = path_parts[-1].split('.')[-2]
    _g_conf.EXPERIMENT_GENERATED_NAME = generate_name(_g_conf)


def get_names(folder):
    alias_in_folder = os.listdir(os.path.join('configs', folder))

    experiments_in_folder = {}
    for experiment_alias in alias_in_folder:

        g_conf.immutable(False)
        merge_with_yaml(os.path.join('configs', folder, experiment_alias))

        experiments_in_folder.update({experiment_alias: g_conf.EXPERIMENT_GENERATED_NAME})

    return experiments_in_folder


def _merge_a_into_b(a, b, stack=None):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """

    assert isinstance(a, AttributeDict) or isinstance(a, dict), 'Argument `a` must be an AttrDict'
    assert isinstance(b, AttributeDict) or isinstance(a, dict), 'Argument `b` must be an AttrDict'

    for k, v_ in a.items():
        full_key = '.'.join(stack) + '.' + k if stack is not None else k
        # a must specify keys that are in b
        if k not in b:
            # if is it more than second stack
            if stack is not None:
                b[k] = v_
            else:
                raise KeyError('Non-existent config key: {}'.format(full_key))

        v = copy.deepcopy(v_)
        v = _decode_cfg_value(v)

        v = _check_and_coerce_cfg_value_type(v, b[k], k, full_key)

        # Recursively merge dicts

        b[k] = v


def _decode_cfg_value(v):
    """Decodes a raw config value (e.g., from a yaml config files or command
    line argument) into a Python object.
    """
    # Configs parsed from raw yaml will contain dictionary keys that need to be
    # converted to AttrDict objects


    # All remaining processing is only applied to strings
    if not isinstance(v, str):
        return v
    # Try to interpret `v` as a:
    #   string, number, tuple, list, dict, boolean, or None
    try:
        v = literal_eval(v)
    # The following two excepts allow v to pass through when it represents a
    # string.
    #

    except ValueError:
        pass
    except SyntaxError:
        pass
    return v


def _check_and_coerce_cfg_value_type(value_a, value_b, key, full_key):
    """Checks that `value_a`, which is intended to replace `value_b` is of the
    right type. The type is correct if it matches exactly or is one of a few
    cases in which the type can be easily coerced.
    """
    # The types must match (with some exceptions)
    type_b = type(value_b)
    type_a = type(value_a)
    if type_a is type_b:
        return value_a

    # Exceptions: numpy arrays, strings, tuple<->list
    if isinstance(value_b, type(None)):
        value_a = value_a
    elif isinstance(value_b, np.ndarray):
        value_a = np.array(value_a, dtype=value_b.dtype)
    elif isinstance(value_b, str):
        value_a = str(value_a)
    elif isinstance(value_a, tuple) and isinstance(value_b, list):
        value_a = list(value_a)
    elif isinstance(value_a, list) and isinstance(value_b, tuple):
        value_a = tuple(value_a)
    elif isinstance(value_b, range) and not isinstance(value_a, list):
        value_a = eval(value_a)
    elif isinstance(value_b, range) and isinstance(value_a, list):
        value_a = list(value_a)
    elif isinstance(value_b, dict):
        value_a = eval(value_a)
    else:
        raise ValueError(
            'Type mismatch ({} vs. {}) with values ({} vs. {}) for config '
            'key: {}'.format(type_b, type_a, value_b, value_a, full_key)
        )
    return value_a


g_conf = _g_conf

