import os
import time

from configs import g_conf
from logger import monitorer

from coilutils.general import sort_nicely


def is_open(file_name):
    if os.path.exists(file_name):
        file1 = os.stat(file_name)  # initial file size
        file1_size = file1.st_size

        # your script here that collects and writes data (increase file size)
        time.sleep(0.5)
        file2 = os.stat(file_name)  # updated file size
        file2_size = file2.st_size
        comp = file2_size - file1_size  # compares sizes
        if comp == 0:
            return False
        else:
            return True

    raise NameError



def maximun_checkpoint_reach(iteration, checkpoint_schedule):
    if iteration is None:
        return False

    if iteration >= max(checkpoint_schedule):
        return True
    else:
        return False


""" FUNCTIONS FOR SAVING THE CHECKPOINTS """


def is_ready_to_save(iteration):
    """ Returns if the iteration is a iteration for saving a checkpoint

    """
    if iteration in set(g_conf.SAVE_SCHEDULE):
        return True
    else:
        return False

def get_latest_saved_checkpoint():
    """
        Returns the , latest checkpoint number that was saved

    """
    checkpoint_files = os.listdir(os.path.join('_logs', g_conf.EXPERIMENT_BATCH_NAME,
                                               g_conf.EXPERIMENT_NAME, 'checkpoints'))
    if checkpoint_files == []:
        return None
    else:
        sort_nicely(checkpoint_files)
        return checkpoint_files[-1]


""" FUNCTIONS FOR GETTING THE CHECKPOINTS"""

def get_latest_evaluated_checkpoint(filename=None):

    """
        Get the latest checkpoint that was validated or tested.
    Args:
    """

    return monitorer.get_latest_checkpoint(filename)


def is_next_checkpoint_ready(checkpoint_schedule, control_filename=None):

    # IT needs
    ltst_check = get_latest_evaluated_checkpoint(control_filename)

    # This means that we got the last one, so we return false and go back to the loop
    if ltst_check == g_conf.TEST_SCHEDULE[-1]:
        return False
    if ltst_check is None:  # This means no checkpoints were evaluated
        next_check = checkpoint_schedule[0]  # Return the first one
    else:
        next_check = checkpoint_schedule[checkpoint_schedule.index(ltst_check)+1]

    # Check if the file is in the checkpoints list.
    if os.path.exists(os.path.join('_logs', g_conf.EXPERIMENT_BATCH_NAME,
                                            g_conf.EXPERIMENT_NAME, 'checkpoints')):

        # test if the file exist:
        if str(next_check) + '.pth' in os.listdir(os.path.join('_logs', g_conf.EXPERIMENT_BATCH_NAME,
                                                               g_conf.EXPERIMENT_NAME, 'checkpoints')):
            # now check if someone is writing to it, if it is the case return false
            return not is_open(os.path.join('_logs', g_conf.EXPERIMENT_BATCH_NAME,
                               g_conf.EXPERIMENT_NAME, 'checkpoints', str(next_check) + '.pth'))

        else:
            return False
    else:
        # This mean the training part has not created the checkpoints yet.
        return False


def get_next_checkpoint(checkpoint_schedule, filename=None):
    ltst_check = get_latest_evaluated_checkpoint(filename)
    if ltst_check is None:
        return checkpoint_schedule[0]

    if checkpoint_schedule.index(ltst_check) + 1 == len(checkpoint_schedule):
        raise RuntimeError("Not able to get next checkpoint, maximum checkpoint is reach")

    print(checkpoint_schedule.index(ltst_check))
    print (ltst_check)
    return checkpoint_schedule[checkpoint_schedule.index(ltst_check) + 1]


def check_loss_validation_stopped(checkpoint, validation_name):
    """
     Check if validation has already found a point that the curve is not going down
     AND
     check if the training iteration is bigger than than the stale checkpoint

    """

    stale_file_name = "validation_" + validation_name + "_stale.csv"
    full_path = os.path.join('_logs', g_conf.EXPERIMENT_BATCH_NAME,
                                            g_conf.EXPERIMENT_NAME, stale_file_name)

    if os.path.exists(full_path):
        with open(full_path, 'r') as f:
            # So if training ran more iterations more than the stale point of validation
            if checkpoint > int(f.read()):
                return True
            else:
                return False

    else:
        return False


def validation_stale_point(validation_name):

    stale_file_name = "validation_" + validation_name + "_stale.csv"
    full_path = os.path.join('_logs', g_conf.EXPERIMENT_BATCH_NAME,
                                            g_conf.EXPERIMENT_NAME, stale_file_name)

    if os.path.exists(full_path):
        with open(full_path, 'r') as f:
            #  Return the stale iteration of the validation
            return int(f.read())

    else:
        return None
