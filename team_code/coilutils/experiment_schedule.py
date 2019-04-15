
from logger import monitorer
import heapq


def get_remainig_exps(executing_processes, experiment_list):

    executing_list = []
    for process in executing_processes:
        executing_list.append(process['experiment'])

    return list(set(experiment_list)- set(executing_list))


def get_gpu_resources(gpu_resources, executing_processes, allocation_params):

    """

    Args:
        allocated_gpus:
        executing_processes:
        allocation_params:

    Returns:

    """
    still_executing_processes = []
    for process_specs in executing_processes:
        # Make process name:
        if process_specs['type'] == 'drive':

            name = 'drive_' + process_specs['environment']

        elif process_specs['type'] == 'validation':
            name = 'validation_' + process_specs['dataset']
        else:
            name = process_specs['type']

        status = monitorer.get_status(process_specs['folder'], process_specs['experiment'],
                                     name)[0]

        if status == "Finished" or status == 'Error':

            gpu_resources[process_specs['gpu']] += allocation_params[process_specs['type']+'_cost']

        else:
            still_executing_processes.append(process_specs)


    return gpu_resources, max(gpu_resources.values()), still_executing_processes


def allocate_gpu_resources(gpu_resources, amount_to_allocate):
    """
        On GPU management allocate gpu resources considering a dictionary with resources
        for each gpu
    Args:
        gpu_resources:
        amount_to_allocate:

    Returns:

    """

    for gpu, resource in gpu_resources.items():
        if resource >= amount_to_allocate:
            gpu_resources[gpu] -= amount_to_allocate
            return gpu_resources, max(gpu_resources.values()), gpu

    raise ValueError("Not enough gpu resources to allocate")


def dict_to_namevec(process_dict):
    """
    Converts a process dict to a name vec

    """
    name_vec = ''

    name_vec += process_dict['type']
    if 'environment' in process_dict:
        name_vec += process_dict['environment']

    if 'dataset' in process_dict:
        name_vec += process_dict['dataset']

    name_vec += process_dict['experiment']
    return name_vec


def execvec_to_names(executing_processes):
    """
    Creates a name_vec for each the dictonary on the list of executing processss
    Args:
        List of executing process
        
    returns 
        List of name vecs 
    """




    process_name_vec = []
    for process_dict in executing_processes:
        process_name_vec.append(dict_to_namevec(process_dict))

    return process_name_vec


#TODO refactor.
def mount_experiment_heap(folder, experiments_list, is_training, executing_processes, old_tasks_queue,
                          validation_datasets, drive_environments, restart_error=True):
    """
        Function that will add all the experiments to a heap. These experiments will
        be consumed when there is enough resources
    Args:
        folder: The folder with the experiments
        experiments_list: The list of all experiments to be executed
        is_training: If Training is being add also ( NOT IMPLEMENTED)
        executing_processes: The current processes being executed
        old_tasks_queue: Current process on the task queue
        validation_datasets: The validation datasets to be evaluated
        drive_environments: All the driving environments where the models are going to be tested.
        restart_error: If you are going to restart experiments that are not working (NOT implemented)

    Returns:

    """

    tasks_queue = []
    exec_name_vec = execvec_to_names(executing_processes)
    for experiment in experiments_list:

        # Train is always priority.
        task_to_add = None

        if is_training:
            if monitorer.get_status(folder, experiment, 'train')[0] == "Not Started":

                task_to_add = (0, experiment + '_train',
                                         {'type': 'train', 'folder': folder,
                                          'experiment': experiment})

            elif restart_error and monitorer.get_status(folder, experiment, 'train')[0] \
                            == "Error":

                task_to_add = (0, experiment + '_train',
                               {'type': 'train', 'folder': folder,
                                'experiment': experiment})

            if task_to_add is not None:
                task_name_vec = dict_to_namevec(task_to_add[2])
                if task_name_vec in exec_name_vec:

                    continue

            if task_to_add is not None and task_to_add not in old_tasks_queue:
                heapq.heappush(tasks_queue, task_to_add)

        for val_data in validation_datasets:
            task_to_add = None
            if monitorer.get_status(folder, experiment, 'validation_' + val_data)[0] == "Not Started":
                task_to_add = (1, experiment + '_validation_' + val_data,
                                             {'type': 'validation', 'folder': folder,
                                              'experiment': experiment, 'dataset': val_data})

            elif restart_error and monitorer.get_status(folder, experiment, 'validation_'
                                                                + val_data)[0] == "Error":
                task_to_add = (1, experiment + '_validation_' + val_data,
                                             {'type': 'validation', 'folder': folder,
                                              'experiment': experiment, 'dataset': val_data})

            if task_to_add is not None:
                task_name_vec = dict_to_namevec(task_to_add[2])
                if task_name_vec in exec_name_vec:

                    continue

            if task_to_add is not None and task_to_add not in old_tasks_queue:
                heapq.heappush(tasks_queue, task_to_add)

        for drive_env in drive_environments:
            task_to_add = None
            if monitorer.get_status(folder, experiment, 'drive_' + drive_env)[0] == "Not Started":
                task_to_add = (2, experiment + '_drive_' + drive_env,
                                             {'type': 'drive', 'folder': folder,
                                              'experiment': experiment, 'environment': drive_env})

            elif restart_error and monitorer.get_status(folder, experiment, 'drive_' + drive_env)\
                                                        [0] == "Error":
                task_to_add = (2, experiment + '_drive_' + drive_env,
                               {'type': 'drive', 'folder': folder,
                                'experiment': experiment, 'environment': drive_env})


            if task_to_add is not None:
                task_name_vec = dict_to_namevec(task_to_add[2])
                if task_name_vec in exec_name_vec:

                    continue

            if task_to_add is not None and task_to_add not in old_tasks_queue:
                heapq.heappush(tasks_queue, task_to_add)

    return tasks_queue