import re
import os
import smtplib
import numpy as np

from email.mime.text import MIMEText
from PIL import Image


def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    temp = 10
    x = x/temp
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()



def tryint(s):
    try:
        return int(s)
    except:
        return s

def unique(sequence):
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]


def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)


# TODO: there should be a more natural way to do that
def command_number_to_index(command_vector):

    return command_vector-2

def camelcase_to_snakecase(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def snakecase_to_camelcase(column):
    first, *rest = column.split('_')
    first = list(first)
    first[0] = first[0].upper()
    first = ''.join(first)
    return first + ''.join(word.capitalize() for word in rest)


def plot_test_image(image, name):

    image_to_plot = Image.fromarray(image)
    image_to_plot.save(name)


def create_log_folder(exp_batch_name):
    """
        Only the train creates the path. The validation should wait for the training anyway,
        so there is no need to create any path for the logs. That avoids race conditions.
    Returns:

    """
    # This is hardcoded the logs always stay on the _logs folder
    root_path = '_logs'

    if not os.path.exists(root_path):
        os.mkdir(root_path)

    if not os.path.exists(os.path.join(root_path, exp_batch_name)):
        os.mkdir(os.path.join(root_path, exp_batch_name))


def create_exp_path(exp_batch_name, experiment_name):
    # This is hardcoded the logs always stay on the _logs folder
    root_path = '_logs'

    if not os.path.exists(os.path.join(root_path, exp_batch_name, experiment_name)):
        os.mkdir(os.path.join(root_path, exp_batch_name, experiment_name))


def get_validation_datasets(exp_batch_name):
    root_path = '_logs'

    experiments = os.listdir(os.path.join(root_path, exp_batch_name))

    validation_datasets = set()
    for exp in experiments:
        if os.path.isdir(os.path.join(root_path, exp_batch_name, exp)):
            experiments = os.listdir(os.path.join(root_path, exp_batch_name, exp))
            for log in experiments:
                folder_file = os.path.join(root_path, exp_batch_name, exp, log)
                if  os.path.isdir(folder_file) and 'validation' in folder_file:
                    validation_datasets.add(folder_file.split('_')[-1])

    return list(validation_datasets)


def get_driving_environments(exp_batch_name):
    root_path = '_logs'

    experiments = os.listdir(os.path.join(root_path, exp_batch_name))

    driving_environments = set()
    for exp in experiments:
        if os.path.isdir(os.path.join(root_path, exp_batch_name, exp)):
            experiments = os.listdir(os.path.join(root_path, exp_batch_name, exp))
            for log in experiments:
                folder_file = os.path.join(root_path, exp_batch_name, exp, log)
                if not os.path.isdir(folder_file) and 'drive' in folder_file:
                    driving_environments.add(folder_file.split('_')[-2]+'_'+folder_file.split('_')[-1])

    return list(driving_environments)


def erase_logs(exp_batch_name):

    root_path = '_logs'

    experiments = os.listdir(os.path.join(root_path, exp_batch_name))

    for exp in experiments:
        if os.path.isdir(os.path.join(root_path, exp_batch_name, exp)):
            experiments_logs = os.listdir(os.path.join(root_path, exp_batch_name, exp))
            for log in experiments_logs:
                if not os.path.isdir(os.path.join(root_path, exp_batch_name, exp, log))\
                        and '.csv' not in log:
                    os.remove(os.path.join(root_path, exp_batch_name, exp, log))


def erase_wrong_plotting_summaries(exp_batch_name, validation_data_list, ):

    root_path = '_logs'

    experiments = os.listdir(os.path.join(root_path, exp_batch_name))

    # Get the correct files sizes for each validation
    # open the csv file with the ground_truth
    validation_sizes = {}
    for validation_data in validation_data_list:
        val_size = len(np.loadtxt(os.path.join(os.environ["COIL_DATASET_PATH"],
                                               validation_data, 'ground_truth.csv'),
                                  delimiter=","))
        validation_sizes.update({validation_data: val_size})

    for exp in experiments:
        print ("exp", exp)
        for validation_log in validation_data_list:
            folder_name = 'validation_' + validation_log + '_csv'
            print(' VALIDATION ----- ', folder_name)
            validation_folder_path = os.path.join(root_path, exp_batch_name, exp, folder_name)
            if not os.path.exists(validation_folder_path):
                continue
            csv_files = os.listdir(validation_folder_path)
            for csv_result in csv_files:
                print ("    csv_file", csv_result)
                csv_file_path = os.path.join(root_path, exp_batch_name, exp,
                                             folder_name, csv_result)

                try:
                    len_of_csv_file = len(np.loadtxt(csv_file_path, delimiter=","))
                except:
                    print ("    wrong file")
                    print ("    deleting")
                    os.remove(csv_file_path)
                    continue

                print ('    len data', validation_sizes[validation_log])
                print ('    len csv ', len_of_csv_file)
                if validation_sizes[validation_log] != len_of_csv_file:

                    print ("    deleting")
                    os.remove(csv_file_path)


def erase_validations(exp_batch_name, validation_data_list ):
    # TODO: eventually add that for driving

    # Erase wrong plotting for validation!

    root_path = '_logs'


    experiments = os.listdir(os.path.join(root_path, exp_batch_name))

    # Get the correct files sizes for each validation
    # open the csv file with the ground_truth

    for exp in experiments:
        print ("exp", exp)
        for validation_log in validation_data_list:
            folder_name = 'validation_' + validation_log + '_csv'
            print(' VALIDATION ----- ', folder_name)
            validation_folder_path = os.path.join(root_path, exp_batch_name, exp, folder_name)
            if not os.path.exists(validation_folder_path):
                continue
            csv_files = os.listdir(validation_folder_path)
            for csv_result in csv_files:
                print("    csv_file", csv_result)
                csv_file_path = os.path.join(root_path, exp_batch_name, exp,
                                             folder_name, csv_result)
                os.remove(csv_file_path)



def get_latest_path(path):
    """ Considering a certain path for experiments, get the latest one."""
    import glob
    files_list = glob.glob(os.path.join('_benchmarks_results', path+'*'))
    sort_nicely(files_list)

    return files_list[-1]


def send_email(address, message):
    msg = MIMEText(message)
    msg['Subject'] = 'The experiment is finished '
    msg['From'] = address
    msg['To'] = address

    s = smtplib.SMTP('localhost')
    s.sendmail(address, [address], msg.as_string())
    s.quit()


def compute_average_std(dic_list, weathers, number_of_tasks=1):

    """
    There are two types of outputs, these come packed in a dictionary

    Success metrics, these are averaged between weathers, is basically the percentage of completion for a
    single task.

    Infractions, these are summed and divided by the total number of driven kilometers


    For this you have the concept of averaging all the weathers from the experiment suite.

    """

    metrics_to_average = [
        'episodes_fully_completed',
        'episodes_completion'

    ]

    metrics_to_sum = [
        'end_pedestrian_collision',
        'end_vehicle_collision',
        'end_other_collision'
    ]

    infraction_metrics = [
        'collision_pedestrians',
        'collision_vehicles',
        'collision_other',
        'intersection_offroad',
        'intersection_otherlane'

    ]
    weather_name_dict = {1: 'Clear Noon', 3: 'After Rain Noon',
                         6: 'Heavy Rain Noon', 8: 'Clear Sunset',
                         4: 'Cloudy After Rain', 14: 'Soft Rain Sunset'}

    number_of_episodes = len(list(dic_list[0]['episodes_fully_completed'].items())[0][1])

    # The average results between the dictionaries.
    average_results_matrix = {}


    for metric_name in (metrics_to_average+infraction_metrics+metrics_to_sum):
        average_results_matrix.update({metric_name: np.zeros((number_of_tasks, len(dic_list)))})

    count_dic_pos = 0
    for metrics_summary in dic_list:


        for metric in metrics_to_average:


            values = metrics_summary[metric]
            #print values

            metric_sum_values = np.zeros(number_of_episodes)
            for weather, tasks in values.items():
                if float(weather) in set(weathers):
                    count = 0
                    for t in tasks:
                        # if isinstance(t, np.ndarray) or isinstance(t, list):

                        if t == []:
                            print('    Metric Not Computed')
                        else:
                            metric_sum_values[count] += (float(sum(t)))

                        count += 1

            for i in range(len(metric_sum_values)):
                average_results_matrix[metric][i][count_dic_pos] = metric_sum_values[i]



        # For the metrics we sum over all the weathers here, this is to better subdivide the driving envs
        for metric in infraction_metrics:
            values_driven = metrics_summary['driven_kilometers']
            values = metrics_summary[metric]
            metric_sum_values = np.zeros(number_of_episodes)
            summed_driven_kilometers = np.zeros(number_of_episodes)


            # print (zip(values.items(), values_driven.items()))
            for items_metric, items_driven in zip(values.items(), values_driven.items()):
                weather = items_metric[0]
                tasks = items_metric[1]
                tasks_driven = items_driven[1]

                if float(weather) in set(weathers):

                    count = 0
                    for t, t_driven in zip(tasks, tasks_driven):
                        # if isinstance(t, np.ndarray) or isinstance(t, list):
                        if t == []:
                            print('Metric Not Computed')
                        else:

                            metric_sum_values[count] += float(sum(t))
                            summed_driven_kilometers[count] += t_driven

                        count += 1


            # On this part average results matrix basically assume the number of infractions.
            for i in range(len(metric_sum_values)):
                if metric_sum_values[i] == 0:
                    average_results_matrix[metric][i][count_dic_pos] = 1
                else:
                    average_results_matrix[metric][i][count_dic_pos] = metric_sum_values[i]


        for metric in metrics_to_sum:
            values = metrics_summary[metric]
            metric_sum_values = np.zeros(number_of_episodes)


            # print (zip(values.items(), values_driven.items()))
            for items_metric  in values.items():
                weather = items_metric[0]
                tasks = items_metric[1]

                if float(weather) in set(weathers):

                    count = 0
                    for t in tasks:
                        # if isinstance(t, np.ndarray) or isinstance(t, list):
                        if t == []:
                            print('Metric Not Computed')
                        else:

                            metric_sum_values[count] += float(sum(t))

                        count += 1


            # On this part average results matrix basically assume the number of infractions.
            for i in range(len(metric_sum_values)):
                    average_results_matrix[metric][i][count_dic_pos] = metric_sum_values[i]
        count_dic_pos += 1



    print(metrics_summary['average_speed'])
    average_speed_task = sum(metrics_summary['average_speed'][str(float(list(weathers)[0]))])


    average_results_matrix.update({'driven_kilometers': np.array(sum(summed_driven_kilometers))})


    average_results_matrix.update({'average_speed': np.array([average_speed_task])})
    print(average_results_matrix)


    for metric, vectors in average_results_matrix.items():

        if metric in metrics_to_average:
            average_results_matrix[metric] =  np.sum(average_results_matrix[metric])/\
                                              (len(average_results_matrix[metric])*25)

        if metric in infraction_metrics:
            average_results_matrix[metric] = average_results_matrix['driven_kilometers']/np.sum(average_results_matrix[metric])

        if metric in metrics_to_sum:
            average_results_matrix[metric] = np.sum(average_results_matrix[metric])

        """
        for i in range(len(vectors)):

            
            average_results_matrix[metric][i] = np.mean(average_results_matrix[metric][i])

        """

    return average_results_matrix

"""
    Writing for the driving summary calculation.
"""
def write_header_control_summary(path, task):

    filename = os.path.join(path + '_' + task + '.csv')

    print (filename)

    csv_outfile = open(filename, 'w')

    csv_outfile.write("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n"
                      % ('step', 'episodes_completion', 'intersection_offroad',
                          'collision_pedestrians', 'collision_vehicles', 'episodes_fully_completed',
                         'driven_kilometers', 'end_pedestrian_collision',
                         'end_vehicle_collision',  'end_other_collision', 'intersection_otherlane' ))
    csv_outfile.close()



def write_data_point_control_summary(path, task, averaged_dict, step, pos):

    filename = os.path.join(path + '_' + task + '.csv')

    print (filename)

    if not os.path.exists(filename):
        raise ValueError("The filename does not yet exists")

    csv_outfile = open(filename, 'a')

    csv_outfile.write("%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n"
                      % (step,
                         averaged_dict['episodes_completion'][pos][0],
                         averaged_dict['intersection_offroad'][pos][0],
                         averaged_dict['collision_pedestrians'][pos][0],
                         averaged_dict['collision_vehicles'][pos][0],
                         averaged_dict['episodes_fully_completed'][pos][0],
                         averaged_dict['driven_kilometers'][pos],
                         averaged_dict['end_pedestrian_collision'][pos][0],
                         averaged_dict['end_vehicle_collision'][pos][0],
                         averaged_dict['end_other_collision'][pos][0],
                         averaged_dict['intersection_otherlane'][pos][0]))

    csv_outfile.close()

# TODO: Needs refactoring
def compute_average_std_separatetasks(dic_list, weathers, number_of_tasks=1):
    """
    There are two types of outputs, these come packed in a dictionary

    Success metrics, these are averaged between weathers, is basically the percentage of completion for a
    single task.

    Infractions, these are summed and divided by the total number of driven kilometers


    For this you have the concept of averaging all the weathers from the experiment suite.

    """

    metrics_to_average = [
        'episodes_fully_completed',
        'episodes_completion'

    ]

    metrics_to_sum = [
        'end_pedestrian_collision',
        'end_vehicle_collision',
        'end_other_collision'
    ]

    infraction_metrics = [
        'collision_pedestrians',
        'collision_vehicles',
        'collision_other',
        'intersection_offroad',
        'intersection_otherlane'

    ]
    weather_name_dict = {1: 'Clear Noon', 3: 'After Rain Noon',
                         6: 'Heavy Rain Noon', 8: 'Clear Sunset',
                         4: 'Cloudy After Rain', 14: 'Soft Rain Sunset'}

    number_of_episodes = len(list(dic_list[0]['episodes_fully_completed'].items())[0][1])

    # The average results between the dictionaries.
    average_results_matrix = {}

    for metric_name in (metrics_to_average + infraction_metrics + metrics_to_sum):
        average_results_matrix.update({metric_name: np.zeros((number_of_tasks, len(dic_list)))})

    count_dic_pos = 0
    for metrics_summary in dic_list:

        for metric in metrics_to_average:

            values = metrics_summary[metric]
            # print values

            metric_sum_values = np.zeros(number_of_episodes)
            for weather, tasks in values.items():
                if float(weather) in set(weathers):
                    count = 0
                    for t in tasks:
                        # if isinstance(t, np.ndarray) or isinstance(t, list):

                        if t == []:
                            print('    Metric Not Computed')
                        else:
                            metric_sum_values[count] += (float(sum(t)))

                        count += 1

            for i in range(len(metric_sum_values)):
                average_results_matrix[metric][i][count_dic_pos] = metric_sum_values[i]/(25*len(weathers))

        # For the metrics we sum over all the weathers here, this is to better subdivide the driving envs
        for metric in infraction_metrics:
            values_driven = metrics_summary['driven_kilometers']
            values = metrics_summary[metric]
            metric_sum_values = np.zeros(number_of_episodes)
            summed_driven_kilometers = np.zeros(number_of_episodes)

            # print (zip(values.items(), values_driven.items()))
            for items_metric, items_driven in zip(values.items(), values_driven.items()):
                weather = items_metric[0]
                tasks = items_metric[1]
                tasks_driven = items_driven[1]

                if float(weather) in set(weathers):

                    count = 0
                    for t, t_driven in zip(tasks, tasks_driven):
                        # if isinstance(t, np.ndarray) or isinstance(t, list):
                        if t == []:
                            print('Metric Not Computed')
                        else:

                            metric_sum_values[count] += float(sum(t))
                            summed_driven_kilometers[count] += t_driven

                        count += 1

            # On this part average results matrix basically assume the number of infractions.
            for i in range(len(metric_sum_values)):
                if metric_sum_values[i] == 0:
                    average_results_matrix[metric][i][count_dic_pos] = 1
                else:
                    average_results_matrix[metric][i][count_dic_pos] = metric_sum_values[i]

        for metric in metrics_to_sum:
            values = metrics_summary[metric]
            metric_sum_values = np.zeros(number_of_episodes)

            # print (zip(values.items(), values_driven.items()))
            for items_metric in values.items():
                weather = items_metric[0]
                tasks = items_metric[1]

                if float(weather) in set(weathers):

                    count = 0
                    for t in tasks:
                        # if isinstance(t, np.ndarray) or isinstance(t, list):
                        if t == []:
                            print('Metric Not Computed')
                        else:

                            metric_sum_values[count] += float(sum(t))

                        count += 1

            # On this part average results matrix basically assume the number of infractions.
            print (" metric sum ", metric_sum_values)
            for i in range(len(metric_sum_values)):
                average_results_matrix[metric][i][count_dic_pos] = metric_sum_values[i]/(25*len(weathers))

        count_dic_pos += 1



    average_speed_task = sum(metrics_summary['average_speed'][str(float(list(weathers)[0]))])

    average_results_matrix.update({'driven_kilometers': np.array(summed_driven_kilometers)})

    average_results_matrix.update({'average_speed': np.array([average_speed_task])})
    print(average_results_matrix)


    return average_results_matrix