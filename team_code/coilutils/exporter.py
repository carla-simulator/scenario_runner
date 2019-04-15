import os
import numpy as np
from visualization.data_reading import read_control_csv, read_summary_csv

from configs.coil_global import get_names, merge_with_yaml, g_conf

def export_csv(exp_batch, variables_to_export):
    # TODO: add parameter for auto versus auto.

    root_path = '_logs'

    experiments = os.listdir(os.path.join(root_path, exp_batch))

    # TODO: for now it always takes the position of maximun succes
    if 'episodes_fully_completed' not in set(variables_to_export):

        raise ValueError(" export csv needs the episodes fully completed param on variables")

    # Make the header of the exported csv
    csv_outfile = os.path.join(root_path, exp_batch, 'result.csv')


    with open(csv_outfile, 'w') as f:
        f.write("experiment,environment")
        for variable in variables_to_export:
            f.write(",%s" % variable)

        f.write("\n")

    for exp in experiments:
        if os.path.isdir(os.path.join(root_path, exp_batch, exp)):
            experiments_logs = os.listdir(os.path.join(root_path, exp_batch, exp))
            for log in experiments_logs:
                if 'drive' in log and '_csv' in log:
                    csv_file_path = os.path.join(root_path, exp_batch, exp, log, 'control_output.csv')

                    if not os.path.exists(csv_file_path):
                        continue
                    control_csv = read_summary_csv(csv_file_path)
                    if control_csv is None:
                        continue
                    print (control_csv)
                    with open(csv_outfile, 'a') as f:
                        f.write("%s,%s" % (exp, log.split('_')[1]))

                        print (' var', variable)
                        print (control_csv[variable])

                        position_of_max_success = np.argmax(control_csv['episodes_fully_completed'])
                        for variable in variables_to_export:
                            f.write(",%f" % control_csv[variable][position_of_max_success])

                        f.write("\n")


def export_csv_separate(exp_batch, variables_to_export, task_list):
    # TODO: add parameter for auto versus auto.

    root_path = '_logs'

    experiments = os.listdir(os.path.join(root_path, exp_batch))

    # TODO: for now it always takes the position of maximun succes
    if 'episodes_fully_completed' not in set(variables_to_export):

        raise ValueError(" export csv needs the episodes fully completed param on variables")

    # Make the header of the exported csv
    csv_outfile = os.path.join(root_path, exp_batch, 'result.csv')


    with open(csv_outfile, 'w') as f:
        f.write("experiment,environment")
        for variable in variables_to_export:
            f.write(",%s" % variable)

        f.write("\n")




    experiment_list = []
    for exp in experiments:
        if os.path.isdir(os.path.join(root_path, exp_batch, exp)):
            experiments_logs = os.listdir(os.path.join(root_path, exp_batch, exp))
            scenario = []
            for log in experiments_logs:
                dicts_to_write = {}
                for task in task_list:
                    dicts_to_write.update({task: {}})

                for task in task_list:
                    if 'drive' in log and '_csv' in log:
                        csv_file_path = os.path.join(root_path, exp_batch, exp, log, 'control_output_' + task + '.csv')

                        if not os.path.exists(csv_file_path):
                            continue
                        control_csv = read_summary_csv(csv_file_path)
                        if control_csv is None:
                            continue
                        print (control_csv)

                        position_of_max_success = np.argmax(control_csv['episodes_fully_completed'])
                        print (dicts_to_write)


                        for variable in variables_to_export:
                            dicts_to_write[task].update({variable: control_csv[variable][position_of_max_success]})

                scenario.append(dicts_to_write)
            experiment_list.append(scenario)


    print (" FULL DICT")
    print (experiment_list)

    with open(csv_outfile, 'a') as f:


        for exp in experiments:
            print ("EXP ", exp)
            if os.path.isdir(os.path.join(root_path, exp_batch, exp)):
                experiments_logs = os.listdir(os.path.join(root_path, exp_batch, exp))
                count = 0
                for log in experiments_logs:
                    if 'drive' in log and '_csv' in log:

                        f.write("%s,%s" % (exp, log.split('_')[1]))
                        for variable in variables_to_export:


                            f.write(",")
                            for task in task_list:
                                if experiment_list[experiments.index(exp)][count][task]:
                                    f.write("%.2f/" % experiment_list[experiments.index(exp)][count][task][variable])


                        f.write("\n")
                    count += 1



def export_status(exp_batch, validation_datasets, driving_environments):

    root_path = '_logs'

    experiments = os.listdir(os.path.join(root_path, exp_batch))



    # Make the header of the exported csv
    csv_outfile = os.path.join(root_path, exp_batch, 'status.csv')

    with open(csv_outfile, 'w') as f:
        f.write("experiment_alias,experiment_name")
        for validation in validation_datasets:
            f.write("," + validation)
        for driving in driving_environments:
            f.write("," + driving)

        f.write('\n')


        names_list = get_names(exp_batch)
        count = 0
        print (names_list)
        for exp in experiments:

            if os.path.isdir(os.path.join(root_path, exp_batch, exp)):

                f.write("%s,%s" % (exp, names_list[exp+'.yaml']))
                count += 1
                print (exp)
                merge_with_yaml(os.path.join('configs', exp_batch, exp + '.yaml'))

                experiments_logs = os.listdir(os.path.join(root_path, exp_batch, exp))

                for validation in validation_datasets:

                    log = 'validation_' + validation + '_csv'


                    if log in os.listdir(os.path.join(root_path, exp_batch, exp)):

                        if (str(g_conf.TEST_SCHEDULE[-1]) + '.csv') in os.listdir(os.path.join(root_path,
                                                                                               exp_batch, exp, log)):
                            f.write(",Done")
                        else:
                            f.write(", ")
                    else:
                        f.write(", ")

                for driving in driving_environments:
                    log = 'drive_' + driving + '_csv'
                    if log in os.listdir(os.path.join(root_path, exp_batch, exp)):

                        if g_conf.USE_ORACLE:
                            output_name = 'control_output_auto.csv'
                        else:

                            output_name = 'control_output.csv'


                        print (log)
                        if output_name in os.listdir(os.path.join(root_path, exp_batch, exp, log)):

                            csv_file_path = os.path.join(root_path, exp_batch, exp, log, output_name)
                            control_csv = read_summary_csv(csv_file_path)


                            if control_csv is not None and (g_conf.TEST_SCHEDULE[-1]) == int(control_csv['step'][-1]):

                                f.write(",Done")
                            else:
                                f.write(", ")
                        else:
                            f.write(", ")
                    else:
                        f.write(", ")



                f.write("\n")