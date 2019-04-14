import collections

def get_dropout_sum(model_configuration):
    return (sum(model_configuration['branches']['fc']['dropouts']) +
            sum(model_configuration['speed_branch']['fc']['dropouts']) +
            sum(model_configuration['measurements']['fc']['dropouts'])+
            sum(model_configuration['join']['fc']['dropouts'])+
            sum(model_configuration['perception']['fc']['dropouts']))


# TODO: THIS FUNCTION IS REPEATED FROM MAIN
def parse_split_configuration(configuration):

    """
    Turns the configuration line of sliptting into a name and a set of params.

    """
    if configuration is None:
        return "None", None
    print ('conf', configuration)
    conf_dict = collections.OrderedDict(configuration)

    name = 'split'
    for key in conf_dict.keys():
        if key != 'weights':
            name += '_'
            name += key



    return name, conf_dict

def generate_name(g_conf):
    # TODO: Make a cool name generator, maybe in another class
    """

        The name generator is currently formed by the following parts
        Dataset_name.
        THe type of network used, got directly from the class.
        The regularization
        The strategy with respect to time
        The type of output
        The preprocessing made in the data
        The type of loss function
        The parts  of data that where used.

        Take into account if the variable was not set, it set the default name, from the global conf



    Returns:
        a string containing the name


    """



    final_name_string = ""
    # Addind dataset
    final_name_string += g_conf.TRAIN_DATASET_NAME
    # Model type
    final_name_string += '_' + g_conf.MODEL_TYPE
    # Model Size
    #TODO: for now is just saying the number of convs, add a layer counting
    if 'conv' in g_conf.MODEL_CONFIGURATION['perception']:
        final_name_string += '_' + str(len(g_conf.MODEL_CONFIGURATION['perception']['conv']['kernels'])) +'conv'
    else:  # FOR NOW IT IS A RES MODEL
        final_name_string += '_' + str(g_conf.MODEL_CONFIGURATION['perception']['res']['name'])

    # Model Regularization
    # We start by checking if there is some kind of augmentation, and the schedule name.

    if 'conv' in g_conf.MODEL_CONFIGURATION['perception']:
        if g_conf.AUGMENTATION is not None and g_conf.AUGMENTATION != 'None':
            final_name_string += '_' + g_conf.AUGMENTATION
        else:
            # We check if there is dropout
            if get_dropout_sum(g_conf.MODEL_CONFIGURATION) > 4:
                final_name_string += '_highdropout'
            elif get_dropout_sum(g_conf.MODEL_CONFIGURATION) > 2:
                final_name_string += '_milddropout'
            elif get_dropout_sum(g_conf.MODEL_CONFIGURATION) > 0:
                final_name_string += '_lowdropout'
            else:
                final_name_string += '_none'


    # Temporal

    if g_conf.NUMBER_FRAMES_FUSION > 1 and g_conf.NUMBER_IMAGES_SEQUENCE > 1:
        final_name_string += '_lstm_fusion'
    elif g_conf.NUMBER_FRAMES_FUSION > 1:
        final_name_string += '_fusion'
    elif g_conf.NUMBER_IMAGES_SEQUENCE > 1:
        final_name_string += '_lstm'
    else:
        final_name_string += '_single'

    # THe type of output

    if 'waypoint1_angle' in set(g_conf.TARGETS):

        final_name_string += '_waypoints'
    else:
        final_name_string += '_control'

    # The pre processing ( Balance or not )
    if g_conf.BALANCE_DATA and len(g_conf.STEERING_DIVISION) > 0:
        final_name_string += '_balancesteer'
    elif g_conf.BALANCE_DATA and g_conf.PEDESTRIAN_PERCENTAGE > 0:
        final_name_string += '_balancepedestrian'
    elif g_conf.BALANCE_DATA and len(g_conf.SPEED_DIVISION) > 0:
        final_name_string += '_balancespeed'
    else:
        final_name_string += '_random'


    # The type of loss function

    final_name_string += '_' + g_conf.LOSS_FUNCTION

    # the parts of the data that were used.

    if g_conf.USE_NOISE_DATA:
        final_name_string += '_noise_'
    else:
        final_name_string += '_'

    final_name_string += g_conf.DATA_USED

    final_name_string += '_' + str(g_conf.AUGMENT_LATERAL_STEERINGS)
    name_splitter, _ = parse_split_configuration(g_conf.SPLIT)
    final_name_string += '_' + name_splitter


    final_name_string += '_' + str(g_conf.NUMBER_OF_HOURS) + 'hours'


    if g_conf.USE_FULL_ORACLE:
        return 'ORACLE'

    return final_name_string