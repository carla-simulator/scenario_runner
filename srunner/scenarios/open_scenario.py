#!/usr/bin/env python

# Copyright (c) 2018-2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Basic scenario class using the OpenSCENARIO definition
"""

import py_trees

from srunner.scenarios.basic_scenario import BasicScenario
from srunner.tools.openscenario_parser import OpenScenarioParser


OPENSCENARIO = ["OpenScenario"]


def oneshot_behavior(behaviour, name=None):
    """
    This is taken from py_trees.idiom.oneshot. However, we use a different
    clearing policy to work around some issues for setting up StartConditions
    of OpenSCENARIO
    """
    if not name:
        name = behaviour.name
    variable_name = get_py_tree_path(behaviour)
    subtree_root = py_trees.composites.Selector(name=name)
    check_flag = py_trees.blackboard.CheckBlackboardVariable(
        name=variable_name + " Done?",
        variable_name=variable_name,
        expected_value=True,
        clearing_policy=py_trees.common.ClearingPolicy.ON_INITIALISE
    )
    set_flag = py_trees.blackboard.SetBlackboardVariable(
        name="Mark Done",
        variable_name=variable_name,
        variable_value=True
    )
    # If it's a sequence, don't double-nest it in a redundant manner
    if isinstance(behaviour, py_trees.composites.Sequence):
        behaviour.add_child(set_flag)
        sequence = behaviour
    else:
        sequence = py_trees.composites.Sequence(name="OneShot")
        sequence.add_children([behaviour, set_flag])

    subtree_root.add_children([check_flag, sequence])
    return subtree_root


def repeatable_behavior(behaviour, name=None):
    """
    This behaviour allows a composite with oneshot ancestors to run multiple
    times, resetting the oneshot variables after each execution
    """
    if not name:
        name = behaviour.name
    clear_descendant_variables = ClearBlackboardVariablesStartingWith(
        name="Clear Descendant Variables of {}".format(name),
        variable_name_beginning=get_py_tree_path(behaviour) + ">"
    )
    # If it's a sequence, don't double-nest it in a redundant manner
    if isinstance(behaviour, py_trees.composites.Sequence):
        behaviour.add_child(clear_descendant_variables)
        sequence = behaviour
    else:
        sequence = py_trees.composites.Sequence(name="RepeatableBehaviour")
        sequence.add_children([behaviour, clear_descendant_variables])
    return sequence


class ClearBlackboardVariablesStartingWith(py_trees.behaviours.Success):

    """
    Clear the values starting with the specified string from the blackboard.

    Args:
        name (:obj:`str`): name of the behaviour
        variable_name_beginning (:obj:`str`): beginning of the names of variable to clear
    """

    def __init__(self,
                 name="Clear Blackboard Variable Starting With",
                 variable_name_beginning="dummy",
                 ):
        super(ClearBlackboardVariablesStartingWith, self).__init__(name)
        self.variable_name_beginning = variable_name_beginning

    def initialise(self):
        """
        Delete the variables from the blackboard.
        """
        blackboard_variables = [key for key, _ in py_trees.blackboard.__dict__.items(
        ) if key.startswith(self.variable_name_beginning)]
        for variable in blackboard_variables:
            delattr(py_trees.blackboard, variable)


def get_py_tree_path(behaviour):
    """
    Accept a behaviour/composite and return a string representation of its full path
    """
    path = ""
    target = behaviour
    while True:
        path = "{}>{}".format(target.name, path)
        target = target.parent
        if not target:
            break

    path = path[:-1]

    return path


class OpenScenario(BasicScenario):

    """
    Implementation of the OpenSCENARIO scenario
    """

    category = "OpenScenario"

    def __init__(self, world, ego_vehicles, config, config_file, debug_mode=False, criteria_enable=True, timeout=300):
        """
        Setup all relevant parameters and create scenario
        """
        self.config = config
        self.target = None
        self.route = None
        self.config_file = config_file
        # Timeout of scenario in seconds
        self.timeout = timeout

        super(OpenScenario, self).__init__("OpenScenario", ego_vehicles=ego_vehicles, config=config,
                                           world=world, debug_mode=debug_mode,
                                           terminate_on_failure=False, criteria_enable=criteria_enable)

    def _create_behavior(self):
        """
        Basic behavior do nothing, i.e. Idle
        """

        story_behavior = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL, name="Story")

        joint_actor_list = self.other_actors + self.ego_vehicles

        for act in self.config.story.iter("Act"):

            if act.attrib.get('name') != 'Behavior':
                continue

            act_sequence = py_trees.composites.Sequence(
                name="Act StartConditions and behaviours")

            start_conditions = py_trees.composites.Parallel(
                policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE, name="StartConditions Group")

            parallel_behavior = py_trees.composites.Parallel(
                policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE, name="Maneuver + EndConditions Group")

            parallel_sequences = py_trees.composites.Parallel(
                policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL, name="Maneuvers")

            for sequence in act.iter("Sequence"):
                sequence_behavior = py_trees.composites.Sequence()
                repetitions = sequence.attrib.get('numberOfExecutions', 1)
                actor_ids = []
                for actor in sequence.iter("Actors"):
                    for entity in actor.iter("Entity"):
                        for k, _ in enumerate(joint_actor_list):
                            if entity.attrib.get('name', None) == joint_actor_list[k].attributes['role_name']:
                                actor_ids.append(k)
                                break

                single_sequence_iteration = py_trees.composites.Parallel(
                    policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL, name=sequence.attrib.get('name'))
                for maneuver in sequence.iter("Maneuver"):
                    maneuver_sequence = py_trees.composites.Parallel(
                        policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL,
                        name="Maneuver " + maneuver.attrib.get('name'))
                    for event in maneuver.iter("Event"):
                        event_sequence = py_trees.composites.Sequence(
                            name="Event " + event.attrib.get('name'))
                        parallel_actions = py_trees.composites.Parallel(
                            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL, name="Actions")
                        for child in event.iter():
                            if child.tag == "Action":
                                for actor_id in actor_ids:
                                    maneuver_behavior = OpenScenarioParser.convert_maneuver_to_atomic(
                                        child, joint_actor_list[actor_id])
                                    parallel_actions.add_child(
                                        oneshot_behavior(maneuver_behavior))

                            if child.tag == "StartConditions":
                                # There is always one StartConditions block per Event
                                parallel_condition_groups = self._create_condition_container(
                                    child, "Parallel Condition Groups")
                                event_sequence.add_child(
                                    parallel_condition_groups)

                        event_sequence.add_child(parallel_actions)
                        maneuver_sequence.add_child(
                            oneshot_behavior(event_sequence))
                    single_sequence_iteration.add_child(
                        oneshot_behavior(maneuver_sequence))

                single_sequence_iteration = repeatable_behavior(
                    single_sequence_iteration)
                for _ in range(int(repetitions)):
                    sequence_behavior.add_child(single_sequence_iteration)

                if sequence_behavior.children:
                    parallel_sequences.add_child(
                        oneshot_behavior(sequence_behavior))

            if parallel_sequences.children:
                parallel_behavior.add_child(parallel_sequences)

            for conditions in act.iter("Conditions"):
                for start_condition in conditions.iter("Start"):
                    parallel_start_criteria = self._create_condition_container(
                        start_condition, "StartConditions Group", oneshot=True)
                    if parallel_start_criteria.children:
                        start_conditions.add_child(parallel_start_criteria)
                for end_condition in conditions.iter("End"):
                    parallel_end_criteria = self._create_condition_container(
                        end_condition, "EndConditions Group")
                    if parallel_end_criteria.children:
                        parallel_behavior.add_child(parallel_end_criteria)
                for cancel_condition in conditions.iter("Cancel"):
                    parallel_cancel_criteria = self._create_condition_container(
                        cancel_condition, "CancelConditions Group")
                    if parallel_cancel_criteria.children:
                        parallel_behavior.add_child(parallel_cancel_criteria)

            if start_conditions.children:
                act_sequence.add_child(start_conditions)
            if parallel_behavior.children:
                act_sequence.add_child(parallel_behavior)

            if act_sequence.children:
                story_behavior.add_child(act_sequence)

        # Build behavior tree
        # sequence.add_child(maneuver_behavior)

        return story_behavior

    def _create_condition_container(self, node, name='Conditions Group', oneshot=False):
        """
        This is a generic function to handle conditions utilising ConditionGroups
        Each ConditionGroup is represented as a Sequence of Conditions
        The ConditionGroups are grouped under a SUCCESS_ON_ONE Parallel
        If oneshot is set to True, oneshot_behaviour will be applied to conditions
        """

        parallel_condition_groups = py_trees.composites.Parallel(name,
                                                                 policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)

        for condition_group in node.iter("ConditionGroup"):
            condition_group_sequence = py_trees.composites.Sequence(
                name="Condition Group")
            for condition in condition_group.iter("Condition"):
                criterion = OpenScenarioParser.convert_condition_to_atomic(
                    condition, self.other_actors + self.ego_vehicles)
                if oneshot:
                    criterion = oneshot_behavior(criterion)
                condition_group_sequence.add_child(criterion)

            if condition_group_sequence.children:
                parallel_condition_groups.add_child(condition_group_sequence)

        return parallel_condition_groups

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        parallel_criteria = py_trees.composites.Parallel("EndConditions (Criteria Group)",
                                                         policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)

        criteria = []
        for endcondition in self.config.storyboard.iter("EndConditions"):
            for condition in endcondition.iter("Condition"):
                if condition.attrib.get('name').startswith('criteria_'):
                    condition.set('name', condition.attrib.get('name')[9:])
                    criteria.append(condition)

        for condition in criteria:

            criterion = OpenScenarioParser.convert_condition_to_atomic(
                condition, self.ego_vehicles)
            parallel_criteria.add_child(criterion)

        return parallel_criteria

    def __del__(self):
        """
        Remove all actors upon deletion
        """
        self.remove_all_actors()
