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
from srunner.scenariomanager.timer import GameTime


OPENSCENARIO = ["OpenScenario"]


class Decorator(py_trees.behaviour.Behaviour):

    """
    A decorator is responsible for handling the lifecycle of a single
    child beneath

    This is taken from py_trees 1.2 to work with our current implementation
    that uses py_trees 0.8.2
    """

    def __init__(self, child, name):
        """
        Common initialisation steps for a decorator - type checks and
        name construction (if None is given).
        Args:
            name (:obj:`str`): the decorator name
            child (:class:`~py_trees.behaviour.Behaviour`): the child to be decorated
        Raises:
            TypeError: if the child is not an instance of :class:`~py_trees.behaviour.Behaviour`
        """
        # Checks
        if not isinstance(child, py_trees.behaviour.Behaviour):
            raise TypeError("A decorator's child must be an instance of py_trees.behaviours.Behaviour")
        # Initialise
        super(Decorator, self).__init__(name=name)
        self.children.append(child)
        # Give a convenient alias
        self.decorated = self.children[0]
        self.decorated.parent = self

    def tick(self):
        """
        A decorator's tick is exactly the same as a normal proceedings for
        a Behaviour's tick except that it also ticks the decorated child node.
        Yields:
            :class:`~py_trees.behaviour.Behaviour`: a reference to itself or one of its children
        """
        self.logger.debug("%s.tick()" % self.__class__.__name__)
        # initialise just like other behaviours/composites
        if self.status != py_trees.common.Status.RUNNING:
            self.initialise()
        # interrupt proceedings and process the child node
        # (including any children it may have as well)
        for node in self.decorated.tick():
            yield node
        # resume normal proceedings for a Behaviour's tick
        new_status = self.update()
        if new_status not in list(py_trees.common.Status):
            self.logger.error(
                "A behaviour returned an invalid status, setting to INVALID [%s][%s]" % (new_status, self.name))
            new_status = py_trees.common.Status.INVALID
        if new_status != py_trees.common.Status.RUNNING:
            self.stop(new_status)
        self.status = new_status
        yield self

    def stop(self, new_status=py_trees.common.Status.INVALID):
        """
        As with other composites, it checks if the child is running
        and stops it if that is the case.
        Args:
            new_status (:class:`~py_trees.common.Status`): the behaviour is transitioning to this new status
        """
        self.logger.debug("%s.stop(%s)" % (self.__class__.__name__, new_status))
        self.terminate(new_status)
        # priority interrupt handling
        if new_status == py_trees.common.Status.INVALID:
            self.decorated.stop(new_status)
        # if the decorator returns SUCCESS/FAILURE and should stop the child
        if self.decorated.status == py_trees.common.Status.RUNNING:
            self.decorated.stop(py_trees.common.Status.INVALID)
        self.status = new_status

    def tip(self):
        """
        Get the *tip* of this behaviour's subtree (if it has one) after it's last
        tick. This corresponds to the the deepest node that was running before the
        subtree traversal reversed direction and headed back to this node.
        """
        if self.decorated.status != py_trees.common.Status.INVALID:
            return self.decorated.tip()

        return super(Decorator, self).tip()


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


class StoryElementStatusToBlackboard(Decorator):

    """
    Reflect the status of the decorator's child story element to the blackboard.

    Args:
        child: the child behaviour or subtree
        story_element_type: the element type [act,scene,maneuver,event,action]
        element_name: the story element's name attribute
    """

    def __init__(
            self,
            child,
            story_element_type,
            element_name
    ):
        super(StoryElementStatusToBlackboard, self).__init__(name=child.name, child=child)
        self.story_element_type = story_element_type
        self.element_name = element_name
        self.blackboard = py_trees.blackboard.Blackboard()

    def initialise(self):
        """
        Record the elements's start time on the blackboard
        """
        self.blackboard.set(
            name="({}){}-{}".format(self.story_element_type.upper(),
                                    self.element_name, "START"),
            value=GameTime.get_time(),
            overwrite=True
        )

    def update(self):
        """
        Reflect the decorated child's status
        Returns: the decorated child's status
        """
        return self.decorated.status

    def terminate(self, new_status):
        """
        Terminate and mark Blackboard entry with END
        """
        # Report whether we ended with End or Cancel
        # If we were ended or cancelled, our state will be INVALID and
        # We will have an ancestor (a parallel SUCCESS_ON_ALL) with a successful child/children
        # It's possible we ENDed AND CANCELled if both condition groups were true simultaneously
        # NOTE 'py_trees.common.Status.INVALID' is the status of a behaviur which was terminated by a parent
        rules = []
        if new_status == py_trees.common.Status.INVALID:
            # We were terminated from above unnaturally
            # Figure out if were ended or cancelled
            terminating_ancestor = self.parent
            while terminating_ancestor.status == py_trees.common.Status.INVALID:
                terminating_ancestor = terminating_ancestor.parent
            # We have found an ancestory which was not terminated by a parent
            # Check what caused it to terminate its children
            if terminating_ancestor.status == py_trees.common.Status.SUCCESS:
                successful_children = [
                    child.name
                    for child
                    in terminating_ancestor.children
                    if child.status == py_trees.common.Status.SUCCESS]
                if "EndConditions" in successful_children:
                    rules.append("END")
                if "CancelConditions" in successful_children:
                    rules.append("CANCEL")

        # END is the default status unless we have a more detailed one
        rules = rules or ["END"]

        for rule in rules:
            self.blackboard.set(
                name="({}){}-{}".format(self.story_element_type.upper(),
                                        self.element_name, rule),
                value=GameTime.get_time(),
                overwrite=True
            )


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
                    maneuver_parallel = py_trees.composites.Parallel(
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
                                    maneuver_behavior = StoryElementStatusToBlackboard(
                                        maneuver_behavior, "ACTION", child.attrib.get('name'))
                                    parallel_actions.add_child(
                                        oneshot_behavior(maneuver_behavior))

                            if child.tag == "StartConditions":
                                # There is always one StartConditions block per Event
                                parallel_condition_groups = self._create_condition_container(
                                    child, "Parallel Condition Groups")
                                event_sequence.add_child(
                                    parallel_condition_groups)

                        parallel_actions = StoryElementStatusToBlackboard(
                            parallel_actions, "EVENT", event.attrib.get('name'))
                        event_sequence.add_child(parallel_actions)
                        maneuver_parallel.add_child(
                            oneshot_behavior(event_sequence))
                    maneuver_parallel = StoryElementStatusToBlackboard(
                        maneuver_parallel, "MANEUVER", maneuver.attrib.get('name'))
                    single_sequence_iteration.add_child(
                        oneshot_behavior(maneuver_parallel))

                # OpenSCENARIO refers to Sequences as Scenes in this instance
                single_sequence_iteration = StoryElementStatusToBlackboard(
                    single_sequence_iteration, "SCENE", sequence.attrib.get('name'))
                single_sequence_iteration = repeatable_behavior(
                    single_sequence_iteration)
                for _ in range(int(repetitions)):
                    sequence_behavior.add_child(single_sequence_iteration)

                if sequence_behavior.children:
                    parallel_sequences.add_child(
                        oneshot_behavior(sequence_behavior))

            if parallel_sequences.children:
                parallel_sequences = StoryElementStatusToBlackboard(
                    parallel_sequences, "ACT", act.attrib.get('name'))
                parallel_behavior.add_child(parallel_sequences)

            for conditions in act.iter("Conditions"):
                for start_condition in conditions.iter("Start"):
                    parallel_start_criteria = self._create_condition_container(
                        start_condition, "StartConditions", oneshot=True)
                    if parallel_start_criteria.children:
                        start_conditions.add_child(parallel_start_criteria)
                for end_condition in conditions.iter("End"):
                    parallel_end_criteria = self._create_condition_container(
                        end_condition, "EndConditions")
                    if parallel_end_criteria.children:
                        parallel_behavior.add_child(parallel_end_criteria)
                for cancel_condition in conditions.iter("Cancel"):
                    parallel_cancel_criteria = self._create_condition_container(
                        cancel_condition, "CancelConditions")
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
