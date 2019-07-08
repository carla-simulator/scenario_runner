#!/usr/bin/env python

#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Basic scenario class using the OpenScenario definition
"""

import py_trees

from srunner.scenariomanager.atomic_scenario_behavior import *
from srunner.scenariomanager.atomic_scenario_criteria import *
from srunner.scenarios.basic_scenario import *

from srunner.tools.openscenario_parser import OpenScenarioParser


OPENSCENARIO = ["OpenScenario"]


def oneshot_behavior(name, variable_name, behaviour):
    """
    This is taken from py_trees.idiom.oneshot. However, we use a different
    clearing policy to work around some issues for setting up StartConditions
    of OpenScenario
    """
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


class OpenScenario(BasicScenario):

    """
    Implementation of a  Master scenario that controls the route.

    This is a single ego vehicle scenario
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

        story_behavior = py_trees.composites.Sequence("Story")

        joint_actor_list = self.other_actors + self.ego_vehicles

        for act in self.config.story.iter("Act"):

            if act.attrib.get('name') != 'Behavior':
                continue

            parallel_behavior = py_trees.composites.Parallel(
                policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE, name="Maneuver + EndConditions Group")

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

                tmp_sequence_behavior = py_trees.composites.Sequence(name=sequence.attrib.get('name'))
                for maneuver in sequence.iter("Maneuver"):
                    maneuver_sequence = py_trees.composites.Sequence(name="Maneuver " + maneuver.attrib.get('name'))
                    for event in maneuver.iter("Event"):
                        event_sequence = py_trees.composites.Sequence(name="Event " + event.attrib.get('name'))
                        parallel_actions = py_trees.composites.Parallel(
                            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL, name="Actions")
                        for child in event.iter():
                            if child.tag == "Action":
                                for actor_id in actor_ids:
                                    maneuver_behavior = OpenScenarioParser.convert_maneuver_to_atomic(
                                        child, joint_actor_list[actor_id])
                                    parallel_actions.add_child(maneuver_behavior)

                            if child.tag == "StartConditions":
                                # There is always on StartConditions block per Event
                                for condition in child.iter('Condition'):
                                    condition_behavior = OpenScenarioParser.convert_condition_to_atomic(
                                        condition, self.other_actors + self.ego_vehicles)

                                    condition_behavior.name += " for {}".format(parallel_actions.name)

                                    if condition_behavior:
                                        event_sequence.add_child(condition_behavior)

                        event_sequence.add_child(parallel_actions)
                        maneuver_sequence.add_child(event_sequence)
                    tmp_sequence_behavior.add_child(maneuver_sequence)

                for _ in range(int(repetitions)):
                    sequence_behavior.add_child(tmp_sequence_behavior)

                if sequence_behavior.children:
                    parallel_behavior.add_child(sequence_behavior)

            for conditions in act.iter("Conditions"):
                start_condition_behavior = py_trees.composites.Parallel(
                    policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL, name="StartConditions Group")
                for start_condition in conditions.iter("Start"):
                    for condition in start_condition.iter('Condition'):
                        condition_behavior = OpenScenarioParser.convert_condition_to_atomic(
                            condition, self.other_actors + self.ego_vehicles)
                        oneshot_idiom = oneshot_behavior(
                            name=condition_behavior.name,
                            variable_name=condition_behavior.name,
                            behaviour=condition_behavior)
                        start_condition_behavior.add_child(oneshot_idiom)
                for end_condition in conditions.iter("End"):
                    for condition in end_condition.iter('Condition'):
                        condition_behavior = OpenScenarioParser.convert_condition_to_atomic(
                            condition, self.other_actors + self.ego_vehicles)
                        parallel_behavior.add_child(condition_behavior)
                for end_condition in conditions.iter("Cancel"):
                    for condition in end_condition.iter('Condition'):
                        condition_behavior = OpenScenarioParser.convert_condition_to_atomic(
                            condition, self.other_actors + self.ego_vehicles)
                        parallel_behavior.add_child(condition_behavior)

            if start_condition_behavior.children:
                story_behavior.add_child(start_condition_behavior)

            if parallel_behavior.children:
                story_behavior.add_child(parallel_behavior)

        # Build behavior tree
        # sequence.add_child(maneuver_behavior)

        return story_behavior

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        parallel_criteria = py_trees.composites.Parallel("EndConditions (Criteria Group)",
                                                         policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)

        for condition in self.config.criteria.iter("Condition"):

            criterion = OpenScenarioParser.convert_condition_to_atomic(condition, self.ego_vehicles)
            parallel_criteria.add_child(criterion)

        return parallel_criteria

    def __del__(self):
        """
        Remove all actors upon deletion
        """
        self.remove_all_actors()
