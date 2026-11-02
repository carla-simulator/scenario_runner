#!/usr/bin/env python

# Copyright (c) 2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides the oneshot idiom used by the OpenSCENARIO support.
"""

import operator

import py_trees

from srunner.tools.blackboard import set_blackboard


def oneshot_behavior(variable_name, behaviour, name=None):
    """
    Runs 'behaviour' once, then short-circuits on the blackboard variable.

    py_trees.idioms.oneshot is not used directly: it latches on its own subtree,
    while the OpenSCENARIO parser resets these variables to replay a story element.
    """
    if not name:
        name = behaviour.name

    subtree_root = py_trees.composites.Selector(name=name, memory=True)

    set_blackboard(variable_name, False)

    check_flag = py_trees.behaviours.CheckBlackboardVariableValue(
        name=variable_name + " Done?",
        check=py_trees.common.ComparisonExpression(
            variable=variable_name, value=True, operator=operator.eq)
    )
    set_flag = py_trees.behaviours.SetBlackboardVariable(
        name="Mark Done",
        variable_name=variable_name,
        variable_value=True,
        overwrite=True
    )
    # If it's a sequence, don't double-nest it in a redundant manner
    if isinstance(behaviour, py_trees.composites.Sequence):
        behaviour.add_child(set_flag)
        sequence = behaviour
    else:
        sequence = py_trees.composites.Sequence(name="OneShot", memory=True)
        sequence.add_children([behaviour, set_flag])

    subtree_root.add_children([check_flag, sequence])
    return subtree_root
