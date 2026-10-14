"""py_trees leaves that run semantic actions and conditions inside the legacy behavior tree."""

from __future__ import absolute_import

import logging

import py_trees

from srunner.openscenario.execution.registry import UnsupportedFeatureError
from srunner.openscenario.model.expressions import EvaluationError
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import AtomicBehavior
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import AtomicCondition

LOGGER = logging.getLogger("[SR:OpenSCENARIO]")


class ExecuteSemanticAction(AtomicBehavior):
    """Execute a semantic action through an ActionExecutorRegistry once per activation.

    Evaluation errors do not escape the tree: the behavior returns FAILURE, logs
    the diagnostic and keeps it in ``self.diagnostic``.
    """

    def __init__(self, action, registry, context, name="SemanticAction"):
        super(ExecuteSemanticAction, self).__init__(name)
        self._action = action
        self._registry = registry
        self._context = context
        self.diagnostic = None
        self.result = None

    def update(self):
        try:
            self.result = self._registry.execute(self._action, self._context)
        except (EvaluationError, UnsupportedFeatureError) as error:
            self.diagnostic = error.diagnostic
            LOGGER.error("%s failed: %s", self.name, error.diagnostic)
            return py_trees.common.Status.FAILURE
        return py_trees.common.Status.SUCCESS


class SemanticValueCondition(AtomicCondition):
    """Compare a parameter or variable with a value; SUCCESS once the rule holds, else RUNNING."""

    PARAMETER = "parameter"
    VARIABLE = "variable"

    def __init__(self, kind, reference, rule, value, context, name="ValueCondition"):
        super(SemanticValueCondition, self).__init__(name)
        if kind not in (self.PARAMETER, self.VARIABLE):
            raise ValueError("Unknown value condition kind {!r}".format(kind))
        self._kind = kind
        self._reference = reference
        self._rule = rule
        self._value = value
        self._context = context
        self.diagnostic = None

    def update(self):
        evaluation = self._context.evaluation
        compare = evaluation.compare_variable if self._kind == self.VARIABLE else evaluation.compare_parameter
        try:
            satisfied = compare(self._reference, self._rule, self._value)
        except (EvaluationError, KeyError) as error:
            self.diagnostic = getattr(error, "diagnostic", None)
            LOGGER.error("%s failed: %s", self.name, error)
            return py_trees.common.Status.FAILURE
        return py_trees.common.Status.SUCCESS if satisfied else py_trees.common.Status.RUNNING
