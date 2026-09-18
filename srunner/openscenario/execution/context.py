"""Execution context passed to semantic action executors."""

from __future__ import absolute_import

from srunner.openscenario.execution.capabilities import BackendCapabilities
from srunner.openscenario.model.parameters import EvaluationContext


class ExecutionContext(object):
    def __init__(self, backend=None, capabilities=None, state=None):
        self.backend = backend
        self.capabilities = capabilities or BackendCapabilities()
        self.state = state if state is not None else {}
        self.evaluation = self.state.get("evaluation") or EvaluationContext()
        self.state["evaluation"] = self.evaluation
