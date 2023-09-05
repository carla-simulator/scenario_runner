from srunner.osc2.symbol_manager.local_scope import LocalScope


class ConstraintScope(LocalScope):
    def __init__(self, scope, constraint_qualifier):
        super().__init__(scope)
        self.constraint_qualifier = constraint_qualifier

    def get_scope_name(self):
        return "constraint"


class KeepScope(ConstraintScope):
    def __init__(self, scope, constraint_qualifier):
        super().__init__(scope, constraint_qualifier)

    def get_scope_name(self):
        return "keep"
