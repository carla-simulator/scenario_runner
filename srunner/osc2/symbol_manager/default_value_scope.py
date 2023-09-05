from srunner.osc2.symbol_manager.local_scope import LocalScope


class DefaultValueScope(LocalScope):
    def __init__(self, scope):
        super().__init__(scope)
        self.name = "="

    def __str__(self):
        buf = self.name
        return buf
