from srunner.osc2.symbol_manager.local_scope import LocalScope

# compositionOperator : 'serial' | 'one_of' | 'parallel';


class DoDirectiveScope(LocalScope):
    def __init__(self, scope):
        super().__init__(scope)
        self.name = "do"

    def __str__(self):
        buf = self.__class__.__name__
        return buf
