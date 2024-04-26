from srunner.osc2.symbol_manager.base_symbol import BaseSymbol

# compositionOperator : 'serial' | 'one_of' | 'parallel';


class CompositionSymbol(BaseSymbol):
    def __init__(self, name, scope):
        super().__init__(name, scope)


class SerialSymbol(CompositionSymbol):
    def __init__(self, scope):
        super().__init__("serial", scope)


class OneOfSymbol(CompositionSymbol):
    def __init__(self, scope):
        super().__init__("one_of", scope)


class ParallelSymbol(CompositionSymbol):
    def __init__(self, scope):
        super().__init__("parallel", scope)
