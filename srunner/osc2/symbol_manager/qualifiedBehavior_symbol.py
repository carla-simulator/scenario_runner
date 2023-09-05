from srunner.osc2.symbol_manager.base_symbol import BaseSymbol
from srunner.osc2.utils.log_manager import *


class QualifiedBehaviorSymbol(BaseSymbol):
    # Applies only to calls to a specified scenario or as a member of that scenario
    # Modifiers can be associated with specific scene types.
    # These modifiers can be applied to calls that associate scenarios in the with block of the call.
    # Modifiers can also be applied as scene members, for example in extensions of associated scene types.
    # The format is qualifiedBehaviorName : (actorName '.')? behaviorName;
    # Here is the distinction that needs to be made, for example, A.A, there will be naming conflicts.
    def __init__(self, name, scope):
        super().__init__(name, scope)

        name_list = name.split(".")
        if len(name_list) == 2:
            self.actor_name = name_list[0]
            self.behavior_name = name_list[1]
        elif len(name_list) == 1:
            self.actor_name = None
            self.behavior_name = name_list[0]
        else:
            self.actor_name = None
            self.behavior_name = None

            # There is no nesting problem in actors, just look for actorName in the contained scope

    def is_actor_name_defined(self):
        if self.actor_name == None:
            return True
        elif self.enclosing_scope.symbols.get(self.actor_name) is not None:
            return True
        else:
            return False

    def is_qualified_behavior_name_valid(self, ctx):
        if self.actor_name == self.behavior_name and self.actor_name:
            error_msg = (
                'behaviorName:"'
                + self.behavior_name
                + '" can not be same with actorName!'
            )
            LOG_ERROR(error_msg, ctx)
        elif self.is_actor_name_defined() is not True:
            error_msg = "actorName: " + self.actor_name + " is not defined!"
            LOG_ERROR(error_msg, ctx)
        elif self.behavior_name == None:
            error_msg = "behaviourName can not be empty!"
            LOG_ERROR(error_msg, ctx)
        else:
            pass

    def __str__(self):
        buf = self.__class__.__name__
        buf += " : "
        buf += self.name
        return buf
