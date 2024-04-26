import srunner.osc2.ast_manager.ast_listener as ASTListener
from srunner.osc2.ast_manager.ast_node import AST


class ASTWalker(object):
    def walk(self, listener: ASTListener, t: AST):
        self.enter_node(listener, t)

        for child in t.get_children():
            if isinstance(child, AST):
                self.walk(listener, child)

        self.exit_node(listener, t)

    def enter_node(self, listener: ASTListener, t: AST):
        t.enter_node(listener)

    def exit_node(self, listener: ASTListener, t: AST):
        t.exit_node(listener)
