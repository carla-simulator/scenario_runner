import re

from antlr4.error.ErrorListener import *
from antlr4.Token import Token

from srunner.osc2.utils.log_manager import LOG_ERROR


class OscErrorListener(ErrorListener):
    def __init__(self, src):
        super(ErrorListener, self).__init__()
        self.src = src

    def getWrongToken(self, t: Token):
        if t is None:
            return "<no token>"
        s = t.text
        if s is None:
            if t.type == Token.EOF:
                s = "<EOF>"
            else:
                s = "<" + str(t.type) + ">"
        return self.escapeWSAndQuote(s)

    def escapeWSAndQuote(self, s: str):
        s = s.replace("\n", "\\n")
        s = s.replace("\r", "\\r")
        s = s.replace("\t", "\\t")
        return "'" + s + "'"

    def reportIndentationError(self, line, column, token_name):
        LOG_ERROR(token_name, line=line, column=column)
        LOG_ERROR("IndentationError: wrong indentation")

    def reportSyntaxErrorWithTokenName(self, line, column, token_name):
        LOG_ERROR(token_name, line=line, column=column)
        LOG_ERROR("SyntaxError: invalid syntax")

    def reportMismatchInputError(self, line, column, token_name):
        error_msg = " mismatched input " + token_name
        LOG_ERROR(error_msg, line=line, column=column)
        LOG_ERROR("SyntaxError: invalid syntax")

    def reportCommonSyntaxError(self, line, column, msg):
        LOG_ERROR(msg, line=line, column=column)
        LOG_ERROR("SyntaxError: invalid syntax")

    # Syntax error
    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
        token = recognizer.getCurrentToken()
        token_name = self.getWrongToken(token)

        if re.match("^extraneous", msg):
            if token_name == "'\\n'":
                self.reportIndentationError(line, column, None)
            else:
                self.reportSyntaxErrorWithTokenName(line, column, token_name)
        elif re.match("^mismatched", msg):
            self.reportMismatchInputError(line, column, token_name)
        elif re.match("^missing", msg):
            if re.search("INDENT", msg):
                self.reportIndentationError(line, column, token_name)
            else:
                self.reportMismatchInputError(line, column, token_name)
        elif re.match("^no viable alternative", msg):
            self.reportCommonSyntaxError(line, column, msg)
        else:
            self.reportCommonSyntaxError(line, column, msg)

    # Ambiguity error
    def reportAmbiguity(
        self, recognizer, dfa, startIndex, stopIndex, exact, ambigAlts, configs
    ):
        pass

    def reportAttemptingFullContext(
        self, recognizer, dfa, startIndex, stopIndex, conflictingAlts, configs
    ):
        pass

    def reportContextSensitivity(
        self, recognizer, dfa, startIndex, stopIndex, prediction, configs
    ):
        pass
