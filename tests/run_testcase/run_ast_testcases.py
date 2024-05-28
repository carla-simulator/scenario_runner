import unittest
import os
import sys
import warnings

try:
    sys.path.insert(0, '../../')
except IndexError:
    pass
# Add the current working directory to the module search path
sys.path.append(os.getcwd())

from tests.run_testcase.test_base import TestASTClass
from tests.run_testcase.log_msg import create_LogMsg as log_msg

log_msg.is_open = True
# modify current working directory
os.chdir('../')


# The class that runs the test cases
# No error message indicates that the AST construction has no syntax errors
class TestStruct(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter('ignore', ResourceWarning)
        self.test_class = TestASTClass()

    # test action
    def test_action_1(self):
        test_case_name = "testcases/test_action_decl.osc"
        self.test_class.testcase(test_case_name)
        self.assertEqual(self.test_class.testcase(test_case_name), True)

    def test_action_2(self):
        test_case_name = "testcases/test_action_inherits.osc"
        self.test_class.testcase(test_case_name)
        self.assertEqual(self.test_class.testcase(test_case_name), True)

    # test actor
    def test_actor_1(self):
        test_case_name = "testcases/test_actor_decl.osc"
        self.test_class.testcase(test_case_name)
        self.assertEqual(self.test_class.testcase(test_case_name), True)

    def test_actor_2(self):
        test_case_name = "testcases/test_actor_inherits.osc"
        self.test_class.testcase(test_case_name)
        self.assertEqual(self.test_class.testcase(test_case_name), True)

    # test physical and unit
    def test_physical_1(self):
        test_case_name = "testcases/test_physical_decl.osc"
        self.test_class.testcase(test_case_name)
        self.assertEqual(self.test_class.testcase(test_case_name), True)

    # test enum
    def test_enum_1(self):
        test_case_name = "testcases/test_enum_decl.osc"
        self.test_class.testcase(test_case_name)
        self.assertEqual(self.test_class.testcase(test_case_name), True)

    def test_enum_2(self):
        test_case_name = "testcases/test_enum_extension.osc"
        self.test_class.testcase(test_case_name)
        self.assertEqual(self.test_class.testcase(test_case_name), True)

    # def test_enum_3(self):
    #     test_case_name = "testcases/test_enum_with_wrong_value.osc"
    #     standard_msg_list = ["[Error] line 3:49 the default value of enum member 'black' is wrong!"]
    #     self.assertListEqual(self.test_class.testcase(test_case_name), standard_msg_list)

    # test struct
    def test_struct_1(self):
        test_case_name = "testcases/test_struct_decl.osc"
        self.test_class.testcase(test_case_name)
        self.assertEqual(self.test_class.testcase(test_case_name), True)

    def test_struct_2(self):
        test_case_name = "testcases/test_struct_inherits.osc"
        self.test_class.testcase(test_case_name)
        self.assertEqual(self.test_class.testcase(test_case_name), True)

    def test_struct_4(self):
        test_case_name = "testcases/test_structured_type_extension.osc"
        self.test_class.testcase(test_case_name)
        self.assertEqual(self.test_class.testcase(test_case_name), True)

    # test modifier
    def test_modifier_1(self):
        test_case_name = "testcases/test_modifier_decl.osc"
        self.test_class.testcase(test_case_name)
        self.assertEqual(self.test_class.testcase(test_case_name), True)

    # test scenario
    def test_scenario_1(self):
        test_case_name = "testcases/test_scenario_decl.osc"
        self.test_class.testcase(test_case_name)
        self.assertEqual(self.test_class.testcase(test_case_name), True)

    #

    # test type extend
    def test_extend_1(self):
        test_case_name = "testcases/test_type_extend.osc"
        self.test_class.testcase(test_case_name)
        self.assertEqual(self.test_class.testcase(test_case_name), True)

    #
    #
    # ################################
    # #       second layer declaration
    # ################################

    # test event
    def test_event_1(self):
        test_case_name = "testcases/test_event_decl.osc"
        self.test_class.testcase(test_case_name)
        self.assertEqual(self.test_class.testcase(test_case_name), True)

    # test field
    def test_field_1(self):
        test_case_name = "testcases/test_parameter_decl.osc"
        self.test_class.testcase(test_case_name)
        self.assertEqual(self.test_class.testcase(test_case_name), True)

    def test_field_2(self):
        test_case_name = "testcases/test_variable_decl.osc"
        self.test_class.testcase(test_case_name)
        self.assertEqual(self.test_class.testcase(test_case_name), True)

    # test constraint
    def test_constrain_1(self):
        test_case_name = "testcases/test_variable_decl.osc"
        self.test_class.testcase(test_case_name)
        self.assertEqual(self.test_class.testcase(test_case_name), True)

    # test modifier invocation
    def test_modifier_invocation_1(self):
        test_case_name = "testcases/test_modifier_invocation.osc"
        self.test_class.testcase(test_case_name)
        self.assertEqual(self.test_class.testcase(test_case_name), True)

    # test method
    def test_method_1(self):
        test_case_name = "testcases/test_method_decl.osc"
        self.test_class.testcase(test_case_name)
        self.assertEqual(self.test_class.testcase(test_case_name), True)

    # test behavior specification
    def test_do_directive(self):
        test_case_name = "testcases/test_do_directive.osc"
        self.test_class.testcase(test_case_name)
        self.assertEqual(self.test_class.testcase(test_case_name), True)

    def test_wait_directive(self):
        test_case_name = "testcases/test_wait_directive.osc"
        self.test_class.testcase(test_case_name)
        self.assertEqual(self.test_class.testcase(test_case_name), True)

    def test_emit_directive(self):
        test_case_name = "testcases/test_emit_directive.osc"
        self.test_class.testcase(test_case_name)
        self.assertEqual(self.test_class.testcase(test_case_name), True)

    def test_call_directive(self):
        test_case_name = "testcases/test_call_directive.osc"
        self.test_class.testcase(test_case_name)
        self.assertEqual(self.test_class.testcase(test_case_name), True)

    # def test_coverage_decl(self):
    #     test_case_name = "testcases/test_coverage_decl.osc"
    #     self.test_class.testcase(test_case_name)
    #     self.assertEqual(self.test_class.testcase(test_case_name), True)

    # ################################
    # #       expression
    # ################################

    def test_logical(self):
        test_case_name = "testcases/test_logical_expression.osc"
        self.test_class.testcase(test_case_name)
        self.assertEqual(self.test_class.testcase(test_case_name), True)

    def test_physical_type(self):
        test_case_name = "testcases/test_physical_type_decl.osc"
        self.test_class.testcase(test_case_name)
        self.assertEqual(self.test_class.testcase(test_case_name), True)

    def test_range(self):
        test_case_name = "testcases/test_range.osc"
        self.test_class.testcase(test_case_name)
        self.assertEqual(self.test_class.testcase(test_case_name), True)

    def test_relation_expression(self):
        test_case_name = "testcases/test_relation_expression.osc"
        self.test_class.testcase(test_case_name)
        self.assertEqual(self.test_class.testcase(test_case_name), True)

    def test_literal(self):
        test_case_name = "testcases/test_exp_literal.osc"
        self.test_class.testcase(test_case_name)
        self.assertEqual(self.test_class.testcase(test_case_name), True)

    def test_list_constructor(self):
        test_case_name = "testcases/test_list_constructor.osc"
        self.test_class.testcase(test_case_name)
        self.assertEqual(self.test_class.testcase(test_case_name), True)

    def test_identifier_reference(self):
        test_case_name = "testcases/test_identifier_reference.osc"
        self.test_class.testcase(test_case_name)
        self.assertEqual(self.test_class.testcase(test_case_name), True)

    def test_enum_reference(self):
        test_case_name = "testcases/test_enum_reference.osc"
        self.test_class.testcase(test_case_name)
        self.assertEqual(self.test_class.testcase(test_case_name), True)

    def test_ternary_exp(self):
        test_case_name = "testcases/test_ternary_exp.osc"
        self.test_class.testcase(test_case_name)
        self.assertEqual(self.test_class.testcase(test_case_name), True)

    def test_unitary_exp(self):
        test_case_name = "testcases/test_unitary_exp.osc"
        self.test_class.testcase(test_case_name)
        self.assertEqual(self.test_class.testcase(test_case_name), True)

    def test_binary_exp(self):
        test_case_name = "testcases/test_binary_expression.osc"
        self.test_class.testcase(test_case_name)
        self.assertEqual(self.test_class.testcase(test_case_name), True)

    def test_list_binary_exp(self):
        test_case_name = "testcases/test_list_binary_exp.osc"
        self.test_class.testcase(test_case_name)
        self.assertEqual(self.test_class.testcase(test_case_name), True)

    def test_binary_logic_exp(self):
        test_case_name = "testcases/test_binary_logic_exp.osc"
        self.test_class.testcase(test_case_name)
        self.assertEqual(self.test_class.testcase(test_case_name), True)

    def test_other_binary_exp(self):
        test_case_name = "testcases/test_type_op.osc"
        self.test_class.testcase(test_case_name)
        self.assertEqual(self.test_class.testcase(test_case_name), True)

    def test_type_op(self):
        test_case_name = "testcases/test_type_op.osc"
        self.test_class.testcase(test_case_name)
        self.assertEqual(self.test_class.testcase(test_case_name), True)

    def tearDown(self):
        del self.test_class


if __name__ == "__main__":
    unittest.main()
