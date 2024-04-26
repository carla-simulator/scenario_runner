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

from tests.run_testcase.test_base import TestSymbolClass
from tests.run_testcase.log_msg import create_LogMsg as log_msg

log_msg.is_open = True
# modify current working directory
os.chdir('../')


# The class that runs the test cases
# No error message indicates that the AST construction has no syntax errors
class TestStruct(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter('ignore', ResourceWarning)
        self.test_class = TestSymbolClass()

    ###############
    # test action #
    ###############
    def test_action_actor_name_not_defined(self):
        test_case_name = "testcases1/test_action_actor_name_not_defined.osc"
        standard_msg_list = ["[Error] line 1:0, actorName: osc_actior is not defined!"]
        result_mag_list = self.test_class.testcase(test_case_name)
        self.assertListEqual(result_mag_list, standard_msg_list)
        log_msg.clean_msg()

    def test_action_enum_ref_not_defined(self):
        test_case_name = "testcases1/test_action_enum_ref_not_defined.osc"
        standard_msg_list = []
        result_mag_list = self.test_class.testcase(test_case_name)
        self.assertListEqual(result_mag_list, standard_msg_list)
        log_msg.clean_msg()

    def test_action_inherits(self):
        test_case_name = "testcases1/test_action_inherits.osc"
        standard_msg_list = []
        result_mag_list = self.test_class.testcase(test_case_name)
        self.assertListEqual(result_mag_list, standard_msg_list)
        log_msg.clean_msg()

    ###############
    # test actor  #
    ###############
    def test_actor_enum_ref_not_defined(self):
        test_case_name = "testcases1/test_actor_enum_ref_not_defined.osc"
        standard_msg_list = []
        result_mag_list = self.test_class.testcase(test_case_name)
        self.assertListEqual(result_mag_list, standard_msg_list)
        log_msg.clean_msg()

    def test_actor_extend_conflict(self):
        test_case_name = "testcases1/test_actor_extend_conflict.osc"
        standard_msg_list = ["[Error] line 9:4, vehicle_type is already defined!"]
        result_mag_list = self.test_class.testcase(test_case_name)
        self.assertListEqual(result_mag_list, standard_msg_list)
        log_msg.clean_msg()

    def test_actor_inherit(self):
        test_case_name = "testcases1/test_actor_inherit.osc"
        standard_msg_list = []
        result_mag_list = self.test_class.testcase(test_case_name)
        self.assertListEqual(result_mag_list, standard_msg_list)
        log_msg.clean_msg()

    def test_actor_multi_field_name_conflict(self):
        test_case_name = "testcases1/test_actor_multi_field_name_conflict.osc"
        standard_msg_list = ["[Error] line 4:4, Can not define same param in same scope!"]
        standard_msg_list.append("[Error] line 4:4, category is already defined!")
        result_mag_list = self.test_class.testcase(test_case_name)
        self.assertListEqual(result_mag_list, standard_msg_list)
        log_msg.clean_msg()

    def test_actor_name_redefined(self):
        test_case_name = "testcases1/test_actor_name_redefined.osc"
        standard_msg_list = ["[Error] line 9:0, dut is already defined!"]
        result_mag_list = self.test_class.testcase(test_case_name)
        self.assertListEqual(result_mag_list, standard_msg_list)
        log_msg.clean_msg()

    def test_actor_same_with_behavior(self):
        test_case_name = "testcases1/test_actor_same_with_behavior.osc"
        standard_msg_list = [
            "[Error] line 7:0, behaviorName:" + '"cut_in_and_slow"' + " can not be same with actorName!"]
        standard_msg_list.append(
            "[Error] line 12:0, behaviorName:" + '"osc_action"' + " can not be same with actorName!")
        result_mag_list = self.test_class.testcase(test_case_name)
        self.assertListEqual(result_mag_list, standard_msg_list)
        log_msg.clean_msg()

    ###############
    # test enum   #
    ###############
    def test_enum_member_index(self):
        test_case_name = "testcases1/test_enum_member_index.osc"
        standard_msg_list = ["[Error] line 7:4, Enum member 'silver' with wrong Value: 3"]
        result_mag_list = self.test_class.testcase(test_case_name)
        self.assertListEqual(result_mag_list, standard_msg_list)
        log_msg.clean_msg()

    def test_enum_member_reference(self):
        test_case_name = "testcases1/test_enum_member_reference.osc"
        standard_msg_list = ["[Error] line 9:23, Enum member green not found!"]
        result_mag_list = self.test_class.testcase(test_case_name)
        self.assertListEqual(result_mag_list, standard_msg_list)
        log_msg.clean_msg()

    def test_enum_name_redefined(self):
        test_case_name = "testcases1/test_enum_name_redefined.osc"
        standard_msg_list = ["[Error] line 3:0, rgb_color is already defined!"]
        result_mag_list = self.test_class.testcase(test_case_name)
        self.assertListEqual(result_mag_list, standard_msg_list)
        log_msg.clean_msg()

    def test_enum_with_wrong_value(self):
        test_case_name = "testcases1/test_enum_with_wrong_value.osc"
        standard_msg_list = ["[Error] line 3:49, Enum member 'black' with wrong Value: 5"]
        result_mag_list = self.test_class.testcase(test_case_name)
        self.assertListEqual(result_mag_list, standard_msg_list)
        log_msg.clean_msg()

    def test_same_enum(self):
        test_case_name = "testcases1/test_same_enum.osc"
        standard_msg_list = ["[Error] line 23:0, color is already defined!"]
        result_mag_list = self.test_class.testcase(test_case_name)
        self.assertListEqual(result_mag_list, standard_msg_list)
        log_msg.clean_msg()

    def test_same_enum_member(self):
        test_case_name = "testcases1/test_same_enum_member.osc"
        standard_msg_list = ["[Error] line 7:4, Enum member 'white' is already defined!"]
        result_mag_list = self.test_class.testcase(test_case_name)
        self.assertListEqual(result_mag_list, standard_msg_list)
        log_msg.clean_msg()

    ###############
    # test extend #
    ###############
    def test_extend_not_defined(self):
        test_case_name = "testcases1/test_extend_not_defined.osc"
        standard_msg_list = ["[Error] line 3:0, my_vehicle is not defined!"]
        standard_msg_list.append("[Error] line 3:7, Type name: my_vehicle is not defined!")
        result_mag_list = self.test_class.testcase(test_case_name)
        self.assertListEqual(result_mag_list, standard_msg_list)
        log_msg.clean_msg()

    #################
    # test physical #
    #################
    def test_physical_not_defined(self):
        test_case_name = "testcases1/test_physical_not_defined.osc"
        standard_msg_list = ["[Error] line 3:0, PhysicalType: angle is not defined!"]
        result_mag_list = self.test_class.testcase(test_case_name)
        self.assertListEqual(result_mag_list, standard_msg_list)
        log_msg.clean_msg()

    def test_physical_type_redefined(self):
        test_case_name = "testcases1/test_physical_type_redefined.osc"
        standard_msg_list = ["[Error] line 3:0, length is already defined!"]
        result_mag_list = self.test_class.testcase(test_case_name)
        self.assertListEqual(result_mag_list, standard_msg_list)
        log_msg.clean_msg()

    #################
    #   test unit   #
    #################
    def test_same_unit_name(self):
        test_case_name = "testcases1/test_same_unit_name.osc"
        standard_msg_list = ["[Error] line 5:0, degree is already defined!"]
        result_mag_list = self.test_class.testcase(test_case_name)
        self.assertListEqual(result_mag_list, standard_msg_list)
        log_msg.clean_msg()

    def test_unit_not_defined(self):
        test_case_name = "testcases1/test_unit_not_defined.osc"
        standard_msg_list = ["[Warning] line 5:20, Unit |foot/s| is not defined!"]
        result_mag_list = self.test_class.testcase(test_case_name)
        self.assertListEqual(result_mag_list, standard_msg_list)
        log_msg.clean_msg()

    ####################
    #  test parameter  #
    ####################
    def test_same_global_parameter(self):
        test_case_name = "testcases1/test_same_global_parameter.osc"
        standard_msg_list = ["[Error] line 4:0, environment is already defined!"]
        result_mag_list = self.test_class.testcase(test_case_name)
        self.assertListEqual(result_mag_list, standard_msg_list)
        log_msg.clean_msg()

    def test_same_parameter(self):
        test_case_name = "testcases1/test_same_parameter.osc"
        standard_msg_list = ["[Error] line 17:4, ego_vehicle is already defined!"]
        result_mag_list = self.test_class.testcase(test_case_name)
        self.assertListEqual(result_mag_list, standard_msg_list)
        log_msg.clean_msg()

    def test_same_parameter_in_scenario(self):
        test_case_name = "testcases1/test_same_parameter_in_scenario.osc"
        standard_msg_list = ["[Error] line 10:4, Can not define same param in same scope!"]
        standard_msg_list.append("[Error] line 10:4, ego_vehicle is already defined!")
        result_mag_list = self.test_class.testcase(test_case_name)
        self.assertListEqual(result_mag_list, standard_msg_list)
        log_msg.clean_msg()

    def test_same_param_in_action(self):
        test_case_name = "testcases1/test_same_param_in_action.osc"
        standard_msg_list = ["[Error] line 4:4, Can not define same param in same scope!"]
        standard_msg_list.append("[Error] line 4:4, ego_vehicle is already defined!")
        result_mag_list = self.test_class.testcase(test_case_name)
        self.assertListEqual(result_mag_list, standard_msg_list)
        log_msg.clean_msg()

    def test_param_without_defined(self):
        test_case_name = "testcases1/test_param_without_defined.osc"
        standard_msg_list = ["[Error] line 2:10, Type name: area_kind is not defined!"]
        result_mag_list = self.test_class.testcase(test_case_name)
        self.assertListEqual(result_mag_list, standard_msg_list)
        log_msg.clean_msg()

    #################
    # test sceanrio #
    #################
    def test_sceanrio_inherits(self):
        test_case_name = "testcases1/test_sceanrio_inherits.osc"
        standard_msg_list = []
        result_mag_list = self.test_class.testcase(test_case_name)
        self.assertListEqual(result_mag_list, standard_msg_list)
        log_msg.clean_msg()

    def test_scenario_actor_name_not_defined(self):
        test_case_name = "testcases1/test_scenario_actor_name_not_defined.osc"
        standard_msg_list = []
        result_mag_list = self.test_class.testcase(test_case_name)
        self.assertListEqual(result_mag_list, standard_msg_list)
        log_msg.clean_msg()

    def test_scenario_enum_ref_not_defined(self):
        test_case_name = "testcases1/test_scenario_enum_ref_not_defined.osc"
        standard_msg_list = ["[Error] line 2:32, Enum color not found!"]
        result_mag_list = self.test_class.testcase(test_case_name)
        self.assertListEqual(result_mag_list, standard_msg_list)
        log_msg.clean_msg()

    def test_scenario_extend_conflict(self):
        test_case_name = "testcases1/test_scenario_extend_conflict.osc"
        standard_msg_list = ["[Error] line 7:4, path is already defined!"]
        standard_msg_list.append("[Error] line 15:4, path is already defined!")
        result_mag_list = self.test_class.testcase(test_case_name)
        self.assertListEqual(result_mag_list, standard_msg_list)
        log_msg.clean_msg()

    def test_same_scenario(self):
        test_case_name = "testcases1/test_same_scenario.osc"
        standard_msg_list = ["[Error] line 86:0, dut.cut_in_and_slow is already defined!"]
        result_mag_list = self.test_class.testcase(test_case_name)
        self.assertListEqual(result_mag_list, standard_msg_list)
        log_msg.clean_msg()

    #################
    #  test struct  #
    #################
    def test_struct_enum_ref_not_defined(self):
        test_case_name = "testcases1/test_struct_enum_ref_not_defined.osc"
        standard_msg_list = ["[Error] line 3:48, Enum car_type not found!"]
        result_mag_list = self.test_class.testcase(test_case_name)
        self.assertListEqual(result_mag_list, standard_msg_list)
        log_msg.clean_msg()

    def test_struct_extension_with_actor_name(self):
        test_case_name = "testcases1/test_struct_extension_with_actor_name.osc"
        standard_msg_list = ["[Error] line 3:0, test.traffic_light2 is Not defined!"]
        result_mag_list = self.test_class.testcase(test_case_name)
        self.assertListEqual(result_mag_list, standard_msg_list)
        log_msg.clean_msg()

    def test_struct_inherits(self):
        test_case_name = "testcases1/test_struct_inherits.osc"
        standard_msg_list = []
        result_mag_list = self.test_class.testcase(test_case_name)
        self.assertListEqual(result_mag_list, standard_msg_list)
        log_msg.clean_msg()

    def test_struct_type_extension(self):
        test_case_name = "testcases1/test_struct_type_extension.osc"
        standard_msg_list = ["[Error] line 13:4, country is already defined!"]
        result_mag_list = self.test_class.testcase(test_case_name)
        self.assertListEqual(result_mag_list, standard_msg_list)
        log_msg.clean_msg()

    def test_same_struct(self):
        test_case_name = "testcases1/test_same_struct.osc"
        standard_msg_list = ["[Error] line 15:0, orientation_3d is already defined!"]
        result_mag_list = self.test_class.testcase(test_case_name)
        self.assertListEqual(result_mag_list, standard_msg_list)
        log_msg.clean_msg()

    #################
    # test variable #
    #################
    def test_variable_field_not_defined(self):
        test_case_name = "testcases1/test_variable_field_not_defined.osc"
        standard_msg_list = ["[Error] line 8:9, x1 is not defined!"]
        result_mag_list = self.test_class.testcase(test_case_name)
        self.assertListEqual(result_mag_list, standard_msg_list)
        log_msg.clean_msg()

    def test_variable_field_value_is_none(self):
        test_case_name = "testcases1/test_variable_field_value_is_none.osc"
        standard_msg_list = ["[Error] line 8:9, x: value is None!"]
        result_mag_list = self.test_class.testcase(test_case_name)
        self.assertListEqual(result_mag_list, standard_msg_list)
        log_msg.clean_msg()

    def test_variable_multi_define(self):
        test_case_name = "testcases1/test_variable_multi_define.osc"
        standard_msg_list = ["[Error] line 8:9, x: value is None!"]
        result_mag_list = self.test_class.testcase(test_case_name)
        self.assertListEqual(result_mag_list, standard_msg_list)
        log_msg.clean_msg()

    def test_variable_not_defined(self):
        test_case_name = "testcases1/test_variable_not_defined.osc"
        standard_msg_list = ["[Error] line 8:9, current_position1 is not defined!"]
        standard_msg_list.append("[Error] line 8:9, x is not defined!")
        result_mag_list = self.test_class.testcase(test_case_name)
        self.assertListEqual(result_mag_list, standard_msg_list)
        log_msg.clean_msg()

    def test_variable_redefined(self):
        test_case_name = "testcases1/test_variable_redefined.osc"
        standard_msg_list = ["[Error] line 8:4, current_position is already defined!"]
        result_mag_list = self.test_class.testcase(test_case_name)
        self.assertListEqual(result_mag_list, standard_msg_list)
        log_msg.clean_msg()

    #################
    # test modifier #
    #################
    def test_same_modifier(self):
        test_case_name = "testcases1/test_same_modifier.osc"
        standard_msg_list = ["[Error] line 6:0, force_lane is already defined!"]
        result_mag_list = self.test_class.testcase(test_case_name)
        self.assertListEqual(result_mag_list, standard_msg_list)
        log_msg.clean_msg()

    def test_modifier_actor_not_defined(self):
        test_case_name = "testcases1/test_modifier_actor_not_defined.osc"
        standard_msg_list = ["[Error] line 3:3, vehicle is already defined!"]
        result_mag_list = self.test_class.testcase(test_case_name)
        self.assertListEqual(result_mag_list, standard_msg_list)
        log_msg.clean_msg()

    #################
    #   test other  #
    #################
    def test_argument_type_not_define(self):
        test_case_name = "testcases1/test_argument_type_not_define.osc"
        standard_msg_list = ["[Error] line 11:47, Argument Type distance is not defined!"]
        result_mag_list = self.test_class.testcase(test_case_name)
        self.assertListEqual(result_mag_list, standard_msg_list)
        log_msg.clean_msg()

    def test_do_directive_name_redefined(self):
        test_case_name = "testcases1/test_do_directive_name_redefined.osc"
        standard_msg_list = ["[Error] line 76:8, get_ahead is already defined!"]
        result_mag_list = self.test_class.testcase(test_case_name)
        self.assertListEqual(result_mag_list, standard_msg_list)
        log_msg.clean_msg()

    def test_same_event(self):
        test_case_name = "testcases1/test_same_event.osc"
        standard_msg_list = ["[Error] line 3:4, event1 is already defined!"]
        result_mag_list = self.test_class.testcase(test_case_name)
        self.assertListEqual(result_mag_list, standard_msg_list)
        log_msg.clean_msg()

    def test_same_si_base_exponent(self):
        test_case_name = "testcases1/test_same_si_base_exponent.osc"
        standard_msg_list = ["[Error] line 2:35, rad is already defined!"]
        result_mag_list = self.test_class.testcase(test_case_name)
        self.assertListEqual(result_mag_list, standard_msg_list)
        log_msg.clean_msg()

    def test_it_not_defined(self):
        test_case_name = "testcases1/test_it_not_defined.osc"
        standard_msg_list = ["[Error] line 8:13, color1 is not defined in scope: car"]
        result_mag_list = self.test_class.testcase(test_case_name)
        self.assertListEqual(result_mag_list, standard_msg_list)
        log_msg.clean_msg()

    def tearDown(self):
        del self.test_class


if __name__ == "__main__":
    unittest.main()
