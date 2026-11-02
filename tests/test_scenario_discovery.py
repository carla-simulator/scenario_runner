#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Checks that every scenario a shipped configuration can name has a class behind
it, and that a user's own scenarios are found through --additionalScenario.

No simulator is needed: nothing here connects to CARLA or builds a behaviour.
Run with: python3 -m unittest tests.test_scenario_discovery
"""

import glob
import os
import tempfile
import unittest
import xml.etree.ElementTree as ET

from srunner.scenarios.basic_scenario import BasicScenario
from srunner.tools.scenario_class_loader import ScenarioClassLoader

SR_ROOT = os.getenv('SCENARIO_RUNNER_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CUSTOM_SCENARIO = '''
from srunner.scenarios.basic_scenario import BasicScenario


class ATestOnlyScenario(BasicScenario):
    pass
'''


def _types_in(pattern, tag):
    """Collects every 'type' attribute of <tag> in the files matching the pattern"""
    types = set()
    for path in glob.glob(pattern):
        for node in ET.parse(path).iter(tag):
            scenario_type = node.attrib.get('type')
            if scenario_type:
                types.add(scenario_type)
    return types


class TestScenarioDiscovery(unittest.TestCase):

    def setUp(self):
        ScenarioClassLoader.set_additional_paths([])

    def tearDown(self):
        ScenarioClassLoader.set_additional_paths([])

    def test_shipped_scenarios_are_found(self):
        """The scenarios folder yields classes, and only BasicScenario subclasses"""
        classes = ScenarioClassLoader.get_all_scenario_classes()
        self.assertTrue(classes, "no scenario classes were found")
        for name, scenario_class in classes.items():
            self.assertTrue(issubclass(scenario_class, BasicScenario),
                            "{} is not a BasicScenario".format(name))
            self.assertIsNot(scenario_class, BasicScenario)

    def test_example_configs_resolve(self):
        """Every type named in srunner/examples has a class"""
        classes = ScenarioClassLoader.get_all_scenario_classes()
        missing = sorted(_types_in(SR_ROOT + '/srunner/examples/*.xml', 'scenario') - set(classes))
        self.assertEqual(missing, [], "example configs name types with no class")

    def test_route_scenarios_resolve(self):
        """Every type named in the shipped route files has a class"""
        classes = ScenarioClassLoader.get_all_scenario_classes()
        missing = sorted(_types_in(SR_ROOT + '/srunner/data/routes_*.xml', 'scenario') - set(classes))
        self.assertEqual(missing, [], "routes name types with no class")

    def test_additional_scenario_file(self):
        """A single .py passed to --additionalScenario is found"""
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, 'a_test_only_scenario.py')
            with open(path, 'w') as handle:
                handle.write(CUSTOM_SCENARIO)
            ScenarioClassLoader.set_additional_paths([path])
            self.assertIn('ATestOnlyScenario', ScenarioClassLoader.get_all_scenario_classes())

    def test_additional_scenario_folder(self):
        """A folder passed to --additionalScenario is searched for .py files"""
        with tempfile.TemporaryDirectory() as folder:
            with open(os.path.join(folder, 'a_test_only_scenario.py'), 'w') as handle:
                handle.write(CUSTOM_SCENARIO)
            ScenarioClassLoader.set_additional_paths([folder])
            self.assertIn('ATestOnlyScenario', ScenarioClassLoader.get_all_scenario_classes())

    def test_additional_scenarios_are_comma_separated(self):
        """Several paths can be given in one argument"""
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, 'a_test_only_scenario.py')
            with open(path, 'w') as handle:
                handle.write(CUSTOM_SCENARIO)
            ScenarioClassLoader.set_additional_paths(['{},{}'.format(folder, path)])
            self.assertIn('ATestOnlyScenario', ScenarioClassLoader.get_all_scenario_classes())

    def test_unknown_type_returns_none(self):
        """A type with no class resolves to None instead of raising"""
        self.assertIsNone(ScenarioClassLoader.get_scenario_class('NoSuchScenarioAnywhere'))


if __name__ == '__main__':
    unittest.main()
