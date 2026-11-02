#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides the discovery of scenario classes, both the ones shipped
with ScenarioRunner and the ones a user supplies through --additionalScenario.
"""

import glob
import importlib
import inspect
import os
import sys

from srunner.scenarios.basic_scenario import BasicScenario


class ScenarioClassLoader(object):

    """
    Finds the BasicScenario subclasses a scenario or route configuration can name.

    The additional paths are registered once, by scenario_runner.py, so that both
    the --scenario and the --route paths resolve the same set of classes.
    """

    _additional_paths = []
    _cache = None

    @staticmethod
    def set_additional_paths(paths):
        """
        Registers the user's scenario implementations. Each entry is a .py file, a
        directory searched for .py files, or several of either separated by commas.
        """
        ScenarioClassLoader._cache = None
        ScenarioClassLoader._additional_paths = []
        for entry in paths or []:
            for item in str(entry).split(','):
                item = item.strip()
                if item:
                    ScenarioClassLoader._additional_paths.append(item)

    @staticmethod
    def _expand(path):
        """Turns a directory into the .py files inside it, leaving a file as it is"""
        if os.path.isdir(path):
            return sorted(glob.glob(os.path.join(path, '*.py')))
        return [path] if path else []

    @staticmethod
    def get_all_scenario_classes(debug=False):
        """
        Returns {class name: class} for every BasicScenario subclass found, the
        shipped ones first so a user's implementation of the same name wins.
        """
        if ScenarioClassLoader._cache is not None:
            return ScenarioClassLoader._cache

        all_scenario_classes = {}

        # The shipped scenarios are imported as part of the package, so that their
        # folder is never added to sys.path, where it would shadow other modules.
        root = os.getenv('SCENARIO_RUNNER_ROOT', ".")
        for scenario_file in sorted(glob.glob("{}/srunner/scenarios/*.py".format(root))):
            module_name = os.path.basename(scenario_file)[:-3]
            if module_name == '__init__':
                continue
            module = ScenarioClassLoader._import(
                'srunner.scenarios.' + module_name, scenario_file, debug)
            ScenarioClassLoader._collect(module, all_scenario_classes)

        for path in ScenarioClassLoader._additional_paths:
            for scenario_file in ScenarioClassLoader._expand(path):
                if not os.path.isfile(scenario_file):
                    print("WARNING: '{}' is not a scenario file or folder".format(scenario_file))
                    continue
                module = ScenarioClassLoader._import_by_path(scenario_file, debug)
                ScenarioClassLoader._collect(module, all_scenario_classes)

        ScenarioClassLoader._cache = all_scenario_classes
        return all_scenario_classes

    @staticmethod
    def _import(module_name, scenario_file, debug):
        """Imports a module of the srunner package, returning None if it cannot be"""
        try:
            return importlib.import_module(module_name)
        except Exception as import_error:  # pylint: disable=broad-except
            # A scenario format whose parser isn't installed (OpenSCENARIO 2.0 and
            # its antlr4 runtime) must not stop the rest from being found.
            if debug:
                print("Skipping {}: {}".format(scenario_file, import_error))
            return None

    @staticmethod
    def _import_by_path(scenario_file, debug):
        """Imports a .py file that lives outside the srunner package"""
        module_name = os.path.basename(scenario_file)[:-3]
        spec = importlib.util.spec_from_file_location(module_name, scenario_file)
        if spec is None:
            print("WARNING: could not load '{}'".format(scenario_file))
            return None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception as import_error:  # pylint: disable=broad-except
            sys.modules.pop(module_name, None)
            print("WARNING: could not load '{}': {}".format(scenario_file, import_error))
            if debug:
                import traceback
                print(traceback.format_exc())
            return None
        return module

    @staticmethod
    def _collect(module, all_scenario_classes):
        """Adds the module's BasicScenario subclasses to the registry"""
        if module is None:
            return
        for _, member in inspect.getmembers(module, inspect.isclass):
            if issubclass(member, BasicScenario) and member is not BasicScenario:
                all_scenario_classes[member.__name__] = member

    @staticmethod
    def get_scenario_class(scenario_type, debug=False):
        """Returns the class for 'scenario_type', or None if there is none"""
        return ScenarioClassLoader.get_all_scenario_classes(debug).get(scenario_type)
