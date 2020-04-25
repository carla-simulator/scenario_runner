#!/usr/bin/env python

# Copyright (c) 2018-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides the Scenario and ScenarioManager implementations.
These must not be modified and are for reference only!
"""

from __future__ import print_function
import sys
import time
import json
import carla

import py_trees

from srunner.autoagents.agent_wrapper import AgentWrapper
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider, CarlaActorPool
from srunner.scenariomanager.result_writer import ResultOutputProvider
from srunner.scenariomanager.timer import GameTime, TimeOut
from srunner.scenariomanager.watchdog import Watchdog


class Scenario(object):

    """
    Basic scenario class. This class holds the behavior_tree describing the
    scenario and the test criteria.

    The user must not modify this class.

    Important parameters:
    - behavior: User defined scenario with py_tree
    - criteria_list: List of user defined test criteria with py_tree
    - timeout (default = 60s): Timeout of the scenario in seconds
    - terminate_on_failure: Terminate scenario on first failure
    """

    def __init__(self, behavior, criteria, name, timeout=60, terminate_on_failure=False):
        self.behavior = behavior
        self.test_criteria = criteria
        self.timeout = timeout
        self.name = name

        if self.test_criteria is not None and not isinstance(self.test_criteria, py_trees.composites.Parallel):
            # list of nodes
            for criterion in self.test_criteria:
                criterion.terminate_on_failure = terminate_on_failure

            # Create py_tree for test criteria
            self.criteria_tree = py_trees.composites.Parallel(name="Test Criteria")
            self.criteria_tree.add_children(self.test_criteria)
            self.criteria_tree.setup(timeout=1)
        else:
            self.criteria_tree = criteria

        # Create node for timeout
        self.timeout_node = TimeOut(self.timeout, name="TimeOut")

        # Create overall py_tree
        self.scenario_tree = py_trees.composites.Parallel(name, policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        if behavior is not None:
            self.scenario_tree.add_child(self.behavior)
        self.scenario_tree.add_child(self.timeout_node)
        if criteria is not None:
            self.scenario_tree.add_child(self.criteria_tree)
        self.scenario_tree.setup(timeout=1)

    def _extract_nodes_from_tree(self, tree):  # pylint: disable=no-self-use
        """
        Returns the list of all nodes from the given tree
        """
        node_list = [tree]
        more_nodes_exist = True
        while more_nodes_exist:
            more_nodes_exist = False
            for node in node_list:
                if node.children:
                    node_list.remove(node)
                    more_nodes_exist = True
                    for child in node.children:
                        node_list.append(child)

        if len(node_list) == 1 and isinstance(node_list[0], py_trees.composites.Parallel):
            return []

        return node_list

    def get_criteria(self):
        """
        Return the list of test criteria (all leave nodes)
        """
        criteria_list = self._extract_nodes_from_tree(self.criteria_tree)
        return criteria_list

    def terminate(self):
        """
        This function sets the status of all leaves in the scenario tree to INVALID
        """
        # Get list of all nodes in the tree
        node_list = self._extract_nodes_from_tree(self.scenario_tree)

        # Set status to INVALID
        for node in node_list:
            node.terminate(py_trees.common.Status.INVALID)


class ScenarioManager(object):

    """
    Basic scenario manager class. This class holds all functionality
    required to start, and analyze a scenario.

    The user must not modify this class.

    To use the ScenarioManager:
    1. Create an object via manager = ScenarioManager()
    2. Load a scenario via manager.load_scenario()
    3. Trigger the execution of the scenario manager.execute()
       This function is designed to explicitly control start and end of
       the scenario execution
    4. Trigger a result evaluation with manager.analyze()
    5. Cleanup with manager.stop_scenario()
    """

    def __init__(self, client, debug_mode=False, timeout=2.0, log=None, playback=None):
        """
        Init requires scenario as input
        """
        self.scenario = None
        self.scenario_tree = None
        self.scenario_class = None
        self.ego_vehicles = None
        self.other_actors = None

        self._debug_mode = debug_mode
        self._agent = None
        self._running = False
        self._timestamp_last_run = 0.0
        self._timeout = timeout
        self._watchdog = Watchdog(float(self._timeout))

        self.scenario_duration_system = 0.0
        self.scenario_duration_game = 0.0
        self.start_system_time = None
        self.end_system_time = None

        # self._log = log
        # self._playback = playback
        # self._client = client

    def _reset(self):
        """
        Reset all parameters
        """
        self._running = False
        self._timestamp_last_run = 0.0
        self.scenario_duration_system = 0.0
        self.scenario_duration_game = 0.0
        self.start_system_time = None
        self.end_system_time = None
        GameTime.restart()

    def cleanup(self):
        """
        This function triggers a proper termination of a scenario
        """

        if self.scenario is not None:
            self.scenario.terminate()

        if self._agent is not None:
            self._agent.cleanup()
            self._agent = None

        CarlaDataProvider.cleanup()
        CarlaActorPool.cleanup()

    def load_scenario(self, scenario, agent=None):
        """
        Load a new scenario
        """
        self._reset()
        self._agent = AgentWrapper(agent) if agent else None
        self.scenario_class = scenario
        self.scenario = scenario.scenario
        self.scenario_tree = self.scenario.scenario_tree
        self.ego_vehicles = scenario.ego_vehicles
        self.other_actors = scenario.other_actors

        CarlaDataProvider.register_actors(self.ego_vehicles)
        CarlaDataProvider.register_actors(self.other_actors)
        # To print the scenario tree uncomment the next line
        # py_trees.display.render_dot_tree(self.scenario_tree)

        if self._agent is not None:
            self._agent.setup_sensors(self.ego_vehicles[0], self._debug_mode)

    def run_scenario(self):
        """
        Trigger the start of the scenario and wait for it to finish/fail
        """
        print("ScenarioManager: Running scenario {}".format(self.scenario_tree.name))
        self.start_system_time = time.time()
        start_game_time = GameTime.get_time()

        self._light_state = self.ego_vehicles[0].get_light_state()

        # if self._playback:
        #     time_step, vehicle_name = self.read_json()

        #     # Check if the time step matches the playback one
        #     settings = self.world.get_settings()
        #     current_time_step = settings.fixed_delta_seconds

        #     if time step != current_time_step:
        #         print("WARNING: Time steps are different, switching from {} to {}".format())
        #         settings.fixed_delta_seconds = time_step
        #         self.world.apply_settings(settings)

        # if self._log:
        #     settings = self.world.get_settings()
        #     vehicle_name = self.ego_vehicles[0].type_id
        #     self._log_data = {'records': [], 'vehicle': vehicle_name, 'time_step': settings.fixed_delta_seconds}
        #     self._delay = 0
        #     self._index = 0

        self._watchdog.start()
        self._running = True

        while self._running:
            timestamp = None
            world = CarlaDataProvider.get_world()
            if world:
                snapshot = world.get_snapshot()
                if snapshot:
                    timestamp = snapshot.timestamp
            if timestamp:
                self._tick_scenario(timestamp)

        self._watchdog.stop()

        # if self._log:
        #     with open(self._log, 'w') as fd:
        #         json.dump(self._log_data, fd, indent=4, sort_keys=True)

        self.cleanup()

        self.end_system_time = time.time()
        end_game_time = GameTime.get_time()

        self.scenario_duration_system = self.end_system_time - \
            self.start_system_time
        self.scenario_duration_game = end_game_time - start_game_time

        if self.scenario_tree.status == py_trees.common.Status.FAILURE:
            print("ScenarioManager: Terminated due to failure")

    # def read_json(self):
        
    #     self._control_list = []
    #     self._index = 0
    #     control_records = None
    #     time_step = None
    #     vehicle_name = None

    #     if self._playback:
    #         with open(self._playback) as fd:
    #             try:
    #                 control_records = json.load(fd)
    #             except json.JSONDecodeError:
    #                 pass

    #     if control_records and control_records['records']:
    #         # transform strs into VehicleControl commands
    #         for entry in control_records['records']:
    #             control = carla.VehicleControl(throttle=entry['control']['throttle'],
    #                                            steer=entry['control']['steer'],
    #                                            brake=entry['control']['brake'],
    #                                            hand_brake=entry['control']['hand_brake'],
    #                                            reverse=entry['control']['reverse'],
    #                                            manual_gear_shift=entry['control']['manual_gear_shift'],
    #                                            gear=entry['control']['gear'])
    #             self._control_list.append(control)
    #     if control_records and control_records['time_step']:
    #         time_step = control_records['time_step']
    #     if control_records and control_records['vehicle']:
    #         vehicle_name = control_records['vehicle']
        
    #     return time_step, vehicle_name

    # def write_json(self, control=None):

    #     if control is None:
    #         control = self.ego_vehicles[0].get_control()
    #         # As this is the one applied 1 (currently 2) frames behind, delay it
    #         if self._delay < 2:
    #             self._delay += 1
    #             return

    #     new_record = {'control':
    #                     {'throttle': control.throttle,
    #                     'steer': control.steer,
    #                     'brake': control.brake,
    #                     'hand_brake': control.hand_brake,
    #                     'reverse': control.reverse,
    #                     'manual_gear_shift': control.manual_gear_shift,
    #                     'gear': control.gear
    #                     }}
    #     self._log_data['records'].append(new_record)

    def _tick_scenario(self, timestamp):
        """
        Run next tick of scenario
        This function is a callback for world.on_tick()

        Important:
        - It has to be ensured that the scenario has not yet completed/failed
          and that the time moved forward.
        - A thread lock should be used to avoid that the scenario tick is performed
          multiple times in parallel.
        """

        if self._timestamp_last_run < timestamp.elapsed_seconds and self._running:
            self._timestamp_last_run = timestamp.elapsed_seconds

            self._watchdog.update()

            if self._debug_mode:
                print("\n--------- Tick ---------\n")

            # Update game time and actor information
            GameTime.on_carla_tick(timestamp)
            CarlaDataProvider.on_carla_tick()

            if self._agent is not None:
                ego_action = self._agent()

            # Tick scenario
            self.scenario_tree.tick_once()

            if self._debug_mode:
                print("\n")
                py_trees.display.print_ascii_tree(self.scenario_tree, show_status=True)
                sys.stdout.flush()

            if self.scenario_tree.status != py_trees.common.Status.RUNNING:
                self._running = False

            # Tick the agent
            # if self._playback:
            #     if self._index < len(self._control_list):
            #         self._client.apply_batch_sync(
            #             [carla.command.ApplyVehicleControl(self.ego_vehicles[0].id, self._control_list[self._index])]
            #         )
            #         self._index += 1
            #     else:
            #         print("JSON file has no more entries")
            if self._agent is not None:
                self._client.apply_batch_sync(
                    [carla.command.ApplyVehicleControl(self.ego_vehicles[0].id, ego_action)]
                )

            # Add the agent control
            # if self._log:
            #     if self._agent:
            #         self.write_json(ego_action)
            #     else:
            #         self.write_json()

        if CarlaDataProvider.is_sync_mode() and self._running and self._watchdog.get_status():
            while True:
                light_state = self.ego_vehicles[0].get_light_state()
                time.sleep(1)
                if light_state != self._light_state:
                    print("Waiting for manual control")
                    time.sleep(0.05)
                else:
                    break
            CarlaDataProvider.get_world().tick()
            print("{} -- {}".format(self.ego_vehicles[0].get_transform(), CarlaDataProvider.get_world().get_snapshot().timestamp.frame))
            # self.ego_vehicles[0].set_light_state(carla.VehicleLightState.Position)

    def get_running_status(self):
        """
        returns:
           bool:  False if watchdog exception occured, True otherwise
        """
        return self._watchdog.get_status()

    def stop_scenario(self):
        """
        This function is used by the overall signal handler to terminate the scenario execution
        """
        self._running = False

    def analyze_scenario(self, stdout, filename, junit):
        """
        This function is intended to be called from outside and provide
        the final statistics about the scenario (human-readable, in form of a junit
        report, etc.)
        """

        failure = False
        timeout = False
        result = "SUCCESS"

        if self.scenario.test_criteria is None:
            return True

        for criterion in self.scenario.get_criteria():
            if (not criterion.optional and
                    criterion.test_status != "SUCCESS" and
                    criterion.test_status != "ACCEPTABLE"):
                failure = True
                result = "FAILURE"
            elif criterion.test_status == "ACCEPTABLE":
                result = "ACCEPTABLE"

        if self.scenario.timeout_node.timeout and not failure:
            timeout = True
            result = "TIMEOUT"

        output = ResultOutputProvider(self, result, stdout, filename, junit)
        output.write()

        return failure or timeout
