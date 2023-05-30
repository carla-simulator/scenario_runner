# Details

Date : 2023-05-23 20:48:51

Directory f:\\C\\V6\\scenario_runner

Total : 143 files,  28352 codes, 6973 comments, 6114 blanks, all 41439 lines

[Summary](results.md) / Details / [Diff Summary](diff.md) / [Diff Details](diff-details.md)

## Files
| filename | language | code | comment | blank | total |
| :--- | :--- | ---: | ---: | ---: | ---: |
| [.pylintrc](/.pylintrc) | Ini | 7 | 0 | 1 | 8 |
| [.readthedocs.yml](/.readthedocs.yml) | YAML | 8 | 2 | 7 | 17 |
| [Dockerfile](/Dockerfile) | Docker | 23 | 19 | 10 | 52 |
| [Docs/CHANGELOG.md](/Docs/CHANGELOG.md) | Markdown | 423 | 0 | 24 | 447 |
| [Docs/CODE_OF_CONDUCT.md](/Docs/CODE_OF_CONDUCT.md) | Markdown | 55 | 0 | 19 | 74 |
| [Docs/CONTRIBUTING.md](/Docs/CONTRIBUTING.md) | Markdown | 66 | 0 | 31 | 97 |
| [Docs/FAQ.md](/Docs/FAQ.md) | Markdown | 26 | 0 | 9 | 35 |
| [Docs/agent_evaluation.md](/Docs/agent_evaluation.md) | Markdown | 57 | 0 | 30 | 87 |
| [Docs/coding_standard.md](/Docs/coding_standard.md) | Markdown | 27 | 0 | 10 | 37 |
| [Docs/creating_new_scenario.md](/Docs/creating_new_scenario.md) | Markdown | 76 | 0 | 19 | 95 |
| [Docs/extra.css](/Docs/extra.css) | CSS | 97 | 11 | 25 | 133 |
| [Docs/getting_scenariorunner.md](/Docs/getting_scenariorunner.md) | Markdown | 168 | 0 | 73 | 241 |
| [Docs/getting_started.md](/Docs/getting_started.md) | Markdown | 86 | 0 | 27 | 113 |
| [Docs/index.md](/Docs/index.md) | Markdown | 26 | 0 | 23 | 49 |
| [Docs/list_of_scenarios.md](/Docs/list_of_scenarios.md) | Markdown | 64 | 0 | 15 | 79 |
| [Docs/metrics_module.md](/Docs/metrics_module.md) | Markdown | 361 | 0 | 109 | 470 |
| [Docs/openscenario_support.md](/Docs/openscenario_support.md) | Markdown | 137 | 0 | 52 | 189 |
| [Docs/requirements.txt](/Docs/requirements.txt) | pip requirements | 3 | 0 | 1 | 4 |
| [Docs/ros_agent.md](/Docs/ros_agent.md) | Markdown | 26 | 0 | 23 | 49 |
| [Jenkinsfile](/Jenkinsfile) | Groovy | 177 | 3 | 7 | 187 |
| [README.md](/README.md) | Markdown | 45 | 0 | 20 | 65 |
| [actor_info.py](/actor_info.py) | Python | 30 | 0 | 10 | 40 |
| [manual_control.py](/manual_control.py) | Python | 756 | 118 | 131 | 1,005 |
| [metrics_manager.py](/metrics_manager.py) | Python | 70 | 52 | 33 | 155 |
| [mkdocs.yml](/mkdocs.yml) | YAML | 26 | 0 | 4 | 30 |
| [no_rendering_mode.py](/no_rendering_mode.py) | Python | 1,097 | 120 | 278 | 1,495 |
| [requirements.txt](/requirements.txt) | pip requirements | 12 | 0 | 1 | 13 |
| [scenario.py](/scenario.py) | Python | 153 | 64 | 70 | 287 |
| [scenario_runner.py](/scenario_runner.py) | Python | 533 | 137 | 222 | 892 |
| [srunner/__init__.py](/srunner/__init__.py) | Python | 0 | 0 | 1 | 1 |
| [srunner/autoagents/__init__.py](/srunner/autoagents/__init__.py) | Python | 0 | 0 | 1 | 1 |
| [srunner/autoagents/agent_wrapper.py](/srunner/autoagents/agent_wrapper.py) | Python | 52 | 29 | 16 | 97 |
| [srunner/autoagents/autonomous_agent.py](/srunner/autoagents/autonomous_agent.py) | Python | 38 | 47 | 24 | 109 |
| [srunner/autoagents/dummy_agent.py](/srunner/autoagents/dummy_agent.py) | Python | 34 | 34 | 16 | 84 |
| [srunner/autoagents/human_agent.py](/srunner/autoagents/human_agent.py) | Python | 147 | 78 | 52 | 277 |
| [srunner/autoagents/npc_agent.py](/srunner/autoagents/npc_agent.py) | Python | 43 | 32 | 23 | 98 |
| [srunner/autoagents/ros_agent.py](/srunner/autoagents/ros_agent.py) | Python | 302 | 95 | 54 | 451 |
| [srunner/autoagents/sensor_interface.py](/srunner/autoagents/sensor_interface.py) | Python | 82 | 51 | 33 | 166 |
| [srunner/data/routes_debug.xml](/srunner/data/routes_debug.xml) | XML | 91 | 0 | 2 | 93 |
| [srunner/data/routes_devtest.xml](/srunner/data/routes_devtest.xml) | XML | 1,060 | 0 | 0 | 1,060 |
| [srunner/data/routes_training.xml](/srunner/data/routes_training.xml) | XML | 1,923 | 0 | 0 | 1,923 |
| [srunner/examples/ChangeLane.xml](/srunner/examples/ChangeLane.xml) | XML | 12 | 1 | 1 | 14 |
| [srunner/examples/ControlLoss.xml](/srunner/examples/ControlLoss.xml) | XML | 48 | 0 | 1 | 49 |
| [srunner/examples/CutIn.xml](/srunner/examples/CutIn.xml) | XML | 13 | 1 | 1 | 15 |
| [srunner/examples/FollowLeadingVehicle.xml](/srunner/examples/FollowLeadingVehicle.xml) | XML | 70 | 0 | 1 | 71 |
| [srunner/examples/FreeRide.xml](/srunner/examples/FreeRide.xml) | XML | 23 | 0 | 1 | 24 |
| [srunner/examples/LeadingVehicle.xml](/srunner/examples/LeadingVehicle.xml) | XML | 33 | 0 | 1 | 34 |
| [srunner/examples/NoSignalJunction.xml](/srunner/examples/NoSignalJunction.xml) | XML | 7 | 0 | 1 | 8 |
| [srunner/examples/ObjectCrossing.xml](/srunner/examples/ObjectCrossing.xml) | XML | 57 | 0 | 2 | 59 |
| [srunner/examples/OppositeDirection.xml](/srunner/examples/OppositeDirection.xml) | XML | 15 | 0 | 1 | 16 |
| [srunner/examples/RunningRedLight.xml](/srunner/examples/RunningRedLight.xml) | XML | 23 | 0 | 0 | 23 |
| [srunner/examples/SignalizedJunctionLeftTurn.xml](/srunner/examples/SignalizedJunctionLeftTurn.xml) | XML | 27 | 0 | 1 | 28 |
| [srunner/examples/SignalizedJunctionRightTurn.xml](/srunner/examples/SignalizedJunctionRightTurn.xml) | XML | 31 | 0 | 1 | 32 |
| [srunner/examples/VehicleTurning.xml](/srunner/examples/VehicleTurning.xml) | XML | 51 | 0 | 1 | 52 |
| [srunner/metrics/examples/basic_metric.py](/srunner/metrics/examples/basic_metric.py) | Python | 7 | 29 | 7 | 43 |
| [srunner/metrics/examples/criteria_filter.py](/srunner/metrics/examples/criteria_filter.py) | Python | 17 | 19 | 11 | 47 |
| [srunner/metrics/examples/distance_between_vehicles.py](/srunner/metrics/examples/distance_between_vehicles.py) | Python | 27 | 26 | 17 | 70 |
| [srunner/metrics/examples/distance_to_lane_center.py](/srunner/metrics/examples/distance_to_lane_center.py) | Python | 27 | 24 | 18 | 69 |
| [srunner/metrics/tools/metrics_log.py](/srunner/metrics/tools/metrics_log.py) | Python | 154 | 192 | 83 | 429 |
| [srunner/metrics/tools/metrics_parser.py](/srunner/metrics/tools/metrics_parser.py) | Python | 378 | 47 | 116 | 541 |
| [srunner/openscenario/0.9.x/OpenSCENARIO_Catalog.xsd](/srunner/openscenario/0.9.x/OpenSCENARIO_Catalog.xsd) | XML | 33 | 0 | 7 | 40 |
| [srunner/openscenario/0.9.x/OpenSCENARIO_TypeDefs.xsd](/srunner/openscenario/0.9.x/OpenSCENARIO_TypeDefs.xsd) | XML | 1,394 | 3 | 61 | 1,458 |
| [srunner/openscenario/0.9.x/OpenSCENARIO_v0.9.1.xsd](/srunner/openscenario/0.9.x/OpenSCENARIO_v0.9.1.xsd) | XML | 213 | 0 | 6 | 219 |
| [srunner/openscenario/0.9.x/migration0_9_1to1_0.xslt](/srunner/openscenario/0.9.x/migration0_9_1to1_0.xslt) | XSL | 3,644 | 7 | 28 | 3,679 |
| [srunner/openscenario/OpenSCENARIO.xsd](/srunner/openscenario/OpenSCENARIO.xsd) | XML | 1,494 | 12 | 1 | 1,507 |
| [srunner/scenarioconfigs/__init__.py](/srunner/scenarioconfigs/__init__.py) | Python | 0 | 0 | 1 | 1 |
| [srunner/scenarioconfigs/openscenario_configuration.py](/srunner/scenarioconfigs/openscenario_configuration.py) | Python | 260 | 94 | 79 | 433 |
| [srunner/scenarioconfigs/route_scenario_configuration.py](/srunner/scenarioconfigs/route_scenario_configuration.py) | Python | 18 | 17 | 16 | 51 |
| [srunner/scenarioconfigs/scenario_configuration.py](/srunner/scenarioconfigs/scenario_configuration.py) | Python | 44 | 21 | 22 | 87 |
| [srunner/scenariomanager/__init__.py](/srunner/scenariomanager/__init__.py) | Python | 0 | 0 | 1 | 1 |
| [srunner/scenariomanager/actorcontrols/__init__.py](/srunner/scenariomanager/actorcontrols/__init__.py) | Python | 0 | 0 | 1 | 1 |
| [srunner/scenariomanager/actorcontrols/actor_control.py](/srunner/scenariomanager/actorcontrols/actor_control.py) | Python | 60 | 97 | 26 | 183 |
| [srunner/scenariomanager/actorcontrols/basic_control.py](/srunner/scenariomanager/actorcontrols/basic_control.py) | Python | 32 | 72 | 15 | 119 |
| [srunner/scenariomanager/actorcontrols/carla_autopilot.py](/srunner/scenariomanager/actorcontrols/carla_autopilot.py) | Python | 9 | 31 | 11 | 51 |
| [srunner/scenariomanager/actorcontrols/external_control.py](/srunner/scenariomanager/actorcontrols/external_control.py) | Python | 9 | 25 | 10 | 44 |
| [srunner/scenariomanager/actorcontrols/npc_vehicle_control.py](/srunner/scenariomanager/actorcontrols/npc_vehicle_control.py) | Python | 65 | 47 | 30 | 142 |
| [srunner/scenariomanager/actorcontrols/pedestrian_control.py](/srunner/scenariomanager/actorcontrols/pedestrian_control.py) | Python | 32 | 30 | 16 | 78 |
| [srunner/scenariomanager/actorcontrols/simple_vehicle_control.py](/srunner/scenariomanager/actorcontrols/simple_vehicle_control.py) | Python | 169 | 130 | 47 | 346 |
| [srunner/scenariomanager/actorcontrols/vehicle_longitudinal_control.py](/srunner/scenariomanager/actorcontrols/vehicle_longitudinal_control.py) | Python | 28 | 30 | 16 | 74 |
| [srunner/scenariomanager/actorcontrols/visualizer.py](/srunner/scenariomanager/actorcontrols/visualizer.py) | Python | 60 | 50 | 21 | 131 |
| [srunner/scenariomanager/carla_data_provider.py](/srunner/scenariomanager/carla_data_provider.py) | Python | 509 | 193 | 127 | 829 |
| [srunner/scenariomanager/result_writer.py](/srunner/scenariomanager/result_writer.py) | Python | 183 | 65 | 43 | 291 |
| [srunner/scenariomanager/scenario_manager.py](/srunner/scenariomanager/scenario_manager.py) | Python | 132 | 60 | 56 | 248 |
| [srunner/scenariomanager/scenarioatomics/__init__.py](/srunner/scenariomanager/scenarioatomics/__init__.py) | Python | 0 | 0 | 1 | 1 |
| [srunner/scenariomanager/scenarioatomics/atomic_behaviors.py](/srunner/scenariomanager/scenarioatomics/atomic_behaviors.py) | Python | 1,620 | 1,143 | 658 | 3,421 |
| [srunner/scenariomanager/scenarioatomics/atomic_criteria.py](/srunner/scenariomanager/scenarioatomics/atomic_criteria.py) | Python | 1,181 | 471 | 433 | 2,085 |
| [srunner/scenariomanager/scenarioatomics/atomic_trigger_conditions.py](/srunner/scenariomanager/scenarioatomics/atomic_trigger_conditions.py) | Python | 617 | 447 | 309 | 1,373 |
| [srunner/scenariomanager/scenarioatomics/test.py](/srunner/scenariomanager/scenarioatomics/test.py) | Python | 7 | 2 | 12 | 21 |
| [srunner/scenariomanager/tempCodeRunnerFile.py](/srunner/scenariomanager/tempCodeRunnerFile.py) | Python | 1 | 0 | 0 | 1 |
| [srunner/scenariomanager/timer.py](/srunner/scenariomanager/timer.py) | Python | 108 | 102 | 75 | 285 |
| [srunner/scenariomanager/traffic_events.py](/srunner/scenariomanager/traffic_events.py) | Python | 33 | 34 | 18 | 85 |
| [srunner/scenariomanager/watchdog.py](/srunner/scenariomanager/watchdog.py) | Python | 42 | 30 | 15 | 87 |
| [srunner/scenariomanager/weather_sim.py](/srunner/scenariomanager/weather_sim.py) | Python | 57 | 81 | 29 | 167 |
| [srunner/scenarios/__init__.py](/srunner/scenarios/__init__.py) | Python | 0 | 0 | 1 | 1 |
| [srunner/scenarios/background_activity.py](/srunner/scenarios/background_activity.py) | Python | 49 | 26 | 19 | 94 |
| [srunner/scenarios/basic_scenario.py](/srunner/scenarios/basic_scenario.py) | Python | 171 | 88 | 60 | 319 |
| [srunner/scenarios/change_lane.py](/srunner/scenarios/change_lane.py) | Python | 98 | 52 | 32 | 182 |
| [srunner/scenarios/construction_crash_vehicle.py](/srunner/scenarios/construction_crash_vehicle.py) | Python | 105 | 36 | 25 | 166 |
| [srunner/scenarios/control_loss.py](/srunner/scenarios/control_loss.py) | Python | 122 | 50 | 27 | 199 |
| [srunner/scenarios/cut_in.py](/srunner/scenarios/cut_in.py) | Python | 83 | 42 | 34 | 159 |
| [srunner/scenarios/follow_leading_vehicle.py](/srunner/scenarios/follow_leading_vehicle.py) | Python | 183 | 89 | 54 | 326 |
| [srunner/scenarios/freeride.py](/srunner/scenarios/freeride.py) | Python | 28 | 26 | 15 | 69 |
| [srunner/scenarios/junction_crossing_route.py](/srunner/scenarios/junction_crossing_route.py) | Python | 91 | 72 | 41 | 204 |
| [srunner/scenarios/maneuver_opposite_direction.py](/srunner/scenarios/maneuver_opposite_direction.py) | Python | 108 | 39 | 26 | 173 |
| [srunner/scenarios/master_scenario.py](/srunner/scenarios/master_scenario.py) | Python | 60 | 27 | 28 | 115 |
| [srunner/scenarios/no_signal_junction_crossing.py](/srunner/scenarios/no_signal_junction_crossing.py) | Python | 89 | 46 | 30 | 165 |
| [srunner/scenarios/object_crash_intersection.py](/srunner/scenarios/object_crash_intersection.py) | Python | 342 | 152 | 113 | 607 |
| [srunner/scenarios/object_crash_vehicle.py](/srunner/scenarios/object_crash_vehicle.py) | Python | 259 | 83 | 63 | 405 |
| [srunner/scenarios/open_scenario.py](/srunner/scenarios/open_scenario.py) | Python | 353 | 102 | 122 | 577 |
| [srunner/scenarios/opposite_vehicle_taking_priority.py](/srunner/scenarios/opposite_vehicle_taking_priority.py) | Python | 131 | 52 | 45 | 228 |
| [srunner/scenarios/other_leading_vehicle.py](/srunner/scenarios/other_leading_vehicle.py) | Python | 79 | 46 | 27 | 152 |
| [srunner/scenarios/route_scenario.py](/srunner/scenarios/route_scenario.py) | Python | 313 | 95 | 109 | 517 |
| [srunner/scenarios/signalized_junction_left_turn.py](/srunner/scenarios/signalized_junction_left_turn.py) | Python | 88 | 40 | 23 | 151 |
| [srunner/scenarios/signalized_junction_right_turn.py](/srunner/scenarios/signalized_junction_right_turn.py) | Python | 98 | 39 | 26 | 163 |
| [srunner/tests/__init__.py](/srunner/tests/__init__.py) | Python | 0 | 0 | 1 | 1 |
| [srunner/tests/carla_mocks/README.md](/srunner/tests/carla_mocks/README.md) | Markdown | 3 | 0 | 1 | 4 |
| [srunner/tests/carla_mocks/__init__.py](/srunner/tests/carla_mocks/__init__.py) | Python | 0 | 0 | 1 | 1 |
| [srunner/tests/carla_mocks/agents/__init__.py](/srunner/tests/carla_mocks/agents/__init__.py) | Python | 0 | 0 | 1 | 1 |
| [srunner/tests/carla_mocks/agents/navigation/__init__.py](/srunner/tests/carla_mocks/agents/navigation/__init__.py) | Python | 0 | 0 | 1 | 1 |
| [srunner/tests/carla_mocks/agents/navigation/basic_agent.py](/srunner/tests/carla_mocks/agents/navigation/basic_agent.py) | Python | 161 | 90 | 51 | 302 |
| [srunner/tests/carla_mocks/agents/navigation/behavior_agent.py](/srunner/tests/carla_mocks/agents/navigation/behavior_agent.py) | Python | 193 | 125 | 57 | 375 |
| [srunner/tests/carla_mocks/agents/navigation/behavior_types.py](/srunner/tests/carla_mocks/agents/navigation/behavior_types.py) | Python | 24 | 6 | 8 | 38 |
| [srunner/tests/carla_mocks/agents/navigation/controller.py](/srunner/tests/carla_mocks/agents/navigation/controller.py) | Python | 117 | 101 | 40 | 258 |
| [srunner/tests/carla_mocks/agents/navigation/global_route_planner.py](/srunner/tests/carla_mocks/agents/navigation/global_route_planner.py) | Python | 277 | 77 | 39 | 393 |
| [srunner/tests/carla_mocks/agents/navigation/local_planner.py](/srunner/tests/carla_mocks/agents/navigation/local_planner.py) | Python | 174 | 113 | 47 | 334 |
| [srunner/tests/carla_mocks/agents/tools/__init__.py](/srunner/tests/carla_mocks/agents/tools/__init__.py) | Python | 0 | 0 | 1 | 1 |
| [srunner/tests/carla_mocks/agents/tools/misc.py](/srunner/tests/carla_mocks/agents/tools/misc.py) | Python | 68 | 69 | 35 | 172 |
| [srunner/tests/carla_mocks/carla.py](/srunner/tests/carla_mocks/carla.py) | Python | 207 | 8 | 94 | 309 |
| [srunner/tests/test_xosc_load.py](/srunner/tests/test_xosc_load.py) | Python | 28 | 14 | 10 | 52 |
| [srunner/tools/__init__.py](/srunner/tools/__init__.py) | Python | 0 | 0 | 1 | 1 |
| [srunner/tools/history.py](/srunner/tools/history.py) | Python | 50 | 0 | 45 | 95 |
| [srunner/tools/openscenario_parser.py](/srunner/tools/openscenario_parser.py) | Python | 995 | 181 | 220 | 1,396 |
| [srunner/tools/py_trees_port.py](/srunner/tools/py_trees_port.py) | Python | 61 | 59 | 18 | 138 |
| [srunner/tools/route_manipulation.py](/srunner/tools/route_manipulation.py) | Python | 72 | 49 | 36 | 157 |
| [srunner/tools/route_parser.py](/srunner/tools/route_parser.py) | Python | 194 | 77 | 55 | 326 |
| [srunner/tools/scenario_helper.py](/srunner/tools/scenario_helper.py) | Python | 414 | 125 | 109 | 648 |
| [srunner/tools/scenario_parser.py](/srunner/tools/scenario_parser.py) | Python | 71 | 24 | 30 | 125 |
| [srunner/utilities/code_check_and_formatting.sh](/srunner/utilities/code_check_and_formatting.sh) | Shell Script | 14 | 1 | 4 | 19 |
| [storyboard.py](/storyboard.py) | Python | 676 | 127 | 320 | 1,123 |
| [temp.xml](/temp.xml) | XML | 212 | 0 | 4 | 216 |
| [tempCodeRunnerFile.py](/tempCodeRunnerFile.py) | Python | 1 | 0 | 0 | 1 |
| [test.py](/test.py) | Python | 6 | 0 | 2 | 8 |
| [test2.py](/test2.py) | Python | 3 | 0 | 1 | 4 |

[Summary](results.md) / Details / [Diff Summary](diff.md) / [Diff Details](diff-details.md)