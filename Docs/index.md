[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![GitHub tag (latest SemVer)](https://img.shields.io/github/tag/carla-simulator/scenario_runner.svg)

# ScenarioRunner

ScenarioRunner is the scenario engine for [CARLA](http://carla.org/). A scenario is a scripted traffic situation — a lead vehicle braking hard, a pedestrian stepping out, a car running a red light — together with the criteria that decide whether the ego vehicle handled it. ScenarioRunner spawns the actors, plays the situation out on a running CARLA server and reports a pass or fail for every criterion.

Scenarios are written in Python, in the ASAM [OpenSCENARIO](http://www.openscenario.org/) standard, or embedded in a route, and the same scenario can be replayed against a human driver, the autopilot or your own driving stack as many times as needed.

ScenarioRunner is a pure Python client: nothing is added to the server and nothing has to be compiled.

---

## Quick start

[**Get ScenarioRunner**](<getting_scenariorunner>) — Pick the branch for your CARLA, install it and set the environment.<br>

[**First steps**](<getting_started>) — The four modes and how to run each one.<br>

[**Create a new scenario**](<creating_new_scenario>) — Write a scenario class and its configuration.<br>

[**Agent evaluation**](<agent_evaluation>) — Drive a route with your own agent and read the route criteria.<br>

[**Metrics module**](<metrics_module>) — Compute your own metrics from a recording, without simulating again.<br>

[**F.A.Q.**](<FAQ>) — The errors that come up most often.<br>

[**Release notes**](<CHANGELOG>) — Features, fixes and other changes listed per release.<br>

## References

[**List of scenarios**](<list_of_scenarios>) — The 19 standalone types and the 26 route types that ship.<br>

[**OpenSCENARIO support**](<openscenario_support>) — Which parts of OpenSCENARIO 1.x are implemented.<br>

[**OpenSCENARIO 2.0**](<README_OpenSCENARIO_2.0.md>) — The experimental `.osc` support.<br>

[**ROS agent**](<ros_agent>) — Drive the ego from a ROS stack.<br>

## Contributing

[**Code of conduct**](<CODE_OF_CONDUCT>) — Standard rights and duties for contributors.<br>

[**Coding standard**](<coding_standard>) — Guidelines to write proper code.<br>

[**Contribution guidelines**](<CONTRIBUTING>) — The different ways to contribute to ScenarioRunner.<br>

---

Questions about ScenarioRunner are welcome on the CARLA forum.

<div class="build-buttons">
<a href="https://github.com/carla-simulator/carla/discussions" target="_blank" class="btn btn-neutral" title="Go to the CARLA discussions">
CARLA discussions</a>
</div>
