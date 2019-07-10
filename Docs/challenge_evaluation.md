Challenge evaluator
=================


 *This tutorial shows how to put some sample   agent to run the
 challenge evaluation.*

The idea of the evaluation for the challenge is to put
the hero agent to perform on several scenarios described in a XML file.
A scenario is defined by a certain trajectory that the hero
agent has to follow  and certain events
that are going to take effect during this trajectory.
The scenario also control the termination criteria, and the
score criteria.

At the end of a route, the system gives a result (fail or success)
and a final score (numeric).

### Getting started

#### Installation

Please, to install the system, follow the general [installation steps for
the scenario runner repository](getting_started.md/#install_prerequisites)

To run the challenge, several environment variables need to be provided:
```Bash
export CARLA_ROOT=/path/to/your/carla/installation
export ROOT_SCENARIO_RUNNER=`pwd`
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/:${ROOT_SCENARIO_RUNNER}:`pwd`:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.5-py2.7-linux-x86_64.egg:${CARLA_ROOT}/PythonAPI/carla/agents
```

#### Running sample agents
In general, both Python 2 and Python 3 are supported. In the following, we just refer to "python" as representative. Please replace it with your python executable.

You can run the challenge evaluation as follows:
```
python srunner/challenge/challenge_evaluator_routes.py --scenarios=${ROOT_SCENARIO_RUNNER}/srunner/challenge/all_towns_traffic_scenarios1_3_4.json --agent=${ROOT_SCENARIO_RUNNER}/srunner/challenge/autoagents/DummyAgent.py
```

After running the evaluator, you should see the CARLA simulator being started
and the following type of output should continuously  appear on the terminal screen:

    =====================>
    [Left -- 00000] with shape (600, 800, 3)
    [Center -- 00000] with shape (600, 800, 3)
    [GPS -- 000000] with shape (3,)
    [Right -- 000000] with shape (600, 800, 3)
    [LIDAR -- 000000] with shape (2751, 3)
    <=====================

This output shows the sensor data received by the sample agent.

You can add --file option to save logs with respect to the challenge
evaluation results.


## Implement your own Agent

Finally, you can also add your own agent into the system by following [this tutorial](agent_evaluation.md)


### ROS-based Agent

Implement an Agent for a ROS-based stack is described [here](ros_agent.md).
