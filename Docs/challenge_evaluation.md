Challenge evaluator
=================


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


Run the setup environment script in order to point where the root folder of
 the CARLA 0.9.3 release is located:

    ./setup_environment.sh --carl-root <path_to_the_CARLA_root_folder>


#### Running sample agents

You can see the list of supported scenarios before you run the evaluation:

    python3 challenge_evaluator.py --list
 

To control, using the keyboard, an agent running a basic scenario run:

    ...

To run for instance the dummy agent on the basic scenario you should
run the following command: 


    python3 srunner/challenge/challenge_evaluator.py --scenario group:ChallengeBasic -a srunner/challenge/agents/DummyAgent.py --carla-root <path_to_carla_root>



After running this command you should see the CARLA simulator being started
and the following type of output should continuously  appear on the terminal screen:

    =====================>
    [Left -- 00000] with shape (600, 800, 3)
    [Center -- 00000] with shape (600, 800, 3)
    [GPS -- 000000] with shape (3,)
    [Right -- 000000] with shape (600, 800, 3)
    [LIDAR -- 000000] with shape (2751, 3)
    <=====================

This output shows the sensor data received by the sample agent.


You can also try to complete the basic scenario as a human driver.


    python3 challenge_evaluator.py --scenario config:ChallengeBasic -a srunner/challenge/agents/HumanAgent.py --carla-root <path_to_carla_root>



Finally, you can also add your own agent into the system by following [this tutorial](Docs/agent_evaluation.md)




