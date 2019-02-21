
### Challenge evaluator



The idea of the evaluation for the challenge is to run an XML file
describing a route that agent will have to go through . These files
also specify the scenarios that are going to be happen during the route.

The scenario is evaluatedtor 


#### Getting started

Dummy agent execution  example.


### Installation

Please to install the system, follow the general [installation steps for 
the scenario runner repository ](getting_started.md/#install_pre)



Define the carla root variable, where the root folder of
 the CARLA 0.9.3 release is located:

    export CARLA_ROOT=<path_to_carla_root_folder>


You can see the list of supported scenarios before you run the evaluation:

    python3 challenge_evaluator.py --list
 

To run for instance the dummy agent on the basic scenario you should
run the following command: 


    python3 challenge_evaluator.py --scenario config:ChallengeBasic
    -a srunner/challenge/agents/DummyAgent.py --carla-root <path_to_carla_root>


With tbhis command we are





#### Adding your Agent to be evaluated ..


There are two main parts that need to be defined in order to set your agent to run


##### setup function:

This function is were you set all the sensors required by your agent,
you can configure the exact transformation of the sensor in the world.

This step is also 

(SHOW AN example of part of the step )





You must redefine your the autonomous agent class 



