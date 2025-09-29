# OpenScenario 2.0

We can use [OpenScenario 2.0](https://www.asam.net/static_downloads/public/asam-openscenario/2.0.0/welcome.html) as the scenario description language specification to design and implement the corresponding compilation system, which can automatically convert the test scenario described with OpenScenario 2.0 into a test scenario based on scenario runner, so as to use carla for autonomous driving testing.

## Installation

**1 Install JDK**

```
sudo apt install openjdk-17-jdk
```

Confirm installation:

```
$ java -version
```

Output:

```
openjdk version "17.0.5" 2022-10-18
OpenJDK Runtime Environment (build 17.0.5+8-Ubuntu-2ubuntu120.04)
OpenJDK 64-Bit Server VM (build 17.0.5+8-Ubuntu-2ubuntu120.04, mixed mode, sharing)
```

**3 Install Antlr 4.10.1**

[Antlr4](https://github.com/antlr/antlr4) is used to build AST and is the core component of OSC2 parsing.

```
sudo apt install curl
curl -O https://www.antlr.org/download/antlr-4.10.1-complete.jar
```

Put the .jar file into local/lib

```
$ sudo cp antlr-4.10.1-complete.jar /usr/local/lib/
```

The following three steps are used to configure environment variables and create aliases so that antlr4 can be easily used from the command line.

```
$ sudo gedit ~/.bashrc
```

Add the following at the end:

```
export CLASSPATH=".:/usr/local/lib/antlr-4.10.1-complete.jar:$CLASSPATH"
alias antlr4='java -jar /usr/local/lib/antlr-4.10.1-complete.jar'
alias grun='java org.antlr.v4.gui.TestRig'
```

And then:

```
source ~/.bashrc
```

Enter antlr4 for verifying in the terminal:

```
$ antlr4
ANTLR Parser Generator  Version 4.10.1
 -o ___              specify output directory where all output is generated
 -lib ___            specify location of grammars, tokens files
 -atn                generate rule augmented transition network diagrams
 -encoding ___       specify grammar file encoding; e.g., euc-jp
 -message-format ___ specify output style for messages in antlr, gnu, vs2005
 -long-messages      show exception details when available for errors and warnings
 -listener           generate parse tree listener (default)
 -no-listener        don't generate parse tree listener
 -visitor            generate parse tree visitor
 -no-visitor         don't generate parse tree visitor (default)
 -package ___        specify a package/namespace for the generated code
 -depend             generate file dependencies
 -D<option>=value    set/override a grammar-level option
 -Werror             treat warnings as errors
 -XdbgST             launch StringTemplate visualizer on generated code
 -XdbgSTWait         wait for STViz to close before continuing
 -Xforce-atn         use the ATN simulator for all predictions
 -Xlog               dump lots of logging info to antlr-timestamp.log
 -Xexact-output-dir  all output goes into -o dir regardless of paths/package
```

**4 Install antlr4 runtime**

```
pip install antlr4-python3-runtime==4.10
```

**5 Install python dependencies**

```
pip install -r requirements.txt
```

**6 Install graphviz**

[graphviz](https://graphviz.org/) is used for visualizing AST and is not required to install, but can be installed for debugging with the following command.

```
sudo apt-get install graphviz
```

**7 Configurate carla**

1. Download [carla release](https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.13.tar.gz)

2. Extract the carla installation package to a directory.

On Ubuntu systems, the Carla environment variable is configured as follows:

```bash
export CARLA_ROOT=/home/dut-aiid/CARLA_0.9.13
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg:${CARLA_ROOT}/PythonAPI/carla/agents:${CARLA_ROOT}/PythonAPI/carla/agents/navigation:${CARLA_ROOT}/PythonAPI/carla:${CARLA_ROOT}/PythonAPI/examples:${CARLA_ROOT}/PythonAPI
```

## Quickstart

**1 Run carla**

Launch the CARLA simulation.

```bash
cd /home/xxx/CARLA_0.9.16
./CarlaUE4.sh
```

**2 Start manual_control**

Run the Scenario Runner's `manual_control.py` script

```
python manual_control.py -a --rolename=ego_vehicle
```

!!! note

    There is a script with the same name in `PythonAPI/examples` within CARLA, here we need to run the `manual_control.py` in the ScenarioRunner repo.

**3 Run a OpenSCENARIO 2.0 scenario**

Run the scenario indicating the `--openscenario2` flag.

```
python scenario_runner.py --sync  --openscenario2 srunner/examples/overtake_concrete.osc --reloadWorld
```

To avoid working with two terminals, you can launch a separate process for the manual control in the same command as:

```sh
gnome-terminal -- bash -c  "python manual_control.py -a --rolename=ego_vehicle"; python scenario_runner.py --sync  --openscenario2 srunner/examples/overtake_concrete.osc --reloadWorld
```

## OSC2 Syntax

Here we show how to define a basic scenario. We will take as an example an overtaking maneuver. See the full example in [`overtake_concrete.osc`](https://github.com/carla-simulator/scenario_runner/blob/master/srunner/examples/overtake_concrete.osc).

Define the scenario setup:

```
import basic.osc

scenario top:
    path: Path                      	# A path in the map
    path.set_map("Town04")              # Map to employ
    path.path_min_driving_lanes(2)      # Path should have at least two lanes

    ego_vehicle: Model3                	# Ego car
    npc: Rubicon                        # The other car

    event start                         # Define start and end events
    event end

```

Define the behaviors:

```
    do parallel(duration: 18s):                     # Execute the following lines in parallel
        npc.drive(path) with:                       # The NPC will drive in the second lane at a constant speed
            speed(30kph)
            lane(2, at: start)                      # Lanes go left to right: [1..n]

        serial:                                     # The following commands are serially executed
            get_ahead: parallel(duration: 3s):      # Set the initial ego vehicle position between 10 and 20m behind the NPC
                ego_vehicle.drive(path) with:
                    speed(30kph)
                    lane(same_as: npc, at: start)
                    position([10m..20m], behind: npc, at: start)

            change_lane1: parallel(duration: 5s):   # Change lane
                ego_vehicle.drive(path) with:
                    lane(left_of: npc, at: end)

            acceleration: parallel(duration: 5s):   # Accelerate for 5s
                ego_vehicle.drive(path) with:
                    speed(45kph)

            change_lane2: parallel(duration: 5s):   # Come back to the initial lane
                ego_vehicle.drive(path) with:
                    lane(same_as: npc, at: end)

```

#### Composition operators

An OSC2 scenario can invoke one or more behaviors using three supported composition operators:

- `serial`: The serial (sequential) composition of scenarios.
- `parallel`: The parallel composition of scenarios.
- `one_of`: The one-of composition of scenarios (at least one of a set of scenarios must hold).

#### Modifiers

The OSC2 standard allows for different movement modifiers. This list shows those supported in CARLA:

- `speed`: set target speed, e.g. `speed(30kph)`.
- `position`: set position, e.g. `position(10m, behind: npc, at: start)`.
- `lane`: set lane, e.g. `lane(1, at: start)`.
- `acceleration`: set the vehicle acceleration, e.g. `acceleration(15kphps)`.
- `keep_lane`: maintain the same lane, e.g. `keep_lane()`.
- `change_speed`: change the current speed, `change_speed(3kph)`.
- `change_lane`: change to a different lane, e.g. `change_lane(lane_changes:[1..2], side: left)`.

#### Units

You can define custom units:

```
type velocity is SI(m: 1, s: -1)
unit mps                 of velocity is SI(m: 1, s: -1, factor: 1)
unit kmph                of velocity is SI(m: 1, s: -1, factor: 0.277777778)
unit mph                 of velocity is SI(m: 1, s: -1, factor: 0.447038889)
```

#### Expressions

Define custom expressions, data structures and functions:

```
struct speeds:
    def compute(x:velocity, y:velocity) -> velocity is expression x-y
```

## Custom vehicles

The following steps explain how to use custom vehicles in a scenario. For example, to add an ambulance to the scenario, the following steps are required.

1. Edit `srunner/osc2_stdlib/vehicle.py` and add a custom vehicle class:

```py
class Ambulance(Car):
    def __init__(self) -> None:
        super().__init__()
        self.set_model("vehicle.ambulance.ford")
```

2. Edit `srunner/scenarioconfigs/osc2_scenario_configuration.py`

```py
vehicle_type = ["Car", "Model3", "Mkz2017", "Carlacola", "Rubicon", "Ambulance"]
```

3. Edit `srunner/examples/basic.osc` and add a line of code at the end:

```
actor Ambulance
```

4. Specify the vehicle in the OSC2 file you want to run

```
ego_vehicle: Ambulance
```
