# osc2-runner

We use OpenScenario 2.0 as the scenario description language specification to design and implement the corresponding compilation system, which can automatically convert the test scenario described with OpenScenario 2.0 into a test scenario based on scenario runner, so as to use carla for autonomous driving testing.

## Installation

**1. Install JDK**

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

**3、Install Antlr 4.10.1**

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

**4、Install antlr4 runtime**
```
pip install antlr4-python3-runtime==4.10
```

**5、Install python dependency**
```
pip install -r requirements.txt
```

**6.Install graphviz**

```
sudo apt-get install graphviz
```


**7、Configurate carla**
    
(1) Download [carla release](https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.13.tar.gz)


(2) Extract the carla installation package to a directory.

On Ubuntu systems, the Carla environment variable is configured as follows:
```bash
export CARLA_ROOT=/home/dut-aiid/CARLA_0.9.13
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg:${CARLA_ROOT}/PythonAPI/carla/agents:${CARLA_ROOT}/PythonAPI/carla/agents/navigation:${CARLA_ROOT}/PythonAPI/carla:${CARLA_ROOT}/PythonAPI/examples:${CARLA_ROOT}/PythonAPI
```

## Quickstart

**1、Run carla**

```bash
cd /home/xxx/CARLA_0.9.13
./CarlaUE4.sh
```

**2、Start manual_control**

```
python manual_control.py -a --rolename=ego_vehicle
```

**3、Run a OpenSCENARIO 2.0 scenario**
```
python scenario_runner.py --sync  --openscenario2 srunner/examples/cut_in_and_slow_right.osc --reloadWorld 
```