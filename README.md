# osc2-runner

This project is aimed at autonomous driving simulation, 
using OpenScenario2.0 as the scene description language specification, 
designing and implementing a corresponding compilation system, 
which can automatically convert the test scenario described in OpenScenario2.0 
into a scenario runner-based test scenario, 
thereby using Carla for autonomous driving testing.

## Install

**1. Install Carla**

Currently, the project is being developed using scenario runner 0.9.13, so it is necessary to install Carla-0.9.13 accordingly.

```bash
wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.13.tar.gz
tar -zxvf CARLA_0.9.13.tar.gz
```

Add at the end of "~/.bashrc".

```bash
export CARLA_ROOT=/home/xxx/CARLA_0.9.13
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg:${CARLA_ROOT}/PythonAPI/carla/agents:${CARLA_ROOT}/PythonAPI/carla/agents/navigation:${CARLA_ROOT}/PythonAPI/carla:${CARLA_ROOT}/PythonAPI/examples:${CARLA_ROOT}/PythonAPI
```

**2. Install JDK**
```
sudo apt install openjdk-17-jdk
```
After installation is complete, execute the command to confirm.
```
$ java -version
```
The output is as follows.
```
openjdk version "17.0.5" 2022-10-18
OpenJDK Runtime Environment (build 17.0.5+8-Ubuntu-2ubuntu120.04)
OpenJDK 64-Bit Server VM (build 17.0.5+8-Ubuntu-2ubuntu120.04, mixed mode, sharing)
```

**3. Install Antlr**
```
sudo apt install curl
curl -O https://www.antlr.org/download/antlr-4.10.1-complete.jar
```
```
$ sudo cp antlr-4.10.1-complete.jar /usr/local/lib/
```
The following three steps are used to configure environment variables and create aliases so that antlr4 can be easily used from the command line.
```
$ sudo gedit ~/.bashrc
```
Add the following content at the end.
```
export CLASSPATH=".:/usr/local/lib/antlr-4.10.1-complete.jar:$CLASSPATH"
alias antlr4='java -jar /usr/local/lib/antlr-4.10.1-complete.jar'
alias grun='java org.antlr.v4.gui.TestRig'
```
Execute the command to make the changes take effect.
```
source ~/.bashrc
```
Enter "antlr4" in the terminal to verify.
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

**4. Install python dependencies**

Execute in the project directory.
```
pip install -r requirements.txt
```

**5. Install graphviz**
```
sudo apt-get install graphviz
```

## QuickStart

**1. Run carla**

Execute in the Carla directory.
```bash
./CarlaUE4.sh &
```

**2. Start manual_control**
```
python manual_control.py -a 
```

**3. Run a scenario**
```
python scenario_runner.py --sync  --osc2 srunner/examples/cut_in_and_slow_right.osc --reloadWorld 
```

