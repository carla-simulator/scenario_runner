# osc2-runner

本项目面向自动驾驶测试，以OpenScenario2.0为场景描述语言规范，设计并实现了相应的编译系统，可自动将用OpenScenario2.0描述的测试场景转换为基于scenario runner的测试场景，从而利用carla进行自动驾驶测试


## 安装


**1、安装carla** 

目前项目使用scenario runner 0.9.13进行开发，所以需对应安装Carla-0.9.13，到[Carla官网](http://carla.org/2021/11/16/release-0.9.13/)下载后解压即可。   

配置要求：

- 操作系统 Windows / Linux
- GPU 最低运行需要 6 GB / 推荐 8 GB 
- 硬盘空间 20 GB以上
- Python 3.8
- pip version 20.3 or higher


**2、安装JDK**

```
sudo apt install openjdk-17-jdk
```    
安装完成后执行命令确认
```
$ java -version
```

输出如下
```
openjdk version "17.0.5" 2022-10-18
OpenJDK Runtime Environment (build 17.0.5+8-Ubuntu-2ubuntu120.04)
OpenJDK 64-Bit Server VM (build 17.0.5+8-Ubuntu-2ubuntu120.04, mixed mode, sharing)
```

**3、安装Antlr 4.10.1**

```
sudo apt install curl
curl -O https://www.antlr.org/download/antlr-4.10.1-complete.jar
```

如果下载速度过慢，可尝试使用如下地址手动安装
[Antlr tools下载地址](https://www.antlr.org/download/antlr-4.10.1-complete.jar)

把下载的jar放到local/lib下
```
$ sudo cp antlr-4.10.1-complete.jar /usr/local/lib/
```

以下三步用于配置环境变量和创建别名，以便可以方便地从命令行使用antlr4

打开配置文件
```
$ sudo gedit ~/.bashrc
```
在最后添加如下内容
```
export CLASSPATH=".:/usr/local/lib/antlr-4.10.1-complete.jar:$CLASSPATH"
alias antlr4='java -jar /usr/local/lib/antlr-4.10.1-complete.jar'
alias grun='java org.antlr.v4.gui.TestRig'
```
更新资源使配置生效
```
source ~/.bashrc
```

在终端中输入antlr4验证
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

**4、安装antlr4运行时**
```
pip install antlr4-python3-runtime==4.10
```

**5、安装python依赖**

在主目录下执行
```
pip install -r requirements.txt
```

**6.安装graphviz**

安装graphviz工具
```
sudo apt-get install graphviz
```
安装python包
```
pip install graphviz
```


**7、carla配置**
    
(1) 下载[carla安装包](https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.13.tar.gz)


(2) 将carla的安装包到解压到目录下，例如：/home/dut-aiid/CARLA_0.9.13

在ubuntu系统上，carla环境变量配置如下：
```bash
添加
export CARLA_ROOT=/home/dut-aiid/CARLA_0.9.13
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg:${CARLA_ROOT}/PythonAPI/carla/agents:${CARLA_ROOT}/PythonAPI/carla/agents/navigation:${CARLA_ROOT}/PythonAPI/carla:${CARLA_ROOT}/PythonAPI/examples:${CARLA_ROOT}/PythonAPI
到 ~/.bashrc
# 执行source生效
source ~/.bashrc

```

在windows系统上，采用在代码中动态包含carla的python包的方式使用python API。在项目的scenario_runner.py和manual_control.py开头加入如下代码，从而确保可以正确导入carla包。其中carla的安装路径需根据安装情况调整。
```
try:
    sys.path.append(glob.glob('D:/CARLA_0.9.13/WindowsNoEditor/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])

    sys.path.insert(0, 'D:/CARLA_0.9.13/WindowsNoEditor/PythonAPI/carla')
    sys.path.insert(1, 'D:/CARLA_0.9.13/WindowsNoEditor/PythonAPI/')
except IndexError:
    pass
```

(3) carla启动及测试

```bash
# 安装包方式启动
cd /home/dut-aiid/CARLA_0.9.13
bash CarlaUE4.sh

# Carla将在地图中生40辆自动驾驶的npc车辆
cd PythonAPI/examples
python generate_traffic.py -n 40
```

## Quickstart

**1、运行carla**

在carla的安装目录下，双击运行CarlaUE4.exe。

**2、启动manual_control**

执行如下代码，启动manual_control，其中‘-a’选项表示自动仿真模式，不加则可手动控制。

```
python manual_control.py -a 
```

**3、加载场景**
    
运行scenario_runner，加载需要仿真的场景，如cut_in_and_slow_right.osc，代码如下：

```
python scenario_runner.py --sync  --osc2 srunner/examples/cut_in_and_slow_right.osc --reloadWorld 
```
carla-0.9.13的load_world在加载town时会出现断言错误，
```
Assertion failed: px != 0, file c:\jenkins\workspace\carla_0.9.13\Build\boost-1.72.0-install\include\boost/smart_ptr/shared_ptr.hpp, line 734
```
但town是成功加载的，所以需再次执行上述命令。
同时，执行上述命令也可能出现spawn_points失败的情况，同样重复执行上述命令即可！

```
RuntimeError: no valid position to spawn npc car
```
运行完上述代码，即可在pygame窗口看到仿真车辆及其运行过程。 

*注意：在windows上，步骤2和3不可调换顺序，否则pygame窗口可能一直是黑框，处于等待状态！*

**4、生成与场景相关的trace**

加载场景完成后，在osc2-runner目录下的trace.json文件即为相应的轨迹文件


**5、对trace进行法规断言**
```
mv trace.json Law_Judgement/example_trace/
```
后续操作步骤见Law_Judgement/readme.md

## 生成lawbreaker所需trace

**1、 闯红灯场景测试**
```
python scenario_runner.py --sync  --osc2 srunner/examples/force_over_signal.osc --reloadWorld 
```

**2、 超速场景测试**
```
python scenario_runner.py --sync  --osc2 srunner/examples/overspeed.osc --reloadWorld 
```

**3、 碰撞场景测试**
```
python scenario_runner.py --sync  --osc2 srunner/examples/cut_in_and_slow_right.osc --reloadWorld 
```

运行后会在主目录下生成trace.json文件

*注意：在场景初始化阶段，无法获取到路径规划，该情况发生时，trace中的direction为默认值0（方向为直行）。在成功找到初始位置，获取到路径规划并开始运动，direction可以被正常检测到*


## FAQ

### 问题1

缺少 package libxerces-c-3.2.so 而报错
```
sudo apt-get install libxerces-c-dev
```

### 问题2

使用CARLA_0.9.13时，若因python版本问题，不能直接pip install carla
```
cd ~/CARLA_0.9.13/PythonAPI/carla/dist/
unzip carla-0.9.13-py3.7-linux-x86_64.egg -d carla-0.9.13-py3.7-linux-x86_64
cd carla-0.9.13-py3.7-linux-x86_64
 
接下来打开文档编辑器，新建setup.py文件, 将以下内容复制进去。
from distutils.core import setup
setup(name='carla',
    version='0.9.13',
    py_modules=['carla'],
    )

然后通过pip安装到python3当中，从此可以直接import carla了。
pip3 install -e ~/carla/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64

```
### 问题3

运行脚本时出现 RuntimeError: trying to create rpc server for traffic manager; but the system failed to create because of bind error
```
【分析原因】
 因为在脚本里默认使用8000端口 car_actor.set_autopilot(enabled=True)

【解决办法】
 1 查询端口占用情况 netstat -tnlp | grep :8000
 2 kill对应端口 kill -9 pid
 例如:我的pid通过第一步查询到是 12332， 然后执行kill -9 12332

 最简单的解决办法 pkill -9 python
 ```