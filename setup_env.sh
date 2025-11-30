#!/bin/bash

# 激活虚拟环境
source /home/zhang/carla_env/bin/activate

# 设置CARLA根目录，请根据你的安装路径修改
CARLA_ROOT=/home/zhang/下载/CARLA_0.9.16

# 设置Scenario Runner根目录
SCENARIO_RUNNER_ROOT=/home/zhang/scenario_runner

# 设置 EGG 文件路径。注意：Scenario Runner 通常需要的是 .egg 文件，而不是 .whl 文件。
# 使用通配符 * 来匹配版本和 Python 编译版本。
EGG_FILE=$(ls $CARLA_ROOT/PythonAPI/carla/dist/carla-*-py3.10-*.egg 2>/dev/null)

# 如果找不到 .egg，尝试查找 .whl (某些安装可能只有 .whl)
if [ -z "$EGG_FILE" ]; then
    EGG_FILE=$(ls $CARLA_ROOT/PythonAPI/carla/dist/*.whl 2>/dev/null)
fi

# 检查是否找到文件
if [ -z "$EGG_FILE" ]; then
    echo "错误: 无法在 $CARLA_ROOT/PythonAPI/carla/dist/ 中找到 CARLA Python 包 (.egg 或 .whl)。请检查 CARLA 安装版本和 Python 版本是否匹配。"
fi

# 设置PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$EGG_FILE
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/agents
export PYTHONPATH=$PYTHONPATH:$SCENARIO_RUNNER_ROOT

