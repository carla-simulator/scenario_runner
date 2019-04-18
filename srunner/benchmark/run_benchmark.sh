#!/usr/bin/env bash

BENCHMARK_DIR=${ROOT_SCENARIO_RUNNER}/srunner/benchmark
CHALLENGE_EVALUATOR=${ROOT_SCENARIO_RUNNER}/srunner/challenge/challenge_evaluator_routes.py

TOWNS="Town01 Town02"
WEATHERS="train test"

function usage {
    echo "usage: $0 --benchmark <corl2017|carla100> --out-dir <directory> [scenario runner arguments]"
    echo "Will run challenge_evaluator_routes.py multiple times for each task in the benchmark."
    echo "Results will be saved to out dir, with any extra arguments passed to the scenario runner."
    exit 1
}

SCENARIO_RUNNER_ARGS=""

while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    --benchmark)
    BENCHMARK_SUITE="$2"
    shift
    shift
    ;;
    --out-dir)
    OUT_DIR="$2"
    shift
    shift
    ;;
    *)
    SCENARIO_RUNNER_ARGS="${SCENARIO_RUNNER_ARGS} $1"
    shift
    ;;
esac
done

if [[ $BENCHMARK_SUITE == "corl2017" ]]; then
    TASKS="straight,straight,absent one_curve,one_curve,absent \
           navigation,navigation,absent navigation_dynamic,navigation,present"
elif [[ $BENCHMARK_SUITE == "carla100" ]]; then
    TASKS="navigation_static,navigation,absent \
           navigation_light,navigation,light \
           navigation_heavy,navigation,heavy"
else
    echo "Unrecognized benchmark '${BENCHMARK_SUITE}'"
    usage
fi

if [[ $OUT_DIR == "" ]]; then
    echo "You must specify an output directory."
    usage
fi
mkdir -p ${OUT_DIR}

set -x

for town in ${TOWNS}; do
    for weather in ${WEATHERS}; do
        for task_cfg in ${TASKS}; do
            IFS=',' read task_name route_name background <<< "${task_cfg}"
            python3 ${CHALLENGE_EVALUATOR} \
                    --scenarios ${BENCHMARK_DIR}/common.json ${BENCHMARK_DIR}/${BENCHMARK_SUITE}/${weather}_weather.json \
                                ${BENCHMARK_DIR}/${BENCHMARK_SUITE}/background_${background}.json \
                    --routes=${BENCHMARK_DIR}/${BENCHMARK_SUITE}/${town}_${route_name}.xml \
                    --filename=${OUT_DIR}/${BENCHMARK_SUITE}_${town}_${weather}_${task_name}.json \
                    ${SCENARIO_RUNNER_ARGS}
        done
    done
done