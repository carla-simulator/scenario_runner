#!/usr/bin/env python3
"""Generate and *simulate* Scenic scenes for multiple parameter variants held in a
single YAML file.

Example usage
-------------
    python run_variations.py -s stop_sign_intersection.scenic \
                             -c params_variations.yaml \
                             --amount 3 --repeat 5

CLI Flags
~~~~~~~~~
-s / --scenario   Path to the .scenic scenario file (required)
-c / --config     YAML file with one or more parameter sets (required)
--amount          How many **different scenes** to sample per variant (default 1)
--repeat          How many times to run **the exact same scene** through the
                  simulator (default 1)

Behaviour
~~~~~~~~~
* **amount** → calls `scenario.generate()` repeatedly, producing new *random*
  scenes under the same parameter set.
* **repeat** → re‑simulates **the same scene** multiple times, giving identical
  initial conditions.
"""
import argparse
import yaml
import scenic
from srunner.scenic.models.simulator import CarlaSimulator
import random

###############################################################################
# Helpers
###############################################################################

def load_variants(path: str):
    """Load YAML file and return a list of parameter dictionaries."""
    with open(path, "r", encoding="utf-8") as f:
        return list(yaml.safe_load_all(f))

###############################################################################
# Main
###############################################################################

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate & simulate Scenic scenes from multiple parameter variants.")
    parser.add_argument("-s", "--scenario", required=True, help="Path to .scenic file")
    parser.add_argument("-c", "--config", required=True, help="YAML parameter file")
    parser.add_argument("--amount", type=int, default=1,
                        help="Number of different scenes to sample per variant")
    parser.add_argument("--repeat", type=int, default=1,
                        help="Re‑simulate the exact same scene this many times")
    args = parser.parse_args()

    random.seed(1)
    variants = load_variants(args.config)

    for vidx, params in enumerate(variants):
        print(f"=== Variant {vidx}: {params}")
        for s in range(args.amount):
            try:
                scenario = scenic.scenarioFromFile(args.scenario, params=params, mode2D=True)
                scene, _ = scenario.generate()

            except Exception as exc:
                import traceback
                print(f"[!] Failed to generate scene for variant {vidx}, sample {s}: {exc}")
                print(traceback.format_exc())
                continue

            simulator = None
            for r in range(args.repeat):
                try:
                    simulator = CarlaSimulator(params["carla_map"], params["map"])
                    sim = simulator.simulate(scene, maxSteps=1000, maxIterations=1000)
                    if sim is None:
                        print(f"    [!] Simulation returned None for variant {vidx} sample {s} run {r}")

                except Exception as exc:
                    print(f"    [!] Simulation error for variant {vidx} sample {s} run {r}: {exc}")

                finally:
                    if simulator:
                        simulator.destroy()


if __name__ == "__main__":
    main()
