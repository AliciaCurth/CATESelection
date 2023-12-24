#!/bin/bash

set -e

python run_experiments.py --setup "A"

python run_experiments.py --setup "B" --file_name "B"

python run_experiments.py --setup "C" --file_name "C"

python run_experiments.py --setup "D" --file_name "D"
