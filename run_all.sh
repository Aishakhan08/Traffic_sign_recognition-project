#!/usr/bin/env bash
# Full pipeline on Linux/macOS: install, (re)generate data, train, evaluate
set -e
python3 -m pip install -r requirements.txt
[ -d dataset ] || python3 generate_dataset.py
python3 train.py
python3 evaluate.py
