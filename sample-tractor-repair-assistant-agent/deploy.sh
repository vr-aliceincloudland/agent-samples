#!/bin/bash
python3.11 -m venv .venv
source .venv/bin/activate
.venv/bin/python3 -m pip install --upgrade pip
pip install -r ./requirements.txt