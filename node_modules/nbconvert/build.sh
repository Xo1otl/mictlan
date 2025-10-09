#!/bin/sh

python -m virtualenv .venv
source .venv/bin/activate

python -m pip install -r requirements.txt

python -m pex nbconvert~=$npm_package_version -m nbconvert -o nbconvert.pex
