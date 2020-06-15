#!/bin/bash

# Fail on any error.
set -e

# Install project prerequisites
pip install -r requirements.txt

make pylint
make test

