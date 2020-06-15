#!/bin/bash

# Display commands being run.
set -x

# Check Ubuntu version
lsb_release -a  # Should be 16.04

echo 'Update package manager'

# Add necessary key/s
# nvidia.github.io
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 6ED91CA3AC1160CD

sudo apt update
sudo apt install software-properties-common -y
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update

echo 'Install Python 3.7'
sudo apt install -y python3.7
# TODO(cezequiel): Check if python3.5 install is still necessary.
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.5 1
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 2
python3 -V   # should be 3.7.x

echo 'Update Python packages'
sudo apt install -y python3.7-dev
sudo apt install -y python3.7-gdbm
sudo apt install -y python-pip
sudo pip install --upgrade pip

echo 'Setup Python virtual environment'
export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python
sudo pip install virtualenvwrapper
export WORKON_HOME=$HOME/envs
source /usr/local/bin/virtualenvwrapper.sh
mkvirtualenv kokoro --python=/usr/bin/python3

echo $(python -V)  # should be Python 3

# Display commands being run.
# WARNING: please only enable 'set -x' if necessary for debugging, and be very
#  careful if you handle credentials (e.g. from Keystore) with 'set -x':
#  statements like "export VAR=$(cat /tmp/keystore/credentials)" will result in
#  the credentials being printed in build logs.
#  Additionally, recursive invocation with credentials as command-line
#  parameters, will print the full command, with credentials, in the build logs.
# set -x

# Code under repo is checked out to ${KOKORO_ARTIFACTS_DIR}/git.
# The final directory name in this path is determined by the scm name specified
# in the job configuration.
cd ${KOKORO_ARTIFACTS_DIR}/git/tfrutil

./build.sh
