#!/bin/bash
set -e

export CASADIPATH=/home/jgillis/programs/casadi/python_install

# Build your plugin

rm -rf build
mkdir build

pushd build
cmake ..
make
popd

# Test your plugin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/build
python test.py




