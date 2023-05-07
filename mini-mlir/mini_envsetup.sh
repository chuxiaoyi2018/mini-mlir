#!/bin/bash
chmod -R +777 .
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

export PROJECT_ROOT=$DIR
export BUILD_PATH=${BUILD_PATH:-$PROJECT_ROOT/build}
export INSTALL_PATH=${INSTALL_PATH:-$PROJECT_ROOT/install}

# add python file to path
export PATH=$INSTALL_PATH/bin:$PATH
export PATH=$PROJECT_ROOT/python/tools:$PATH

# add module for import
export PYTHONPATH=$PROJECT_ROOT/python:$PYTHONPATH
export PYTHONPATH=$PROJECT_ROOT/third_party/llvm/python_packages/mlir_core:$PYTHONPATH

#export LD_LIBRARY_PATH=$INSTALL_PATH/lib:$LD_LIBRARY_PATH
