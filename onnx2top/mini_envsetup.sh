#!/bin/bash
chmod -R +777 .
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

export PROJECT_ROOT=$DIR

# add python file to path
export PATH=$PROJECT_ROOT/python/tools:$PATH

# add module for import
export PYTHONPATH=$PROJECT_ROOT/python:$PYTHONPATH
export PYTHONPATH=$PROJECT_ROOT/third_party/llvm/python_packages/mlir_core:$PYTHONPATH
