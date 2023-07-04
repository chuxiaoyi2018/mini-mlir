#!/bin/bash
chmod -R +777 .
ln -s /usr/include/python3.7m /usr/include/python3.6m

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

export PROJECT_ROOT=$DIR
export BUILD_PATH=${BUILD_PATH:-$PROJECT_ROOT/build}
export INSTALL_PATH=${INSTALL_PATH:-$PROJECT_ROOT/install}

# add python file to path
export PATH=$INSTALL_PATH/bin:$PATH
export PATH=$PROJECT_ROOT/python/tools:$PATH
export PATH=$PROJECT_ROOT/third_party/llvm/bin:$PATH

# dynamic link lib
export LD_LIBRARY_PATH=$INSTALL_PATH/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$INSTALL_PATH/lib:$PROJECT_ROOT/capi/lib:$LD_LIBRARY_PATH

# add module for import
export PYTHONPATH=$INSTALL_PATH/python:$PYTHONPATH # pymlir
export PYTHONPATH=$PROJECT_ROOT/python:$PYTHONPATH
export PYTHONPATH=$PROJECT_ROOT/third_party/llvm/python_packages/mlir_core:$PYTHONPATH


#export LD_LIBRARY_PATH=$INSTALL_PATH/lib:$LD_LIBRARY_PATH
