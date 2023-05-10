#!/bin/bash
set -e

PROJECT_ROOT="$( cd "$(dirname "$0")" ; pwd -P )"

if [[ -z "$INSTALL_PATH" ]]; then
  echo "Please source envsetup.sh firstly."
  exit 1
fi


BUILD_FLAG="-DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS=-ggdb"


echo "BUILD_PATH: $BUILD_PATH"
echo "INSTALL_PATH: $INSTALL_PATH"

# prepare install/build dir
mkdir -p $BUILD_PATH

pushd $BUILD_PATH
cmake -G Ninja \
    $BUILD_FLAG \
    -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH \
    $PROJECT_ROOT
cmake --build . --target install mlir-doc
popd

# Clean up some files for release build
if [ "$1" = "RELEASE" ]; then
  # strip mlir tools
  pushd $INSTALL_PATH
  find ./ -name "*.so" |xargs strip
  find ./ -name "*.a" |xargs rm
  popd
fi

mv $INSTALL_PATH/python/pymlir.cpython-36m-x86_64-linux-gnu.so $INSTALL_PATH/python/pymlir.cpython-37m-x86_64-linux-gnu.so

echo "mv $INSTALL_PATH/python/pymlir.cpython-36m-x86_64-linux-gnu.so $INSTALL_PATH/python/pymlir.cpython-37m-x86_64-linux-gnu.so"
