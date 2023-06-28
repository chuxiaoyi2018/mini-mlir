
cd ../mini-mlir
mkdir third_party
cd third_party

# llvm
git clone git@github.com:llvm/llvm-project.git
cd llvm-project && git checkout c39dcf54bb0494f319cdc712d24607433bd7f30a && cd ..
mkdir -vp llvm-project/build
cd llvm-project/build
cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS="mlir" \
   -DLLVM_TARGETS_TO_BUILD="" \
   -DLLVM_ENABLE_RTTI=ON \
   -DLLVM_ENABLE_EH=ON \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DLLVM_INCLUDE_TESTS=OFF \
   -DMLIR_INCLUDE_TESTS=OFF \
   -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
   -DCMAKE_BUILD_TYPE=Release \
   -DCMAKE_INSTALL_PREFIX=llvm
cmake --build . --target install
cd ../..
#cp -r llvm-project/build/install temp
#rm -rf llvm-project/
#mv temp llvm

# oneDNN
#git clone https://github.com/oneapi-src/oneDNN.git
#cd oneDNN && git checkout 48d956c10ec6cacf338e6a812c1ae1e3087b920c && cd ..
#mkdir -p oneDNN/build
#pushd oneDNN/build
#cmake .. \
#   -DDNNL_CPU_RUNTIME=OMP \
#   -DCMAKE_INSTALL_PREFIX=install
#cmake --build . --target install
#popd
#cp -r oneDNN/build/install temp
#rm -rf oneDNN
#mv temp oneDNN

# cnpy
#git clone https://github.com/rogersce/cnpy.git
#cd cnpy && git checkout 4e8810b1a8637695171ed346ce68f6984e585ef4
