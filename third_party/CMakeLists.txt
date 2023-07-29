add_subdirectory(cnpy)

install(FILES oneDNN/lib/libdnnl.so
              oneDNN/lib/libdnnl.so.3
              oneDNN/lib/libdnnl.so.3.1
        DESTINATION lib)

install(PROGRAMS llvm/bin/mlir-opt
                 llvm/bin/mlir-translate
                 llvm/bin/llc
        DESTINATION bin)
install(DIRECTORY llvm/python_packages/mlir_core/mlir DESTINATION python)
