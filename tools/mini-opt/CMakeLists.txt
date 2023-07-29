set(LLVM_LINK_COMPONENTS
  Core
  Support
  )

set(LIBS
  MLIRFuncDialect
  MLIRTosaDialect
  MLIROptLib
  MiniInitAll
  )

add_llvm_executable(mini-opt
  mini-opt.cpp

  DEPENDS
  ${LIBS}
  )

target_link_libraries(mini-opt PRIVATE ${LIBS})
llvm_update_compile_flags(mini-opt)

mlir_check_all_link_libraries(mini-opt)

install(TARGETS mini-opt DESTINATION bin)
