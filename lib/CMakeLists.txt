add_subdirectory(Interfaces)
add_subdirectory(Support)
add_subdirectory(Dialect)
add_subdirectory(Conversion)


add_mlir_library(MiniInitAll
  InitAll.cpp
  ModuleInterpreter.cpp

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRFuncDialect
  MLIRSupport
  MLIRTosaDialect
  MLIRInferenceInterface
  MiniTop
  MiniSupport
  MiniTopToTosa
)

