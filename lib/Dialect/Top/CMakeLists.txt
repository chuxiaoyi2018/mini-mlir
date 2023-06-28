file(GLOB _sources IR/*.cpp Transforms/*.cpp)

add_mlir_library(MiniTop
  ${_sources}

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/mini_mlir/Dialect/Top

  DEPENDS
  MLIRTopPassIncGen
  MLIRTopOpsIncGen
  MLIRInferenceInterfaceIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRParser
  MLIRFuncDialect
  MLIRQuantDialect
  MLIRRewrite
  MLIRTransforms
  MLIRInferenceInterface
  MiniSupport
)

