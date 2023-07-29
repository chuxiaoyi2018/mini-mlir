#pragma once

#include "mini_mlir/Conversion/TopToTosa/TopLowering.h"

namespace mini_mlir {

void populateTosaOpFoldPatterns(RewritePatternSet *patterns);

//===------------------------------------------------------------===//
// GetConstValue
//===------------------------------------------------------------===//
static std::vector<float> get_const_value(mlir::tosa::ConstOp const_op) {
  auto denseAttr = const_op->getAttr("value").dyn_cast<DenseFPElementsAttr>();
  auto vec = llvm::to_vector(llvm::map_range(
      denseAttr.getValues<APFloat>(),
      [&](APFloat value) -> float { return value.convertToFloat(); }));
  return std::vector<float>(vec.begin(), vec.end());
}

} // namespace mini_mlir
