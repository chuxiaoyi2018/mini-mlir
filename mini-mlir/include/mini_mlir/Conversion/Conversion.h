#ifndef MINI_MLIR_CONVERSION_H
#define MINI_MLIR_CONVERSION_H

#include "mini_mlir/Conversion/TopToTosa/TopLowering.h"

namespace mlir {
#define GEN_PASS_DECL
#include "mini_mlir/Conversion/Passes.h.inc"
} // namespace mlir

namespace mini_mlir {

std::unique_ptr<Pass> createConvertTopToTosa();

} // namespace mini_mlir

#endif // MINI_MLIR_CONVERSION_H
