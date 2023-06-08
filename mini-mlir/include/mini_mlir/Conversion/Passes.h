#ifndef MINI_MLIR_CONVERSION_PASSES_H
#define MINI_MLIR_CONVERSION_PASSES_H

#include "mini_mlir/Conversion/Conversion.h"

namespace mlir {

/// Generate the code for registering conversion passes.
#define GEN_PASS_REGISTRATION
#include "mini_mlir/Conversion/Passes.h.inc"

} // namespace mlir

#endif // MINI_MLIR_CONVERSION_PASSES_H