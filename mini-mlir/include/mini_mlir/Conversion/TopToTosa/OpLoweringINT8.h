#pragma once

#include "mini_mlir/Conversion/TopToTosa/TopLowering.h"

namespace mini_mlir {

void populateTopToTosaConversionINT8Patterns(RewritePatternSet *patterns);

#define OpLoweringINT8(OP)                                                         \
  struct OP##LoweringINT8 : public TopLoweringToTosa<top::OP##Op> {                \
    OP##LoweringINT8(MLIRContext *ctx) : TopLoweringToTosa<top::OP##Op>(ctx) {}    \
    void Lowering(PatternRewriter &rewriter, top::OP##Op op) const override;   \
  };
// clang-format off
OpLoweringINT8(Add)
// clang-format on
} // namespace mini_mlir
