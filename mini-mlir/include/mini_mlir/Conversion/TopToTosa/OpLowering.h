#pragma once

#include "mini_mlir/Conversion/TopToTosa/TopLowering.h"

namespace mini_mlir {

void populateTopToTosaConversionPatterns(RewritePatternSet *patterns);

#define OpLowering(OP)                                                         \
  struct OP##Lowering : public TopLoweringToTosa<top::OP##Op> {                \
    OP##Lowering(MLIRContext *ctx) : TopLoweringToTosa<top::OP##Op>(ctx) {}    \
    void Lowering(PatternRewriter &rewriter, top::OP##Op op) const override;   \
  };
// clang-format off
OpLowering(Input)
OpLowering(Add)
OpLowering(Softmax)
OpLowering(Conv)
OpLowering(Reshape)
OpLowering(Permute)
OpLowering(Concat)
OpLowering(ReduceMean)
OpLowering(Sub)
OpLowering(Mul)
OpLowering(Div)
OpLowering(Sqrt)
// clang-format on
} // namespace mini_mlir
