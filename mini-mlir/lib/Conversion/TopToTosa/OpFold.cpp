#include "mini_mlir/Conversion/TopToTosa/OpFold.h"

namespace mini_mlir {



struct TosaFoldDoubleReciprocal : public OpRewritePattern<mlir::tosa::ReciprocalOp> {

  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::tosa::ReciprocalOp recip,
                                PatternRewriter &rewriter) const override {

    return success();
  }
};


void populateTosaFoldDoubleReciprocalPatterns(RewritePatternSet *patterns) {
  patterns->add<TosaFoldDoubleReciprocal>(patterns->getContext());
}

} // namespace mini_mlir
