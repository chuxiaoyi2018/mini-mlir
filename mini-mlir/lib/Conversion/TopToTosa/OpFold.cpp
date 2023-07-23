#include "mini_mlir/Conversion/TopToTosa/OpFold.h"

namespace mini_mlir {



struct TosaFoldDoubleReciprocal : public OpRewritePattern<mlir::tosa::ReciprocalOp> {

  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::tosa::ReciprocalOp recip_op1,
                                PatternRewriter &rewriter) const override {
    mlir::Value value1 = recip_op1->getOperand(0);
    if (auto recip_op2 = value1.getDefiningOp<mlir::tosa::ReciprocalOp>()) {
      // 找到连续的两个reciprocal操作
      mlir::Value value2 = recip_op2->getOperand(0);

      // 用操作数替换第一个reciprocal的结果
      rewriter.replaceAllUsesWith(recip_op2, value2);
      // 删除第一个reciprocal操作
      rewriter.eraseOp(recip_op2);

      rewriter.replaceAllUsesWith(recip_op1, value2);
      rewriter.eraseOp(recip_op1);
      return success();
    }
    return failure();
  }
};


void populateTosaFoldDoubleReciprocalPatterns(RewritePatternSet *patterns) {
  patterns->add<TosaFoldDoubleReciprocal>(patterns->getContext());
}

} // namespace mini_mlir
