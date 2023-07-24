#include "mini_mlir/Conversion/TopToTosa/OpFold.h"

namespace mini_mlir {

struct TosaFoldDoubleReciprocal
    : public OpRewritePattern<mlir::tosa::ReciprocalOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(mlir::tosa::ReciprocalOp op1,
                                PatternRewriter &rewriter) const override {
    mlir::Value value1 = op1->getOperand(0);
    if (auto op2 = value1.getDefiningOp<mlir::tosa::ReciprocalOp>()) {
      // 找到连续的两个reciprocal操作
      mlir::Value value2 = op2->getOperand(0);

      // 用操作数替换第一个reciprocal的结果
      rewriter.replaceAllUsesWith(op2, value2);
      // 删除第一个reciprocal操作
      rewriter.eraseOp(op2);

      rewriter.replaceAllUsesWith(op1, value2);
      rewriter.eraseOp(op1);
      return success();
    }
    return failure();
  }
};

struct TosaFoldDoubleMul : public OpRewritePattern<mlir::tosa::MulOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(mlir::tosa::MulOp op1,
                                PatternRewriter &rewriter) const override {
    mlir::Value left_value1 = op1->getOperand(0);
    mlir::Value right_value1 = op1->getOperand(1);
    if (auto op2 = left_value1.getDefiningOp<mlir::tosa::MulOp>()) {
      if (auto const1 = right_value1.getDefiningOp<mlir::tosa::ConstOp>()) {
        mlir::Value left_value2 = op2->getOperand(0);
        mlir::Value right_value2 = op2->getOperand(1);
        if (auto const2 = right_value2.getDefiningOp<mlir::tosa::ConstOp>()) {
          auto vec1 = get_const_value(const1);
          auto v1 = vec1[0];
          auto vec2 = get_const_value(const2);
          auto v2 = vec2[0];
          if (std::fabs(v1 * v2 - 1) < 0.001) {
            //Remove Double Mul
            mlir::Value value2 = op2->getOperand(0);
            rewriter.replaceAllUsesWith(op2, value2);
            rewriter.eraseOp(op2);
            rewriter.replaceAllUsesWith(op1, value2);
            rewriter.eraseOp(op1);
            return success();
          }
        }
      }
    }
    return failure();
  }
};

void populateTosaOpFoldPatterns(RewritePatternSet *patterns) {
  patterns->add<TosaFoldDoubleReciprocal, TosaFoldDoubleMul>(
      patterns->getContext());
}

} // namespace mini_mlir
