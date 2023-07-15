#include "mini_mlir/Conversion/TopToTosa/OpLoweringINT8.h"

namespace mini_mlir {

void populateTopToTosaConversionINT8Patterns(RewritePatternSet *patterns) {
  patterns->add<
      // clang-format off
        AddLoweringINT8
      // clang-format on
      >(patterns->getContext());
}


//===------------------------------------------------------------===//
// AddLowering
//===------------------------------------------------------------===//
void AddLoweringINT8::Lowering(PatternRewriter &rewriter, top::AddOp op) const {
  assert(op->getNumResults() == 1);
  auto newType = op->getResult(0).getType();
  auto coeff = op.getCoeffAttr();
  std::vector<Value> operands;
  for (auto in : op->getOperands()) {
    operands.push_back(in);
  }
  rewriter.replaceOpWithNewOp<mlir::tosa::AddOp>(op, newType, operands);
}

} // namespace mini_mlir
