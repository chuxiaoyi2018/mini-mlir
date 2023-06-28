#include "mini_mlir/Conversion/TopToTosa/OpLowering.h"

namespace mini_mlir {

void populateTopToTosaConversionPatterns(RewritePatternSet *patterns) {
  patterns->add<
      // clang-format off
        InputLowering,
        AddLowering
      // clang-format on
      >(patterns->getContext());
}

//===------------------------------------------------------------===//
// InputLowering
//===------------------------------------------------------------===//
void InputLowering::Lowering(PatternRewriter &rewriter, top::InputOp op) const {
  assert(op->getNumResults() == 1);
  auto outType = change_dataformat(op->getResult(0).getType());
  std::vector<Value> operands;
  operands.push_back(op->getOperand(0));
  std::vector<int32_t> perms = {0, 2, 3, 1};
  auto const_ty = RankedTensorType::get({4}, rewriter.getI32Type());
  DenseElementsAttr attr = DenseElementsAttr::get(
      const_ty, llvm::ArrayRef(perms.data(), perms.size()));
  auto constop =
      rewriter.create<mlir::tosa::ConstOp>(op->getLoc(), const_ty, attr);
  operands.push_back(constop->getResult(0));
  rewriter.replaceOpWithNewOp<mlir::tosa::TransposeOp>(op, outType, operands);
}

//===------------------------------------------------------------===//
// AddLowering
//===------------------------------------------------------------===//
void AddLowering::Lowering(PatternRewriter &rewriter, top::AddOp op) const {
  assert(op->getNumResults() == 1);
  auto newType = change_dataformat(op->getResult(0).getType());
  std::vector<Value> operands;
  for (auto in : op->getOperands()) {
    operands.push_back(in);
    /*
    // if(auto InOp =
    //          dyn_cast<top::InputOp>(in.getDefiningOp())){
    if (isa<top::InputOp>(in.getDefiningOp())) {
      auto InOp = in.getDefiningOp<top::InputOp>();
      operands.push_back(InOp->getOperand(0));
    } else
      operands.push_back(in);
    */
  }
  rewriter.replaceOpWithNewOp<mlir::tosa::AddOp>(op, newType, operands);
}

} // namespace mini_mlir
