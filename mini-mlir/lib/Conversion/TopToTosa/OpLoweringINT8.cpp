#include "mini_mlir/Conversion/TopToTosa/OpLoweringINT8.h"

namespace mini_mlir {

void populateTopToTosaConversionINT8Patterns(RewritePatternSet *patterns, std::map<std::string, float> calibration_map) {
  patterns->add<
      // clang-format off
        ReshapeLoweringINT8
      // clang-format on
      >(patterns->getContext(), calibration_map);
}


//===------------------------------------------------------------===//
// AddLoweringINT8
//===------------------------------------------------------------===//
// void AddLoweringINT8::Lowering(PatternRewriter &rewriter, top::AddOp op) const {
//   assert(op->getNumResults() == 1);
//   std::string left_name = 
//     op->getOperand(0).getDefiningOp()->getAttr("name").cast<StringAttr>().getValue().str();
//   std::string right_name = 
//     op->getOperand(1).getDefiningOp()->getAttr("name").cast<StringAttr>().getValue().str();
//   auto name = op->getAttr("name").cast<StringAttr>().getValue().str();


//   auto opname = op->getName();

//   if (calibration_map.find(left_name) != calibration_map.end()) {
//     auto newType = op->getResult(0).getType();
//     auto coeff = op.getCoeffAttr();
//     std::vector<Value> operands;
//     for (auto in : op->getOperands()) {
//       operands.push_back(in);
//     }
//     rewriter.replaceOpWithNewOp<mlir::tosa::AddOp>(op, newType, operands);
//   }
// }

//===------------------------------------------------------------===//
// ReshapeLoweringINT8
//===------------------------------------------------------------===//
void ReshapeLoweringINT8::Lowering(PatternRewriter &rewriter, top::ReshapeOp op) const {
  assert(op->getNumResults() == 1);
  std::string in_name = 
    op->getOperand(0).getDefiningOp()->getAttr("name").cast<StringAttr>().getValue().str();
  float threshold = calibration_map.at(in_name);
  // float threshold = 1;

  auto inType = op->getOperand(0).getType();
  std::vector<int64_t> in_shape(inType.cast<RankedTensorType>().getShape());
  int in_size = in_shape.size();
  auto outType = op->getResult(0).getType();
  std::vector<int64_t> out_shape(outType.cast<RankedTensorType>().getShape());
  int out_size = out_shape.size();

  // ConstOp inv_scale
  std::vector<float> inv_scale;
  inv_scale.push_back(127./threshold);
  std::vector<int64_t> const_shape;
  for (int i = 0; i < in_size ; i++) {
    const_shape.push_back(1);
  }
  auto const_ty = RankedTensorType::get({const_shape}, rewriter.getF32Type());
  DenseElementsAttr attr = DenseElementsAttr::get(
      const_ty, llvm::ArrayRef(inv_scale.data(), inv_scale.size()));
  auto const_inv_scale =
      rewriter.create<mlir::tosa::ConstOp>(op->getLoc(), const_ty, attr);

  // MulOp for int8
  auto mul_inv_scale_op = rewriter.create<mlir::tosa::MulOp>(op->getLoc(), inType,
      op->getOperand(0), const_inv_scale->getResult(0), rewriter.getI32IntegerAttr(0));

  // CastOp fp32->int8
  auto cast2int8_ty = RankedTensorType::get({in_shape}, rewriter.getI8Type());
  auto cast2int8_op =
      rewriter.create<mlir::tosa::CastOp>(op->getLoc(), cast2int8_ty, mul_inv_scale_op->getResult(0));
  
  // ReshapeOp
  auto reshapeType = RankedTensorType::get({out_shape}, rewriter.getI8Type());
  auto reshape_op =
      rewriter.create<mlir::tosa::ReshapeOp>(op->getLoc(), reshapeType, cast2int8_op->getResult(0), out_shape);

  // CastOp int8->fp32
  auto cast2fp32_ty = RankedTensorType::get({out_shape}, rewriter.getF32Type());
  auto cast2fp32_op =
      rewriter.create<mlir::tosa::CastOp>(op->getLoc(), cast2fp32_ty, reshape_op->getResult(0));

  // ConstOp scale
  std::vector<float> scale;
  scale.push_back(threshold/127.);
  const_shape.clear();
  for (int i = 0; i < out_size ; i++) {
    const_shape.push_back(1);
  }
  const_ty = RankedTensorType::get({const_shape}, rewriter.getF32Type());
  attr = DenseElementsAttr::get(
      const_ty, llvm::ArrayRef(scale.data(), scale.size()));
  auto const_scale =
      rewriter.create<mlir::tosa::ConstOp>(op->getLoc(), const_ty, attr);

  // MulOp for fp32
  auto mul_scale_op = rewriter.create<mlir::tosa::MulOp>(op->getLoc(), outType, 
      cast2fp32_op->getResult(0), const_scale->getResult(0), rewriter.getI32IntegerAttr(0));

  // Replace
  rewriter.replaceOp(op, mul_scale_op->getResult(0));
}

} // namespace mini_mlir
