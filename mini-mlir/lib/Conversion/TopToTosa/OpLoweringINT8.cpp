#include "mini_mlir/Conversion/TopToTosa/OpLoweringINT8.h"

namespace mini_mlir {

void populateTopToTosaConversionINT8Patterns(RewritePatternSet *patterns, std::map<std::string, float> calibration_map) {
  patterns->add<
      // clang-format off
        AddLoweringINT8,
        ReshapeLoweringINT8
      // clang-format on
      >(patterns->getContext(), calibration_map);
}

//===------------------------------------------------------------===//
// ReshapeLoweringINT8
//===------------------------------------------------------------===//
void ReshapeLoweringINT8::Lowering(PatternRewriter &rewriter, top::ReshapeOp op) const {
  assert(op->getNumResults() == 1);
  std::string in_name = 
    op->getOperand(0).getDefiningOp()->getAttr("name").cast<StringAttr>().getValue().str();
  float threshold = calibration_map.at(in_name);

  auto inType = op->getOperand(0).getType();
  std::vector<int64_t> in_shape(inType.cast<RankedTensorType>().getShape());
  auto outType = op->getResult(0).getType();
  std::vector<int64_t> out_shape(outType.cast<RankedTensorType>().getShape());
  Location loc = op->getLoc();
  
  // quantize
  auto cast2int8_op =  lowering_quantize(rewriter, op->getOperand(0), inType, loc, in_shape, threshold);

  // ReshapeOp
  auto reshapeType = RankedTensorType::get({out_shape}, rewriter.getI8Type());
  auto reshape_op =
      rewriter.create<mlir::tosa::ReshapeOp>(op->getLoc(), reshapeType, cast2int8_op->getResult(0), out_shape);
  
  // dequantize
  auto mul_scale_op = lowering_dequantize(rewriter, reshape_op->getResult(0), outType, loc, out_shape, threshold);
  // Replace
  rewriter.replaceOp(op, mul_scale_op->getResult(0));
}

//===------------------------------------------------------------===//
// AddLoweringINT8
//===------------------------------------------------------------===//
void AddLoweringINT8::Lowering(PatternRewriter &rewriter, top::AddOp op) const {
  assert(op->getNumResults() == 1);

  auto right_value = op->getOperand(1);
  float percentile;
  if (auto weight_op = dyn_cast<top::WeightOp>(right_value.getDefiningOp())) {
    percentile = weight_threshold(weight_op);
  }

  std::string left_name = 
    op->getOperand(0).getDefiningOp()->getAttr("name").cast<StringAttr>().getValue().str();
  std::string right_name = 
    op->getOperand(0).getDefiningOp()->getAttr("name").cast<StringAttr>().getValue().str();

  
  float threshold = calibration_map.at(left_name);

  auto inType = op->getOperand(0).getType();
  std::vector<int64_t> in_shape(inType.cast<RankedTensorType>().getShape());
  auto outType = op->getResult(0).getType();
  std::vector<int64_t> out_shape(outType.cast<RankedTensorType>().getShape());
  Location loc = op->getLoc();
  
  // quantize
  auto cast2int8_op =  lowering_quantize(rewriter, op->getOperand(0), inType, loc, in_shape, threshold);

  // ReshapeOp
  auto reshapeType = RankedTensorType::get({out_shape}, rewriter.getI8Type());
  auto reshape_op =
      rewriter.create<mlir::tosa::ReshapeOp>(op->getLoc(), reshapeType, cast2int8_op->getResult(0), out_shape);
  
  // dequantize
  auto mul_scale_op = lowering_dequantize(rewriter, reshape_op->getResult(0), outType, loc, out_shape, threshold);
  // Replace
  rewriter.replaceOp(op, mul_scale_op->getResult(0));
}

} // namespace mini_mlir
