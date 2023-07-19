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

  Location loc = op->getLoc();
  auto inType = op->getOperand(0).getType();
  auto outType = op->getResult(0).getType();
  std::vector<int64_t> out_shape(outType.cast<RankedTensorType>().getShape());
  
  // quantize
  auto cast2int8_op =  lowering_quantize(rewriter, op->getOperand(0), inType, rewriter.getI8Type(), loc, threshold);

  // ReshapeOp
  auto actionType = RankedTensorType::get({out_shape}, rewriter.getI8Type());
  auto action_op =
      rewriter.create<mlir::tosa::ReshapeOp>(op->getLoc(), actionType, cast2int8_op->getResult(0), out_shape);
  
  // dequantize
  float scale_value = threshold/127.;
  auto mul_scale_op = lowering_dequantize(rewriter, action_op->getResult(0), outType, loc, scale_value);

  // Replace
  rewriter.replaceOp(op, mul_scale_op->getResult(0));
}

//===------------------------------------------------------------===//
// AddLoweringINT8
//===------------------------------------------------------------===//
void AddLoweringINT8::Lowering(PatternRewriter &rewriter, top::AddOp op) const {
  assert(op->getNumResults() == 1);

  Location loc = op->getLoc();
  auto inType = op->getOperand(0).getType();
  auto outType = op->getResult(0).getType();
  std::vector<int64_t> out_shape(outType.cast<RankedTensorType>().getShape());

  std::string left_name = 
    op->getOperand(0).getDefiningOp()->getAttr("name").cast<StringAttr>().getValue().str();
  if (calibration_map.find(left_name) == calibration_map.end()) {return;}
  float left_threshold = calibration_map.at(left_name);

  // right == top::WeightOp
  mlir::Value right_value = op->getOperand(1);
  std::string right_op_name = op->getOperand(1).getDefiningOp()->getName().getStringRef().str();
  if (right_op_name == "top.Weight") {
    auto weight_op = dyn_cast<top::WeightOp>(right_value.getDefiningOp());
    // quantize left op
    auto cast2int32_op =  lowering_quantize(rewriter, op->getOperand(0), inType, rewriter.getI32Type(), loc, left_threshold);
    
    // quantize weight op
    std::vector<int64_t> in_shape(op->getOperand(1).getType().cast<RankedTensorType>().getShape());
    auto const_op = lowering_weight_int32(rewriter, weight_op, rewriter.getI32Type(), loc, left_threshold, in_shape);

    // AddOp
    auto actionType = RankedTensorType::get({out_shape}, rewriter.getI32Type());
    auto action_op =
        rewriter.create<mlir::tosa::AddOp>(loc, actionType, cast2int32_op->getResult(0), const_op->getResult(0));
    
    // dequantize
    float scale_value = left_threshold/127.f;
    auto mul_scale_op = lowering_dequantize(rewriter, action_op->getResult(0), outType, loc, scale_value);

    // Replace
    rewriter.replaceOp(op, mul_scale_op->getResult(0));
  } else {
    // quantize left op
    auto cast2int32_op =  lowering_quantize(rewriter, op->getOperand(0), inType, rewriter.getI32Type(), loc, left_threshold);

    // quantize right op
    auto right_op =  lowering_quantize(rewriter, op->getOperand(1), inType, rewriter.getI32Type(), loc, left_threshold);

    // AddOp
    auto actionType = RankedTensorType::get({out_shape}, rewriter.getI32Type());
    auto action_op =
        rewriter.create<mlir::tosa::AddOp>(loc, actionType, cast2int32_op->getResult(0), right_op->getResult(0));
    
    // dequantize
    float scale_value = left_threshold/127.f;
    auto mul_scale_op = lowering_dequantize(rewriter, action_op->getResult(0), outType, loc, scale_value);

    // Replace
    rewriter.replaceOp(op, mul_scale_op->getResult(0));
  }
}

//===------------------------------------------------------------===//
// MatMulLoweringINT8
//===------------------------------------------------------------===//
// void MatMulLoweringINT8::Lowering(PatternRewriter &rewriter, top::MatMulOp op) const {
//   assert(op->getNumResults() == 1);

//   Location loc = op->getLoc();
//   auto inType = op->getOperand(0).getType();
//   auto outType = op->getResult(0).getType();
//   std::vector<int64_t> out_shape(outType.cast<RankedTensorType>().getShape());

//   std::string left_name = 
//     op->getOperand(0).getDefiningOp()->getAttr("name").cast<StringAttr>().getValue().str();
//   if (calibration_map.find(left_name) == calibration_map.end()) {return;}
//   float left_threshold = calibration_map.at(left_name);

//   // right == top::WeightOp
//   mlir::Value right_value = op->getOperand(1);
//   if (auto weight_op = dyn_cast<top::WeightOp>(right_value.getDefiningOp())) {
//     // quantize left op
//     auto cast2int8_op =  lowering_quantize(rewriter, op->getOperand(0), inType, rewriter.getI8Type(), loc, left_threshold);
    
//     // quantize weight op
//     std::vector<int64_t> in_shape(op->getOperand(1).getType().cast<RankedTensorType>().getShape());
//     float weight_threshold = get_weight_threshold(rewriter, weight_op);
//     auto const_op = lowering_weight_int8(rewriter, weight_op, rewriter.getI8Type(), loc, weight_threshold, in_shape);

//     // MatMulOp
//     auto actionType = RankedTensorType::get({out_shape}, rewriter.getI32Type());
//     auto action_op =
//         rewriter.create<mlir::tosa::MatMulOp>(loc, actionType, cast2int8_op->getResult(0), const_op->getResult(0));
    
//     // dequantize
//     float scale_value = left_threshold/127.f * weight_threshold/127.f;
//     auto mul_scale_op = lowering_dequantize(rewriter, action_op->getResult(0), outType, loc, scale_value);

//     // Replace
//     rewriter.replaceOp(op, mul_scale_op->getResult(0));
//   } else {
//     // quantize left op
//     auto cast2int8_op =  lowering_quantize(rewriter, op->getOperand(0), inType, rewriter.getI8Type(), loc, left_threshold);

//     // quantize right op
//     std::string right_name = 
//       op->getOperand(1).getDefiningOp()->getAttr("name").cast<StringAttr>().getValue().str();
//     if (calibration_map.find(right_name) == calibration_map.end()) {return;}
//     float right_threshold = calibration_map.at(right_name);
//     auto right_op =  lowering_quantize(rewriter, op->getOperand(1), inType, rewriter.getI8Type(), loc, right_threshold);

//     // MatMulOp
//     auto actionType = RankedTensorType::get({out_shape}, rewriter.getI32Type());
//     auto action_op =
//         rewriter.create<mlir::tosa::MatMulOp>(loc, actionType, cast2int8_op->getResult(0), right_op->getResult(0));
    
//     // dequantize
//     float scale_value = left_threshold/127.f * right_threshold/127.f;
//     auto mul_scale_op = lowering_dequantize(rewriter, action_op->getResult(0), outType, loc, scale_value);

//     // Replace
//     rewriter.replaceOp(op, mul_scale_op->getResult(0));
//   }
// }

} // namespace mini_mlir
