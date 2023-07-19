#include "mini_mlir/Conversion/TopToTosa/OpLoweringINT8.h"

namespace mini_mlir {

void populateTopToTosaConversionINT8Patterns(RewritePatternSet *patterns, std::map<std::string, std::vector<float>> calibration_map) {
  patterns->add<
      // clang-format off
      ReshapeLoweringINT8,
      AddLoweringINT8,
      MatMulLoweringINT8
      // clang-format on
      >(patterns->getContext(), calibration_map);
}

//===------------------------------------------------------------===//
// ReshapeLoweringINT8
//===------------------------------------------------------------===//
void ReshapeLoweringINT8::Lowering(PatternRewriter &rewriter, top::ReshapeOp op) const {
  assert(op->getNumResults() == 1);
  std::string out_name = op->getAttr("name").cast<StringAttr>().getValue().str();
  auto threshold = calibration_map.at(out_name)[0];

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
  rewriter.replaceOp(op, mul_scale_op->getResults());
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

  std::string out_name = op->getAttr("name").cast<StringAttr>().getValue().str();
  if (calibration_map.find(out_name) == calibration_map.end()) {return;}
  const float left_threshold = calibration_map.at(out_name)[0];

  // right == top::WeightOp
  mlir::Value right_value = op->getOperand(1);
  std::string right_op_name = op->getOperand(1).getDefiningOp()->getName().getStringRef().str();
  if (right_op_name == "tosa.const") {
    llvm_unreachable("not support now");
  } else if (right_op_name == "top.Weight") {
    auto weight_op = dyn_cast<top::WeightOp>(right_value.getDefiningOp());
    // quantize left op
    auto cast2int32_op =  lowering_quantize(rewriter, op->getOperand(0), inType, rewriter.getI32Type(), loc, left_threshold);
    
    // quantize weight op
    std::vector<int64_t> in_shape(op->getOperand(1).getType().cast<RankedTensorType>().getShape());
    auto const_op = lowering_weight_int32(rewriter, weight_op, rewriter.getI32Type(), loc, left_threshold, in_shape);

    // AddOp
    auto actionType = RankedTensorType::get({out_shape}, rewriter.getI32Type());
    auto action_op =
        rewriter.create<mlir::tosa::AddOp>(op->getLoc(), actionType, cast2int32_op->getResult(0), const_op->getResult(0));
    
    // dequantize
    float scale_value = left_threshold/127.f;
    auto mul_scale_op = lowering_dequantize(rewriter, action_op->getResult(0), outType, loc, scale_value);

    // Replace
    rewriter.replaceOp(op, mul_scale_op->getResults());
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
    rewriter.replaceOp(op, mul_scale_op->getResults());
  }
}

//===------------------------------------------------------------===//
// MatMulLoweringINT8
//===------------------------------------------------------------===//
void MatMulLoweringINT8::Lowering(PatternRewriter &rewriter, top::MatMulOp op) const {
  assert(op->getNumResults() == 1);

  Location loc = op->getLoc();
  auto outType = op->getResult(0).getType();
  auto leftType = op->getOperand(0).getType();
  auto rightType = op->getOperand(1).getType();
  std::vector<int64_t> outShape(outType.cast<RankedTensorType>().getShape());
  std::vector<int64_t> leftShape(leftType.cast<RankedTensorType>().getShape());
  std::vector<int64_t> rightShape(rightType.cast<RankedTensorType>().getShape());
  auto leftSize = leftType.cast<RankedTensorType>().getShape().size();
  auto rightSize = rightType.cast<RankedTensorType>().getShape().size();

  std::string out_name = op->getAttr("name").cast<StringAttr>().getValue().str();
  if (calibration_map.find(out_name) == calibration_map.end()) {return;}
  const float left_threshold = calibration_map.at(out_name)[0];

  // right == top::WeightOp
  mlir::Value right_value = op->getOperand(1);
  std::string right_op_name = op->getOperand(1).getDefiningOp()->getName().getStringRef().str();
  mlir::tosa::ConstOp weight_op;
  mlir::tosa::CastOp right_op;
  mlir::tosa::MatMulOp action_op;
  float right_threshold;

  if (right_op_name == "tosa.const") {
    llvm_unreachable("not support now");
  } else if (right_op_name == "top.Weight") {
    auto top_weight_op = dyn_cast<top::WeightOp>(right_value.getDefiningOp());
    // quantize weight op
    std::vector<int64_t> in_shape(op->getOperand(1).getType().cast<RankedTensorType>().getShape());
    right_threshold = get_weight_threshold(rewriter, top_weight_op);
    weight_op = lowering_weight_int8(rewriter, top_weight_op, rewriter.getI8Type(), loc, right_threshold, in_shape);
  } else {
    // quantize right op
    right_threshold = calibration_map.at(out_name)[1];
    right_op =  lowering_quantize(rewriter, op->getOperand(1), rightType, rewriter.getI8Type(), loc, right_threshold);
  }

  // quantize left op
  mlir::tosa::CastOp cast2int8_op =  lowering_quantize(rewriter, op->getOperand(0), leftType, rewriter.getI8Type(), loc, left_threshold);


  // MatMulOp
  if (leftSize == 3 && rightSize == 3) {
    if (right_op_name == "top.Weight") {
        // MatMulOp
        auto actionType = RankedTensorType::get({outShape}, rewriter.getI32Type());
        auto action_op = rewriter.create<mlir::tosa::MatMulOp>(op->getLoc(), 
                                actionType, cast2int8_op->getResult(0), weight_op->getResult(0));

        // dequantize
        // MulOp
        float scale_value = left_threshold/127.f * right_threshold/127.f;
        auto mul_scale_op = lowering_dequantize(rewriter, action_op->getResult(0), outType, loc, scale_value);
        rewriter.replaceOp(op, mul_scale_op->getResults());
    }
  } else if (leftSize == 4 && rightSize == 4 && leftShape[0] == 1 && rightShape[0] == 1) {
    // ReshapeOp
    std::vector<int64_t> newLeftShape(leftShape.begin() + 1, leftShape.end());
    std::vector<int64_t> newRightShape(rightShape.begin() + 1, rightShape.end());

    auto left_type = RankedTensorType::get(newLeftShape, rewriter.getI8Type());

    auto right_type = RankedTensorType::get(newRightShape, rewriter.getI8Type());

    auto left_reshape_op = rewriter.create<mlir::tosa::ReshapeOp>(op->getLoc(), left_type, cast2int8_op->getResult(0), newLeftShape);
    auto right_reshape_op = rewriter.create<mlir::tosa::ReshapeOp>(op->getLoc(), right_type, right_op->getResult(0), newRightShape);

    // MatMulOp
    std::vector<int64_t> matmulShape = {newLeftShape[0], newLeftShape[1], newRightShape[2]};
    auto matmul_type = RankedTensorType::get(matmulShape, rewriter.getI32Type());
    auto matmul_op = rewriter.create<mlir::tosa::MatMulOp>(
        op->getLoc(), matmul_type, left_reshape_op->getResult(0), right_reshape_op->getResult(0));

    // ReshapeOp
    std::vector<int64_t> finalShape = {1, newLeftShape[0], newLeftShape[1], newRightShape[2]};
    auto reshape_type = RankedTensorType::get(finalShape, rewriter.getI32Type());
    auto reshape_op = rewriter.create<mlir::tosa::ReshapeOp>(op->getLoc(), reshape_type, matmul_op->getResult(0), finalShape);

    // dequantize
    // MulOp
    float scale_value = left_threshold/127.f * right_threshold/127.f;
    auto mul_scale_op = lowering_dequantize(rewriter, reshape_op->getResult(0), outType, loc, scale_value);
    rewriter.replaceOp(op, mul_scale_op->getResults());
  }
}

} // namespace mini_mlir
