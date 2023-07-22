#include "mini_mlir/Conversion/TopToTosa/OpLoweringINT8.h"

namespace mini_mlir {

void populateTopToTosaConversionINT8Patterns(
    RewritePatternSet *patterns,
    std::map<std::string, std::vector<float>> threshold_map,
    std::map<std::string, std::vector<float>> fmin_map,
    std::map<std::string, std::vector<float>> fmax_map) {
  patterns->add<
      // clang-format off
      ReshapeLoweringINT8,
      AddLoweringINT8,
      MatMulLoweringINT8,
      MulLoweringINT8,
      PermuteLoweringINT8,
      GELULoweringINT8
      // clang-format on
      >(patterns->getContext(), threshold_map, fmin_map, fmax_map);
}

//===------------------------------------------------------------===//
// ReshapeLoweringINT8
//===------------------------------------------------------------===//
void ReshapeLoweringINT8::Lowering(PatternRewriter &rewriter,
                                   top::ReshapeOp op) const {
  assert(op->getNumResults() == 1);
  std::string out_name =
      op->getAttr("name").cast<StringAttr>().getValue().str();
  auto threshold = threshold_map.at(out_name)[0];
  float inv_scale_value = 127. / threshold;

  Location loc = op->getLoc();
  auto inType = op->getOperand(0).getType();
  auto outType = op->getResult(0).getType();
  std::vector<int64_t> out_shape(outType.cast<RankedTensorType>().getShape());

  // quantize
  auto cast2int8_op =
      lowering_quantize(rewriter, op->getOperand(0), inType,
                        rewriter.getI8Type(), loc, inv_scale_value);

  // ReshapeOp
  auto actionType = RankedTensorType::get({out_shape}, rewriter.getI8Type());
  auto action_op = rewriter.create<mlir::tosa::ReshapeOp>(
      op->getLoc(), actionType, cast2int8_op->getResult(0), out_shape);

  // dequantize
  float scale_value = threshold / 127.;
  auto mul_scale_op = lowering_dequantize(rewriter, action_op->getResult(0),
                                          outType, loc, scale_value);

  // Replace
  rewriter.replaceOp(op, mul_scale_op->getResults());
}

//===------------------------------------------------------------===//
// PermuteLoweringINT8
//===------------------------------------------------------------===//
void PermuteLoweringINT8::Lowering(PatternRewriter &rewriter,
                                   top::PermuteOp op) const {
  assert(op->getNumResults() == 1);
  std::string out_name =
      op->getAttr("name").cast<StringAttr>().getValue().str();
  auto threshold = threshold_map.at(out_name)[0];
  float inv_scale_value = 127. / threshold;

  Location loc = op->getLoc();
  auto inType = op->getOperand(0).getType();
  auto outType = op->getResult(0).getType();
  std::vector<int64_t> out_shape(outType.cast<RankedTensorType>().getShape());

  // ConstOp
  auto order = module::getI64Array(op.getOrder());
  std::vector<int64_t> &ord = *order;
  int ord_size = ord.size();
  std::vector<int64_t> perms;
  for (int i = 0; i < ord_size; i++) {
    perms.push_back(order->at(i));
  }
  auto const_ty = RankedTensorType::get({ord_size}, rewriter.getI64Type());
  DenseElementsAttr attr = DenseElementsAttr::get(
      const_ty, llvm::ArrayRef(perms.data(), perms.size()));
  auto constop =
      rewriter.create<mlir::tosa::ConstOp>(op->getLoc(), const_ty, attr);

  // quantize
  auto cast2int8_op =
      lowering_quantize(rewriter, op->getOperand(0), inType,
                        rewriter.getI8Type(), loc, inv_scale_value);

  // ReshapeOp
  auto actionType = RankedTensorType::get({out_shape}, rewriter.getI8Type());
  auto action_op = rewriter.create<mlir::tosa::TransposeOp>(
      op->getLoc(), actionType, cast2int8_op->getResult(0),
      constop->getResult(0));

  // dequantize
  float scale_value = threshold / 127.;
  auto mul_scale_op = lowering_dequantize(rewriter, action_op->getResult(0),
                                          outType, loc, scale_value);

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

  std::string out_name =
      op->getAttr("name").cast<StringAttr>().getValue().str();
  if (threshold_map.find(out_name) == threshold_map.end()) {
    return;
  }
  const float left_threshold = threshold_map.at(out_name)[0];
  float inv_scale_value = 127. / left_threshold;

  // right == top::WeightOp
  mlir::Value right_value = op->getOperand(1);
  std::string right_op_name =
      op->getOperand(1).getDefiningOp()->getName().getStringRef().str();
  if (right_op_name == "tosa.const") {
    llvm_unreachable("not support now");
  } else if (right_op_name == "top.Weight") {
    auto weight_op = dyn_cast<top::WeightOp>(right_value.getDefiningOp());
    // quantize left op
    auto cast2int32_op =
        lowering_quantize(rewriter, op->getOperand(0), inType,
                          rewriter.getI32Type(), loc, inv_scale_value);

    // quantize weight op
    std::vector<int64_t> in_shape(
        op->getOperand(1).getType().cast<RankedTensorType>().getShape());
    auto const_op =
        lowering_weight_int32(rewriter, weight_op, rewriter.getI32Type(), loc,
                              left_threshold, in_shape);

    // AddOp
    auto actionType = RankedTensorType::get({out_shape}, rewriter.getI32Type());
    auto action_op = rewriter.create<mlir::tosa::AddOp>(
        op->getLoc(), actionType, cast2int32_op->getResult(0),
        const_op->getResult(0));

    // dequantize
    float scale_value = left_threshold / 127.f;
    auto mul_scale_op = lowering_dequantize(rewriter, action_op->getResult(0),
                                            outType, loc, scale_value);

    // Replace
    rewriter.replaceOp(op, mul_scale_op->getResults());
  } else {
    // quantize left op
    auto cast2int32_op =
        lowering_quantize(rewriter, op->getOperand(0), inType,
                          rewriter.getI32Type(), loc, inv_scale_value);

    // quantize right op
    // use left_threshold
    auto right_op =
        lowering_quantize(rewriter, op->getOperand(1), inType,
                          rewriter.getI32Type(), loc, inv_scale_value);

    // AddOp
    auto actionType = RankedTensorType::get({out_shape}, rewriter.getI32Type());
    auto action_op = rewriter.create<mlir::tosa::AddOp>(
        loc, actionType, cast2int32_op->getResult(0), right_op->getResult(0));

    // dequantize
    float scale_value = left_threshold / 127.f;
    auto mul_scale_op = lowering_dequantize(rewriter, action_op->getResult(0),
                                            outType, loc, scale_value);

    // Replace
    rewriter.replaceOp(op, mul_scale_op->getResults());
  }
}

//===------------------------------------------------------------===//
// MatMulLoweringINT8
//===------------------------------------------------------------===//
void MatMulLoweringINT8::Lowering(PatternRewriter &rewriter,
                                  top::MatMulOp op) const {
  assert(op->getNumResults() == 1);

  Location loc = op->getLoc();
  auto outType = op->getResult(0).getType();
  auto leftType = op->getOperand(0).getType();
  auto rightType = op->getOperand(1).getType();
  std::vector<int64_t> outShape(outType.cast<RankedTensorType>().getShape());
  std::vector<int64_t> leftShape(leftType.cast<RankedTensorType>().getShape());
  std::vector<int64_t> rightShape(
      rightType.cast<RankedTensorType>().getShape());
  auto leftSize = leftType.cast<RankedTensorType>().getShape().size();
  auto rightSize = rightType.cast<RankedTensorType>().getShape().size();

  std::string out_name =
      op->getAttr("name").cast<StringAttr>().getValue().str();
  if (threshold_map.find(out_name) == threshold_map.end()) {
    return;
  }

  mlir::tosa::ConstOp weight_op;
  mlir::Value right_value = op->getOperand(1);
  std::string right_op_name =
      op->getOperand(1).getDefiningOp()->getName().getStringRef().str();
  float right_threshold;

  // quantize left op
  const float left_threshold = threshold_map.at(out_name)[0];
  float inv_scale_value = 127. / left_threshold;
  mlir::tosa::CastOp left_op =
      lowering_quantize(rewriter, op->getOperand(0), leftType,
                        rewriter.getI8Type(), loc, inv_scale_value);

  // MatMulOp
  if (right_op_name == "tosa.const") {
    llvm_unreachable("not support now");
  } else if (leftSize == 3 && rightSize == 3 && right_op_name == "top.Weight") {
    // quantize weight op
    auto top_weight_op = dyn_cast<top::WeightOp>(right_value.getDefiningOp());
    std::vector<int64_t> in_shape(
        op->getOperand(1).getType().cast<RankedTensorType>().getShape());
    right_threshold = get_weight_threshold(rewriter, top_weight_op);
    weight_op =
        lowering_weight_int8(rewriter, top_weight_op, rewriter.getI8Type(), loc,
                             right_threshold, in_shape);

    // MatMulOp
    auto actionType = RankedTensorType::get({outShape}, rewriter.getI32Type());
    auto action_op = rewriter.create<mlir::tosa::MatMulOp>(
        op->getLoc(), actionType, left_op->getResult(0),
        weight_op->getResult(0));

    // dequantize
    float scale_value = left_threshold / 127.f * right_threshold / 127.f;
    auto mul_scale_op = lowering_dequantize(rewriter, action_op->getResult(0),
                                            outType, loc, scale_value);
    rewriter.replaceOp(op, mul_scale_op->getResults());
  } else if (leftSize == 4 && rightSize == 4 && leftShape[0] == 1 &&
             rightShape[0] == 1 && right_op_name != "top.Weight") {
    // quantize right op
    right_threshold = threshold_map.at(out_name)[1];
    float inv_scale_value = 127. / right_threshold;
    auto right_op =
        lowering_quantize(rewriter, op->getOperand(1), rightType,
                          rewriter.getI8Type(), loc, inv_scale_value);

    // ReshapeOp
    std::vector<int64_t> newLeftShape(leftShape.begin() + 1, leftShape.end());
    std::vector<int64_t> newRightShape(rightShape.begin() + 1,
                                       rightShape.end());
    auto left_type = RankedTensorType::get(newLeftShape, rewriter.getI8Type());
    auto right_type =
        RankedTensorType::get(newRightShape, rewriter.getI8Type());
    auto left_reshape_op = rewriter.create<mlir::tosa::ReshapeOp>(
        op->getLoc(), left_type, left_op->getResult(0), newLeftShape);
    auto right_reshape_op = rewriter.create<mlir::tosa::ReshapeOp>(
        op->getLoc(), right_type, right_op->getResult(0), newRightShape);

    // MatMulOp
    auto matmul_type = RankedTensorType::get(
        {newLeftShape[0], newLeftShape[1], newRightShape[2]},
        rewriter.getI32Type());
    auto matmul_op = rewriter.create<mlir::tosa::MatMulOp>(
        op->getLoc(), matmul_type, left_reshape_op->getResult(0),
        right_reshape_op->getResult(0));

    // ReshapeOp
    std::vector<int64_t> finalShape = {1, newLeftShape[0], newLeftShape[1],
                                       newRightShape[2]};
    auto reshape_type =
        RankedTensorType::get(finalShape, rewriter.getI32Type());
    auto reshape_op = rewriter.create<mlir::tosa::ReshapeOp>(
        op->getLoc(), reshape_type, matmul_op->getResult(0), finalShape);

    // dequantize
    float scale_value = left_threshold / 127.f * right_threshold / 127.f;
    auto mul_scale_op = lowering_dequantize(rewriter, reshape_op->getResult(0),
                                            outType, loc, scale_value);
    rewriter.replaceOp(op, mul_scale_op->getResults());
  }
}

//===------------------------------------------------------------===//
// MulLoweringINT8
//===------------------------------------------------------------===//
void MulLoweringINT8::Lowering(PatternRewriter &rewriter, top::MulOp op) const {
  assert(op->getNumResults() == 1);

  Location loc = op->getLoc();
  auto outType = op->getResult(0).getType();
  auto leftType = op->getOperand(0).getType();
  auto rightType = op->getOperand(1).getType();
  std::vector<int64_t> outShape(outType.cast<RankedTensorType>().getShape());

  std::string out_name =
      op->getAttr("name").cast<StringAttr>().getValue().str();
  if (threshold_map.find(out_name) == threshold_map.end()) {
    return;
  }

  std::string right_op_name =
      op->getOperand(1).getDefiningOp()->getName().getStringRef().str();

  // quantize left op
  const float left_threshold = threshold_map.at(out_name)[0];
  float inv_scale_value = 127. / left_threshold;
  mlir::tosa::CastOp left_op =
      lowering_quantize(rewriter, op->getOperand(0), leftType,
                        rewriter.getI8Type(), loc, inv_scale_value);

  // MulOp
  if (right_op_name == "tosa.const") {
    llvm_unreachable("not support now");
  } else if (right_op_name == "top.Weight") {
    // quantize weight op
    auto top_weight_op =
        dyn_cast<top::WeightOp>(op->getOperand(1).getDefiningOp());
    std::vector<int64_t> in_shape(
        op->getOperand(1).getType().cast<RankedTensorType>().getShape());
    float right_threshold = get_weight_threshold(rewriter, top_weight_op);
    auto weight_op =
        lowering_weight_int8(rewriter, top_weight_op, rewriter.getI8Type(), loc,
                             right_threshold, in_shape);

    // MulOp
    auto actionType = RankedTensorType::get({outShape}, rewriter.getI32Type());
    auto action_op = rewriter.create<mlir::tosa::MulOp>(
        op->getLoc(), actionType, left_op->getResult(0),
        weight_op->getResult(0), rewriter.getI32IntegerAttr(0));

    // dequantize
    float scale_value = left_threshold / 127.f * right_threshold / 127.f;
    auto mul_scale_op = lowering_dequantize(rewriter, action_op->getResult(0),
                                            outType, loc, scale_value);
    rewriter.replaceOp(op, mul_scale_op->getResults());
  } else if (right_op_name != "top.Weight") {
    // quantize right op
    float right_threshold = threshold_map.at(out_name)[1];
    float inv_scale_value = 127. / right_threshold;
    auto right_op =
        lowering_quantize(rewriter, op->getOperand(1), rightType,
                          rewriter.getI8Type(), loc, inv_scale_value);

    // MulOp
    auto action_type = RankedTensorType::get(outShape, rewriter.getI32Type());
    auto action_op = rewriter.create<mlir::tosa::MulOp>(
        op->getLoc(), action_type, left_op->getResult(0),
        right_op->getResult(0), rewriter.getI32IntegerAttr(0));

    // dequantize
    float scale_value = left_threshold / 127.f * right_threshold / 127.f;
    auto mul_scale_op = lowering_dequantize(rewriter, action_op->getResult(0),
                                            outType, loc, scale_value);
    rewriter.replaceOp(op, mul_scale_op->getResults());
  }
}

//===------------------------------------------------------------===//
// GELULoweringINT8
//===------------------------------------------------------------===//
void GELULoweringINT8::Lowering(PatternRewriter &rewriter,
                                top::GELUOp op) const {
  assert(op->getNumResults() == 1);
  std::string out_name =
      op->getAttr("name").cast<StringAttr>().getValue().str();
  auto threshold_vec = threshold_map.at(out_name);
  auto fmin_vec = fmin_map.at(out_name);
  auto fmax_vec = fmax_map.at(out_name);

//   auto in_threshold = threshold_vec[0];
  auto in_fmin = fmin_vec[0];
  auto in_fmax = fmax_vec[0];

  // auto out_threshold = threshold_vec[threshold_vec.size() - 1];
  auto out_fmin = fmin_vec[fmin_vec.size() - 1];
  auto out_fmax = fmax_vec[fmax_vec.size() - 1];

  float qmax = 127;
  float qmin = -128;



  Location loc = op->getLoc();
  auto inType = op->getOperand(0).getType();
  auto outType = op->getResult(0).getType();
  std::vector<int64_t> out_shape(outType.cast<RankedTensorType>().getShape());

  // ClampOp
  auto positive_value = gen_clamp_value(rewriter, inType, op->getLoc(), op->getOperand(0), 0.f);
  auto negative_value = gen_clamp_value(rewriter, inType, op->getLoc(), op->getOperand(0), std::numeric_limits<float>::lowest(), 0.f);

  // positive
  auto in_scale = (in_fmax - 0.f) / (qmax - qmin);
  auto in_zp = -0.f/in_scale + qmin;
  auto out_scale = (out_fmax - 0.) / (qmax - qmin);
  auto out_zp = - 0. / out_scale + qmin;
  auto pos_cast2int8_op =
      lowering_quantize(rewriter, positive_value, inType,
                        rewriter.getI8Type(), loc, in_scale, in_zp);
//   mlir::tosa::ConstOp pos_table_op = create_lookup_table(
//       rewriter, op->getLoc(), in_scale, in_zp, out_scale, out_zp, qmax, qmin, [](double x) {
//         return 0.5 * x *
//                (1 + std::tanh(std::sqrt(2.0 / M_PI) *
//                               (x + 0.044715 * std::pow(x, 3))));
//       });
  mlir::tosa::ConstOp pos_table_op = create_lookup_table(
      rewriter, op->getLoc(), in_scale, in_zp, out_scale, out_zp, qmax, qmin);
  auto pos_action_op = rewriter.create<mlir::tosa::TableOp>(
      op->getLoc(), RankedTensorType::get({out_shape}, rewriter.getI8Type()), pos_cast2int8_op->getResult(0),
      pos_table_op->getResult(0));
  auto pos_mul_scale_op = lowering_dequantize(rewriter, pos_action_op->getResult(0),
                                          outType, loc, out_scale, out_zp);

  // negative
  in_scale = (0.f - in_fmin) / (qmax - qmin);
  in_zp = -in_fmin/in_scale + qmin;
  out_scale = (0. - out_fmin) / (qmax - qmin);
  out_zp = - out_fmin / out_scale + qmin;
  auto neg_cast2int8_op =
      lowering_quantize(rewriter, negative_value, inType,
                        rewriter.getI8Type(), loc, in_scale, in_zp);
  mlir::tosa::ConstOp neg_table_op = create_lookup_table(
      rewriter, op->getLoc(), in_scale, in_zp, out_scale, out_zp, qmax, qmin);
  auto neg_action_op = rewriter.create<mlir::tosa::TableOp>(
      op->getLoc(), RankedTensorType::get({out_shape}, rewriter.getI8Type()), neg_cast2int8_op->getResult(0),
      neg_table_op->getResult(0));
  auto neg_mul_scale_op = lowering_dequantize(rewriter, neg_action_op->getResult(0),
                                          outType, loc, out_scale, out_zp);


  // AddOp
auto final_add_op = rewriter.create<mlir::tosa::AddOp>(
    op->getLoc(), outType, pos_mul_scale_op->getResult(0),
    neg_mul_scale_op->getResult(0));

  // Replace
  rewriter.replaceOp(op, final_add_op->getResults());
}

} // namespace mini_mlir
