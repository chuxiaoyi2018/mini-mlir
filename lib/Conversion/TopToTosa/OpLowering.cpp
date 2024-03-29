#include "mini_mlir/Conversion/TopToTosa/OpLowering.h"

namespace mini_mlir {

void populateTopToTosaConversionPatterns(RewritePatternSet *patterns) {
  patterns->add<
      // clang-format off
        InputLowering,
        AddLowering,
        ConvLowering,
	    SoftmaxLowering,
        ReshapeLowering,
        PermuteLowering,
        ConcatLowering,
        ReduceMeanLowering,
        SubLowering,
        MulLowering,
        DivLowering,
        SqrtLowering,
        MatMulLowering,
        MulConstLowering,
        GELULowering,
        SliceLowering,
        ConvPermuteLowering,
        LayerNormLowering,
        RMSNormLowering,
        SigmoidLowering,
        GatherLowering
      // clang-format on
      >(patterns->getContext());
}

//===------------------------------------------------------------===//
// InputLowering
//===------------------------------------------------------------===//
void InputLowering::Lowering(PatternRewriter &rewriter, top::InputOp op) const {
  assert(op->getNumResults() == 1);
  auto outType = op->getResult(0).getType();
  auto outShape = outType.cast<RankedTensorType>().getShape();
  if (outShape.size() == 4) {
    outType = change_dataformat(outType);
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
  } else {
    std::vector<Operation *> next_ops = {op->user_begin(), op->user_end()};
    for (auto next_op : next_ops) {
      int idx = 0;
      for (auto iop : next_op->getOperands()) {
        if (iop == op) {
          next_op->setOperand(idx, op->getOperand(0));
        }
        idx++;
      }
    }
    rewriter.eraseOp(op);
  }
}

//===------------------------------------------------------------===//
// AddLowering
//===------------------------------------------------------------===//
void AddLowering::Lowering(PatternRewriter &rewriter, top::AddOp op) const {
  assert(op->getNumResults() == 1);
  auto newType = op->getResult(0).getType();
  auto coeff = op.getCoeffAttr();
  // TODO: coeff -> constOp
  /*
  if (!coeff) {
    float coeff0 =
  coeff.getValue()[0].cast<mlir::FloatAttr>().getValueAsDouble();

    auto const_ty = RankedTensorType::get({}, rewriter.getI32Type());
    DenseElementsAttr attr = DenseElementsAttr::get(const_ty,
                      llvm::ArrayRef(perms.data(), perms.size()));
    auto constop = rewriter.create<mlir::tosa::ConstOp>(op->getLoc(), const_ty,
  attr); double coeff1 =
  coeff.getValue()[1].cast<mlir::FloatAttr>().getValueAsDouble();
  }
  */
  std::vector<Value> operands;
  for (auto in : op->getOperands()) {
    operands.push_back(in);
  }
  // do_relu
  if (op.getDoRelu()) {
    // Add op
    auto add =
        rewriter.create<mlir::tosa::AddOp>(op->getLoc(), newType, operands);
    auto relu_limit = op.getReluLimit();
    std::vector<NamedAttribute> clamp_attr =
        gen_clamp_attr(rewriter, newType, relu_limit);
    auto out_type = add->getResult(0).getType();
    // Clamp op
    auto clamp = rewriter.create<mlir::tosa::ClampOp>(
        op->getLoc(), out_type, add->getResults(), clamp_attr);
    rewriter.replaceOp(op, clamp->getResults());
  } else {
    rewriter.replaceOpWithNewOp<mlir::tosa::AddOp>(op, newType, operands);
  }
}

//===------------------------------------------------------------===//
// ConvLowering
//===------------------------------------------------------------===//
void ConvLowering::Lowering(PatternRewriter &rewriter, top::ConvOp op) const {
  assert(op->getNumResults() == 1);
  auto newType = change_dataformat(op->getResult(0).getType());
  std::vector<NamedAttribute> attrs;
  auto pads = module::getI64Array(op.getPads());
  std::vector<int64_t> newValues{pads->at(0), pads->at(2), pads->at(1),
                                 pads->at(3)};
  attrs.push_back(
      rewriter.getNamedAttr("pad", rewriter.getDenseI64ArrayAttr(newValues)));
  auto strides = module::getI64Array(op.getStrides());
  attrs.push_back(
      rewriter.getNamedAttr("stride", rewriter.getDenseI64ArrayAttr(*strides)));
  auto dilations = module::getI64Array(op.getDilations(), 2, 1);
  attrs.push_back(rewriter.getNamedAttr(
      "dilation", rewriter.getDenseI64ArrayAttr(*dilations)));
  std::vector<Value> operands;
  auto ic = op->getOperand(0).getType().cast<RankedTensorType>().getShape()[1];
  auto oc = op->getResult(0).getType().cast<RankedTensorType>().getShape()[1];
  auto kc = op->getOperand(1).getType().cast<RankedTensorType>().getShape()[1];
  auto group = op.getGroup();
  // depth_wise conv
  if (ic == oc && oc == group && kc == 1) {
    auto weight = op->getOperand(1);
    auto weightTy = weight.getType().cast<RankedTensorType>(); // NCHW
    // NCHW -> HWCM(HWCN)  In this case, "N"->"C", "C"="M"=1
    // std::vector<int32_t> perms = {2, 3, 0, 1};
    // NHWC -> HWCM(HWCN)  In this case, "N"->"C", "C"="M"=1
    std::vector<int32_t> perms = {1, 2, 0, 3};
    auto const_ty = RankedTensorType::get({4}, rewriter.getI32Type());
    DenseElementsAttr attr = DenseElementsAttr::get(
        const_ty, llvm::ArrayRef(perms.data(), perms.size()));
    auto constop =
        rewriter.create<mlir::tosa::ConstOp>(op->getLoc(), const_ty, attr);
    std::vector<int64_t> newWeightShape;
    auto weightShape = weightTy.getShape(); // NCHW
    newWeightShape.push_back(weightShape[2]);
    newWeightShape.push_back(weightShape[3]);
    newWeightShape.push_back(weightShape[0]);
    newWeightShape.push_back(weightShape[1]); // HWCM(HWCN)
    auto newWeightTy =
        RankedTensorType::get(newWeightShape, weightTy.getElementType());
    auto newweight =
        rewriter
            .create<mlir::tosa::TransposeOp>(op->getLoc(), newWeightTy, weight,
                                             constop->getResult(0))
            ->getResult(0);
    operands.push_back(op->getOperand(0));
    operands.push_back(newweight);
    if (op->getOperand(2).getType().isa<mlir::NoneType>()) {
      std::vector<float> bias(oc, 0);
      auto const_ty = RankedTensorType::get({oc}, rewriter.getF32Type());
      DenseElementsAttr attr = DenseElementsAttr::get(
          const_ty, llvm::ArrayRef(bias.data(), bias.size()));
      auto constop =
          rewriter.create<mlir::tosa::ConstOp>(op->getLoc(), const_ty, attr);
      operands.push_back(constop->getResult(0));
    } else {
      operands.push_back(op->getOperand(2));
    }
    // do_relu
    if (op.getDoRelu()) {
      // Conv op
      auto conv = rewriter.create<mlir::tosa::DepthwiseConv2DOp>(
          op->getLoc(), newType, operands, attrs);
      auto relu_limit = op.getReluLimit();
      std::vector<NamedAttribute> clamp_attr =
          gen_clamp_attr(rewriter, newType, relu_limit);
      auto out_type = conv->getResult(0).getType();
      // Clamp op
      auto clamp = rewriter.create<mlir::tosa::ClampOp>(
          op->getLoc(), out_type, conv->getResults(), clamp_attr);
      rewriter.replaceOp(op, clamp->getResults());
    } else {
      rewriter.replaceOpWithNewOp<mlir::tosa::DepthwiseConv2DOp>(
          op, newType, operands, attrs);
    }
  }
  // normal conv
  else if (group == 1) {
    for (auto in : op->getOperands()) {
      if (in.getType().isa<mlir::NoneType>()) { // bias
        std::vector<float> bias(oc, 0);
        auto const_ty = RankedTensorType::get({oc}, rewriter.getF32Type());
        DenseElementsAttr attr = DenseElementsAttr::get(
            const_ty, llvm::ArrayRef(bias.data(), bias.size()));
        auto constop =
            rewriter.create<mlir::tosa::ConstOp>(op->getLoc(), const_ty, attr);
        operands.push_back(constop->getResult(0));
      } else {
        operands.push_back(in);
      }
    }
    // do_Relu
    if (op.getDoRelu()) {
      // Conv op
      auto conv = rewriter.create<mlir::tosa::Conv2DOp>(op->getLoc(), newType,
                                                        operands, attrs);
      auto relu_limit = op.getReluLimit();
      std::vector<NamedAttribute> clamp_attr =
          gen_clamp_attr(rewriter, newType, relu_limit);
      auto out_type = conv->getResult(0).getType();
      // Clamp op
      auto clamp = rewriter.create<mlir::tosa::ClampOp>(
          op->getLoc(), out_type, conv->getResults(), clamp_attr);
      rewriter.replaceOp(op, clamp->getResults());
    } else {
      auto conv_op = rewriter.create<mlir::tosa::Conv2DOp>(
          op->getLoc(), newType, operands, attrs);
      rewriter.replaceOp(op, conv_op->getResults());
    }
  }
  // TODO: support for group conv
  else
    ;
}

//===------------------------------------------------------------===//
// ConvPermuteLowering
//===------------------------------------------------------------===//
void ConvPermuteLowering::Lowering(PatternRewriter &rewriter,
                                   top::ConvPermuteOp op) const {
  assert(op->getNumResults() == 1);
  auto outType = op->getResult(0).getType();

  // ConvOp
  std::vector<NamedAttribute> attrs;
  auto pads = module::getI64Array(op.getPads());
  std::vector<int64_t> newValues{pads->at(0), pads->at(2), pads->at(1),
                                 pads->at(3)};
  attrs.push_back(
      rewriter.getNamedAttr("pad", rewriter.getDenseI64ArrayAttr(newValues)));
  auto strides = module::getI64Array(op.getStrides());
  attrs.push_back(
      rewriter.getNamedAttr("stride", rewriter.getDenseI64ArrayAttr(*strides)));
  auto dilations = module::getI64Array(op.getDilations(), 2, 1);
  attrs.push_back(rewriter.getNamedAttr(
      "dilation", rewriter.getDenseI64ArrayAttr(*dilations)));

  auto convShapeAttr =
      op->getAttr("channel_last_conv_shape").dyn_cast_or_null<ArrayAttr>();
  std::vector<int64_t> convShape;
  if (convShapeAttr) {
    for (auto convAttr : convShapeAttr.getValue()) {
      convShape.push_back(convAttr.cast<IntegerAttr>().getInt());
    }
  }
  auto convType = RankedTensorType::get(
      convShape, outType.cast<RankedTensorType>().getElementType());
  auto conv_op = rewriter.create<mlir::tosa::Conv2DOp>(
      op->getLoc(), convType, op->getOperands(), attrs);

  // ReshapeOp
  auto newShape = outType.cast<RankedTensorType>().getShape();
  auto reshape_op = rewriter.create<mlir::tosa::ReshapeOp>(
      op->getLoc(), outType, conv_op->getResult(0), newShape);
  rewriter.replaceOp(op, reshape_op->getResults());
}

//===------------------------------------------------------------===//
// ReshapeLowering
//===------------------------------------------------------------===//
void ReshapeLowering::Lowering(PatternRewriter &rewriter,
                               top::ReshapeOp op) const {
  assert(op->getNumResults() == 1);
  auto newType = op->getResult(0).getType();
  auto newShape = newType.cast<RankedTensorType>().getShape();
  rewriter.replaceOpWithNewOp<mlir::tosa::ReshapeOp>(
      op, newType, op->getOperand(0), newShape);
}

//===------------------------------------------------------------===//
// PermuteLowering
//===------------------------------------------------------------===//
void PermuteLowering::Lowering(PatternRewriter &rewriter,
                               top::PermuteOp op) const {
  assert(op->getNumResults() == 1);
  auto outType = op->getResult(0).getType();
  auto outShape = outType.cast<RankedTensorType>().getShape();

  std::vector<Value> operands;
  operands.push_back(op->getOperand(0));

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
  operands.push_back(constop->getResult(0));

  rewriter.replaceOpWithNewOp<mlir::tosa::TransposeOp>(op, outType, operands);
}

//===------------------------------------------------------------===//
// ConcatLowering
//===------------------------------------------------------------===//
void ConcatLowering::Lowering(PatternRewriter &rewriter,
                              top::ConcatOp op) const {
  assert(op->getNumResults() == 1);
  auto outType = op->getResult(0).getType();
  auto size = outType.cast<RankedTensorType>().getShape().size();
  int32_t new_axis, axis = op.getAxis();

  if (axis > 0)
    new_axis = axis;
  else
    new_axis = size + axis;

  std::vector<NamedAttribute> attrs;
  attrs.push_back(
      rewriter.getNamedAttr("axis", rewriter.getI64IntegerAttr(new_axis)));

  std::vector<Value> operands;
  for (auto in : op->getOperands()) {
    operands.push_back(in);
  }
  rewriter.replaceOpWithNewOp<mlir::tosa::ConcatOp>(op, outType, operands,
                                                    attrs);
}

//===------------------------------------------------------------===//
// ReduceMeanLowering
//===------------------------------------------------------------===//
void ReduceMeanLowering::Lowering(PatternRewriter &rewriter,
                                  top::ReduceMeanOp op) const {
  assert(op->getNumResults() == 1);
  // auto newType = change_dataformat(op->getResult(0).getType());
  auto outType = op->getResult(0).getType();
  auto size = outType.cast<RankedTensorType>().getShape().size();
  int32_t new_axis, axis = op.getAxis();

  if (axis > 0)
    new_axis = axis;
  else
    new_axis = size + axis;

  // ReduceSumOp
  std::vector<NamedAttribute> attrs;
  attrs.push_back(
      rewriter.getNamedAttr("axis", rewriter.getI64IntegerAttr(new_axis)));
  std::vector<int64_t> out_shape(outType.cast<RankedTensorType>().getShape());
  out_shape[new_axis] = 1;
  auto out_type = RankedTensorType::get(
      out_shape, outType.cast<RankedTensorType>().getElementType());
  auto reducesum = rewriter.create<mlir::tosa::ReduceSumOp>(
      op->getLoc(), out_type, op->getOperands(), attrs);

  // ConstOp
  auto inType = op->getOperand(0).getType();
  std::vector<int64_t> in_shape(inType.cast<RankedTensorType>().getShape());
  std::vector<float> const_value;
  const_value.push_back(1. / in_shape[in_shape.size() - 1]);
  std::vector<int64_t> const_shape;
  for (int i = 0; i < size; i++) {
    const_shape.push_back(1);
  }
  auto const_ty = RankedTensorType::get(const_shape, rewriter.getF32Type());
  DenseElementsAttr attr = DenseElementsAttr::get(
      const_ty, llvm::ArrayRef(const_value.data(), const_value.size()));
  auto constop =
      rewriter.create<mlir::tosa::ConstOp>(op->getLoc(), const_ty, attr);

  // MulOp
  rewriter.replaceOpWithNewOp<mlir::tosa::MulOp>(
      op, outType, reducesum->getResult(0), constop->getResult(0),
      rewriter.getI32IntegerAttr(0));
}

//===------------------------------------------------------------===//
// SubLowering
//===------------------------------------------------------------===//
void SubLowering::Lowering(PatternRewriter &rewriter, top::SubOp op) const {
  assert(op->getNumResults() == 1);
  auto newType = op->getResult(0).getType();
  rewriter.replaceOpWithNewOp<mlir::tosa::SubOp>(op, newType, op->getOperand(0),
                                                 op->getOperand(1));
}

//===------------------------------------------------------------===//
// MulLowering
//===------------------------------------------------------------===//
void MulLowering::Lowering(PatternRewriter &rewriter, top::MulOp op) const {
  assert(op->getNumResults() == 1);
  auto newType = op->getResult(0).getType();
  rewriter.replaceOpWithNewOp<mlir::tosa::MulOp>(op, newType, op->getOperand(0),
                                                 op->getOperand(1),
                                                 rewriter.getI32IntegerAttr(0));
}

//===------------------------------------------------------------===//
// DivLowering
//===------------------------------------------------------------===//
void DivLowering::Lowering(PatternRewriter &rewriter, top::DivOp op) const {
  assert(op->getNumResults() == 1);
  auto newType = op->getResult(0).getType();

  // ReciprocalOp
  auto reciprocal_ty = op->getOperand(1).getType();
  auto reciprocal = rewriter.create<mlir::tosa::ReciprocalOp>(
      op->getLoc(), reciprocal_ty, op->getOperand(1));
  // MulOp
  auto mul = rewriter.create<mlir::tosa::MulOp>(
      op->getLoc(), newType, op->getOperand(0), reciprocal->getResult(0),
      rewriter.getI32IntegerAttr(0));
  rewriter.replaceOp(op, mul->getResults());
}

//===------------------------------------------------------------===//
// SqrtLowering
//===------------------------------------------------------------===//
void SqrtLowering::Lowering(PatternRewriter &rewriter, top::SqrtOp op) const {
  assert(op->getNumResults() == 1);
  auto newType = op->getResult(0).getType();

  // RsqrtOp
  auto rsqrt = rewriter.create<mlir::tosa::RsqrtOp>(op->getLoc(), newType,
                                                    op->getOperand(0));
  // ReciprocalOp
  auto reciprocal = rewriter.create<mlir::tosa::ReciprocalOp>(
      op->getLoc(), newType, rsqrt->getResult(0));
  rewriter.replaceOp(op, reciprocal->getResults());
}

//===------------------------------------------------------------===//
// MatMulLowering
//===------------------------------------------------------------===//
void MatMulLowering::Lowering(PatternRewriter &rewriter,
                              top::MatMulOp op) const {
  assert(op->getNumResults() == 1);
  auto newType = op->getResult(0).getType();
  auto leftType = op->getOperand(0).getType();
  auto rightType = op->getOperand(1).getType();
  std::vector<int64_t> leftShape(leftType.cast<RankedTensorType>().getShape());
  std::vector<int64_t> rightShape(
      rightType.cast<RankedTensorType>().getShape());
  auto leftSize = leftType.cast<RankedTensorType>().getShape().size();
  auto rightSize = rightType.cast<RankedTensorType>().getShape().size();

  if (leftSize == 3 && rightSize == 3) {
    // MatMulOp
    auto matmul = rewriter.create<mlir::tosa::MatMulOp>(
        op->getLoc(), newType, op->getOperand(0), op->getOperand(1));
    rewriter.replaceOp(op, matmul->getResults());
  } else if (leftSize == 4 && rightSize == 4 && leftShape[0] == 1 &&
             rightShape[0] == 1) {
    // ReshapeOp
    std::vector<int64_t> newLeftShape(leftShape.begin() + 1, leftShape.end());
    std::vector<int64_t> newRightShape(rightShape.begin() + 1,
                                       rightShape.end());

    auto left_type = RankedTensorType::get(
        newLeftShape, leftType.cast<RankedTensorType>().getElementType());

    auto right_type = RankedTensorType::get(
        newRightShape, rightType.cast<RankedTensorType>().getElementType());

    auto left_op = rewriter.create<mlir::tosa::ReshapeOp>(
        op->getLoc(), left_type, op->getOperand(0), newLeftShape);
    auto right_op = rewriter.create<mlir::tosa::ReshapeOp>(
        op->getLoc(), right_type, op->getOperand(1), newRightShape);

    // MatMulOp
    std::vector<int64_t> matmulShape = {newLeftShape[0], newLeftShape[1],
                                        newRightShape[2]};
    auto matmul_type = RankedTensorType::get(
        matmulShape, newType.cast<RankedTensorType>().getElementType());
    auto matmul_op = rewriter.create<mlir::tosa::MatMulOp>(
        op->getLoc(), matmul_type, left_op->getResult(0),
        right_op->getResult(0));

    // ReshapeOp
    std::vector<int64_t> finalShape = {1, newLeftShape[0], newLeftShape[1],
                                       newRightShape[2]};
    auto reshape_type = RankedTensorType::get(
        finalShape, newType.cast<RankedTensorType>().getElementType());
    auto reshape_op = rewriter.create<mlir::tosa::ReshapeOp>(
        op->getLoc(), reshape_type, matmul_op->getResult(0), finalShape);
    rewriter.replaceOp(op, reshape_op->getResults());
  }
}

//===------------------------------------------------------------===//
// MulConstLowering
//===------------------------------------------------------------===//
void MulConstLowering::Lowering(PatternRewriter &rewriter,
                                top::MulConstOp op) const {
  assert(op->getNumResults() == 1);

  // ConstOp
  auto inType = op->getOperand(0).getType();
  auto size = inType.cast<RankedTensorType>().getShape().size();
  std::vector<float> const_value;
  const_value.push_back(op->getAttr("const_val")
                            .dyn_cast_or_null<FloatAttr>()
                            .getValueAsDouble());

  std::vector<int64_t> const_shape;
  for (int i = 0; i < size; i++) {
    const_shape.push_back(1);
  }
  auto const_ty = RankedTensorType::get(const_shape, rewriter.getF32Type());
  DenseElementsAttr attr = DenseElementsAttr::get(
      const_ty, llvm::ArrayRef(const_value.data(), const_value.size()));
  auto constop =
      rewriter.create<mlir::tosa::ConstOp>(op->getLoc(), const_ty, attr);

  // MulOp
  rewriter.replaceOpWithNewOp<mlir::tosa::MulOp>(op, inType, op->getOperand(0),
                                                 constop->getResult(0),
                                                 rewriter.getI32IntegerAttr(0));
}

//===------------------------------------------------------------===//
// GELULowering
//===------------------------------------------------------------===//
void GELULowering::Lowering(PatternRewriter &rewriter, top::GELUOp op) const {
  assert(op->getNumResults() == 1);

  // ConstOp
  auto inType = op->getOperand(0).getType();
  auto size = inType.cast<RankedTensorType>().getShape().size();
  std::vector<int64_t> const_shape;
  for (int i = 0; i < size; i++) {
    const_shape.push_back(1);
  }
  auto const_ty = RankedTensorType::get(const_shape, rewriter.getF32Type());

  // ConstOp const_value_0 = 0.5
  std::vector<float> const_value_0 = {0.5};
  DenseElementsAttr attr0 = DenseElementsAttr::get(
      const_ty, llvm::ArrayRef(const_value_0.data(), const_value_0.size()));
  auto constop_0 =
      rewriter.create<mlir::tosa::ConstOp>(op->getLoc(), const_ty, attr0);

  // ConstOp const_value_1 = 1
  std::vector<float> const_value_1 = {1};
  DenseElementsAttr attr1 = DenseElementsAttr::get(
      const_ty, llvm::ArrayRef(const_value_1.data(), const_value_1.size()));
  auto constop_1 =
      rewriter.create<mlir::tosa::ConstOp>(op->getLoc(), const_ty, attr1);

  // ConstOp const_value_2 = 2
  std::vector<float> const_value_2 = {0.7978845608028654};
  DenseElementsAttr attr2 = DenseElementsAttr::get(
      const_ty, llvm::ArrayRef(const_value_2.data(), const_value_2.size()));
  auto constop_2 =
      rewriter.create<mlir::tosa::ConstOp>(op->getLoc(), const_ty, attr2);

  // ConstOp const_value_3 = 3
  std::vector<float> const_value_3 = {0.044715};
  DenseElementsAttr attr3 = DenseElementsAttr::get(
      const_ty, llvm::ArrayRef(const_value_3.data(), const_value_3.size()));
  auto constop_3 =
      rewriter.create<mlir::tosa::ConstOp>(op->getLoc(), const_ty, attr3);

  // ConstOp for PowOp
  std::vector<float> const_value_pow = {3};
  DenseElementsAttr attr_pow = DenseElementsAttr::get(
      const_ty, llvm::ArrayRef(const_value_pow.data(), const_value_pow.size()));
  auto constop_pow =
      rewriter.create<mlir::tosa::ConstOp>(op->getLoc(), const_ty, attr_pow);

  // PowOp
  auto pow_op = rewriter.create<mlir::tosa::PowOp>(
      op->getLoc(), inType, op->getOperand(0), constop_pow->getResult(0));

  // MulOp
  auto mul_pow_const3_op = rewriter.create<mlir::tosa::MulOp>(
      op->getLoc(), inType, pow_op->getResult(0), constop_3->getResult(0),
      rewriter.getI32IntegerAttr(0));

  // AddOp
  auto add_op_mul_op = rewriter.create<mlir::tosa::AddOp>(
      op->getLoc(), inType, op->getOperand(0), mul_pow_const3_op->getResult(0));

  // MulOp
  auto mul_add_const2_op = rewriter.create<mlir::tosa::MulOp>(
      op->getLoc(), inType, add_op_mul_op->getResult(0),
      constop_2->getResult(0), rewriter.getI32IntegerAttr(0));

  // TanhOp
  auto tanh_op = rewriter.create<mlir::tosa::TanhOp>(
      op->getLoc(), inType, mul_add_const2_op->getResult(0));

  // AddOp
  auto add_tanh_const1_op = rewriter.create<mlir::tosa::AddOp>(
      op->getLoc(), inType, tanh_op->getResult(0), constop_1->getResult(0));

  // MulOp
  auto mul_op_tanh_op = rewriter.create<mlir::tosa::MulOp>(
      op->getLoc(), inType, op->getOperand(0), add_tanh_const1_op->getResult(0),
      rewriter.getI32IntegerAttr(0));

  // MulOp
  auto mul_const0_op = rewriter.create<mlir::tosa::MulOp>(
      op->getLoc(), inType, constop_0->getResult(0),
      mul_op_tanh_op->getResult(0), rewriter.getI32IntegerAttr(0));

  rewriter.replaceOp(op, mul_const0_op->getResults());
}

//===------------------------------------------------------------===//
// SliceLowering
//===------------------------------------------------------------===//
void SliceLowering::Lowering(PatternRewriter &rewriter, top::SliceOp op) const {
  assert(op->getNumResults() == 1);
  auto inType = op->getOperand(0).getType();

  // start_list
  auto startListAttr = op->getAttr("start_list").dyn_cast_or_null<ArrayAttr>();
  std::vector<int64_t> starts;
  if (startListAttr) {
    for (auto startAttr : startListAttr.getValue()) {
      starts.push_back(startAttr.cast<IntegerAttr>().getInt());
    }
  }

  // size_list
  auto sizeListAttr = op->getAttr("size_list").dyn_cast_or_null<ArrayAttr>();
  std::vector<int64_t> sizes;
  if (sizeListAttr) {
    for (auto sizeAttr : sizeListAttr.getValue()) {
      sizes.push_back(sizeAttr.cast<IntegerAttr>().getInt());
    }
  }

  // SliceOp
  std::vector<int64_t> outShape = {sizes.begin(), sizes.end()};
  auto outType = RankedTensorType::get(
      outShape, inType.cast<RankedTensorType>().getElementType());

  std::vector<NamedAttribute> attrs;
  attrs.push_back(
      rewriter.getNamedAttr("start", rewriter.getDenseI64ArrayAttr(starts)));
  attrs.push_back(
      rewriter.getNamedAttr("size", rewriter.getDenseI64ArrayAttr(sizes)));
  rewriter.replaceOpWithNewOp<mlir::tosa::SliceOp>(op, outType,
                                                   op->getOperand(0), attrs);
}

//===------------------------------------------------------------===//
// SoftmaxLowering
//===------------------------------------------------------------===//
void SoftmaxLowering::Lowering(PatternRewriter &rewriter,
                               top::SoftmaxOp op) const {
  assert(op->getNumResults() == 1);
  auto preType = op->getResult(0).getType();
  // auto newType = change_dataformat(preType);
  auto newType = op->getOperand(0).getType();
  auto size = preType.cast<RankedTensorType>().getShape().size();
  int32_t new_axis, axis = op.getAxis();
  if (size == 4) {
    if (axis == 1 || axis == -3)
      new_axis = 1; // C
    else if (axis == 2 || axis == -2)
      new_axis = 2; // H
    else if (axis == 3 || axis == -1)
      new_axis = 3; // W
    else
      new_axis = axis; // N
  }
  bool log_attr_val = op.getLog();
  // op.getBeta() (beta = 1 by default)
  // ReduceMaxOp
  std::vector<NamedAttribute> attrs;
  attrs.push_back(
      rewriter.getNamedAttr("axis", rewriter.getI64IntegerAttr(new_axis)));
  std::vector<int64_t> out_shape(newType.cast<RankedTensorType>().getShape());
  out_shape[new_axis] = 1;
  auto out_type = RankedTensorType::get(
      out_shape, newType.cast<RankedTensorType>().getElementType());
  auto reducemax = rewriter.create<mlir::tosa::ReduceMaxOp>(
      op->getLoc(), out_type, op->getOperands(), attrs);
  // SubOp
  std::vector<Value> operands;
  operands.push_back(op->getOperand(0));
  operands.push_back(reducemax->getResult(0));
  auto sub =
      rewriter.create<mlir::tosa::SubOp>(op->getLoc(), newType, operands);
  // ExpOp
  auto sub_ty = sub->getResult(0).getType();
  auto exp = rewriter.create<mlir::tosa::ExpOp>(op->getLoc(), sub_ty,
                                                sub->getResults());
  // ReduceSumOp ( out_type & attrs same as ReduceMaxOp)
  auto reducesum = rewriter.create<mlir::tosa::ReduceSumOp>(
      op->getLoc(), out_type, exp->getResults(), attrs);
  // LogSoftmax ? Softmax ?
  if (log_attr_val) {
    // LogOp
    auto reducesum_ty = reducesum->getResult(0).getType();
    auto log = rewriter.create<mlir::tosa::LogOp>(op->getLoc(), reducesum_ty,
                                                  reducesum->getResults());
    // SubOp
    operands.clear();
    operands.push_back(sub->getResult(0));
    operands.push_back(log->getResult(0));
    auto sub2 =
        rewriter.create<mlir::tosa::SubOp>(op->getLoc(), newType, operands);
    rewriter.replaceOp(op, sub->getResults());
  } else {
    // ReciprocalOp
    auto reducesum_ty = reducesum->getResult(0).getType();
    auto reciprocal = rewriter.create<mlir::tosa::ReciprocalOp>(
        op->getLoc(), reducesum_ty, reducesum->getResults());
    // MulOp
    auto mul = rewriter.create<mlir::tosa::MulOp>(
        op->getLoc(), newType, exp->getResult(0), reciprocal->getResult(0),
        rewriter.getI32IntegerAttr(0));
    rewriter.replaceOp(op, mul->getResults());
  }
}

//===------------------------------------------------------------===//
// LayerNormLowering
//===------------------------------------------------------------===//
void LayerNormLowering::Lowering(PatternRewriter &rewriter,
                                 top::LayerNormOp op) const {
  assert(op->getNumResults() == 1);
  // auto newType = change_dataformat(op->getResult(0).getType());
  auto outType = op->getResult(0).getType();
  auto size = outType.cast<RankedTensorType>().getShape().size();
  int32_t new_axis, axis = op.getAxis();

  if (axis > 0)
    new_axis = axis;
  else
    new_axis = size + axis;

  // ReduceSumOp
  std::vector<NamedAttribute> attrs;
  attrs.push_back(
      rewriter.getNamedAttr("axis", rewriter.getI64IntegerAttr(new_axis)));
  std::vector<int64_t> out_shape(outType.cast<RankedTensorType>().getShape());
  out_shape[new_axis] = 1;
  auto out_type = RankedTensorType::get(
      out_shape, outType.cast<RankedTensorType>().getElementType());
  auto reducesum = rewriter.create<mlir::tosa::ReduceSumOp>(
      op->getLoc(), out_type, op->getOperand(0), attrs);

  // ConstOp
  auto inType = op->getOperand(0).getType();
  std::vector<int64_t> in_shape(inType.cast<RankedTensorType>().getShape());
  std::vector<float> const_value;
  const_value.push_back(1. / in_shape[new_axis]);
  std::vector<int64_t> const_shape;
  for (int i = 0; i < size; i++) {
    const_shape.push_back(1);
  }
  auto const_ty = RankedTensorType::get(const_shape, rewriter.getF32Type());
  DenseElementsAttr attr = DenseElementsAttr::get(
      const_ty, llvm::ArrayRef(const_value.data(), const_value.size()));
  auto constop =
      rewriter.create<mlir::tosa::ConstOp>(op->getLoc(), const_ty, attr);

  // Mean
  // MulOp
  auto mul_op_0 = rewriter.create<mlir::tosa::MulOp>(
      op->getLoc(), out_type, reducesum->getResult(0), constop->getResult(0),
      rewriter.getI32IntegerAttr(0));

  // SubOp
  auto sub_op = rewriter.create<mlir::tosa::SubOp>(
      op->getLoc(), outType, op->getOperand(0), mul_op_0->getResult(0));

  // ConstOp for PowOp
  std::vector<float> const_value_pow = {2};
  DenseElementsAttr attr_pow = DenseElementsAttr::get(
      const_ty, llvm::ArrayRef(const_value_pow.data(), const_value_pow.size()));
  auto constop_pow =
      rewriter.create<mlir::tosa::ConstOp>(op->getLoc(), const_ty, attr_pow);

  // PowOp
  auto pow_op = rewriter.create<mlir::tosa::PowOp>(
      op->getLoc(), outType, sub_op->getResult(0), constop_pow->getResult(0));
      
  // ReduceSumOp
  auto reducesum_op_2 = rewriter.create<mlir::tosa::ReduceSumOp>(
      op->getLoc(), out_type, pow_op->getResult(0), attrs);

  // MulOp
  auto mul_op_1 = rewriter.create<mlir::tosa::MulOp>(
      op->getLoc(), out_type, reducesum_op_2->getResult(0), constop->getResult(0),
      rewriter.getI32IntegerAttr(0));

  // Epsilon
  // ConstOp for AddOp
  float eps = op->getAttr("eps").cast<FloatAttr>().getValueAsDouble();
  std::vector<float> const_value_add = {eps};
  DenseElementsAttr attr_add = DenseElementsAttr::get(
      const_ty, llvm::ArrayRef(const_value_add.data(), const_value_add.size()));
  auto constop_add =
      rewriter.create<mlir::tosa::ConstOp>(op->getLoc(), const_ty, attr_add);

  // AddOp
  auto add_op_0 = rewriter.create<mlir::tosa::AddOp>(
      op->getLoc(), out_type, mul_op_1->getResult(0), constop_add->getResult(0));

  // Var
  // RsqrtOp
  auto rsqrt_op = rewriter.create<mlir::tosa::RsqrtOp>(op->getLoc(), out_type, add_op_0->getResult(0));

  // MulOp
  auto mul_op_2 = rewriter.create<mlir::tosa::MulOp>(
      op->getLoc(), outType, sub_op->getResult(0), rsqrt_op->getResult(0),
      rewriter.getI32IntegerAttr(0));

  // MulOp
  auto mul_op_3 = rewriter.create<mlir::tosa::MulOp>(
      op->getLoc(), outType, mul_op_2->getResult(0), op->getOperand(1),
      rewriter.getI32IntegerAttr(0));

  // AddOp
  auto add_op_1 = rewriter.create<mlir::tosa::AddOp>(
      op->getLoc(), outType, mul_op_3->getResult(0), op->getOperand(2));

  // Output
  rewriter.replaceOp(op, add_op_1->getResults());
}

//===------------------------------------------------------------===//
// RMSNormLowering
//===------------------------------------------------------------===//
void RMSNormLowering::Lowering(PatternRewriter &rewriter,
                                 top::RMSNormOp op) const {
  assert(op->getNumResults() == 1);
  // auto newType = change_dataformat(op->getResult(0).getType());
  auto outType = op->getResult(0).getType();
  auto size = outType.cast<RankedTensorType>().getShape().size();
  int32_t new_axis, axis = op.getAxis();

  if (axis > 0)
    new_axis = axis;
  else
    new_axis = size + axis;

  // Params
  std::vector<NamedAttribute> attrs;
  attrs.push_back(
      rewriter.getNamedAttr("axis", rewriter.getI64IntegerAttr(new_axis)));
  std::vector<int64_t> out_shape(outType.cast<RankedTensorType>().getShape());
  out_shape[new_axis] = 1;
  auto out_type = RankedTensorType::get(
      out_shape, outType.cast<RankedTensorType>().getElementType());

  // ConstOp
  auto inType = op->getOperand(0).getType();
  std::vector<int64_t> in_shape(inType.cast<RankedTensorType>().getShape());
  std::vector<float> const_value;
  const_value.push_back(1. / in_shape[new_axis]);
  std::vector<int64_t> const_shape;
  for (int i = 0; i < size; i++) {
    const_shape.push_back(1);
  }
  auto const_ty = RankedTensorType::get(const_shape, rewriter.getF32Type());
  DenseElementsAttr attr = DenseElementsAttr::get(
      const_ty, llvm::ArrayRef(const_value.data(), const_value.size()));
  auto constop =
      rewriter.create<mlir::tosa::ConstOp>(op->getLoc(), const_ty, attr);

  // ConstOp for PowOp
  std::vector<float> const_value_pow = {2};
  DenseElementsAttr attr_pow = DenseElementsAttr::get(
      const_ty, llvm::ArrayRef(const_value_pow.data(), const_value_pow.size()));
  auto constop_pow =
      rewriter.create<mlir::tosa::ConstOp>(op->getLoc(), const_ty, attr_pow);

  // PowOp
  auto pow_op = rewriter.create<mlir::tosa::PowOp>(
      op->getLoc(), outType, op->getOperand(0), constop_pow->getResult(0));
      
  // ReduceSumOp
  auto reducesum_op = rewriter.create<mlir::tosa::ReduceSumOp>(
      op->getLoc(), out_type, pow_op->getResult(0), attrs);

  // MulOp
  auto mul_op_1 = rewriter.create<mlir::tosa::MulOp>(
      op->getLoc(), out_type, reducesum_op->getResult(0), constop->getResult(0),
      rewriter.getI32IntegerAttr(0));

  // Epsilon
  // ConstOp for AddOp
  float eps = op->getAttr("eps").cast<FloatAttr>().getValueAsDouble();
  std::vector<float> const_value_add = {eps};
  DenseElementsAttr attr_add = DenseElementsAttr::get(
      const_ty, llvm::ArrayRef(const_value_add.data(), const_value_add.size()));
  auto constop_add =
      rewriter.create<mlir::tosa::ConstOp>(op->getLoc(), const_ty, attr_add);

  // AddOp
  auto add_op_0 = rewriter.create<mlir::tosa::AddOp>(
      op->getLoc(), out_type, mul_op_1->getResult(0), constop_add->getResult(0));

  // Var
  // RsqrtOp
  auto rsqrt_op = rewriter.create<mlir::tosa::RsqrtOp>(op->getLoc(), out_type, add_op_0->getResult(0));

  // MulOp
  auto mul_op_2 = rewriter.create<mlir::tosa::MulOp>(
      op->getLoc(), outType, op->getOperand(0), rsqrt_op->getResult(0),
      rewriter.getI32IntegerAttr(0));

  // Output
  rewriter.replaceOp(op, mul_op_2->getResults());
}

//===------------------------------------------------------------===//
// SigmoidLowering
//===------------------------------------------------------------===//
void SigmoidLowering::Lowering(PatternRewriter &rewriter,
                               top::SigmoidOp op) const {
  assert(op->getNumResults() == 1);
  auto outType = op->getResult(0).getType();
  auto outShape = outType.cast<RankedTensorType>().getShape();
  rewriter.replaceOpWithNewOp<mlir::tosa::SigmoidOp>(
      op, outType, op->getOperand(0));
}

//===------------------------------------------------------------===//
// GatherLowering
//===------------------------------------------------------------===//
void GatherLowering::Lowering(PatternRewriter &rewriter,
                               top::GatherOp op) const {
  assert(op->getNumResults() == 1);
  auto outType = op->getResult(0).getType();
  auto outShape = outType.cast<RankedTensorType>().getShape();
  rewriter.replaceOpWithNewOp<mlir::tosa::GatherOp>(
      op, outType, op->getOperands());
}

} // namespace mini_mlir
