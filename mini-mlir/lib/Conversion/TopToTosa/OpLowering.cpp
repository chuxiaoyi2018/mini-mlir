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
        ConcatLowering
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
      if (in.getType().isa<mlir::NoneType>()){  //bias
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
      rewriter.replaceOpWithNewOp<mlir::tosa::Conv2DOp>(op, newType, operands,
                                                        attrs);
    }
  }
  // TODO: support for group conv
  else
    ;
}

//===------------------------------------------------------------===//
// ReshapeLowering
//===------------------------------------------------------------===//
void ReshapeLowering::Lowering(PatternRewriter &rewriter,
                               top::ReshapeOp op) const {
  assert(op->getNumResults() == 1);
  auto newType = change_dataformat(op->getResult(0).getType());
  auto newShape = newType.cast<RankedTensorType>().getShape();
  //auto attr = rewriter.getNamedAttr("new_shape", rewriter.getDenseI64ArrayAttr(newShape));
  rewriter.replaceOpWithNewOp<mlir::tosa::ReshapeOp>(op, newType, op->getOperand(0), newShape);
}

//===------------------------------------------------------------===//
// PermuteLowering
//===------------------------------------------------------------===//
void PermuteLowering::Lowering(PatternRewriter &rewriter,
                               top::PermuteOp op) const {
  assert(op->getNumResults() == 1);
  auto outType = change_dataformat(op->getResult(0).getType());

  std::vector<Value> operands;
  operands.push_back(op->getOperand(0));

  auto order = module::getI64Array(op.getOrder());
  std::vector<int64_t>& ord = *order;
  int ord_size = ord.size();

  std::vector<int64_t> perms;
  for (int i=0; i < ord_size; i++) {
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
  auto outType = change_dataformat(op->getResult(0).getType());
  auto preType = op->getResult(0).getType();
  auto size = preType.cast<RankedTensorType>().getShape().size();
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
  rewriter.replaceOpWithNewOp<mlir::tosa::ConcatOp>(op, outType, operands, attrs);
}

//===------------------------------------------------------------===//
// SoftmaxLowering
//===------------------------------------------------------------===//
void SoftmaxLowering::Lowering(PatternRewriter &rewriter,
                               top::SoftmaxOp op) const {
  assert(op->getNumResults() == 1);
  auto preType = op->getResult(0).getType();
  auto newType = change_dataformat(preType);
  auto size = preType.cast<RankedTensorType>().getShape().size();
  int32_t new_axis, axis = op.getAxis();
  if (size == 4) {
    if (axis == 1 || axis == -3)
      new_axis = 3; // C
    else if (axis == 2 || axis == -2)
      new_axis = 1; // H
    else if (axis == 3 || axis == -1)
      new_axis = 2; // W
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


} // namespace mini_mlir
