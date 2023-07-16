#pragma once

#include "mini_mlir/Dialect/Top/IR/TopOps.h"
#include "mini_mlir/Support/Module.h"


#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace llvm;

namespace mini_mlir {

template <typename OpTy>
class TopLoweringToTosa : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy opTy,
                                PatternRewriter &rewriter) const override {
    Lowering(rewriter, opTy);
    return success();
  }

  virtual void Lowering(PatternRewriter &rewriter, OpTy opTy) const {
    llvm_unreachable("Not Implemented");
  }
};

// NCHW -> NHWC
static Type change_dataformat(Type ty_){
  auto ty = ty_.cast<RankedTensorType>();
  if (ty.getShape().size() != 4) return ty;
  auto n = ty.getShape()[0]; // N
  auto h = ty.getShape()[2]; // H
  auto w = ty.getShape()[3]; // W
  auto c = ty.getShape()[1]; // C
  std::vector<int64_t> newShape{n, h, w, c};
  return RankedTensorType::get(newShape, ty.getElementType());
}

// NCH -> NHC
static Type change_dataformat_3D(Type ty_){
  auto ty = ty_.cast<RankedTensorType>();
  if (ty.getShape().size() != 3) return ty;
  auto n = ty.getShape()[0]; // N
  auto h = ty.getShape()[2]; // H
  auto c = ty.getShape()[1]; // C
  std::vector<int64_t> newShape{n, h, c};
  return RankedTensorType::get(newShape, ty.getElementType());
}


// reorder weight for tosa  [N,C,H,W] -> [N,H,W,C]
static float* change_weight(std::shared_ptr<std::vector<float>> valptr, 
                                    Type ty_) {
  auto ty = ty_.cast<RankedTensorType>();
  if (ty.getShape().size() != 4) return valptr->data();
  auto n = ty.getShape()[0]; 
  auto h = ty.getShape()[2]; 
  auto w = ty.getShape()[3]; 
  auto c = ty.getShape()[1]; 
  float* new_val = new float[valptr->size()];
  int dst, src, ds_1, d_2, d_3, s_3;
  int a_ds = h*w*c, b_d = w*c, b_s = h*w;
  for (int i = 0; i < n; i++) {
    ds_1 = i * a_ds; 
    for (int j = 0; j < h; j++) {
      d_2 = j * b_d;
      s_3 = j * w;
      for (int k = 0; k < w; k++) {
        d_3 = k * c;
        for (int p = 0; p < c; p++){
          dst = ds_1 + d_2   + d_3 + p;    // nhwc
          src = ds_1 + p*b_s + s_3 + k;    // nchw
          new_val[dst] = valptr->data()[src];
        }
      }
    }
  }
  return new_val;
}

static std::vector<NamedAttribute> gen_clamp_attr(PatternRewriter &rewriter,
                      Type newType, ::llvm::APFloat relu_limit) {
  std::vector<NamedAttribute> clamp_attr;
  clamp_attr.push_back(rewriter.getNamedAttr("min_int", rewriter.getI64IntegerAttr(0)));
  clamp_attr.push_back(rewriter.getNamedAttr("max_int", rewriter.getI64IntegerAttr(0)));
  clamp_attr.push_back(rewriter.getNamedAttr("min_fp", rewriter.getF32FloatAttr(0)));
  auto floatType = newType.cast<RankedTensorType>().getElementType().cast<FloatType>();
  const llvm::fltSemantics &semantic = floatType.getFloatSemantics();
  auto zero = llvm::APFloat::getZero(relu_limit.getSemantics());   // Negative = false
  if (relu_limit < zero) {
    clamp_attr.push_back(rewriter.getNamedAttr("max_fp",
        rewriter.getFloatAttr(floatType, APFloat::getInf(semantic)))); // Negative = false
  } else {
    clamp_attr.push_back(rewriter.getNamedAttr("max_fp",
        rewriter.getFloatAttr(floatType, relu_limit)));
  }
  return clamp_attr;
}



static mlir::tosa::CastOp lowering_quantize(PatternRewriter &rewriter, mlir::Value in_value, mlir::Type inType, 
                                     mlir::Location loc, std::vector<int64_t> in_shape, float threshold) {
  // ConstOp inv_scale
  int in_size = in_shape.size();
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
      rewriter.create<mlir::tosa::ConstOp>(loc, const_ty, attr);

  // MulOp for int8
  auto mul_inv_scale_op = rewriter.create<mlir::tosa::MulOp>(loc, inType,
      in_value, const_inv_scale->getResult(0), rewriter.getI32IntegerAttr(0));

  // CastOp fp32->int8
  auto cast2int8_ty = RankedTensorType::get({in_shape}, rewriter.getI8Type());
  auto cast2int8_op =
      rewriter.create<mlir::tosa::CastOp>(loc, cast2int8_ty, mul_inv_scale_op->getResult(0));
  return cast2int8_op;
}

static mlir::tosa::MulOp lowering_dequantize(PatternRewriter &rewriter, mlir::Value in_value, mlir::Type outType, 
                                      mlir::Location loc, std::vector<int64_t> out_shape, float threshold) {
  // CastOp int8->fp32
  auto cast2fp32_ty = RankedTensorType::get({out_shape}, rewriter.getF32Type());
  auto cast2fp32_op =
      rewriter.create<mlir::tosa::CastOp>(loc, cast2fp32_ty, in_value);
  
  // ConstOp scale
  int out_size = out_shape.size();
  std::vector<float> scale;
  scale.push_back(threshold/127.);
  std::vector<int64_t> const_shape;
  for (int i = 0; i < out_size ; i++) {
    const_shape.push_back(1);
  }
  auto const_ty = RankedTensorType::get({const_shape}, rewriter.getF32Type());
  DenseElementsAttr attr = DenseElementsAttr::get(
      const_ty, llvm::ArrayRef(scale.data(), scale.size()));
  auto const_scale =
      rewriter.create<mlir::tosa::ConstOp>(loc, const_ty, attr);

  // MulOp for fp32
  auto mul_scale_op = rewriter.create<mlir::tosa::MulOp>(loc, outType, 
      cast2fp32_op->getResult(0), const_scale->getResult(0), rewriter.getI32IntegerAttr(0));
  return mul_scale_op;
}

static float weight_threshold(top::WeightOp weight_op) {
  auto valptr = weight_op.read_as_float();
  std::vector<float> data = *valptr.get();
  std::sort(data.begin(), data.end());
  int index = data.size() * 0.95;
  float percentile = data[index];
  return percentile;
}

} // namespace mini_mlir
