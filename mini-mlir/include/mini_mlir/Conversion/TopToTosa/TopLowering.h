#pragma once

#include <cmath>
#include <algorithm>
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
                                mlir::Type eleType, mlir::Location loc, float threshold) {
  // ConstOp inv_scale
  std::vector<int64_t> in_shape(inType.cast<RankedTensorType>().getShape());
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

  // CastOp fp32->int8/int32
  auto cast2int_ty = RankedTensorType::get({in_shape}, eleType);
  auto cast2int_op =
      rewriter.create<mlir::tosa::CastOp>(loc, cast2int_ty, mul_inv_scale_op->getResult(0));
  return cast2int_op;
}

static mlir::tosa::MulOp lowering_dequantize(PatternRewriter &rewriter, mlir::Value in_value, mlir::Type outType, 
                              mlir::Location loc, float scale_value) {
  std::vector<int64_t> out_shape(outType.cast<RankedTensorType>().getShape());
  
  // CastOp int->fp32
  auto cast2fp32_ty = RankedTensorType::get({out_shape}, rewriter.getF32Type());
  auto cast2fp32_op =
      rewriter.create<mlir::tosa::CastOp>(loc, cast2fp32_ty, in_value);
  
  // ConstOp scale
  int out_size = out_shape.size();
  std::vector<float> scale;
  scale.push_back(scale_value);
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


// weight lowering to INT8
static float get_weight_threshold(PatternRewriter &rewriter, top::WeightOp weight_op) {
  auto valptr = weight_op.read_as_float();
  std::vector<float> data = *valptr.get();
  std::sort(data.begin(), data.end());
  float min_value = data[static_cast<int>(data.size() * 0.001)];
  float max_value = data[static_cast<int>(data.size() * 0.999)];
  float threshold = std::min(fabs(min_value), fabs(max_value));
  return threshold;
}

static mlir::tosa::ConstOp lowering_weight_int8(PatternRewriter &rewriter, top::WeightOp weight_op, 
        mlir::Type castType, Location loc, float threshold, std::vector<int64_t> in_shape) {
  auto valptr = weight_op.read_as_float();
  float inv_scale = 127./threshold;
  std::vector<float>& val = *valptr.get(); 
  std::vector<int8_t> const_value;

  for (int i = 0; i < val.size(); i++) {
    float tmp = std::clamp(floor(val[i] * inv_scale + 0.5), -128., 127.);
    const_value.push_back(static_cast<int8_t>(tmp));
  }

  
  // ConstOp weight
  auto const_ty = RankedTensorType::get({in_shape}, castType);
  DenseElementsAttr attr = DenseElementsAttr::get(
      const_ty, llvm::ArrayRef(const_value.data(), const_value.size()));
  auto const_op =
      rewriter.create<mlir::tosa::ConstOp>(loc, const_ty, attr);

  return const_op;
}

static mlir::tosa::ConstOp lowering_weight_int32(PatternRewriter &rewriter, top::WeightOp weight_op, 
        mlir::Type castType, Location loc, float threshold, std::vector<int64_t> in_shape) {
  auto valptr = weight_op.read_as_float();
  float inv_scale = 127./threshold;
  std::vector<float>& val = *valptr.get(); 
  std::vector<int32_t> const_value;

  for (int i = 0; i < val.size(); i++) {
    float tmp = std::clamp(floor(val[i] * inv_scale + 0.5), -128., 127.);
    const_value.push_back(static_cast<int32_t>(tmp));
  }
  
  // ConstOp weight
  auto const_ty = RankedTensorType::get({in_shape}, castType);
  DenseElementsAttr attr = DenseElementsAttr::get(
      const_ty, llvm::ArrayRef(const_value.data(), const_value.size()));
  auto const_op =
      rewriter.create<mlir::tosa::ConstOp>(loc, const_ty, attr);

  return const_op;
}

template <typename Func>
static DenseElementsAttr create_lookup_table(PatternRewriter &rewriter, 
                  Location loc, float threshold, Func func) {
  int64_t min_th = 0;
  int64_t max_th = 255;

  float scale = threshold / static_cast<float>(max_th - min_th);
  float inv_scale = 1. / scale;

  std::vector<int8_t> table;
  for (auto i = min_th; i <= max_th; i++) {
    int8_t data = static_cast<int8_t>(func(i * scale) * inv_scale);
    table.push_back(data);
  }

  auto table_type = RankedTensorType::get(
      {256}, rewriter.getI8Type());
  DenseElementsAttr attr = DenseElementsAttr::get(
      table_type, llvm::ArrayRef(table.data(), table.size()));
  return attr;
}
} // namespace mini_mlir
