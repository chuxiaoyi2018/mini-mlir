#pragma once

#include "mini_mlir/Conversion/TopToTosa/TopLowering.h"

namespace mini_mlir {

void populateTopToTosaConversionINT8Patterns(
    RewritePatternSet *patterns,
    std::map<std::string, std::vector<float>> threshold_map,
    std::map<std::string, std::vector<float>> fmin_map,
    std::map<std::string, std::vector<float>> fmax_map);

#define OpLoweringINT8(OP)                                                     \
  struct OP##LoweringINT8 : public TopLoweringToTosa<top::OP##Op> {            \
  public:                                                                      \
    OP##LoweringINT8(MLIRContext *ctx,                                         \
                     std::map<std::string, std::vector<float>> threshold_map,  \
                     std::map<std::string, std::vector<float>> fmin_map,       \
                     std::map<std::string, std::vector<float>> fmax_map)       \
        : TopLoweringToTosa<top::OP##Op>(ctx), threshold_map(threshold_map),   \
          fmin_map(fmin_map), fmax_map(fmax_map) {}                            \
    void Lowering(PatternRewriter &rewriter, top::OP##Op op) const override;   \
                                                                               \
  private:                                                                     \
    std::map<std::string, std::vector<float>> threshold_map;                   \
    std::map<std::string, std::vector<float>> fmin_map;                        \
    std::map<std::string, std::vector<float>> fmax_map;                        \
  };
// clang-format off
OpLoweringINT8(Reshape)
OpLoweringINT8(Add)
OpLoweringINT8(MatMul)
OpLoweringINT8(Mul)
OpLoweringINT8(Permute)
OpLoweringINT8(GELU)
// clang-format on

mlir::Value GELULoweringINT8_v1(PatternRewriter &rewriter, mlir::Value in_value,
                                mlir::Type inType, mlir::Type outType, Location loc,
                                float in_fmin, float in_fmax, float out_fmin,
                                float out_fmax, float qmin, float qmax);

mlir::Value GELULoweringINT8_v2(PatternRewriter &rewriter, mlir::Value in_value,
                                mlir::Type inType, mlir::Type outType,
                                Location loc, float in_fmin, float in_fmax,
                                float out_fmin, float out_fmax, float qmin,
                                float qmax);

} // namespace mini_mlir
