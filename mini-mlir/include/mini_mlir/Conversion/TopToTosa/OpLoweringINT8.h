#pragma once

#include "mini_mlir/Conversion/TopToTosa/TopLowering.h"

namespace mini_mlir {

void populateTopToTosaConversionINT8Patterns(RewritePatternSet *patterns, std::map<std::string, float> calibration_map);

#define OpLoweringINT8(OP)                                                       \
  struct OP##LoweringINT8 : public TopLoweringToTosa<top::OP##Op> {              \
    public:                                                                      \
      OP##LoweringINT8(MLIRContext *ctx, std::map<std::string, float> calibration_map)          \
        : TopLoweringToTosa<top::OP##Op>(ctx), calibration_map(calibration_map){}          \
      void Lowering(PatternRewriter &rewriter, top::OP##Op op) const override;   \
    private:                                                                     \
      std::map<std::string, float> calibration_map;                                             \
  };
// clang-format off
OpLoweringINT8(Add)
OpLoweringINT8(Reshape)
// clang-format on

} // namespace mini_mlir
