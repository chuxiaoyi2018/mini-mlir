#include "mini_mlir/Conversion/Conversion.h"
#include "mini_mlir/Conversion/TopToTosa/OpFold.h"
#include "mini_mlir/Support/Module.h"

namespace mlir {
#define GEN_PASS_DEF_TOSAOPFOLDPASS
#include "mini_mlir/Conversion/Passes.h.inc"
} // namespace mlir

namespace mini_mlir {

struct TosaOpFoldPass
    : public ::impl::TosaOpFoldPassBase<TosaOpFoldPass> {
public:
  void runOnOperation() override {
    module_ = getOperation();
    ctx_ = &getContext();
    mainFunc_ = module::getMainFuncOp();

    RewritePatternSet patterns(ctx_);
    ConversionTarget target(*ctx_);
    target.addLegalDialect<mlir::tosa::TosaDialect, mlir::func::FuncDialect>();

    auto config = GreedyRewriteConfig();
    config.maxIterations = 1;

    populateTosaFoldDoubleReciprocalPatterns(&patterns);

    // // Match Order: int8 -> fp32 -> weight
    // // Lowering to INT8
    // if (weightType == "INT8") {
    //   populateTopToTosaConversionINT8Patterns(
    //       &patterns, threshold_map_with_parent, fmin_map_with_parent,
    //       fmax_map_with_parent);
    //   applyPatternsAndFoldGreedily(module_, std::move(patterns), config);
    //   patterns.clear();
    // }

    // // Lowering to FP32
    // populateTopToTosaConversionPatterns(&patterns);
    // applyPatternsAndFoldGreedily(module_, std::move(patterns), config);
    // patterns.clear();

    // // Lower weight
    // patterns.add<WeightLowering>(patterns.getContext(), includeWeight);
    // applyPatternsAndFoldGreedily(module_, std::move(patterns), config);
    // patterns.clear();

    // module::updateModuleTypes();
    // module::setState(module::State::TOSA_F32);
  }

protected:
  ModuleOp module_;
  FuncOp mainFunc_;
  MLIRContext *ctx_;
};

std::unique_ptr<Pass> createTosaOpFoldPass() {
  return std::make_unique<TosaOpFoldPass>();
}

} // namespace mini_mlir