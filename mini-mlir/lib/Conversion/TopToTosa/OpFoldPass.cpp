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
    applyPatternsAndFoldGreedily(module_, std::move(patterns), config);
    patterns.clear();

    module::updateModuleTypes();
    module::setState(module::State::TOSA_F32);
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