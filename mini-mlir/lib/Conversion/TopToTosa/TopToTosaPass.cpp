#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include "mini_mlir/Conversion/TopToTosa/OpLowering.h"
#include "mini_mlir/Conversion/TopToTosa/OpLoweringINT8.h"
#include "mini_mlir/Conversion/Conversion.h"
#include "mini_mlir/Support/Module.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTTOPTOTOSA
#include "mini_mlir/Conversion/Passes.h.inc"
} // namespace mlir

namespace mini_mlir {

struct LowerTopWeightOp : public OpRewritePattern<top::WeightOp> {
public:
  LowerTopWeightOp(MLIRContext *ctx, bool include_weight)
      : OpRewritePattern(ctx), include_weight(include_weight) {}

  LogicalResult matchAndRewrite(top::WeightOp op,
                                PatternRewriter &rewriter) const override {
    assert(op->getNumResults() == 1);
    auto outType = change_dataformat(op->getResult(0).getType());
    auto has_weight = include_weight;
    for (auto user : op.getOutput().getUsers()) {
      if (isa<tosa::TransposeOp>(user)) {
        has_weight = true;
      }
    }
    if (has_weight) {
      auto valptr = op.read_as_float();
      auto new_val = change_weight(valptr, op->getResult(0).getType());
      auto attr = DenseElementsAttr::get(
          outType.cast<RankedTensorType>(), llvm::ArrayRef(new_val, valptr->size()));
      rewriter.replaceOpWithNewOp<mlir::tosa::ConstOp>(op, outType, attr);
    } else {
      auto attr = DenseElementsAttr::get(
          RankedTensorType::get({}, rewriter.getI64Type()),
          llvm::ArrayRef<int64_t>({0}));
      rewriter.replaceOpWithNewOp<mlir::tosa::ConstOp>(op, outType, attr);
    }
    return success();
  }

private:
  bool include_weight;
};

struct ConvertTopToTosa
    : public ::impl::ConvertTopToTosaBase<ConvertTopToTosa> {
public:
  void runOnOperation() override {
    module_ = getOperation();
    ctx_ = &getContext();
    mainFunc_ = module::getMainFuncOp();

    RewritePatternSet patterns(ctx_);
    ConversionTarget target(*ctx_);
    target.addLegalDialect<mlir::tosa::TosaDialect, mlir::func::FuncDialect>();

    // Read Calibration Table
    if (tableFile.size() > 0) {
      std::ifstream infile(tableFile);
    }
    

    // Lower TOP Ops
    patterns.add<LowerTopWeightOp>(patterns.getContext(), includeWeight);

    // Lowering to INT8
    if (weightType == "INT8") {
      populateTopToTosaConversionINT8Patterns(&patterns);
    }
    // Lowering to FP32
    populateTopToTosaConversionPatterns(&patterns);

    auto config = GreedyRewriteConfig();
    config.maxIterations = 1;
    applyPatternsAndFoldGreedily(module_, std::move(patterns), config);
    patterns.clear();

    module::updateModuleTypes();
    module::setState(module::State::TOSA_F32);
  }

protected:
  ModuleOp module_;
  FuncOp mainFunc_;
  MLIRContext *ctx_;
  std::string table_file;
};

std::unique_ptr<Pass> createConvertTopToTosa() {
  return std::make_unique<ConvertTopToTosa>();
}

} // namespace mini_mlir