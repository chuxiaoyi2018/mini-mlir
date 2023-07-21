#include "mini_mlir/Conversion/Conversion.h"
#include "mini_mlir/Conversion/TopToTosa/OpLowering.h"
#include "mini_mlir/Conversion/TopToTosa/OpLoweringINT8.h"
#include "mini_mlir/Support/Module.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

namespace mlir {
#define GEN_PASS_DEF_CONVERTTOPTOTOSA
#include "mini_mlir/Conversion/Passes.h.inc"
} // namespace mlir

namespace mini_mlir {

struct WeightLowering : public OpRewritePattern<top::WeightOp> {
public:
  WeightLowering(MLIRContext *ctx, bool include_weight)
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
      auto attr =
          DenseElementsAttr::get(outType.cast<RankedTensorType>(),
                                 llvm::ArrayRef(new_val, valptr->size()));
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
    std::ifstream infile(this->tableFile);
    if (!infile) {
      llvm_unreachable("can't open calibration table file!");
    }

    std::map<std::string, float> threshold_map;
    std::map<std::string, float> fmin_map;
    std::map<std::string, float> fmax_map;

    std::string line;
    while (std::getline(infile, line)) {
      std::stringstream linestream(line);
      std::string name, threshold, fmin, fmax;
      std::getline(linestream, name, ',');
      std::getline(linestream, threshold, ',');
      std::getline(linestream, fmin, ',');
      std::getline(linestream, fmax, ',');

      threshold_map[name] = atof(threshold.c_str());
      fmin_map[name] = atof(fmin.c_str());
      fmax_map[name] = atof(fmax.c_str());
    }

    infile.close();

    std::map<std::string, std::vector<float>> threshold_map_with_parent;
    std::map<std::string, std::vector<float>> fmin_map_with_parent;
    std::map<std::string, std::vector<float>> fmax_map_with_parent;

    auto mainFunc = module::getMainFuncOp();
    mainFunc.walk([&](Operation *op) {
      if (!isa<top::NoneOp, top::InputOp, ReturnOp, FuncOp>(op)) {
        std::vector<float> threshold_vec{0., 0., 0., 0.};
        std::vector<float> fmin_vec{0., 0., 0., 0.};
        std::vector<float> fmax_vec{0., 0., 0., 0.};
        int operand_num = static_cast<int>(op->getNumOperands());
        for (int i = 0; i < 2 && operand_num > 0; i++, operand_num--) {
          std::string operand_name = op->getOperand(i)
                                         .getDefiningOp()
                                         ->getAttr("name")
                                         .cast<StringAttr>()
                                         .getValue()
                                         .str();
          if (threshold_map.find(operand_name) != threshold_map.end()) {
            threshold_vec[i] = threshold_map.at(operand_name);
            fmin_vec[i] = fmin_map.at(operand_name);
            fmax_vec[i] = fmax_map.at(operand_name);
          }
        }
        std::string node_name =
            op->getAttr("name").cast<StringAttr>().getValue().str();
        if (threshold_map.find(node_name) != threshold_map.end() &&
            threshold_vec[0] != 0 && threshold_vec[0] < 20 &&
            threshold_vec[3] < 20) {
          threshold_vec[threshold_vec.size() - 1] = threshold_map.at(node_name);
          fmin_vec[threshold_vec.size() - 1] = fmin_map.at(node_name);
          fmax_vec[threshold_vec.size() - 1] = fmax_map.at(node_name);

          threshold_map_with_parent[node_name] = threshold_vec;
          fmin_map_with_parent[node_name] = fmin_vec;
          fmax_map_with_parent[node_name] = fmax_vec;
        }
      }
    });

    auto config = GreedyRewriteConfig();
    config.maxIterations = 1;

    // Match Order: int8 -> fp32 -> weight
    // Lowering to INT8
    if (weightType == "INT8") {
      populateTopToTosaConversionINT8Patterns(
          &patterns, threshold_map_with_parent, fmin_map_with_parent,
          fmax_map_with_parent);
      applyPatternsAndFoldGreedily(module_, std::move(patterns), config);
      patterns.clear();
    }

    // Lowering to FP32
    populateTopToTosaConversionPatterns(&patterns);
    applyPatternsAndFoldGreedily(module_, std::move(patterns), config);
    patterns.clear();

    // Lower weight
    patterns.add<WeightLowering>(patterns.getContext(), includeWeight);
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