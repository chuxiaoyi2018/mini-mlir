#include "mini_mlir/Dialect/Top/Transforms/Passes.h"
#include "mini_mlir/Support/Module.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace llvm;


namespace mini_mlir {
namespace top {

class DeinitPass : public DeinitBase<DeinitPass> {
public:
  DeinitPass() {}
  void runOnOperation() override {
    auto state = module::getState();
    if (state >= module::State::TOSA_F32) {
      return;
    }
    module::removeUnusedOp();
    module::saveWeight();
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createDeinitPass() {
  return std::make_unique<DeinitPass>();
}
} // namespace top
} // namespace mini_mlir
