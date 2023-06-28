#include "mini_mlir/Dialect/Top/Transforms/Passes.h"
#include "mini_mlir/Support/Module.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <set>

using namespace llvm;


namespace mini_mlir {
namespace top {

class InitPass : public InitBase<InitPass> {
public:
  InitPass() {}
  void runOnOperation() override {
    auto mOp = getOperation();
    module::init(mOp);
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createInitPass() {
  return std::make_unique<InitPass>();
}
} // namespace top
} // namespace mini_mlir
