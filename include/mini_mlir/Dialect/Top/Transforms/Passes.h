#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/QuantOps.h"
#include "mini_mlir/Dialect/Top/IR/TopOps.h"


namespace mini_mlir {
namespace top {

std::unique_ptr<OperationPass<ModuleOp>> createInitPass();
std::unique_ptr<OperationPass<ModuleOp>> createDeinitPass();
#define GEN_PASS_REGISTRATION
#define GEN_PASS_CLASSES
#include "mini_mlir/Dialect/Top/Transforms/Passes.h.inc"

} // namespace top
} // namespace mini_mlir
