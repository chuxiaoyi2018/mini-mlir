//===----------------------------------------------------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "mini_mlir/InitAll.h"
#include "mini_mlir/Dialect/Top/IR/TopOps.h"
#include "mini_mlir/Conversion/Passes.h"
#include "mini_mlir/Dialect/Top/Transforms/Passes.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"


namespace mini_mlir {
void registerAllDialects(mlir::DialectRegistry &registry) {
  registry.insert<mlir::tosa::TosaDialect, mlir::func::FuncDialect, top::TopDialect>();
  //registry.insert<mlir::StandardOpsDialect>();
}

void registerAllPasses() {
  mlir::registerConversionPasses();
  top::registerTopPasses();
  //mlir::registerCanonicalizerPass();
}
} // namespace mini_mlir

