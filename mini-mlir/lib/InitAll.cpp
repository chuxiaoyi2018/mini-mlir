//===----------------------------------------------------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "mini_mlir/InitAll.h"
#include "mini_mlir/Dialect/Tops/IR/TopsOps.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dialect.h"

void mlir::mini_mlir::registerAllDialects(mlir::DialectRegistry &registry) {
  registry.insert<mlir::func::FuncDialect, mlir::tops::TopsDialect>();
  //registry.insert<mlir::StandardOpsDialect>();
}

void mlir::mini_mlir::registerAllPasses() {
  //mlir::registerCanonicalizerPass();
}
