//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "mini_mlir/Dialect/Top/IR/TopOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Casting.h"
#include <numeric>

using namespace mlir;
using namespace mini_mlir::top;

//===----------------------------------------------------------------------===//
// Dialect initialize method.
//===----------------------------------------------------------------------===//
#include "mini_mlir/Dialect/Top/IR/TopOpsDialect.cpp.inc"

void TopDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mini_mlir/Dialect/Top/IR/TopOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Top Operator Definitions.
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mini_mlir/Dialect/Top/IR/TopOps.cpp.inc"