//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "mini_mlir/Dialect/Tops/IR/TopsOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Casting.h"
#include <numeric>

using namespace mlir;
using namespace mlir::tops;

//===----------------------------------------------------------------------===//
// Dialect initialize method.
//===----------------------------------------------------------------------===//
#include "mini_mlir/Dialect/Tops/IR/TopsOpsDialect.cpp.inc"

void TopsDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mini_mlir/Dialect/Tops/IR/TopsOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Tops Operator Definitions.
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mini_mlir/Dialect/Tops/IR/TopsOps.cpp.inc"


// void tops::MatMulOp::parseParam(int64_t &batch, int64_t &M, int64_t &K,
//                                 int64_t &N) {
//   auto i_s = input().getType().cast<RankedTensorType>().getShape();
//   auto r_s = right().getType().cast<RankedTensorType>().getShape();
//   auto o_s = output().getType().cast<RankedTensorType>().getShape();
//   auto r_dims = r_s.size();
//   auto i_dims = i_s.size();
//   N = r_s[r_dims - 1];
//   K = r_s[r_dims - 2];
//   if (r_dims > 2) {
//     M = i_s[i_dims - 2];
//     assert(i_s[i_dims - 1] == K);
//     batch = std::accumulate(r_s.begin(), r_s.begin() + r_dims - 2, 1,
//                             std::multiplies<int64_t>());
//   } else {
//     batch = 1;
//     M = std::accumulate(i_s.begin(), i_s.begin() + i_dims - 1, 1,
//                         std::multiplies<int64_t>());
//   }
// }
