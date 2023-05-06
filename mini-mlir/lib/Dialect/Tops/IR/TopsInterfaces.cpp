//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "sophgo/Dialect/Tops/IR/TopsOps.h"
#include "sophgo/Interfaces/InferenceInterface.h"
#include "sophgo/Support/DnnlConv.h"
#include "sophgo/Support/DnnlPool.h"
#include "sophgo/Support/DnnlMatMul.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Casting.h"
#include "dnnl.hpp"
#include "omp.h"
using namespace mlir;

int omp_schedule(int count) {
  return (count + omp_get_num_threads() - 1) / omp_get_num_threads();
}

template <typename T> static void relu(T *src, T *dst, size_t size) {
#pragma omp parallel for schedule(static, omp_schedule(size))
  for (size_t i = 0; i < size; ++i) {
    dst[i] = src[i] > 0 ? src[i] : 0;
  }
}

LogicalResult tops::ReluOp::init(InferenceParameter &p) { return success(); }
void tops::ReluOp::deinit(InferenceParameter &p) {}

LogicalResult tops::ReluOp::inference(InferenceParameter &p) {
  auto num_elem = getInput().getType().cast<RankedTensorType>().getNumElements();
  relu(p.inputs[0], p.outputs[0], num_elem);
  return success();
}

// LogicalResult tops::MatMulOp::init(InferenceParameter &p) {
//   auto matmul = new dnnl::MatMul();
//   int64_t batch, M, K, N;
//   parseParam(batch, M, K, N);
//   matmul->setup(p.inputs[0], p.inputs[1], p.inputs[2], p.outputs[0], batch, M,
//                 K, N, do_relu());
//   p.handle = (void *)matmul;
//   return success();
// }

// void tops::MatMulOp::deinit(InferenceParameter &p) {
//   if (p.handle != nullptr) {
//     auto matmul = (dnnl::MatMul *)p.handle;
//     delete matmul;
//     p.handle = nullptr;
//   }
//   return;
// }

// LogicalResult tops::MatMulOp::inference(InferenceParameter &p) {
//   if (p.handle == nullptr) {
//     return failure();
//   }
//   auto matmul = (dnnl::MatMul *)p.handle;
//   matmul->run();
//   return success();
// }
