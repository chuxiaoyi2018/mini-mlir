//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "mini_mlir/Dialect/Tops/IR/TopsOps.h"
#include "mini_mlir/Interfaces/InferenceInterface.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Casting.h"
#include "dnnl.hpp"
#include "omp.h"

// Add
#include "mini_mlir/Support/DnnlBinary.h"
#include "mini_mlir/Support/MathUtils.h"


using namespace mlir;

int omp_schedule(int count) {
  return (count + omp_get_num_threads() - 1) / omp_get_num_threads();
}


// relu
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


// Add
// see lib/Support/MathUtils.cpp   bianry_add
LogicalResult tops::AddOp::init(InferenceParameter &p) {
  if (getInputs().size() == 2) {
    auto binary = new mini_mlir::Binary();
    auto lhs_shape = getInputs()[0].getType().cast<RankedTensorType>().getShape();
    auto rhs_shape = getInputs()[1].getType().cast<RankedTensorType>().getShape();

    (*binary)
        .hs(p.inputs[0], p.inputs[1], lhs_shape, rhs_shape)
        .dst(p.outputs[0], getOutput().getType().cast<RankedTensorType>().getShape())
        .do_relu(getDoRelu())
        .relu_limit(getReluLimit().convertToDouble())
        .algorithem(algorithm::binary_add)
        .setup();

    p.handle = (void *)binary;
  } else {
    p.handle = nullptr;
  }
  return success();
}
void tops::AddOp::deinit(InferenceParameter &p) {
  if (p.handle != nullptr) {
    auto binary = (mini_mlir::Binary *)p.handle;
    delete binary;
    p.handle = nullptr;
  }
}

LogicalResult tops::AddOp::inference(InferenceParameter &p) {
  if (getInputs().size() == 2) {
    if (p.handle == nullptr) {
      return failure();
    }
    auto binary = (mini_mlir::Binary *)p.handle;
    binary->run();
  } else {
    auto num_elem = getOutput().getType().cast<RankedTensorType>().getNumElements();
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
    for (int64_t i = 0; i < num_elem; i++) {
      p.outputs[0][i] = 0;
      for (auto in : p.inputs) {
        if (in != nullptr) {
          p.outputs[0][i] += in[i];
        }
      }
    }
  }
  return success();
}


// Sub
// see third_party/oneDNN/include/oneapi/dnnl/dnnl.hpp   bianry_sub
LogicalResult tops::SubOp::init(InferenceParameter &p) {
  if (getInputs().size() == 2) {
    auto binary = new mini_mlir::Binary();
    auto lhs_shape = getInputs()[0].getType().cast<RankedTensorType>().getShape();
    auto rhs_shape = getInputs()[1].getType().cast<RankedTensorType>().getShape();

    (*binary)
        .hs(p.inputs[0], p.inputs[1], lhs_shape, rhs_shape)
        .dst(p.outputs[0], getOutput().getType().cast<RankedTensorType>().getShape())
        .do_relu(getDoRelu())
        .relu_limit(getReluLimit().convertToDouble())
        .algorithem(algorithm::binary_sub)
        .setup();

    p.handle = (void *)binary;
  } else {
    p.handle = nullptr;
  }
  return success();
}
void tops::SubOp::deinit(InferenceParameter &p) {
  if (p.handle != nullptr) {
    auto binary = (mini_mlir::Binary *)p.handle;
    delete binary;
    p.handle = nullptr;
  }
}

LogicalResult tops::SubOp::inference(InferenceParameter &p) {
  if (p.handle == nullptr) {
    return failure();
  }
  auto binary = (mini_mlir::Binary *)p.handle;
  binary->run();
  return success();
}


// Div
// see third_party/oneDNN/include/oneapi/dnnl/dnnl.hpp   bianry_div
LogicalResult tops::DivOp::init(InferenceParameter &p) {
  if (getInputs().size() == 2) {
    auto binary = new mini_mlir::Binary();
    auto lhs_shape = getInputs()[0].getType().cast<RankedTensorType>().getShape();
    auto rhs_shape = getInputs()[1].getType().cast<RankedTensorType>().getShape();

    (*binary)
        .hs(p.inputs[0], p.inputs[1], lhs_shape, rhs_shape)
        .dst(p.outputs[0], getOutput().getType().cast<RankedTensorType>().getShape())
        .do_relu(getDoRelu())
        .relu_limit(getReluLimit().convertToDouble())
        .algorithem(algorithm::binary_div)
        .setup();

    p.handle = (void *)binary;
  } else {
    p.handle = nullptr;
  }
  return success();
}
void tops::DivOp::deinit(InferenceParameter &p) {
  if (p.handle != nullptr) {
    auto binary = (mini_mlir::Binary *)p.handle;
    delete binary;
    p.handle = nullptr;
  }
}

LogicalResult tops::DivOp::inference(InferenceParameter &p) {
  if (p.handle == nullptr) {
    return failure();
  }
  auto binary = (mini_mlir::Binary *)p.handle;
  binary->run();
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
