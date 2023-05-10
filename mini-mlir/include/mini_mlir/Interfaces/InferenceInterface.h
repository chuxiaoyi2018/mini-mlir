//===- Inference.h - Interface for tiling operations ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the definitions of the TilingInterface defined in
// `TilingInterface.td`.
//
//===----------------------------------------------------------------------===//

#ifndef MINI_INTERFACES_INFERENCEINTERFACE_H_
#define MINI_INTERFACES_INFERENCEINTERFACE_H_

// #include "mlir/IR/Builders.h"
// #include "mlir/IR/BuiltinTypes.h"
// #include "mlir/IR/Operation.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir {
  struct InferenceParameter {
    std::vector<float*> inputs;
    std::vector<float*> outputs;
    void * handle = nullptr;
  };
}
/// Include the ODS generated interface header files.
#include "mini_mlir/Interfaces/InferenceInterface.h.inc"

#endif // MINI_INTERFACES_INFERENCEINTERFACE_H_
