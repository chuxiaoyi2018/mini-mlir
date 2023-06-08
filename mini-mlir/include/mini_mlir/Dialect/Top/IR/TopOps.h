//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mini_mlir/Interfaces/InferenceInterface.h"
#include "mini_mlir/Support/Module.h"

#include "mini_mlir/Dialect/Top/IR/TopOpsDialect.h.inc"
#define GET_OP_CLASSES
#include "mini_mlir/Dialect/Top/IR/TopOps.h.inc"

