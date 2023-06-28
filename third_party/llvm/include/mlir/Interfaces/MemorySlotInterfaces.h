//===-- Mem2RegInterfaces.h - Mem2Reg interfaces ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_MEMORYSLOTINTERFACES_H
#define MLIR_INTERFACES_MEMORYSLOTINTERFACES_H

#include "mlir/IR/Dominance.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir {

/// Represents a slot in memory. This is generated by an allocating operation
/// (for example alloca).
struct MemorySlot {
  /// Pointer to the memory slot, used by operations to refer to it.
  Value ptr;
  /// Type of the value contained in the slot.
  Type elemType;
};

/// Returned by operation promotion logic requesting the deletion of an
/// operation.
enum class DeletionKind {
  /// Keep the operation after promotion.
  Keep,
  /// Delete the operation after promotion.
  Delete,
};

} // namespace mlir

#include "mlir/Interfaces/MemorySlotOpInterfaces.h.inc"

#endif // MLIR_INTERFACES_MEMORYSLOTINTERFACES_H
