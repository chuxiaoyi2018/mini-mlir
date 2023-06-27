//===- Utils.h - Transform dialect utilities --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TRANSFORMS_UTILS_UTILS_H
#define MLIR_DIALECT_TRANSFORMS_UTILS_UTILS_H

#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
class OpAsmPrinter;

namespace transform {
class TransformState;

/// Printer hook for custom directive in assemblyFormat.
///
///   custom<PackedOrDynamicIndexList>($packed, $values, $integers)
///
/// where `values` are variadic Index values, `integers` is an `I64ArrayAttr`
/// and `packed` is a single transform dialect handle who's mapped payload ops
/// have a single Index result and represent the index list. Either `packed`
/// or the other two parameters may be specified.
///
/// This allows idiomatic printing of mixed value and integer attributes in a
/// list or with a single handle. E.g., `[%arg0, 7, 42, %arg42]` or just `%h`.
void printPackedOrDynamicIndexList(OpAsmPrinter &printer, Operation *op,
                                   Value packed, OperandRange values,
                                   ArrayRef<int64_t> integers);

/// Pasrer hook for custom directive in assemblyFormat.
///
///   custom<PackedOrDynamicIndexList>($packed, $values, $integers)
///
/// See `printPackedOrDynamicIndexList` for details.
ParseResult parsePackedOrDynamicIndexList(
    OpAsmParser &parser, std::optional<OpAsmParser::UnresolvedOperand> &packed,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &values,
    DenseI64ArrayAttr &integers);
} // namespace transform
} // namespace mlir

#endif // MLIR_DIALECT_TRANSFORMS_UTILS_UTILS_H
