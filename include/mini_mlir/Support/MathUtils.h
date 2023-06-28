#pragma once

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

namespace mini_mlir {

template <typename T>
std::vector<int64_t> shape_expand_dim(llvm::ArrayRef<T> shape, int dims);
template <typename T>
std::vector<int64_t> shape_expand_dim(const std::vector<T> &shape, int dims);

} // namespace mini_mlir
