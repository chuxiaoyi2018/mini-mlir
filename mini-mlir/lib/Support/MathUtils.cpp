#include "mini_mlir/Support/MathUtils.h"
#include "mini_mlir/Support/DnnlBinary.h"
#include "float.h"
#include "mlir/IR/PatternMatch.h"
#include "omp.h"
#include "llvm/Support/Debug.h"
#include <algorithm>
#include <map>
#include <numeric>
#include <queue>

#define DEBUG_TYPE "math_utils"
namespace mini_mlir {



template <typename T>
std::vector<int64_t> shape_expand_dim(const std::vector<T> &shape, int dims) {
  int diff = dims - shape.size();
  std::vector<int64_t> shape_v(shape.begin(), shape.end());
  if (diff == 0)
    return shape_v;
  shape_v.insert(shape_v.begin(), diff, 1);
  return shape_v;
}
template std::vector<int64_t> shape_expand_dim(const std::vector<float> &shape,
                                               int dims);
template std::vector<int64_t>
shape_expand_dim(const std::vector<int64_t> &shape, int dims);

template <typename T>
std::vector<int64_t> shape_expand_dim(llvm::ArrayRef<T> shape, int dims) {
  int diff = dims - shape.size();
  std::vector<int64_t> shape_v(shape.begin(), shape.end());
  if (diff == 0)
    return shape_v;
  shape_v.insert(shape_v.begin(), diff, 1);
  return shape_v;
}
template std::vector<int64_t> shape_expand_dim(llvm::ArrayRef<float> shape,
                                               int dims);
template std::vector<int64_t> shape_expand_dim(llvm::ArrayRef<int64_t> shape,
                                               int dims);




std::shared_ptr<std::vector<float>>
binary_add(float *a, float *b, const llvm::ArrayRef<int64_t> &a_shape,
           const llvm::ArrayRef<int64_t> &b_shape,
           std::vector<int64_t> &o_shape) {
  auto max_ndim = std::max(a_shape.size(), b_shape.size());
  auto a_shape_ = shape_expand_dim(a_shape, max_ndim);
  auto b_shape_ = shape_expand_dim(b_shape, max_ndim);
  o_shape.clear();
  for (int i = 0; i < max_ndim; i++) {
    o_shape.push_back(std::max(a_shape_[i], b_shape_[i]));
  }
  auto num_output = std::accumulate(o_shape.begin(), o_shape.end(), 1, std::multiplies<int64_t>());
  auto output = std::make_shared<std::vector<float>>(num_output);
  Binary add;
  add.lhs(a, a_shape_)
      .rhs(b, b_shape_)
      .dst(output->data(),o_shape)
      .algorithem(algorithm::binary_add)
      .setup();
  add.run();
  return std::move(output);
}


} // namespace mini_mlir
