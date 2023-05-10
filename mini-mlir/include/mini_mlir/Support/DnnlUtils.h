#pragma once
#include "oneapi/dnnl/dnnl.hpp"
using namespace dnnl;
namespace mini_mlir {

void post_relu(primitive_attr &attr, bool &do_relu, double &relu_limit);
} // namespace mini_mlir
