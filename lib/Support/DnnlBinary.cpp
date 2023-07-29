#include "mini_mlir/Support/DnnlBinary.h"
#include "oneapi/dnnl/dnnl.hpp"
#include "mini_mlir/Support/DnnlUtils.h"

using namespace dnnl;

namespace mini_mlir {
Binary::Binary() {
  eng = dnnl::engine(engine::kind::cpu, 0);
  engine_stream = dnnl::stream(eng);
}

void Binary::setup() {
  primitive_attr relu_attr;
  post_relu(relu_attr, do_relu_, relu_limit_);
  auto pd =
      binary::primitive_desc(eng, algorithm_, lhs_mem.get_desc(),
                             rhs_mem.get_desc(), dst_mem.get_desc(), relu_attr);
  binary_prim = binary(pd);
}

void Binary::run() {
  binary_prim.execute(engine_stream, {{DNNL_ARG_SRC_0, lhs_mem},
                                      {DNNL_ARG_SRC_1, rhs_mem},
                                      {DNNL_ARG_DST, dst_mem}});
  engine_stream.wait();
}

} // namespace mini_mlir
