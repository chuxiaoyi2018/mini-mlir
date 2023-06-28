module attributes {top.weight_file = "gemm_top_weight.npz"} {
  func.func @main(%arg0: tensor<2x3xf32>, %arg1: tensor<3x4xf32>) -> tensor<2x4xf32> {
    %0 = "top.None"() : () -> none
    %1 = "top.Input"(%arg0) {name = "a"} : (tensor<2x3xf32>) -> tensor<2x3xf32>
    %2 = "top.Input"(%arg1) {name = "b"} : (tensor<3x4xf32>) -> tensor<3x4xf32>
    %3 = "top.MatMul"(%1, %2, %0) {do_relu = false, name = "z_Gemm"} : (tensor<2x3xf32>, tensor<3x4xf32>, none) -> tensor<2x4xf32>
    return %3 : tensor<2x4xf32>
  }
}
