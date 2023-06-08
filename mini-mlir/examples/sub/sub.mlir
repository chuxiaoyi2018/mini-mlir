module attributes {top.weight_file = "sub_top_weight.npz"} {
  func.func @main(%arg0: tensor<2x72xf32>, %arg1: tensor<2x72xf32>) -> tensor<2x72xf32> {
    %0 = "top.None"() : () -> none
    %1 = "top.Input"(%arg0) {name = "onnx::Sub_0"} : (tensor<2x72xf32>) -> tensor<2x72xf32>
    %2 = "top.Input"(%arg1) {name = "onnx::Sub_1"} : (tensor<2x72xf32>) -> tensor<2x72xf32>
    %3 = "top.Sub"(%1, %2) {do_relu = false, name = "2_Sub"} : (tensor<2x72xf32>, tensor<2x72xf32>) -> tensor<2x72xf32>
    return %3 : tensor<2x72xf32>
  }
}
