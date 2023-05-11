module attributes {tops.weight_file = "sub_tops_weight.npz"} {
  func.func @main(%arg0: tensor<2x72xf32>, %arg1: tensor<2x72xf32>) -> tensor<2x72xf32> {
    %0 = "tops.None"() : () -> none
    %1 = "tops.Input"(%arg0) {name = "onnx::Sub_0"} : (tensor<2x72xf32>) -> tensor<2x72xf32>
    %2 = "tops.Input"(%arg1) {name = "onnx::Sub_1"} : (tensor<2x72xf32>) -> tensor<2x72xf32>
    %3 = "tops.Sub"(%1, %2) {do_relu = false, name = "2_Sub", relu_limit = -1.000000e+00 : f64} : (tensor<2x72xf32>, tensor<2x72xf32>) -> tensor<2x72xf32>
    return %3 : tensor<2x72xf32>
  }
}

