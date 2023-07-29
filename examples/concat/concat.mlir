module attributes {top.weight_file = "concat_top_weight.npz"} {
  func.func @main(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>) -> tensor<2x6xf32> {
    %0 = "top.None"() : () -> none
    %1 = "top.Input"(%arg0) {name = "onnx::Concat_0"} : (tensor<2x3xf32>) -> tensor<2x3xf32>
    %2 = "top.Input"(%arg1) {name = "onnx::Concat_1"} : (tensor<2x3xf32>) -> tensor<2x3xf32>
    %3 = "top.Concat"(%1, %2) {axis = 1 : i32, name = "2_Concat"} : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x6xf32>
    return %3 : tensor<2x6xf32>
  }
}
