module attributes {top.weight_file = "softmax_top_weight.npz"} {
  func.func @main(%arg0: tensor<2x72xf32>) -> tensor<2x72xf32> {
    %0 = "top.None"() : () -> none
    %1 = "top.Input"(%arg0) {name = "x"} : (tensor<2x72xf32>) -> tensor<2x72xf32>
    %2 = "top.Softmax"(%1) {axis = 1 : i32, name = "1_Softmax"} : (tensor<2x72xf32>) -> tensor<2x72xf32>
    return %2 : tensor<2x72xf32>
  }
}
