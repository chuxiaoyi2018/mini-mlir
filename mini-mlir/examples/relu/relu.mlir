module attributes {tops.weight_file = "relu_tops_weight.npz"} {
  func.func @main(%arg0: tensor<2x72xf32>) -> tensor<2x72xf32> {
    %0 = "tops.None"() : () -> none
    %1 = "tops.Input"(%arg0) {name = "x"} : (tensor<2x72xf32>) -> tensor<2x72xf32>
    %2 = "tops.Relu"(%1) {name = "1_Relu"} : (tensor<2x72xf32>) -> tensor<2x72xf32>
    return %2 : tensor<2x72xf32>
  }
}
