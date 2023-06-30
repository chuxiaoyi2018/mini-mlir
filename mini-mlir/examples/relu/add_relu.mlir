module attributes {module.state = "TOSA_F32", top.weight_file = "relu_top_weight.npz"} {
  func.func @main(%arg0: tensor<1x2x2x72xf32>) -> tensor<1x2x2x72xf32> {
    %0 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1 = "tosa.transpose"(%arg0, %0) : (tensor<1x2x2x72xf32>, tensor<4xi32>) -> tensor<1x2x72x2xf32>
    %2 = "top.Relu"(%1) {name = "1_Relu", relu_limit = -1.000000e+00 : f64} : (tensor<1x2x72x2xf32>) -> tensor<1x2x2x72xf32>
    return %2 : tensor<1x2x2x72xf32>
  }
}

