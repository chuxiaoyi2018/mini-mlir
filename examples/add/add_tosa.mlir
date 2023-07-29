module attributes {module.state = "TOSA_F32", top.weight_file = "add_top_weight.npz"} {
  func.func @main(%arg0: tensor<1x2x2x72xf32>, %arg1: tensor<1x2x2x72xf32>) -> tensor<1x2x72x2xf32> {
    %0 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1 = "tosa.transpose"(%arg0, %0) : (tensor<1x2x2x72xf32>, tensor<4xi32>) -> tensor<1x2x72x2xf32>
    %2 = "tosa.transpose"(%arg1, %0) : (tensor<1x2x2x72xf32>, tensor<4xi32>) -> tensor<1x2x72x2xf32>
    %3 = "tosa.add"(%1, %2) : (tensor<1x2x72x2xf32>, tensor<1x2x72x2xf32>) -> tensor<1x2x72x2xf32>
    return %3 : tensor<1x2x72x2xf32>
  }
}

