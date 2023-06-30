module attributes {module.state = "TOSA_F32", top.weight_file = "Softmax_top_weight.npz"} {
  func.func @main(%arg0: tensor<1x2x2x72xf32>) -> tensor<1x2x72x2xf32> {
    %0 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1 = "tosa.transpose"(%arg0, %0) : (tensor<1x2x2x72xf32>, tensor<4xi32>) -> tensor<1x2x72x2xf32>
    %2 = "tosa.reduce_max"(%1) <{axis = 2 : i64}> : (tensor<1x2x72x2xf32>) -> tensor<1x2x1x2xf32>
    %3 = "tosa.sub"(%1, %2) : (tensor<1x2x72x2xf32>, tensor<1x2x1x2xf32>) -> tensor<1x2x72x2xf32>
    %4 = "tosa.exp"(%3) : (tensor<1x2x72x2xf32>) -> tensor<1x2x72x2xf32>
    %5 = "tosa.reduce_sum"(%4) <{axis = 2 : i64}> : (tensor<1x2x72x2xf32>) -> tensor<1x2x1x2xf32>
    %6 = "tosa.reciprocal"(%5) : (tensor<1x2x1x2xf32>) -> tensor<1x2x1x2xf32>
    %7 = "tosa.mul"(%4, %6) <{shift = 0 : i32}> : (tensor<1x2x72x2xf32>, tensor<1x2x1x2xf32>) -> tensor<1x2x72x2xf32>
    return %7 : tensor<1x2x72x2xf32>
  }
}

