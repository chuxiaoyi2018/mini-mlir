module attributes {module.FLOPs = 144 : i64, module.chip = "ALL", module.name = "add", module.platform = "ONNX", module.state = "TOSA_F32", module.weight_file = "add_top_f32_all_weight.npz"} {
  func.func @main(%arg0: tensor<2x72xf32>, %arg1: tensor<2x72xf32>) -> tensor<2x72xf32> {
    %0 = "tosa.const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
    %1 = "tosa.transpose"(%arg0, %0) : (tensor<2x72xf32>, tensor<4xi32>) -> tensor<2x72xf32>
    %2 = "tosa.transpose"(%arg1, %0) : (tensor<2x72xf32>, tensor<4xi32>) -> tensor<2x72xf32>
    %3 = "tosa.add"(%1, %2) : (tensor<2x72xf32>, tensor<2x72xf32>) -> tensor<2x72xf32>
    return %3 : tensor<2x72xf32>
  }
}

