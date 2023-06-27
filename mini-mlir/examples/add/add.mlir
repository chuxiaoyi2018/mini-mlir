module attributes {top.weight_file = "add_top_weight.npz"} {
  func.func @main(%arg0: tensor<1x2x2x72xf32>, %arg1: tensor<1x2x2x72xf32>) -> tensor<1x2x2x72xf32> {
    %0 = "top.None"() : () -> none
    %1 = "top.Input"(%arg0) {name = "onnx::Add_0"} : (tensor<1x2x2x72xf32>) -> tensor<1x2x2x72xf32>
    %2 = "top.Input"(%arg1) {name = "onnx::Add_1"} : (tensor<1x2x2x72xf32>) -> tensor<1x2x2x72xf32>
    %3 = "top.Add"(%1, %2) {do_relu = false, name = "2_Add"} : (tensor<1x2x2x72xf32>, tensor<1x2x2x72xf32>) -> tensor<1x2x2x72xf32>
    return %3 : tensor<1x2x2x72xf32>
  }
}
