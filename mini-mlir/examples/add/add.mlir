#loc = loc(unknown)
module attributes {module.FLOPs = 144 : i64, module.chip = "ALL", module.name = "add", module.platform = "ONNX", module.state = "TOP_F32", module.weight_file = "add_top_f32_all_weight.npz"} {
  func.func @main(%arg0: tensor<2x72xf32> loc(unknown), %arg1: tensor<2x72xf32> loc(unknown)) -> tensor<2x72xf32> {
    %0 = "top.Input"(%arg0) {name = "onnx::Add_0"}: (tensor<2x72xf32>) -> tensor<2x72xf32> loc(#loc1)
    %1 = "top.Input"(%arg1) {name = "onnx::Add_1"}: (tensor<2x72xf32>) -> tensor<2x72xf32> loc(#loc2)
    %2 = "top.Add"(%0, %1) {name = "2_Add", do_relu = false, relu_limit = -1.000000e+00 : f64} : (tensor<2x72xf32>, tensor<2x72xf32>) -> tensor<2x72xf32> loc(#loc3)
    return %2 : tensor<2x72xf32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("onnx::Add_0")
#loc2 = loc("onnx::Add_1")
#loc3 = loc("2_Add")
