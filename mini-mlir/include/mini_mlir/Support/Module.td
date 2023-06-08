// =============================================================================
//
// Defines TOP Dialect operations.
//
//===----------------------------------------------------------------------===//

#ifndef MINI_MLIR_MODULE_ATTR
#define MINI_MLIR_MODULE_ATTR

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/BuiltinDialect.td"

def Module_State: I32EnumAttr<"State", "module state",[
      I32EnumAttrCase<"TOP_F32", 0>,
      I32EnumAttrCase<"TOP_CALIBRATED", 1>,
      I32EnumAttrCase<"TOP_QUANTIZED", 2>,
      I32EnumAttrCase<"TOSA_F32", 1000>
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mini_mlir::module";
}

def Module_Mode: I32EnumAttr<"Mode", "module mode",[
      I32EnumAttrCase<"INT8",  0>,
      I32EnumAttrCase<"UINT8", 1>,
      I32EnumAttrCase<"INT4",  2>,
      I32EnumAttrCase<"BF16",  3>,
      I32EnumAttrCase<"F16",   4>,
      I32EnumAttrCase<"F32",   5>
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mini_mlir::module";
}

def Module_Platform: I32EnumAttr<"Platform", "module platform",[
      I32EnumAttrCase<"ONNX",    0>,
      I32EnumAttrCase<"TORCH",   1>,
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mini_mlir::module";
}

#endif // MINI_MLIR_MODULE_ATTR
