#ifndef MINI_MLIR_CONVERSION_PASSES
#define MINI_MLIR_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// TopToTosa
//===----------------------------------------------------------------------===//
def ConvertTopToTosa : Pass<"convert-top-to-tosa", "ModuleOp"> {
  let summary = "Convert top-level Top Ops to Tosa Ops";
  let constructor = "mini_mlir::createConvertTopToTosa()";
  let dependentDialects =
      [ "mini_mlir::top::TopDialect", "mlir::tosa::TosaDialect"];
  let options = [
    Option<"includeWeight", "includeWeight", "bool", /*default=*/"false",
           "true for including weight datas in tosa.mlir, or false for not">,
    Option<"weightType", "weightType", "std::string", /*default=*/"",
           "whether to use quantization, you can choose FP32 or INT8">,
    Option<"tableFile", "tableFile", "std::string", /*default=*/"",
           "calibration table file path">,
    Option<"isAsymmetric", "isAsymmetric", "bool", /*default=*/"true",
           "true for asymmetric quantization, or false for symmetric">,
  ];
}

//===----------------------------------------------------------------------===//
// TopToTosa
//===----------------------------------------------------------------------===//
def TosaOpFoldPass : Pass<"tosa-op-fold", "ModuleOp"> {
  let summary = "Fold layerwise operations on tosa";
  let description = [{
    Pass that enables folding of full-layer operations.
  }];
  let dependentDialects =
      [ "mini_mlir::top::TopDialect", "mlir::tosa::TosaDialect"];
  let constructor = "mini_mlir::createTosaOpFoldPass()";
}

#endif // MINI_MLIR_CONVERSION_PASSES
