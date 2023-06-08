#ifndef MINI_MLIR_DIALECT_TOPPASSES
#define MINI_MLIR_DIALECT_TOPPASSES

include "mlir/Pass/PassBase.td"

def Init : Pass<"init", "ModuleOp"> {
  let summary = "init module";
  let constructor = "createInitPass()";
  let dependentDialects = ["TopDialect"];
}

def Deinit : Pass<"deinit", "ModuleOp"> {
  let summary = "deinit module";
  let constructor = "createDeinitPass()";
  let dependentDialects = ["TopDialect"];
}

#endif
