#pragma once
#include "mlir/IR/BuiltinOps.h" // ModuleOp
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/QuantTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mini_mlir/Dialect/Top/IR/TopOps.h"
#include "mini_mlir/Support/TensorFile.h"
#include "mini_mlir/Support/ModuleEnum.h.inc"

using namespace mlir;
using namespace mlir::func;
using namespace mini_mlir;

namespace mini_mlir {

//-----------------------------------------------------------------
// Types
//-----------------------------------------------------------------
typedef std::shared_ptr<std::vector<int32_t>> i32_array_t;
typedef std::shared_ptr<std::vector<int64_t>> i64_array_t;
typedef std::shared_ptr<std::vector<double>> f64_array_t;
namespace module {

// init module by ModuleOp in init pass
void init(ModuleOp module);

//-----------------------------------------------------------------
// Helper for get/set Attributes
//-----------------------------------------------------------------
State getState();
void setState(State state);
bool isState(State state);

//-----------------------------------------------------------------
// Helper Functions for ModuleOp
//-----------------------------------------------------------------
ModuleOp getModuleOp();
Location getLoc();
MLIRContext *getCtx();
std::string genWeightFileName(bool &same_name);

uint32_t getIdx(Value v);
NameLoc getLoc(Value v);

// for weight op
Type getStorageType(Value v); // storage type
Type getStorageType(Type type);

func::CallOp getCallOp(FuncOp func);

FuncOp getMainFuncOp();
void updateModuleTypes();
llvm::StringRef getName(Operation *op, int index = 0);
llvm::StringRef getName(Value v);
llvm::StringRef getModuleName();

bool isOpInGroup(Operation *Op);

void removeUnusedOp();

//-----------------------------------------------------------------
// Helper Functions for weight
//-----------------------------------------------------------------
mlir::TensorFile &weightFile();
void setWeightFileName(const std::string &name);
void saveWeight();
void detachWeightFile();

} // namespace module
} // namespace mini_mlir
