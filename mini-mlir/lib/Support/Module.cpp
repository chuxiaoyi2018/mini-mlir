#include "mini_mlir/Support/Module.h"
#include "mini_mlir/Dialect/Top/IR/TopOps.h"
#include "mini_mlir/Support/MathUtils.h"
#include "mini_mlir/Support/ModuleEnum.cpp.inc"

#include "float.h"
#include "mlir/Dialect/Quant/FakeQuantSupport.h"
#include "mlir/IR/PatternMatch.h"
#include <map>


namespace mini_mlir {
namespace module {

struct Attr {
  static constexpr llvm::StringRef NAME = "module.name";
  static constexpr llvm::StringRef STATE = "module.state";
  static constexpr llvm::StringRef CHIP = "module.chip";
  static constexpr llvm::StringRef WEIGHT_FILE = "module.weight_file";
  static constexpr llvm::StringRef FLOPS = "module.FLOPs";
  static constexpr llvm::StringRef COEFF_ADDR = "module.coeff_addr";
  static constexpr llvm::StringRef COEFF_SIZE = "module.coeff_size";
  static constexpr llvm::StringRef NEURON_ADDR = "module.neuron_addr";
  static constexpr llvm::StringRef NEURON_SIZE = "module.neuron_size";
  static constexpr llvm::StringRef GMEM_PRIVATE_SIZE = "module.private_size";
  static constexpr llvm::StringRef ASYMMETRIC = "module.asymmetric";
  static constexpr llvm::StringRef MODE = "module.mode";
  static constexpr llvm::StringRef PLATFORM = "module.platform";
};


static ModuleOp m = nullptr;
static MLIRContext *ctx = nullptr;
static std::unique_ptr<mlir::TensorFile> wFile = nullptr;
static std::string weightFileName = "";
static Platform platform = Platform::ONNX;

// init module by ModuleOp in init pass
void init(ModuleOp module) {
  m = module;
  ctx = m.getContext();
  auto chip_ = m->getAttrOfType<StringAttr>(Attr::CHIP);
  wFile = nullptr;
  platform = Platform::ONNX;
}

//-----------------------------------------------------------------
// Helper for get/set Attributes
//-----------------------------------------------------------------
State getState() {
  auto s = m->getAttrOfType<StringAttr>(Attr::STATE);
  return symbolizeState(s).value_or(State::TOP_F32);
}

void setState(State state) {
  auto s = stringifyState(state);
  m->setAttr(Attr::STATE, StringAttr::get(ctx, s));
}

bool isState(State state) { return state == getState(); }

//-----------------------------------------------------------------
// Helper for Array
//-----------------------------------------------------------------
i64_array_t getI64Array(ArrayAttr arrayAttr) {
  auto data = std::make_shared<std::vector<int64_t>>();
  for (auto en : llvm::enumerate(arrayAttr)) {
    auto attr = en.value().dyn_cast<IntegerAttr>();
    if (attr) {
      data->push_back(attr.getInt());
    } else {
      arrayAttr.dump();
      llvm_unreachable("not int64_t type");
    }
  }
  return std::move(data);
}

i64_array_t getI64Array(Optional<ArrayAttr> arrayAttr, int64_t num_elem,
                        int64_t default_value) {
  if (arrayAttr.has_value()) {
    auto arr = getI64Array(arrayAttr.value());
    assert(arr->size() == num_elem);
    return std::move(arr);
  }
  return std::make_shared<std::vector<int64_t>>(num_elem, default_value);
}

//-----------------------------------------------------------------
// Helper Functions for ModuleOp
//-----------------------------------------------------------------
ModuleOp getModuleOp() { return m; }
Location getLoc() { return m.getLoc(); }
MLIRContext *getCtx() { return ctx; }

uint32_t getIdx(Value v) {
  uint32_t idx = 0;
  if (auto r = v.dyn_cast<OpResult>()) {
    idx = r.getResultNumber();
  } else if (auto r = v.dyn_cast<BlockArgument>()) {
    idx = r.getArgNumber();
  } else {
    v.dump();
    llvm_unreachable("Not Implemented");
  }
  return idx;
}

NameLoc getLoc(Value v) {
  if (auto loc = v.getLoc().dyn_cast<NameLoc>()) {
    return loc;
  } else if (auto fuse_loc = v.getLoc().dyn_cast<FusedLoc>()) {
    auto locs = fuse_loc.getLocations();
    uint32_t idx = getIdx(v);
    if (auto name_loc = locs[idx].dyn_cast<NameLoc>()) {
      return name_loc;
    }
  } else if (auto op = v.getDefiningOp()) {
    auto loc = op->getLoc();
    if (auto name_loc = loc.dyn_cast<NameLoc>()) {
      return name_loc;
    }
    if (auto fuse_loc = loc.dyn_cast<FusedLoc>()) {
      uint32_t idx = getIdx(v);
      auto locs = fuse_loc.getLocations();
      if (auto name_loc = locs[idx].dyn_cast<NameLoc>()) {
        return name_loc;
      }
    }
  }
  v.dump();
  llvm_unreachable("Not Implemented");
  return nullptr;
}

FuncOp getFuncOp(StringRef func_name) {
  for (auto func : m.getOps<FuncOp>()) {
    if (func.getName() == func_name) {
      return func;
    }
  }
  llvm::errs() << "Can't find FuncOp:" << func_name << "\n";
  llvm_unreachable("Error getFuncOp !!\n");
  return nullptr;
}

// for weight op
Type getStorageType(Type type) {
  if (type.isa<RankedTensorType>()) {
    type = type.cast<RankedTensorType>().getElementType();
  }
  if (auto qType = type.dyn_cast<quant::CalibratedQuantizedType>()) {
    return qType.getExpressedType();
  } else if (auto qType = type.dyn_cast<quant::UniformQuantizedType>()) {
    auto stype = qType.getStorageType();
    bool isSign = qType.isSigned();
    if (stype.isSignlessInteger()) {
      auto bits = stype.getIntOrFloatBitWidth();
      auto sign = isSign ? IntegerType::Signed : IntegerType::Unsigned;
      return IntegerType::get(type.getContext(), bits, sign);
    }
    return stype;
  } else if (auto qType = type.dyn_cast<quant::UniformQuantizedPerAxisType>()) {
    return qType.getStorageType();
  }
  return type;
}

Type getStorageType(Value v) { return getStorageType(v.getType()); }


func::CallOp getCallOp(FuncOp func) {
  func::CallOp call = nullptr;
  for (auto each_func : m.getOps<FuncOp>()) {
    WalkResult result = each_func.walk<WalkOrder::PreOrder>([&](func::CallOp op) {
      if (!call && op.getCallee() == func.getName()) {
        call = op;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (result.wasInterrupted())
      break;
  }
  return call;
}

FuncOp getMainFuncOp() { return getFuncOp("main"); }

void updateModuleTypes() {
  Builder builder(ctx);
  // update callee func's return types
  for (auto func : m.getOps<FuncOp>()) {
    if (func.getName() == "main") {
      continue;
    }
    std::vector<Type> returns;
    Block &entryBlock = func.front();
    auto returnOp = dyn_cast<ReturnOp>(entryBlock.back()).getOperation();
    for (uint32_t i = 0; i < returnOp->getNumOperands(); ++i) {
      returns.push_back(returnOp->getOperand(i).getType());
    }
    auto fnType = builder.getFunctionType(func.getArgumentTypes(),
                                          llvm::ArrayRef<Type>{returns});
    func.setType(fnType);
    auto callee = getCallOp(func);
    if (callee) {
      for (auto it : llvm::zip(callee.getResults(), returns)) {
        std::get<0>(it).setType(std::get<1>(it));
      }
    }
  }
  // update callee arg types
  for (auto func : m.getOps<FuncOp>()) {
    if (func.getName() == "main") {
      continue;
    }
    auto callee = getCallOp(func);
    if (!callee) {
      continue;
    }
    std::vector<Type> arguments;
    for (auto it :
         llvm::zip(callee.getOperandTypes(), func.front().getArguments())) {
      arguments.push_back(std::get<0>(it));
      std::get<1>(it).setType(std::get<0>(it));
    }
    auto fnType = builder.getFunctionType(llvm::ArrayRef<Type>(arguments),
                                          func.getResultTypes());
    func.setType(fnType);
  }
  // update main op return types
  auto mainFunc = getMainFuncOp();
  Block &entryBlock = mainFunc.front();
  auto returnOp = dyn_cast<ReturnOp>(entryBlock.back()).getOperation();
  std::vector<Type> returns;
  for (uint32_t i = 0; i < returnOp->getNumOperands(); ++i) {
    returns.push_back(returnOp->getOperand(i).getType());
  }
  auto fnType = builder.getFunctionType(mainFunc.getArgumentTypes(),
                                        llvm::ArrayRef<Type>{returns});
  mainFunc.setType(fnType);
}

StringRef getName(Operation *op, int index) {
  if (auto module = dyn_cast<ModuleOp>(op)) {
    return getName(module);
  }
  if (auto loc = op->getLoc().dyn_cast<NameLoc>()) {
    return loc.getName();
  }
  if (auto loc = op->getLoc().dyn_cast<FusedLoc>()) {
    auto locs = loc.getLocations();
    assert(index < locs.size());
    if (auto name_loc = locs[index].dyn_cast<NameLoc>()) {
      return name_loc.getName();
    }
  }
  op->print(llvm::errs(), OpPrintingFlags().useLocalScope().enableDebugInfo());
  llvm::errs() << "\n";
  llvm_unreachable("op has no name location!!!");
  return "";
}

StringRef getName(Value v) { return getLoc(v).getName().strref(); }

llvm::StringRef getModuleName() {
  return m->getAttrOfType<StringAttr>(Attr::NAME).getValue();
}


void removeUnusedOp() {
  std::vector<Operation *> all_ops;
  for (auto func : m.getOps<FuncOp>()) {
    for (auto &op : func.getOps()) {
      if (false == isa<ReturnOp, FuncOp>(op)) {
        all_ops.push_back(&op);
      }
    }
  }
  for (auto iter = all_ops.rbegin(); iter != all_ops.rend(); iter++) {
    if ((*iter)->use_empty()) {
      (*iter)->erase();
    }
  }
}


//-----------------------------------------------------------------
// Helper Functions for weight
//-----------------------------------------------------------------
std::string genWeightFileName(bool &same_name) {
  auto name = getModuleName();
  // auto state = getState();
  auto old_name = m->getAttrOfType<StringAttr>(Attr::WEIGHT_FILE).getValue();
  std::string file_name = name.lower() + std::string("_");

  auto new_name = file_name + "_weight.npz";
  same_name = (old_name == new_name);
  if (same_name) {
    new_name = file_name + "_weight_fix.npz";
  }
  return new_name;
}

void setWeightFileName(const std::string &name) { weightFileName = name; }
void saveWeight() {
  // check name conflict
  std::set<StringRef> all_names;
  for (auto func : m.getOps<FuncOp>()) {
    func.walk([&](Operation *op) {
      if (op->getLoc().dyn_cast<NameLoc>()) {
        auto name = module::getName(op);
        //if op have more than two regions, it can have the same op Name
        // if (all_names.find(name) != all_names.end() && !isa<tpu::YieldOp, tpu::IfOp>(op)) {
        if (all_names.find(name) != all_names.end()) {
          op->dump();
          llvm_unreachable("op name conflict");
        }
        all_names.insert(name);
      }
    });
  }
  bool same_name = true;
  std::string filename_;
  if (weightFileName == "") {
    filename_ = module::genWeightFileName(same_name);
  } else {
    same_name = false;
    filename_ = weightFileName;
  }
  // weight remove unused in npz
  if (wFile == nullptr) {
    if (!same_name) {
      weightFile().save(filename_);
      m->setAttr(Attr::WEIGHT_FILE, StringAttr::get(ctx, filename_));
    }
    return;
  }
  if (wFile->changed() == false && same_name) {
    return;
  }
  std::set<StringRef> weight_names;
  for (auto func : m.getOps<FuncOp>()) {
    func.walk([&](top::WeightOp op) {
      weight_names.insert(module::getName(op.getOperation()));
    });
  }
  std::set<StringRef> npz_names;
  wFile->getAllNames(npz_names);
  std::set<StringRef> dif_names;
  for (auto name : npz_names) {
    if (weight_names.find(name) == weight_names.end()) {
      dif_names.insert(name);
    }
  }
  for (auto &name : dif_names) {
    wFile->deleteTensor<float>(name);
  }
  if (wFile->changed() == false && same_name) {
    return;
  }
  wFile->save(filename_);
  m->setAttr(Attr::WEIGHT_FILE, StringAttr::get(ctx, filename_));
}
void detachWeightFile() { wFile = nullptr; }

mlir::TensorFile &weightFile() {
  if (wFile == nullptr) {
    auto name = m->getAttrOfType<StringAttr>(Attr::WEIGHT_FILE).getValue();
    wFile = openTensorFile(name);
  }
  return *wFile;
}


} // namespace module
} // namespace mini_mlir
