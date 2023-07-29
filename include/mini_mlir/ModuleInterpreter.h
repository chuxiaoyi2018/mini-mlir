//===- ModuleInterpreter.h - Interpreter ------------------------------*- C++
//-*-===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// This header file defines prototypes that expose interpreter constructors.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_MODULEINTERPRETER_H_
#define MLIR_MODULEINTERPRETER_H_

#include "mini_mlir/Interfaces/InferenceInterface.h"


#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/Support/Debug.h"

#include <fstream>
#include <iostream>
#include <map>
#include <unordered_map>

#define DEBUG_TYPE "interpreter"

using namespace mlir;

namespace mini_mlir {
// Implementation class for module interpreter.
class ModuleInterpreter {

public:
  // Interpret the given MLIR module expressed in MLIR TPU IR dialect
  explicit ModuleInterpreter(ModuleOp module) : module(module) {}
  virtual ~ModuleInterpreter();
  void allocate_resources();
  void invoke();
  void setTensor(const std::string &name, const void *data, size_t size);
  std::shared_ptr<std::vector<float>> getTensor(const std::string &name);
  llvm::ArrayRef<int64_t> getTensorShape(const std::string &name);
  std::vector<std::string> getAllTensorName();

public:
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  std::vector<std::string>
      all_tensor_names; // activation tensor, without weight

private:
  ModuleOp module;
  std::unordered_map<std::string, mlir::Value> value_map; // activation value
  std::map<std::string, std::shared_ptr<InferenceParameter>> inference_map;
  std::map<std::string, std::shared_ptr<std::vector<float>>> mem_map;
};

} // namespace mini_mlir

#endif // MLIR_DIALECT_TPU_MODULEINTERPRETER_H_
