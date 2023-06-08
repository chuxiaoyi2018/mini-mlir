#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vector>

// -------------
// pure C++ code
// -------------
#include "mini_mlir/Dialect/Top/IR/TopOps.h"
#include "mini_mlir/ModuleInterpreter.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/QuantOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;
using namespace mini_mlir;

#define OP_NAME "name"
#define OP_TYPE "type"
#define OP_QUANT "quant"

typedef std::map<std::string, std::shared_ptr<std::vector<float>>> tensor_map_t;
typedef std::map<std::string, std::vector<int64_t>> shape_map_t;

// ----------------
// Python interface
// ----------------

namespace py = pybind11;

// Warning: buffer in C++. New inference will erase old output
static py::array getPyArray(float *ptr, const std::vector<int64_t> &shape) {
  py::capsule do_nothing((void *)ptr, [](void *f) {});
  return py::array_t<float>(shape, ptr, do_nothing);
}


template <typename Dtype>
static py::array getPythonArray(std::vector<Dtype> *vec,
                                const std::vector<int64_t> &shape) {
  std::vector<unsigned> stride_v(shape.size(), sizeof(Dtype));
  for (int i = shape.size() - 1; i > 0; i--) {
    for (int j = 0; j < i; j++) {
      stride_v[j] *= shape[i];
    }
  }

  return py::array(
      py::buffer_info(vec->data(),   /* data as contiguous array  */
                      sizeof(Dtype), /* size of one scalar        */
                      py::format_descriptor<Dtype>::format(), /* data type */
                      shape.size(),                           // ndim/
                      shape,                                  // shape
                      stride_v                                // strides
                      ));
}

class py_module {
public:
  py_module() {}
  ~py_module() {
    interpreter_.reset();
    auto module = module_.release();
    if (module) {
      module.erase();
    }
    context_.reset();
  }

  void load(std::string filename) {
    if (context_) {
      context_.reset();
    }

    DialectRegistry registry;
    registry.insert<top::TopDialect, func::FuncDialect>();
    context_ = std::make_unique<MLIRContext>(registry);

    module_ = parseSourceFile<ModuleOp>(filename, context_.get());
    if (!module_) {
      llvm::errs() << "Error, parse :" << filename << "\n";
      llvm_unreachable("could not parse mlir file\n");
    }
    if (interpreter_) {
      interpreter_.reset();
    }

    interpreter_ = std::make_unique<ModuleInterpreter>(module_.get());
    interpreter_->allocate_resources();
  }

  py::dict getAllTensor() {
    py::dict py_ret;
    for (auto &name : interpreter_->all_tensor_names) {
      auto tensor = interpreter_->getTensor(name);
      auto shape = interpreter_->getTensorShape(name);
      py::str py_s(name);
      py_ret[py_s] = getPyArray(tensor->data(), shape);
    }
    return py_ret;
  }

  py::list get_input_names() {
    py::list ret;
    auto &inputs = interpreter_->input_names;
    for (auto &i : inputs) {
      ret.append(i);
    }
    return ret;
  }
  py::list get_output_names() {
    py::list ret;
    auto &outputs = interpreter_->output_names;
    for (auto &i : outputs) {
      ret.append(i);
    }
    return ret;
  }

  void set_tensor(
      std::string name,
      py::array_t<float, py::array::c_style | py::array::forcecast> data) {
    interpreter_->setTensor(name, data.data(), data.size() * sizeof(float));
  }
  py::array get_tensor(std::string name) {
    auto tensor = interpreter_->getTensor(name);
    std::vector<int64_t> shape = interpreter_->getTensorShape(name);
    return getPythonArray(tensor.get(), shape);
  }
  void invoke() { interpreter_->invoke(); }

public:
  py::list opInfo_;
  static std::string version;

private:
  std::unique_ptr<mlir::MLIRContext> context_;
  OwningOpRef<ModuleOp> module_;
  std::string weightFilePath_;
  std::unique_ptr<ModuleInterpreter> interpreter_;
};

std::string py_module::version = MLIR_VERSION;

// wrap as Python module
PYBIND11_MODULE(pymlir, m) {
  m.doc() = "pybind11 for mlir";

  py::class_<py_module>(m, "module", "MLIR Module")
      .def(py::init<>())
      .def("load", &py_module::load, "load module from IR")
      .def("get_all_tensor", &py_module::getAllTensor, "dump all tensor data")
      .def("set_tensor", &py_module::set_tensor)
      .def("get_tensor", &py_module::get_tensor, "get one tensor data")
      .def_readwrite("op_info", &py_module::opInfo_)
      .def("invoke", &py_module::invoke)
      .def("get_input_names", &py_module::get_input_names)
      .def("get_output_names", &py_module::get_output_names)
      .def_readonly_static("version", &py_module::version);
}
