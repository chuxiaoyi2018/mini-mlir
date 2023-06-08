// =============================================================================
//
// Defines TOP Dialect operations.
//
//===----------------------------------------------------------------------===//

#ifndef MINI_MLIR_TOP_OPS
#define MINI_MLIR_TOP_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mini_mlir/Interfaces/InferenceInterface.td"

def Top_Dialect : Dialect {
  let name = "top";
  let summary = "A top dialect for the MINI specification";
  let cppNamespace = "::mini_mlir::top";
}

def AnyTensorOrNone: AnyTypeOf<[AnyTensor, NoneType]>;

class Top_Op<string mnemonic, list<Trait> traits = []> :
    Op<Top_Dialect, mnemonic, traits> ;

def Top_NoneOp : Top_Op<"None", [Pure]> {
  let summary = "none operator";

  let description = [{
    A none Op to return a NoneType.
  }];
  let results = (outs NoneType);
}

def Top_WeightOp : Top_Op<"Weight", [Pure]> {
  let summary = "load weight operator";

  let description = [{
    Load weight from a file. The file should be a valid .npz format file.
    This Op does not take any input, and the location captures the tensor name.
    The Output is an n-dimensional tensor whose type matches
    the tensor type in the .npz file.
  }];

  let arguments = (ins
    OptionalAttr<F64ArrayAttr>:$scale,
    OptionalAttr<BoolAttr>:$do_compress
  );

  let results = (outs AnyRankedTensor:$output);
  let extraClassDeclaration = [{
  template<typename T>
  std::shared_ptr<std::vector<T>> read();
  std::shared_ptr<std::vector<float>> read_as_float();
  template<typename T>
  static mlir::Value create(mlir::Operation * OwnerOp,
                            llvm::StringRef suffix,
                            const std::vector<T>& data,
                            mlir::RankedTensorType& type);
  template<typename T>
  mlir::LogicalResult update(const std::vector<T>& data, size_t count);
  mlir::Value clone_int(mlir::Operation *OwnerOp);
  }];
}

def Top_InputOp: Top_Op<"Input",[Pure]> {
  let summary = "Input operator";

  let description = [{
  }];

  let arguments = (
    ins AnyTensor:$input,
    StrAttr:$name
  );

  let results = (outs AnyTensor:$output);
}

def Top_ReluOp: Top_Op<"Relu", [Pure,
  DeclareOpInterfaceMethods<InferenceInterface,["init","deinit"]>]> {
  let summary = "Relu operator";

  let description = [{
     ReLU with a scalar maximum value. if limit is zero, do not use upper limit.
  }];

  let arguments = (
    ins AnyTensor:$input,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit
  );

  let results = (outs AnyTensor:$output);

  //let hasCanonicalizer = 1;
}

def Top_AddOp: Top_Op<"Add", [Pure,
  DeclareOpInterfaceMethods<InferenceInterface,["init","deinit"]>]> {
  let summary = "Add operator";

  let description = [{
    Elementwise addition of input1 and input2. Axis of size 1 will be broadcast,
    as necessary.
  }];

  let arguments = (ins
    Variadic<AnyTensor>:$inputs,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit,
    OptionalAttr<F64ArrayAttr>:$coeff
  );

  let results = (outs AnyTensor:$output);
}


def Top_DivOp: Top_Op<"Div", [Pure,
  DeclareOpInterfaceMethods<InferenceInterface,["init","deinit"]>]> {
  let summary = "Div operator";

  let description = [{
    Elementwise division of input1 and input2. Axis of size 1 will be broadcast,
    as necessary.
  }];

  let arguments = (ins
    Variadic<AnyTensor>:$inputs,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit,
    OptionalAttr<F64ArrayAttr>:$coeff
  );

  let results = (outs AnyTensor:$output);
}

def Top_SubOp: Top_Op<"Sub", [Pure,
  DeclareOpInterfaceMethods<InferenceInterface,["init","deinit"]>]> {
  let summary = "Sub operator";

  let description = [{
    Elementwise subtraction of input1 and input2. Axis of size 1 will be broadcast,
    as necessary.
  }];

  let arguments = (ins
    Variadic<AnyTensor>:$inputs,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit,
    OptionalAttr<F64ArrayAttr>:$coeff
  );

  let results = (outs AnyTensor:$output);
}

#endif // Top_OPS
