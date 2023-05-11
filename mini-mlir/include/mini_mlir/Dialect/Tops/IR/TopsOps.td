// =============================================================================
//
// Defines TOPS Dialect operations.
//
//===----------------------------------------------------------------------===//

#ifndef TOPS_OPS
#define TOPS_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mini_mlir/Interfaces/InferenceInterface.td"

def Tops_Dialect : Dialect {
  let name = "tops";
  let summary = "A tops dialect for the SOPHGO specification";
  let cppNamespace = "::mlir::tops";
}

def AnyTensorOrNone: AnyTypeOf<[AnyTensor, NoneType]>;

class Tops_Op<string mnemonic, list<Trait> traits = []> :
    Op<Tops_Dialect, mnemonic, traits> ;

def Tops_NoneOp : Tops_Op<"None", [Pure]> {
  let summary = "none operator";

  let description = [{
    A none Op to return a NoneType.
  }];
  let results = (outs NoneType);
}

def Tops_WeightOp : Tops_Op<"Weight", [Pure]> {
  let summary = "load weight operator";

  let description = [{
    Load weight from a file. The file should be a valid .npy format file.
    This Op does not take any input, and the attribute captures the file name.
    The Output is an n-dimensional tensor whose type matches
    the tensor type in the .npy file.
  }];

  let arguments = (
    StrAttr:$name
  );

  let results = (outs AnyTensor:$output);
}

def Tops_InputOp: Tops_Op<"Input",[Pure]> {
  let summary = "Input operator";

  let description = [{
  }];

  let arguments = (
    ins AnyTensor:$input,
    StrAttr:$name
  );

  let results = (outs AnyTensor:$output);
}

def Tops_ReluOp: Tops_Op<"Relu", [Pure,
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

def Tops_AddOp: Tops_Op<"Add", [Pure,
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


def Tops_DivOp: Tops_Op<"Div", [Pure,
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

def Tops_SubOp: Tops_Op<"Sub", [Pure,
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

#endif // Tops_OPS
