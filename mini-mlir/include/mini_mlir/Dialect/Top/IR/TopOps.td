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

def Top_SoftmaxOp:Top_Op<"Softmax"> {
  let summary = "Softmax operation";
  let description = [{
    Integrates some operations related to softmax.
  }];
  let arguments = (ins
    AnyTensor:$input,
    I32Attr:$axis,
    DefaultValuedAttr<BoolAttr, "false">:$log,
    DefaultValuedAttr<F64Attr, "1.0">:$beta
  );
  let results = (outs AnyTensor:$output);
}

def Top_ReshapeOp:Top_Op<"Reshape"> {
  let summary = "Reshape operation";
  let description = [{
    Returns a tensor with the same type/values as the input, with a new shape
    specified by the shape argument. Reshape may operate on tensors of any rank.
    No data conversion happens during a reshape operation.
    0: keep dim from input
    -1: left dim from input
  }];
  let arguments = (ins
    AnyTensor:$input,
    OptionalAttr<I64ArrayAttr>:$shape
  );
  let results = (outs AnyTensor:$output);
  //let hasCanonicalizer = 1;
}

def Top_LayerNormOp : Top_Op<"LayerNorm"> {
  let summary = "LayerNorm operation";
  let description = [{
    layer normalization
  }];
  let arguments = (ins
    AnyTensor:$input,
    AnyTensorOrNone:$weight,
    AnyTensorOrNone:$bias,
    I64ArrayAttr:$normalized_shape,
    I32Attr:$axis,
    F32Attr:$eps
  );
  let results = (outs
    AnyTensor:$output
  );
  //let hasCanonicalizer = 1;
}

def Top_TransposeOp: Top_Op<"Transpose"> {

  let summary = "Transpose operator";

  let description = [{
      Transpose on input.
  }];

  let arguments = (
    ins AnyTensor:$input,
    SI32Attr:$dim0,
    SI32Attr:$dim1
  );
  let results = (outs AnyTensor:$output);
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

def Top_PermuteOp: Top_Op<"Permute"> {

  let summary = "Permute operator";

  let description = [{
      Perform permute on input.
  }];

  let arguments = (
    ins AnyTensor:$input,
    I64ArrayAttr:$order
  );
  let results = (outs AnyTensor:$output);
  //let hasCanonicalizer = 1;
  //let extraClassDeclaration = [{
  //  permute_attr_t parseParam();
  //}];
}

def Top_SliceOp: Top_Op<"Slice"> {
  let summary = "Slice operator";

  let description = [{
    Slice Operation on input.
  }];

  let arguments = (
    ins AnyTensor:$input,
    AnyTensorOrNone:$offsetT,
    AnyTensorOrNone:$endsT,
    AnyTensorOrNone:$stepsT,
    I64ArrayAttr:$offset,
    I64ArrayAttr:$steps,
    I64ArrayAttr:$ends
  );
  let results = (outs AnyTensor:$output);
  //let hasCanonicalizer = 1;
}

def Top_SplitOp: Top_Op<"Split"> {
  let summary = "Split operator";

  let description = [{
    Split input tensor into a list of tensors.
  }];

  let arguments = (
    ins AnyTensor:$input,
    SI32Attr:$axis,
    I64Attr:$num,
    OptionalAttr<I64ArrayAttr>:$split_size
  );
  let results = (outs Variadic<AnyTensor>:$outputs);
  //let hasCanonicalizer = 1;
}

def Top_MulConstOp: Top_Op<"MulConst"> {
  let summary = "Mul Const operator";

  let description = [{
    Y = X * const_val
  }];

  let arguments = (
    ins AnyTensor:$input,
    F64Attr: $const_val
  );

  let results = (outs AnyTensor:$output);
  //let hasCanonicalizer = 1;
}

def Top_GELUOp : Top_Op<"GELU"> {
  let summary = " GELU operator,  0.5x * (1.0 + tf.erf(x / tf.sqrt(2.0)))";
  let description = [{
     Y = 0.5x * (1.0 + tf.erf(x / tf.sqrt(2.0)))
  }];
  let arguments = (
    ins AnyTensor:$input
  );

  let results = (outs AnyTensor:$output);
}

def Top_GatherOp: Top_Op<"Gather"> {
  let summary = "Gather operator";
  let description = [{
    Perform Gather operation on the given axis.
  }];

  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$indices,

    DefaultValuedAttr<I64Attr, "0">:$axis
  );

  let results = (outs AnyTensor:$output);
  //let hasCanonicalizer = 1;
}

#endif // Top_OPS
