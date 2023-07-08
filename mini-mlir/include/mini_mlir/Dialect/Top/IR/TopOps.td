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

def Top_ConvOp: Top_Op<"Conv"> {
  let summary = "Convolution operator";

  let description = [{
    In the simplest case, the output value of the layer with input size
    $$(N, C_{\text{in}}, H, W)$$ and output $$(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})$$
    can be precisely described as:

    ```math
        \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k = 0}^{C_{\text{in}} - 1} \text{weight}(C_{\text{out}_j}, k) \star \text{input}(N_i, k)
    ```


    where $$\star$$ is the valid 2D cross-correlation operator,
    $$N$$ is a batch size, $$C$$ denotes a number of channels,
    $$H$$ is a height of input planes in pixels, and $$W$$ is
    width in pixels.

    weight (Tensor): the learnable weights of the module of shape
    $$(\text{out\_channels}, \frac{\text{in\_channels}}{\text{groups}},
    \text{kernel\_size[0]}, \text{kernel\_size[1]})$$

    bias (Tensor optional): the learnable bias of the module of shape (out_channels).

    - **stride**: controls the stride for the cross-correlation, a single
      number or a tuple.

    - **padding**: controls the amount of padding applied to the input. It
      contains four ints with top, left, bottom, right respectively.

    - **dilation**: controls the spacing between the kernel points; also
      known as the à trous algorithm. It is harder to describe, but this
      [Link](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md)
      has a nice visualization of what **dilation** does.

    - **groups**: (optional): Number of blocked connections from input
            channels to output channels. Default: 1


    Shape:
        - Input: $$(N, C_{in}, H_{in}, W_{in})$$
        - Output: $$(N, C_{out}, H_{out}, W_{out})$$ where

          ```math
              H_{out} = \left\lfloor\frac{H_{in}  + \text{padding}[0] + \text{padding}[2] - \text{dilation}[0]
                        \times (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor
          ```
          ```math
              W_{out} = \left\lfloor\frac{W_{in}  + \text{padding}[1] + \text{padding}[3] - \text{dilation}[1]
                        \times (\text{kernel\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor
          ```
  }];

  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$filter,
    AnyTensorOrNone:$bias,
    I64ArrayAttr:$kernel_shape,
    I64ArrayAttr:$strides,
    I64ArrayAttr:$pads, // top,left,bottom,right
    DefaultValuedAttr<I64Attr, "1">:$group,
    OptionalAttr<I64ArrayAttr>:$dilations,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit
  );

  let results = (outs AnyTensor:$output);
  //let hasCanonicalizer = 1;
}


def Top_ConcatOp: Top_Op<"Concat"> {
  let summary = "Concat operator";

  let description = [{
  Concatenates the given sequence of seq tensors in the given dimension.
  All tensors must either have the same shape (except in the concatenating dimension) or be empty.
  }];

  let arguments = (
    ins Variadic<AnyTensor>:$inputs,
    DefaultValuedAttr<I32Attr, "1">:$axis,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit
  );

  let results = (outs AnyTensor:$output);
  //let hasCanonicalizer = 1;
}


def Top_MatMulOp: Top_Op<"MatMul"> {
  let summary = "matmul operator";

  let description = [{
    Performs a two dimensional matrix multiplication. This allows both inputs to
    be activations, rather than reserving weights as an attribute in the
    FULLY_CONNECTED operator.
  }];

  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$right,
    AnyTensorOrNone:$bias,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit
  );

  let results = (outs AnyTensor:$output);
  //let hasCanonicalizer = 1;
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
    I64ArrayAttr:$offset,
    I32Attr:$axis,
    I64ArrayAttr:$start_list,
    I64ArrayAttr:$size_list
    // AnyTensorOrNone:$endsT,
    // AnyTensorOrNone:$stepsT,
    // I64ArrayAttr:$offset,
    // I64ArrayAttr:$steps,
    // I64ArrayAttr:$ends
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

def Top_ReduceMeanOp : Top_Op<"ReduceMean"> {
  let summary = "ReduceMean operation";
  let description = [{
    Computes the mean of the input tensor's element along the provided axes.
  }];
  let arguments = (ins
    AnyTensor:$input,
    I32Attr:$axis
  );
  let results = (outs AnyTensor:$output);
  //let hasCanonicalizer = 1;
}

def Top_MulOp: Top_Op<"Mul"> {
  let summary = "Mul operator";

  let description = [{
    Elementwise multiplication of input1 and input2. input1 and input2 are tensors.
  }];

  let arguments = (ins
    Variadic<AnyTensor>:$inputs,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit
  );

  let results = (outs AnyTensor:$output);
  //let hasCanonicalizer = 1;
}

def Top_SqrtOp : Top_Op<"Sqrt"> {
  let summary = "Sqrt operation";
  let description = [{
    Computes the square root of the input tensor's element.
  }];
  let arguments = (ins
    AnyTensor:$input
  );
  let results = (outs AnyTensor:$output);
}

def Top_ErfOp : Top_Op<"Erf"> {
  let summary = "Erf operation";
  let description = [{
    Computes the error function of the given input tensor element-wise.
  }];
  let arguments = (ins
    AnyTensor:$input
  );
  let results = (outs AnyTensor:$output);
}

#endif // Top_OPS
