import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit as jit
import argparse
import os

from utils.mlir_shell import _os_system


class ONNX_IR_TESTER(object):
    # This class is built for testing single operator transform.
    def __init__(self,
                 chip: str = "",
                 mode: str = "all",
                 ):
        Y, N = True, False
        # yapf: disable
        self.test_cases = {
            #########################################
            # ONNX Test Case, Alphabetically
            #########################################
            # case: (test, cpu x86_64, arm)
            "Softmax":      (self.test_Softmax,       N, N),
            }


    #########################################
    # Softmax
    #########################################
    def export_Softmax_onnx(self, name, shape):
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
            def forward(self, x):
                return F.softmax(x,dim=-1)

        model = Model().eval()
        inputs = tuple(torch.rand(s) for s in shape)
        input_names = [f'x{i}' for i in range(len(shape))]

        torch.onnx.export(
                model,
                inputs,
                f'{name}/{name}.onnx',
                input_names,
                ['Y'],
                opset_version=11
                )

    def test_Softmax(self, name):
        shape = [[1,2,2,72]]
        exec(f"self.export_{name}_onnx(name, shape)")
        self.onnx2mlir(name, shape)

    def onnx2mlir(self, name, shape):
        shape_str = str(shape).replace(' ','') 
        cmd_str = ["model_transform.py", 
                  f"--model_name {name}",
                  f"--model_def {name}/{name}.onnx",
                  f"--input_shapes {shape_str}",
                  f"--mlir {name}/{name}.mlir",
                   "--model_type onnx"]
        _os_system([" ".join(cmd_str)])
    
    def mlir2tosa(self, name):
        cmd_str = f"mini-opt --init --convert-top-to-tosa --deinit {name}/{name}.mlir  -o tosa.mlir"
        _os_system([cmd_str])
    
    def tosa2llvmir(self, name):
        lower_param = (f"mlir-opt {name}/tosa.mlir"
                       "--pass-pipeline=\"builtin.module("
                       "func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-arith, tosa-to-tensor, tosa-to-scf), "
                       "convert-tensor-to-linalg, "
                       "func.func(canonicalize, linalg-bufferize, convert-linalg-to-affine-loops, affine-loop-fusion, affine-simplify-structures, lower-affine), "
                       "func-bufferize, "
                       "func.func(tensor-bufferize, llvm-request-c-wrappers), "
                       "arith-expand, arith-bufferize, normalize-memrefs, convert-scf-to-cf, "
                       "convert-math-to-llvm, convert-arith-to-llvm, convert-func-to-llvm, convert-cf-to-llvm, "
                       "convert-bufferization-to-memref, memref-expand, expand-strided-metadata, finalize-memref-to-llvm, "
                       "canonicalize, llvm-legalize-for-export, reconcile-unrealized-casts)\""
                       "| mlir-translate --mlir-to-llvmir "
                       f"| llc -mtriple=x86_64-unknown-linux-gnu --filetype=obj -o {name}/final.o")
        _os_system(["".join(lower_param)])
    
    def mkdir(self, dir_path):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    def test_single(self, name):
        assert name in self.test_cases
        self.mkdir(name)
        func = self.test_cases[name][0]

        # onnx->mlir
        func(name)

        # mlir->tosa
        self.mlir2tosa(name)

        # tosa->llvmir
        self.tosa2llvmir(name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # yapf: disable
    parser.add_argument("--chip", default="x86", type=str, choices=['x86'],
                        help="chip platform name")
    parser.add_argument("--case", default="all", type=str, help="test one case, if all, then test all cases")
    args = parser.parse_args()

    tester = ONNX_IR_TESTER(args.chip)
    tester.test_single(args.case)

