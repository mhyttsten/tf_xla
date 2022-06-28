#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>
#include <string>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>
#include <stdlib.h>
#include <unistd.h>
class MHTracer_DTPStensorflowPScorePSopsPSrisc_opsDTcc {
public:
   std::string _s;
   int _indent = 0;
   std::string _functionName;
   bool _isFile = false;
   std::string _fileName;
   std::string _envMHIndent;
   int _lineNumber;
   bool _filtered = false;
   bool _otherThread = false;
   MHTracer_DTPStensorflowPScorePSopsPSrisc_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
      _functionName = functionName;
      _lineNumber = lineNumber;

      // Check if tracing is enabled
      const char* env_path = std::getenv("PATH");
      if (env_path != nullptr && std::string(env_path).find("MHTRACER_ENABLE") == std::string::npos) {
         return;
      }
      // Should we trace of filter?
      const char* env_filter = std::getenv("MHTRACER_FILTER");
      if (env_filter != nullptr) {
         std::string sfilter = std::string(env_filter);
         std::string sLineNumber = std::to_string(lineNumber);
         while (true) {
            std::size_t ioE = sfilter.find(";");
            if (sfilter.size() == 0) {
               break;
            }
            std::string cfs = sfilter.substr(0, ioE);
            std::size_t ioFileName = cfs.find("|");
            std::string fFileName  = cfs.substr(0, ioFileName);
            std::size_t ioFunctionName = cfs.find("|", ioFileName+1);
            std::string fFunctionName  = cfs.substr(ioFileName+1, ioFunctionName-ioFileName-1);
            std::string fLineNumber    = cfs.substr(ioFunctionName+1, cfs.size()-ioFunctionName-1);

            if (  (fFileName == "*" || fFileName == fileName)
               && (fFunctionName == "*" || fFunctionName == functionName)
               && (fLineNumber == "*" || fLineNumber == sLineNumber)) {
              _filtered = true;
               return;
            }

            if (ioE == std::string::npos) {
               sfilter = "";
            } else {
               sfilter = sfilter.substr(ioE+1, sfilter.size()-ioE-1);
            }
         }
      }

      // Create log string
      std::string ostr;

      // Assign indent spaces (tied to PID and TID)
      pid_t pid = getpid();
      std::thread::id tid = std::this_thread::get_id();
      std::stringstream pid_dash_tid_ss;
      pid_dash_tid_ss << pid << "-" << tid;
      std::string pid_dash_tid_str = pid_dash_tid_ss.str();
      _envMHIndent = "MHTRACER_INDENT_";
      char* env_indent = std::getenv(_envMHIndent.c_str());
      if (env_indent != nullptr) {
         _indent = std::stoi(std::string(env_indent));
      }
      _s.assign(_indent, ' ');

      // Check that reporting matches pid/tid
      const char* env_pid_dash_tid = std::getenv("MHTRACER_PID_DASH_TID");
      if (env_pid_dash_tid != nullptr) {
         std::string env_pid_dash_tid_str(env_pid_dash_tid);
         if (env_pid_dash_tid_str != pid_dash_tid_str) {
            _otherThread = true;
         }
      }
      else {  // PID-THREAD not set, set it for the first time (starter thread)
         setenv("MHTRACER_PID_DASH_TID", pid_dash_tid_str.c_str(), 1);
      }

      std::string paramStr;
      for (int i=0; i < params.size(); i++) {
         auto e = params[i];
         while (e.find("\n") != std::string::npos) {
            size_t pos = e.find("\n");
            e = e.erase(pos, 1);
            e = e.insert(pos, "<NL>");
         }
         while (e.find("[") != std::string::npos) {
            size_t pos = e.find("[");
            e = e.erase(pos, 1);
            e = e.insert(pos, "<LB>");
         }
         while (e.find("]") != std::string::npos) {
            size_t pos = e.find("]");
            e = e.erase(pos, 1);
            e = e.insert(pos, "<RB>");
         }
         paramStr += e;
         if ((i+1) < params.size()) {
            paramStr += ", ";
         }
      }

      const char* env_dont_print_pid_dash_tid = std::getenv("MHTRACER_DONT_PRINT_PID_DASH_TID");
      if (env_dont_print_pid_dash_tid != nullptr) {
         pid_dash_tid_str = "";
      }
      if (_otherThread) {
         functionName = "MHOT_" + functionName;
      }
      ostr += _s + functionName + 
         + " [1]"
         + " [" + prefix + "]"
         + " [" + paramStr + "]"
         + " [" + pid_dash_tid_str + " "
         +    std::to_string(lineNumber)
         +    " @ " + fileName + "]\n";

      // Log to file
      if (env_path != nullptr && std::string(env_path).find("MHTRACER_USEFILE") != std::string::npos) {
         _isFile = true;
         _fileName = "/tmp/mhtracer_" + pid_dash_tid_str + ".log";
         std::ofstream os;
         os.open(_fileName, std::ofstream::out | std::ofstream::app);
         os << ostr << "";
         os.close();
      }
      // Log to stdout
      else {
         std::cout << ostr << "";
      }

      // Increment indent spaces
      if (_otherThread) {
         return;
      }
      _indent += 3;
      setenv(_envMHIndent.c_str(), std::to_string(_indent).c_str(), 1);
   }
   ~MHTracer_DTPStensorflowPScorePSopsPSrisc_opsDTcc() {
      // Check if tracing is enabled
      char* env_path = std::getenv("PATH");
      if (env_path != nullptr && std::string(env_path).find("MHTRACER_ENABLE") == std::string::npos) {
         return;
      }

      // Don't update indent if tracing was filtered or from another thread
      if (_filtered || _otherThread) {
         return;
      }

      _indent -= 3;
      setenv(_envMHIndent.c_str(), std::to_string(_indent).c_str(), 1);
   }
};

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

namespace {
Status RiscBinaryNonBroadcastOpShapeFn(shape_inference::InferenceContext* c) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSopsPSrisc_opsDTcc mht_0(mht_0_v, 194, "", "./tensorflow/core/ops/risc_ops.cc", "RiscBinaryNonBroadcastOpShapeFn");

  const auto rank = c->Rank(c->input(0));
  if (rank != c->Rank(c->input(1))) {
    return errors::InvalidArgument("Mismatch rank for input.");
  }
  for (int i = 0; i < rank; ++i) {
    if (!c->ValueKnown(c->Dim(c->input(0), i)) ||
        !c->ValueKnown(c->Dim(c->input(1), i))) {
      continue;
    }
    if (c->Value(c->Dim(c->input(0), i)) != c->Value(c->Dim(c->input(1), i))) {
      return errors::InvalidArgument("Mismatch shapes for input.");
    }
  }
  c->set_output(0, c->input(0));
  auto* handle_data = c->input_handle_shapes_and_types(0);
  if (handle_data != nullptr) {
    c->set_output_handle_shapes_and_types(0, *handle_data);
  }
  return Status::OK();
}
}  // namespace

REGISTER_OP("RiscAbs")
    .Input("x: T")
    .Output("y: T")
    .Attr("T: {bfloat16, half, float, double}")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("RiscAdd")
    .Input("x: T")
    .Input("y: T")
    .Output("z: T")
    .Attr("T: {bfloat16, half, float, double}")
    .SetShapeFn(RiscBinaryNonBroadcastOpShapeFn)
    .SetIsAggregate()
    .SetIsCommutative();

// TODO(b/178234771): retire this.
REGISTER_OP("RiscBinaryArithmetic")
    .Input("x: T")
    .Input("y: T")
    .Output("z: T")
    .Attr("op_type: {'ADD', 'SUB', 'MUL', 'DIV', 'REM', 'MIN', 'POW'}")
    .Attr("T: {bfloat16, half, float, double}")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("RiscBinaryComparison")
    .Input("x: T")
    .Input("y: T")
    .Output("z: bool")
    .Attr("op_type: {'EQ', 'NE', 'GE', 'GT', 'LE', 'LT'}")
    .Attr("T: {bfloat16, half, float, double}")
    .SetShapeFn(RiscBinaryNonBroadcastOpShapeFn);

// TODO(b/178234771): change shape function.
REGISTER_OP("RiscBitcast")
    .Input("x: SrcT")
    .Output("y: DstT")
    .Attr("SrcT: type")
    .Attr("DstT: type")
    .SetShapeFn(shape_inference::UnknownShape);

// TODO(b/178234771): change shape function.
REGISTER_OP("RiscBroadcast")
    .Input("input: T")
    .Input("shape: Tidx")
    .Output("output: T")
    .Attr("T: type")
    .Attr("Tidx: {int32, int64} = DT_INT32")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("RiscCast")
    .Input("x: SrcT")
    .Output("y: DstT")
    .Attr("SrcT: type")
    .Attr("DstT: type")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("RiscCeil")
    .Input("x: T")
    .Output("y: T")
    .Attr("T: {bfloat16, half, float, double}")
    .SetShapeFn(shape_inference::UnchangedShape);

// TODO(b/178234771): change shape function.
REGISTER_OP("RiscCholesky")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: {bfloat16, half, float, double}")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("RiscConcat")
    .Input("values: N * T")
    .Input("axis: Tidx")
    .Output("output: T")
    .Attr("N: int >= 2")
    .Attr("T: type")
    .Attr("Tidx: {int32, int64} = DT_INT32")
    .SetShapeFn(shape_inference::ConcatV2Shape);

// TODO(b/178234771): change shape function.
REGISTER_OP("RiscCondition")
    .Input("pred: bool")
    .Input("input_true: SrcT")
    .Input("input_false: SrcT")
    .Output("output: DstT")
    .Attr("func_true: func")
    .Attr("func_false: func")
    .Attr("SrcT: {bfloat16, half, float, double}")
    .Attr("DstT: {bfloat16, half, float, double}")
    .SetShapeFn(shape_inference::UnknownShape);

// TODO(b/178234771): change shape function.
REGISTER_OP("RiscConv")
    .Input("input: T")
    .Input("filter: T")
    .Output("output: T")
    .Attr("T: {bfloat16, half, float, double}")
    .Attr("strides: list(int)")
    .Attr(GetConvnetDataFormatAttrString())
    .SetShapeFn(shape_inference::UnknownShape)
    .Attr("dilations: list(int) = [1, 1, 1, 1]");

REGISTER_OP("RiscCos")
    .Input("x: T")
    .Output("y: T")
    .Attr("T: {bfloat16, half, float, double}")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("RiscDiv")
    .Input("x: T")
    .Input("y: T")
    .Output("z: T")
    .Attr("T: {bfloat16, half, float, double}")
    .SetShapeFn(RiscBinaryNonBroadcastOpShapeFn);

REGISTER_OP("RiscDot")
    .Input("a: T")
    .Input("b: T")
    .Output("product: T")
    .Attr("transpose_a: bool = false")
    .Attr("transpose_b: bool = false")
    .Attr("T: {bfloat16, half, float, double}")
    .SetShapeFn(shape_inference::MatMulShape);

REGISTER_OP("RiscExp")
    .Input("x: T")
    .Output("y: T")
    .Attr("T: {bfloat16, half, float, double}")
    .SetShapeFn(shape_inference::UnchangedShape);

// TODO(b/178234771): change shape function.
REGISTER_OP("RiscFft")
    .Input("input: Tcomplex")
    .Output("output: Tcomplex")
    .Attr("Tcomplex: {complex64, complex128} = DT_COMPLEX64")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("RiscFloor")
    .Input("x: T")
    .Output("y: T")
    .Attr("T: {bfloat16, half, float, double}")
    .SetShapeFn(shape_inference::UnchangedShape);

// TODO(b/178234771): change shape function.
REGISTER_OP("RiscGather")
    .Input("params: Tparams")
    .Input("indices: Tindices")
    .Input("axis: Taxis")
    .Attr("batch_dims: int = 0")
    .Output("output: Tparams")
    .Attr("Tparams: type")
    .Attr("Tindices: {int32,int64}")
    .Attr("Taxis: {int32,int64}")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("RiscImag")
    .Input("input: T")
    .Output("output: Tout")
    .Attr("T: {complex64, complex128} = DT_COMPLEX64")
    .Attr("Tout: {float, double} = DT_FLOAT")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("RiscIsFinite")
    .Input("x: T")
    .Output("y: bool")
    .Attr("T: {bfloat16, half, float, double}")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("RiscLog")
    .Input("x: T")
    .Output("y: T")
    .Attr("T: {bfloat16, half, float, double}")
    .SetShapeFn(shape_inference::UnchangedShape);

// TODO(b/178234771): change shape function.
REGISTER_OP("RiscLogicalAnd")
    .Input("x: bool")
    .Input("y: bool")
    .Output("z: bool")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("RiscLogicalNot")
    .Input("x: bool")
    .Output("z: bool")
    .SetShapeFn(shape_inference::UnchangedShape);

// TODO(b/178234771): change shape function.
REGISTER_OP("RiscLogicalOr")
    .Input("x: bool")
    .Input("y: bool")
    .Output("z: bool")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("RiscMax")
    .Input("x: T")
    .Input("y: T")
    .Output("max: T")
    .Attr("T: {bfloat16, half, float, double}")
    .SetShapeFn(RiscBinaryNonBroadcastOpShapeFn);

REGISTER_OP("RiscMin")
    .Input("x: T")
    .Input("y: T")
    .Output("z: T")
    .Attr("T: {bfloat16, half, float, double}")
    .SetShapeFn(RiscBinaryNonBroadcastOpShapeFn);

REGISTER_OP("RiscMul")
    .Input("x: T")
    .Input("y: T")
    .Output("z: T")
    .Attr("T: {bfloat16, half, float, double}")
    .SetShapeFn(RiscBinaryNonBroadcastOpShapeFn);

REGISTER_OP("RiscNeg")
    .Input("x: T")
    .Output("y: T")
    .Attr("T: {bfloat16, half, float, double}")
    .SetShapeFn(shape_inference::UnchangedShape);

// TODO(b/178234771): change shape function.
REGISTER_OP("RiscPad")
    .Input("input: T")
    .Input("paddings: Tpaddings")
    .Input("constant_values: T")
    .Output("output: T")
    .Attr("T: {bfloat16, half, float, double}")
    .Attr("Tpaddings: {int32, int64} = DT_INT32")
    .SetShapeFn(shape_inference::UnknownShape);

// TODO(b/178234771): change shape function.
REGISTER_OP("RiscPool")
    .Input("value: T")
    .Output("output: T")
    .Attr("ksize: list(int) >= 4")
    .Attr("strides: list(int) >= 4")
    .Attr("pooling_type: {'AVG', 'MAX'}")
    .Attr(GetConvnetDataFormatAttrString())
    .Attr("T: {bfloat16, half, float, double}")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("RiscPow")
    .Input("x: T")
    .Input("y: T")
    .Output("z: T")
    .Attr("T: {bfloat16, half, float, double}")
    .SetShapeFn(RiscBinaryNonBroadcastOpShapeFn);

REGISTER_OP("RiscRandomUniform")
    .Input("shape: T")
    .Output("output: float")
    .Attr("seed: int = 0")
    .Attr("T: {int32, int64}")
    .SetShapeFn(shape_inference::RandomShape);

REGISTER_OP("RiscReal")
    .Input("input: T")
    .Output("output: Tout")
    .Attr("T: {complex64, complex128} = DT_COMPLEX64")
    .Attr("Tout: {float, double} = DT_FLOAT")
    .SetShapeFn(shape_inference::UnchangedShape);

// TODO(b/178234771): change shape function.
REGISTER_OP("RiscReduce")
    .Input("tensor: T")
    .Input("axis: Index")
    .Output("output: T")
    .Attr("reduce_type: {'MEAN', 'SUM'}")
    .Attr("Index: {int32,int64} = DT_INT32")
    .Attr("T: {bfloat16, half, float, double}")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("RiscRem")
    .Input("x: T")
    .Input("y: T")
    .Output("z: T")
    .Attr("T: {bfloat16, half, float, double}")
    .SetShapeFn(RiscBinaryNonBroadcastOpShapeFn);

// TODO(b/178234771): change shape function.
REGISTER_OP("RiscReshape")
    .Input("tensor: T")
    .Input("shape: Tshape")
    .Output("output: T")
    .Attr("T: {bfloat16, half, float, double}")
    .Attr("Tshape: {int32, int64} = DT_INT32")
    .SetShapeFn(shape_inference::UnknownShape);

// TODO(b/178234771): change shape function.
REGISTER_OP("RiscReverse")
    .Input("tensor: T")
    .Input("axis: Tidx")
    .Output("output: T")
    .Attr("Tidx: {int32, int64} = DT_INT32")
    .Attr("T: {bfloat16, half, float, double}")
    .SetShapeFn(shape_inference::UnknownShape);

// TODO(b/178234771): change shape function.
REGISTER_OP("RiscScatter")
    .Input("indices: Tindices")
    .Input("updates: T")
    .Input("shape: Tindices")
    .Output("output: T")
    .Attr("T: {bfloat16, half, float, double}")
    .Attr("Tindices: {int32, int64}")
    .SetShapeFn(shape_inference::UnknownShape);

// TODO(b/178234771): change shape function.
REGISTER_OP("RiscShape")
    .Input("input: T")
    .Output("output: out_type")
    .Attr("T: {bfloat16, half, float, double}")
    .Attr("out_type: {int32, int64} = DT_INT32")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("RiscSign")
    .Input("x: T")
    .Output("y: T")
    .Attr("T: {bfloat16, half, float, double}")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("RiscSlice")
    .Input("input: T")
    .Input("begin: Index")
    .Input("size: Index")
    .Output("output: T")
    .Attr("T: {bfloat16, half, float, double}")
    .Attr("Index: {int32,int64}")
    .SetShapeFn(shape_inference::SliceShape);

REGISTER_OP("RiscSort")
    .Input("input: T")
    .Input("axis: Index")
    .Output("output: T")
    .Attr("Index: {int32,int64} = DT_INT32")
    .Attr("T: {bfloat16, half, float, double}")
    .Attr("direction: {'ASCENDING', 'DESCENDING'}")
    .SetShapeFn(shape_inference::UnchangedShape);

// TODO(b/178234771): change shape function.
REGISTER_OP("RiscSqueeze")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: type")
    .Attr("squeeze_dims: list(int) >= 0 = []")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("RiscSub")
    .Input("x: T")
    .Input("y: T")
    .Output("z: T")
    .Attr("T: {bfloat16, half, float, double}")
    .SetShapeFn(RiscBinaryNonBroadcastOpShapeFn);

// TODO(b/178234771): change shape function.
REGISTER_OP("RiscTranspose")
    .Input("x: T")
    .Input("perm: Tperm")
    .Output("y: T")
    .Attr("T: type")
    .Attr("Tperm: {int32, int64} = DT_INT32")
    .SetShapeFn(shape_inference::UnknownShape);

// TODO(b/178234771): change shape function.
REGISTER_OP("RiscTriangularSolve")
    .Input("matrix: T")
    .Input("rhs: T")
    .Output("output: T")
    .Attr("lower: bool = True")
    .Attr("adjoint: bool = False")
    .Attr("T: {bfloat16, half, float, double}")
    .SetShapeFn(shape_inference::UnknownShape);

// TODO(b/178234771): retire this.
REGISTER_OP("RiscUnary")
    .Input("x: T")
    .Output("y: T")
    .Attr(
        "op_type: {'ABL', 'CEIL', 'COS', 'EXP', 'FLOOR', 'IMAG', 'LOG', 'NEG', "
        "'REAL', 'SIGN'}")
    .Attr("T: {bfloat16, half, float, double}")
    .SetShapeFn(shape_inference::UnchangedShape);

// TODO(b/178234771): change shape function.
REGISTER_OP("RiscWhile")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: list(type) >= 0")
    .Attr("cond: func")
    .Attr("body: func")
    .Attr("output_shapes: list(shape) = []")
    .Attr("parallel_iterations: int = 10")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnknownShape);

}  // namespace tensorflow
