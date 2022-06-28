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
class MHTracer_DTPStensorflowPScorePSopsPSarray_gradDTcc {
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
   MHTracer_DTPStensorflowPScorePSopsPSarray_gradDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSopsPSarray_gradDTcc() {
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

/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include <vector>
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

typedef FunctionDefHelper FDH;

REGISTER_OP_NO_GRADIENT("Shape");
REGISTER_OP_NO_GRADIENT("Rank");
REGISTER_OP_NO_GRADIENT("Size");
REGISTER_OP_NO_GRADIENT("ZerosLike");
REGISTER_OP_NO_GRADIENT("OnesLike");
REGISTER_OP_NO_GRADIENT("Const");
REGISTER_OP_NO_GRADIENT("EditDistance");
REGISTER_OP_NO_GRADIENT("StopGradient");

Status ReshapeGrad(const AttrSlice& attrs, FunctionDef* g) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSopsPSarray_gradDTcc mht_0(mht_0_v, 202, "", "./tensorflow/core/ops/array_grad.cc", "ReshapeGrad");

  // clang-format off
  *g = FDH::Define(
      // Arg defs
      {"x: T", "shape: int32", "dy: T"},
      // Ret val defs
      {"dx: T", "dshape: int32"},
      // Attr defs
      {"T: type"},
      // Nodes
      {
        {{"x_shape"}, "Shape", {"x"}, {{"T", "$T"}}},
        {{"dx"}, "Reshape", {"dy", "x_shape"}, {{"T", "$T"}}},
        {{"dshape"}, "ZerosLike", {"shape"}, {{"T", DT_INT32}}},
      });
  // clang-format on
  return Status::OK();
}
REGISTER_OP_GRADIENT("Reshape", ReshapeGrad);
REGISTER_OP_GRADIENT("ExpandDims", ReshapeGrad);

Status SqueezeGrad(const AttrSlice& attrs, FunctionDef* g) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSopsPSarray_gradDTcc mht_1(mht_1_v, 226, "", "./tensorflow/core/ops/array_grad.cc", "SqueezeGrad");

  // clang-format off
  *g = FDH::Define(
      // Arg defs
      {"x: T", "dy: T"},
      // Ret val defs
      {"dx: T"},
      // Attr defs
      {"T: type"},
      // Nodes
      {
        {{"x_shape"}, "Shape", {"x"}, {{"T", "$T"}}},
        {{"dx"}, "Reshape", {"dy", "x_shape"}, {{"T", "$T"}}},
      });
  // clang-format on
  return Status::OK();
}
REGISTER_OP_GRADIENT("Squeeze", SqueezeGrad);

Status IdentityGrad(const AttrSlice& attrs, FunctionDef* g) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSopsPSarray_gradDTcc mht_2(mht_2_v, 248, "", "./tensorflow/core/ops/array_grad.cc", "IdentityGrad");

  // clang-format off
  *g = FDH::Define(
      // Arg defs
      {"x: T", "dy: T"},
      // Ret val defs
      {"dx: T"},
      // Attr defs
      {"T: type"},
      // Nodes
      {
        {{"dx"}, "Identity", {"dy"}, {{"T", "$T"}}},
      });
  // clang-format on
  VLOG(1) << "IdentityGrad " << DebugString(*g);
  return Status::OK();
}
REGISTER_OP_GRADIENT("Identity", IdentityGrad);

Status PackGrad(const AttrSlice& attrs, FunctionDef* g) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSopsPSarray_gradDTcc mht_3(mht_3_v, 270, "", "./tensorflow/core/ops/array_grad.cc", "PackGrad");

  // clang-format off
  *g = FDH::Create(
      "_",
      // Arg defs
      {"x: N*T", "dy: T"},
      // Ret val defs
      {"dx: N*T"},
      // Attr defs
      {"T: type", "N: int", "axis: int"},
      // Nodes
      {
        {
          {"dx"},
          "Unpack",
          {"dy"},
          {{"T", "$T"}, {"num", "$N"}, {"axis", "$axis"}}
        },
      },
      {{"dx", "dx:output"}});
  // clang-format on
  VLOG(1) << "PackGrad " << DebugString(*g);
  return Status::OK();
}
REGISTER_OP_GRADIENT("Pack", PackGrad);

Status UnpackGrad(const AttrSlice& attrs, FunctionDef* g) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSopsPSarray_gradDTcc mht_4(mht_4_v, 299, "", "./tensorflow/core/ops/array_grad.cc", "UnpackGrad");

  // clang-format off
  *g = FDH::Define(
      // Arg defs
      {"x: T", "dy: num*T"},
      // Ret val defs
      {"dx: T"},
      // Attr defs
      {"T: type", "num: int", "axis: int"},
      // Nodes
      {
        {
          {"dx"},
          "Pack",
          {"dy"},
          {{"T", "$T"}, {"N", "$num"}, {"axis", "$axis"}}
        },
      });
  // clang-format on
  VLOG(1) << "UnpackGrad " << DebugString(*g);
  return Status::OK();
}
REGISTER_OP_GRADIENT("Unpack", UnpackGrad);

Status ConcatGradHelper(const AttrSlice& attrs, FunctionDef* g,
                        bool dim_is_last_arg) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSopsPSarray_gradDTcc mht_5(mht_5_v, 327, "", "./tensorflow/core/ops/array_grad.cc", "ConcatGradHelper");

  int N;
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "N", &N));
  DataType T;
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "T", &T));

  std::vector<string> shape_i;
  std::vector<string> offset_i;
  std::vector<string> dx_i;
  for (int i = 0; i < N; ++i) {
    shape_i.push_back(strings::StrCat("shapes:output:", i));
    offset_i.push_back(strings::StrCat("offset:offset:", i));
    dx_i.push_back(strings::StrCat("dx_", i, ":output:0"));
  }
  DataTypeVector dtype_list(N, T);

  // ConcatGrad(dim, x, dy):
  //   for i in range(N):
  //     dx[i] = Slice(dy, offset[i], shape[x[i]]),
  // where offset[i] is the offset of x[i] in the output y,
  // which is the same as dx[i]'s offset within dy.
  std::vector<FDH::Node> nodes{
      {{"shapes"}, "ShapeN", {"x"}, {{"T", "$T"}, {"N", "$N"}}},
      {{"offset"}, "ConcatOffset", {"dim", "shapes:output"}, {{"N", "$N"}}},
      {{"d_dim"}, "ZerosLike", {"dim"}, {{"T", DT_INT32}}},
      {{"dx"},
       "_ListToArray",
       dx_i,
       {{"T", "$T"}, {"N", "$N"}, {"Tin", DataTypeVector(N, T)}}}};

  // For each dx[i], we take a slice of dy. The offset and size of the
  // slice is given by offset[i] and shape[i].
  for (int i = 0; i < N; ++i) {
    nodes.push_back({{strings::StrCat("dx_", i)},
                     "Slice",
                     {"dy", offset_i[i], shape_i[i]},
                     {{"T", "$T"}, {"Index", DT_INT32}}});
  }
  if (dim_is_last_arg) {
    // clang-format off
    *g = FDH::Create(
        "_",
        // Arg defs
        {"x: N*T", "dim: int32", "dy: T"},
        // Return signature
        {"dx: N*T", "d_dim: int32"},
        // Attr defs
        {"T: type", "N: int"},
        // Nodes
        nodes,
        // Return values
        {{"dx", "dx:output"}, {"d_dim", "d_dim:y:0"}});
    // clang-format on
  } else {
    // clang-format off
    *g = FDH::Create(
        "_",
        // Arg defs
        {"dim: int32", "x: N*T", "dy: T"},
        // Return signature
        {"d_dim: int32", "dx: N*T"},
        // Attr defs
        {"T: type", "N: int"},
        // Nodes
        nodes,
        // Return values
        {{"dx", "dx:output"}, {"d_dim", "d_dim:y:0"}});
    // clang-format on
  }
  VLOG(1) << "ConcatGrad " << DebugString(*g);
  return Status::OK();
}

Status ConcatGrad(const AttrSlice& attrs, FunctionDef* g) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSopsPSarray_gradDTcc mht_6(mht_6_v, 403, "", "./tensorflow/core/ops/array_grad.cc", "ConcatGrad");

  return ConcatGradHelper(attrs, g, false);
}

Status ConcatGradV2(const AttrSlice& attrs, FunctionDef* g) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSopsPSarray_gradDTcc mht_7(mht_7_v, 410, "", "./tensorflow/core/ops/array_grad.cc", "ConcatGradV2");

  return ConcatGradHelper(attrs, g, true);
}

REGISTER_OP_GRADIENT("Concat", ConcatGrad);
REGISTER_OP_GRADIENT("ConcatV2", ConcatGradV2);

Status SplitGrad(const AttrSlice& attrs, FunctionDef* g) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSopsPSarray_gradDTcc mht_8(mht_8_v, 420, "", "./tensorflow/core/ops/array_grad.cc", "SplitGrad");

  // clang-format off
  *g = FDH::Define(
      // Arg defs
      {"dim: int32", "x: T", "dy: num_split*T"},
      // Ret val defs
      {"d_dim: int32", "dx: T"},
      // Attr defs
      {"T: type", "num_split: int"},
      // Nodes
      {
        {{"d_dim"}, "ZerosLike", {"dim"}, {{"T", DT_INT32}}},
        {{"dx"}, "Concat", {"dim", "dy"}, {{"T", "$T"}, {"N", "$num_split"}}}
      });
  // clang-format on
  VLOG(1) << "SplitGrad " << DebugString(*g);
  return Status::OK();
}
REGISTER_OP_GRADIENT("Split", SplitGrad);

Status SplitVGrad(const AttrSlice& attrs, FunctionDef* g) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSopsPSarray_gradDTcc mht_9(mht_9_v, 443, "", "./tensorflow/core/ops/array_grad.cc", "SplitVGrad");

  // clang-format off
  *g = FDH::Define(
      // Arg defs
      {"x: T", "size_splits: Tlen", "dim: int32", "dy: num_split*T"},
      // Ret val defs
      {"dx: T", "d_size_splits: Tlen", "d_dim: int32"},
      // Attr defs
      {"T: type", "Tlen: type", "num_split: int"},
      // Nodes
      {
        {{"dx"}, "Concat", {"dim", "dy"}, {{"T", "$T"}, {"N", "$num_split"}}},
        {{"d_size_splits"}, "ZerosLike", {"size_splits"}, {{"T", "$Tlen"}}},
        {{"d_dim"}, "ZerosLike", {"dim"}, {{"T", DT_INT32}}},
      });
  // clang-format on
  VLOG(1) << "SplitVGrad " << DebugString(*g);
  return Status::OK();
}
REGISTER_OP_GRADIENT("SplitV", SplitVGrad);

Status ArrayToListGrad(const AttrSlice& attrs, FunctionDef* g) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSopsPSarray_gradDTcc mht_10(mht_10_v, 467, "", "./tensorflow/core/ops/array_grad.cc", "ArrayToListGrad");

  int N;
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "N", &N));
  std::vector<string> dys;
  dys.reserve(N);
  for (int i = 0; i < N; ++i) {
    dys.push_back(strings::StrCat("dy:", i));
  }
  // clang-format off
  *g = FDH::Define(
      // Arg defs
      {"x: N*T", "dy: out_types"},
      // Ret val defs
      {"dx: N*T"},
      // Attr defs
      {"T: type", "N: int", "out_types: list(type)"},
      // Nodes
      {
        {{"dx"}, "_ListToArray", dys,
         {{"T", "$T"}, {"N", "$N"}, {"Tin", "$out_types"}}}
      });
  // clang-format on
  VLOG(1) << "ArrayToListGrad " << DebugString(*g);
  return Status::OK();
}
REGISTER_OP_GRADIENT("_ArrayToList", ArrayToListGrad);

Status ListToArrayGrad(const AttrSlice& attrs, FunctionDef* g) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSopsPSarray_gradDTcc mht_11(mht_11_v, 497, "", "./tensorflow/core/ops/array_grad.cc", "ListToArrayGrad");

  // clang-format off
  *g = FDH::Define(
      // Arg defs
      {"x: Tin", "dy: N*T"},
      // Ret val defs
      {"dx: Tin"},
      // Attr defs
      {"T: type", "N: int", "Tin: list(type)"},
      // Nodes
      {
        {{"dx"}, "_ArrayToList", {"dy"},
         {{"T", "$T"}, {"N", "$N"}, {"out_types", "$Tin"}}}
      });
  // clang-format on
  VLOG(1) << "ListToArrayGrad " << DebugString(*g);
  return Status::OK();
}
REGISTER_OP_GRADIENT("_ListToArray", ListToArrayGrad);

Status FillGrad(const AttrSlice& attrs, FunctionDef* g) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSopsPSarray_gradDTcc mht_12(mht_12_v, 520, "", "./tensorflow/core/ops/array_grad.cc", "FillGrad");

  *g = FDH::Define(
      // Arg defs
      {"dims: int32", "x: T", "dy: T"},
      // Ret val defs
      {"d_dims: int32", "dx: T"},
      // Attr defs
      {"T: type"},
      // Nodes
      {
          {{"d_dims"}, "ZerosLike", {"dims"}, {{"T", DT_INT32}}},
          FDH::Const("zero", 0),
          {{"rank"}, "Rank", {"dy"}, {{"T", "$T"}}},
          FDH::Const("one", 1),
          {{"r"}, "Range", {"zero", "rank", "one"}, {}},
          // dx = sum(dy)
          {{"dx"}, "Sum", {"dy", "r"}, {{"T", "$T"}}},
      });
  VLOG(1) << "FillGrad " << DebugString(*g);
  return Status::OK();
}
REGISTER_OP_GRADIENT("Fill", FillGrad);

Status TransposeGrad(const AttrSlice& attrs, FunctionDef* g) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSopsPSarray_gradDTcc mht_13(mht_13_v, 546, "", "./tensorflow/core/ops/array_grad.cc", "TransposeGrad");

  *g = FDH::Define(
      // Arg defs
      {"x: T", "p: int32", "dy: T"},
      // Ret val defs
      {"dx: T", "dp: int32"},
      // Attr defs
      {"T: type"},
      // Nodes
      {
          {{"q"}, "InvertPermutation", {"p"}, {}},
          {{"dx"}, "Transpose", {"dy", "q"}, {{"T", "$T"}}},
          {{"dp"}, "ZerosLike", {"p"}, {{"T", DT_INT32}}},
      });
  VLOG(1) << "TransposeGrad " << DebugString(*g);
  return Status::OK();
}
REGISTER_OP_GRADIENT("Transpose", TransposeGrad);

Status GatherNdGrad(const AttrSlice& attrs, FunctionDef* g) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSopsPSarray_gradDTcc mht_14(mht_14_v, 568, "", "./tensorflow/core/ops/array_grad.cc", "GatherNdGrad");

  // clang-format off
  *g = FDH::Define(
      // Arg defs
      {"params: Tparams", "indices: Tindices", "doutput: Tparams"},
      // Ret val defs
      {"dparams: Tparams", "dindices: Tindices"},
      // Attr defs
      {"Tparams: type", "Tindices: type"},
      // Nodes
      {
        {{"x_shape"}, "Shape", {"params"}, {{"T", "$Tparams"}}},
        {{"dparams"}, "ScatterNd", {"indices", "doutput", "x_shape"},
         {{"T", "$Tparams"}, {"Tindices", "$Tindices"}}},
        {{"dindices"}, "ZerosLike", {"indices"}, {{"T", "$Tindices"}}},
      });
  // clang-format on
  return Status::OK();
}
REGISTER_OP_GRADIENT("GatherNd", GatherNdGrad);

Status ConjugateTransposeGrad(const AttrSlice& attrs, FunctionDef* g) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSopsPSarray_gradDTcc mht_15(mht_15_v, 592, "", "./tensorflow/core/ops/array_grad.cc", "ConjugateTransposeGrad");

  *g = FDH::Define(
      // Arg defs
      {"x: T", "p: int32", "dy: T"},
      // Ret val defs
      {"dx: T", "dp: int32"},
      // Attr defs
      {"T: type"},
      // Nodes
      {
          {{"q"}, "InvertPermutation", {"p"}, {}},
          {{"dx"}, "ConjugateTranspose", {"dy", "q"}, {{"T", "$T"}}},
          {{"dp"}, "ZerosLike", {"p"}, {{"T", DT_INT32}}},
      });
  VLOG(1) << "ConjugateTransposeGrad " << DebugString(*g);
  return Status::OK();
}
REGISTER_OP_GRADIENT("ConjugateTranspose", ConjugateTransposeGrad);

Status ReverseGrad(const AttrSlice& attrs, FunctionDef* g) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSopsPSarray_gradDTcc mht_16(mht_16_v, 614, "", "./tensorflow/core/ops/array_grad.cc", "ReverseGrad");

  *g = FDH::Define(
      // Arg defs
      {"x: T", "d: bool", "dy: T"},
      // Ret val defs
      {"dx: T", "dd: bool"},
      // Attr defs
      {"T: type"},
      // Nodes
      {
          {{"dx"}, "Reverse", {"dy", "d"}, {{"T", "$T"}}},
          {{"dd"}, "ZerosLike", {"d"}, {{"T", DT_BOOL}}},
      });
  VLOG(1) << "ReverseGrad " << DebugString(*g);
  return Status::OK();
}
REGISTER_OP_GRADIENT("Reverse", ReverseGrad);

Status ReverseV2Grad(const AttrSlice& attrs, FunctionDef* g) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSopsPSarray_gradDTcc mht_17(mht_17_v, 635, "", "./tensorflow/core/ops/array_grad.cc", "ReverseV2Grad");

  DataType itype;
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "Tidx", &itype));
  if (itype != DT_INT32) {
    return errors::Unimplemented(
        "ReverseV2Grad for int64 index are not supported.");
  }
  *g = FDH::Define(
      // Arg defs
      {"x: T", "d: int32", "dy: T"},
      // Ret val defs
      {"dx: T", "dd: int32"},
      // Attr defs
      {"T: type", "Tidx: {int32, int64}"},
      // Nodes
      {
          {{"dx"}, "ReverseV2", {"dy", "d"}, {{"T", "$T"}}},
          {{"dd"}, "ZerosLike", {"d"}, {{"T", "$Tidx"}}},
      });
  VLOG(1) << "ReverseGrad " << DebugString(*g);
  return Status::OK();
}
REGISTER_OP_GRADIENT("ReverseV2", ReverseV2Grad);

Status SliceGrad(const AttrSlice& attrs, FunctionDef* g) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSopsPSarray_gradDTcc mht_18(mht_18_v, 662, "", "./tensorflow/core/ops/array_grad.cc", "SliceGrad");

  DataType itype;
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "Index", &itype));
  if (itype != DT_INT32) {
    return errors::Unimplemented(
        "SliceGrad for int64 index are not supported.");
  }
  *g = FDH::Define(
      // Arg defs
      {"x: T", "begin: int32", "size: int32", "dy: T"},
      // Ret val defs
      {"dx: T", "begin_grad: int32", "size_grad: int32"},
      // Attr defs
      {"T: type"},
      // Nodes
      {// paddings = concat(1, [begin, shape(x) - begin - size])
       FDH::Const("one", 1),
       {{"b1"}, "ExpandDims", {"begin", "one"}, {{"T", DT_INT32}}},
       {{"xs"}, "Shape", {"x"}, {{"T", "$T"}}},
       {{"xs_b"}, "Sub", {"xs", "begin"}, {{"T", DT_INT32}}},
       {{"xs_b_s"}, "Sub", {"xs_b", "size"}, {{"T", DT_INT32}}},
       {{"a1"}, "ExpandDims", {"xs_b_s", "one"}, {{"T", DT_INT32}}},
       {{"paddings"},
        "Concat",
        {"one", "b1", "a1"},
        {{"N", 2}, {"T", DT_INT32}}},
       // dx = Pad(dy, paddings)
       {{"dx"}, "Pad", {"dy", "paddings"}, {{"T", "$T"}}},
       {{"begin_grad"}, "ZerosLike", {"begin"}, {{"T", DT_INT32}}},
       {{"size_grad"}, "ZerosLike", {"size"}, {{"T", DT_INT32}}}});
  VLOG(1) << "SliceGrad " << DebugString(*g);
  return Status::OK();
}
REGISTER_OP_GRADIENT("Slice", SliceGrad);

Status StridedSliceGrad(const AttrSlice& attrs, FunctionDef* g) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSopsPSarray_gradDTcc mht_19(mht_19_v, 700, "", "./tensorflow/core/ops/array_grad.cc", "StridedSliceGrad");

  DataType itype;
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "Index", &itype));
  if (itype != DT_INT32) {
    return errors::Unimplemented(
        "SliceGrad for int64 index are not supported.");
  }

  *g = FDH::Define(
      // Arg defs
      {"x: T", "begin: int32", "end: int32", "stride: int32", "dy: T"},
      // Ret val defs
      {"dx: T", "begin_grad: int32", "end_grad: int32", "stride_grad: int32"},
      // Attr defs
      {"T: type", "Index: {int32, int64}", "begin_mask: int", "end_mask: int",
       "ellipsis_mask: int", "new_axis_mask: int", "shrink_axis_mask: int"},
      {// Nodes
       {{{"xs"}, "Shape", {"x"}, {{"T", "$T"}}},
        {{"dx"},
         "StridedSliceGrad",
         {"xs", "begin", "end", "stride", "dy"},
         {{"T", "$T"},
          {"Index", "$Index"},
          {"begin_mask", "$begin_mask"},
          {"end_mask", "$end_mask"},
          {"ellipsis_mask", "$ellipsis_mask"},
          {"new_axis_mask", "$new_axis_mask"},
          {"shrink_axis_mask", "$shrink_axis_mask"}}},
        {{"begin_grad"}, "ZerosLike", {"begin"}, {{"T", DT_INT32}}},
        {{"end_grad"}, "ZerosLike", {"end"}, {{"T", DT_INT32}}},
        {{"stride_grad"}, "ZerosLike", {"stride"}, {{"T", DT_INT32}}}}});

  VLOG(1) << "StridedSliceGrad " << DebugString(*g);
  return Status::OK();
}
REGISTER_OP_GRADIENT("StridedSlice", StridedSliceGrad);

Status StridedSliceGradGrad(const AttrSlice& attrs, FunctionDef* g) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSopsPSarray_gradDTcc mht_20(mht_20_v, 740, "", "./tensorflow/core/ops/array_grad.cc", "StridedSliceGradGrad");

  DataType itype;
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "Index", &itype));
  if (itype != DT_INT32) {
    return errors::Unimplemented(
        "SliceGrad for int64 index are not supported.");
  }

  // TODO(aselle): Shouldn't the int32 tensors return zeros of shape like
  // dy_grad?
  // I'm following slice's behavior for now.
  *g = FDH::Define(
      // Arg defs
      {"shape: int32", "begin: int32", "end: int32", "stride: int32", "dy: T",
       "grad: T"},
      // Ret val defs
      {"shape_grad: int32", "begin_grad: int32", "end_grad: int32",
       "stride_grad: int32", "dy_grad: T"},
      // Attr defs
      {"T: type", "Index: {int32, int64}", "begin_mask: int", "end_mask: int",
       "ellipsis_mask: int", "new_axis_mask: int", "shrink_axis_mask: int"},
      {// Nodes
       {{{"shape_grad"}, "ZerosLike", {"shape"}, {{"T", DT_INT32}}},
        {{"begin_grad"}, "ZerosLike", {"begin"}, {{"T", DT_INT32}}},
        {{"end_grad"}, "ZerosLike", {"end"}, {{"T", DT_INT32}}},
        {{"stride_grad"}, "ZerosLike", {"stride"}, {{"T", DT_INT32}}},
        {{"dy_grad"},
         "StridedSlice",
         {"grad", "begin", "end", "stride"},
         {{"T", "$T"},
          {"Index", "$Index"},
          {"begin_mask", "$begin_mask"},
          {"end_mask", "$end_mask"},
          {"ellipsis_mask", "$ellipsis_mask"},
          {"new_axis_mask", "$new_axis_mask"},
          {"shrink_axis_mask", "$shrink_axis_mask"}}}}});

  VLOG(1) << "StridedSliceGrad " << DebugString(*g);
  return Status::OK();
}
REGISTER_OP_GRADIENT("StridedSliceGrad", StridedSliceGradGrad);

Status BroadcastToGrad(const AttrSlice& attrs, FunctionDef* g) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSopsPSarray_gradDTcc mht_21(mht_21_v, 785, "", "./tensorflow/core/ops/array_grad.cc", "BroadcastToGrad");

  DataType itype;
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "Tidx", &itype));
  if (itype != DT_INT32) {
    return errors::Unimplemented(
        "BroadcastToGrad for int64 index are not supported.");
  }
  std::vector<FDH::Node> nodes = {
      {{"sx"}, "Shape", {"x"}, {{"T", "$T"}}},
      {{"rx", "ry"}, "BroadcastGradientArgs", {"sx", "shape"}},
      {{"sum_gx"}, "Sum", {"dy", "rx"}, {{"T", "$T"}}},
      {{"dx"}, "Reshape", {"sum_gx", "sx"}, {{"T", "$T"}}},
      {{"dshape"}, "ZerosLike", {"shape"}, {{"T", "$Tidx"}}}};
  *g = FDH::Define(
      // Arg defs
      {"x: T", "shape: int32", "dy: T"},
      // Ret val defs
      {"dx: T", "dshape: Tidx"},
      // Attr defs
      {{"T: type"}, {"Tidx: {int32, int64}"}},
      // Nodes
      nodes);
  return Status::OK();
}
REGISTER_OP_GRADIENT("BroadcastTo", BroadcastToGrad);

}  // end namespace tensorflow
