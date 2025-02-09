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
class MHTracer_DTPStensorflowPScorePSopsPSsparse_opsDTcc {
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
   MHTracer_DTPStensorflowPScorePSopsPSsparse_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSopsPSsparse_opsDTcc() {
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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/errors.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

namespace {

Status SparseSparseMinOrMaxShapeFn(InferenceContext* c) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSopsPSsparse_opsDTcc mht_0(mht_0_v, 199, "", "./tensorflow/core/ops/sparse_ops.cc", "SparseSparseMinOrMaxShapeFn");

  ShapeHandle unused;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &unused));  // a_indices
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &unused));  // a_values
  TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &unused));  // a_shape
  TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 2, &unused));  // b_indices
  TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 1, &unused));  // b_values
  TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 1, &unused));  // b_shape
  c->set_output(0, c->Matrix(InferenceContext::kUnknownDim,
                             InferenceContext::kUnknownDim));
  c->set_output(1, c->Vector(InferenceContext::kUnknownDim));
  return Status::OK();
}

}  // namespace

REGISTER_OP("SparseAddGrad")
    .Input("backprop_val_grad: T")
    .Input("a_indices: int64")
    .Input("b_indices: int64")
    .Input("sum_indices: int64")
    .Output("a_val_grad: T")
    .Output("b_val_grad: T")
    .Attr("T: numbertype")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle a_indices;
      ShapeHandle b_indices;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &a_indices));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &b_indices));
      c->set_output(0, c->Vector(c->Dim(a_indices, 0)));
      c->set_output(1, c->Vector(c->Dim(b_indices, 0)));
      return Status::OK();
    });

REGISTER_OP("SparseAdd")
    .Input("a_indices: int64")
    .Input("a_values: T")
    .Input("a_shape: int64")
    .Input("b_indices: int64")
    .Input("b_values: T")
    .Input("b_shape: int64")
    .Input("thresh: Treal")
    .Output("sum_indices: int64")
    .Output("sum_values: T")
    .Output("sum_shape: int64")
    .Attr("T: numbertype")
    .Attr("Treal: realnumbertype")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle a_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &a_shape));
      c->set_output(
          0, c->Matrix(InferenceContext::kUnknownDim, c->Dim(a_shape, 0)));
      c->set_output(1, c->Vector(InferenceContext::kUnknownDim));
      c->set_output(2, a_shape);
      return Status::OK();
    });

REGISTER_OP("SparseTensorDenseMatMul")
    .Input("a_indices: Tindices")
    .Input("a_values: T")
    .Input("a_shape: int64")
    .Input("b: T")
    .Output("product: T")
    .Attr("T: type")
    .Attr("Tindices: {int32,int64} = DT_INT64")
    .Attr("adjoint_a: bool = false")
    .Attr("adjoint_b: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      DimensionHandle unused_dim;
      ShapeHandle unused;
      ShapeHandle b;
      ShapeHandle a_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &unused));  // a_indices
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &unused));  // a_values
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(2, &a_shape));
      TF_RETURN_IF_ERROR(c->WithRank(a_shape, 2, &a_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 2, &b));

      bool adjoint_a;
      bool adjoint_b;
      TF_RETURN_IF_ERROR(c->GetAttr("adjoint_a", &adjoint_a));
      TF_RETURN_IF_ERROR(c->GetAttr("adjoint_b", &adjoint_b));

      DimensionHandle output_right = c->Dim(b, adjoint_b ? 0 : 1);
      DimensionHandle output_left = c->Dim(a_shape, adjoint_a ? 1 : 0);
      DimensionHandle inner_left = c->Dim(a_shape, adjoint_a ? 0 : 1);
      DimensionHandle inner_right = c->Dim(b, adjoint_b ? 1 : 0);
      TF_RETURN_IF_ERROR(c->Merge(inner_left, inner_right, &unused_dim));
      c->set_output(0, c->Matrix(output_left, output_right));
      return Status::OK();
    });

REGISTER_OP("SerializeSparse")
    .Input("sparse_indices: int64")
    .Input("sparse_values: T")
    .Input("sparse_shape: int64")
    .Attr("T: type")
    .Output("serialized_sparse: out_type")
    .Attr("out_type: {string, variant} = DT_STRING")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &unused));
      c->set_output(0, c->Vector(3));
      return Status::OK();
    });

REGISTER_OP("SerializeManySparse")
    .Input("sparse_indices: int64")
    .Input("sparse_values: T")
    .Input("sparse_shape: int64")
    .Attr("T: type")
    .Output("serialized_sparse: out_type")
    .Attr("out_type: {string, variant} = DT_STRING")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &unused));
      c->set_output(0, c->Matrix(InferenceContext::kUnknownDim, 3));
      return Status::OK();
    });

REGISTER_OP("DeserializeSparse")
    .Input("serialized_sparse: Tserialized")
    .Output("sparse_indices: int64")
    .Output("sparse_values: dtype")
    .Output("sparse_shape: int64")
    .Attr("dtype: type")
    .Attr("Tserialized: {string, variant} = DT_STRING")
    .SetShapeFn([](InferenceContext* c) {
      // serialized sparse is [?, ..., ?, 3] vector.
      ShapeHandle unused_shape;
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 1, &unused_shape));
      DimensionHandle unused;
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(c->input(0), -1), 3, &unused));
      c->set_output(0, c->Matrix(InferenceContext::kUnknownDim,
                                 InferenceContext::kUnknownDim));
      c->set_output(1, c->Vector(InferenceContext::kUnknownDim));
      c->set_output(2, c->Vector(InferenceContext::kUnknownDim));
      return Status::OK();
    });

REGISTER_OP("DeserializeManySparse")
    .Input("serialized_sparse: string")
    .Output("sparse_indices: int64")
    .Output("sparse_values: dtype")
    .Output("sparse_shape: int64")
    .Attr("dtype: type")
    .SetShapeFn([](InferenceContext* c) {
      // serialized sparse is [?,3] matrix.
      ShapeHandle serialized_sparse;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &serialized_sparse));
      DimensionHandle unused;
      TF_RETURN_IF_ERROR(
          c->WithValue(c->Dim(serialized_sparse, 1), 3, &unused));

      c->set_output(0, c->Matrix(InferenceContext::kUnknownDim,
                                 InferenceContext::kUnknownDim));
      c->set_output(1, c->Vector(InferenceContext::kUnknownDim));
      c->set_output(2, c->Vector(InferenceContext::kUnknownDim));
      return Status::OK();
    });

REGISTER_OP("SparseToDense")
    .Input("sparse_indices: Tindices")
    .Input("output_shape: Tindices")
    .Input("sparse_values: T")
    .Input("default_value: T")
    .Attr("validate_indices: bool = true")
    .Attr("T: type")
    .Output("dense: T")
    .Attr("Tindices: {int32, int64}")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle out;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(1, &out));
      c->set_output(0, out);
      return Status::OK();
    });

REGISTER_OP("SparseConcat")
    .Input("indices: N * int64")
    .Input("values: N * T")
    .Input("shapes: N * int64")
    .Output("output_indices: int64")
    .Output("output_values: T")
    .Output("output_shape: int64")
    .Attr("concat_dim: int")
    .Attr("N: int >= 2")
    .Attr("T: type")
    .SetShapeFn([](InferenceContext* c) {
      // These accumulates the sum.
      DimensionHandle output_row_count = c->MakeDim(0ll);

      // These are only merged.
      DimensionHandle output_ind_cols = c->UnknownDim();
      ShapeHandle output_shape = c->UnknownShape();

      const int n = c->num_inputs() / 3;
      for (int i = 0; i < n; i++) {
        ShapeHandle ind;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 2, &ind));
        ShapeHandle val;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(i + n), 1, &val));
        ShapeHandle shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(i + 2 * n), 1, &shape));

        // Add to output_ind_rows.
        DimensionHandle num_dim;
        TF_RETURN_IF_ERROR(c->Merge(c->Dim(ind, 0), c->Dim(val, 0), &num_dim));
        TF_RETURN_IF_ERROR(
            c->Add(output_row_count, num_dim, &output_row_count));

        // Merge into output_ind_cols and output_shape.
        TF_RETURN_IF_ERROR(
            c->Merge(output_ind_cols, c->Dim(ind, 1), &output_ind_cols));
        TF_RETURN_IF_ERROR(c->Merge(output_shape, shape, &output_shape));
      }

      c->set_output(0, c->Matrix(output_row_count, output_ind_cols));
      c->set_output(1, c->Vector(output_row_count));
      c->set_output(2, output_shape);
      return Status::OK();
    });

REGISTER_OP("SparseCross")
    .Input("indices: N * int64")
    .Input("values: sparse_types")
    .Input("shapes: N * int64")
    .Input("dense_inputs: dense_types")
    .Output("output_indices: int64")
    .Output("output_values: out_type")
    .Output("output_shape: int64")
    .Attr("N: int >= 0")
    .Attr("hashed_output: bool")
    .Attr("num_buckets: int >= 0")
    .Attr("hash_key: int")
    .Attr("sparse_types: list({int64, string}) >= 0")
    .Attr("dense_types: list({int64, string}) >= 0")
    .Attr("out_type: {int64, string}")
    .Attr("internal_type: {int64, string}")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Matrix(c->UnknownDim(), 2));
      c->set_output(1, c->Vector(c->UnknownDim()));
      c->set_output(2, c->Vector(2));
      return Status::OK();
    });

REGISTER_OP("SparseCrossV2")
    .Input("indices: N * int64")
    .Input("values: sparse_types")
    .Input("shapes: N * int64")
    .Input("dense_inputs: dense_types")
    .Input("sep: string")
    .Output("output_indices: int64")
    .Output("output_values: string")
    .Output("output_shape: int64")
    .Attr("N: int >= 0")
    .Attr("sparse_types: list({int64, string}) >= 0")
    .Attr("dense_types: list({int64, string}) >= 0")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Matrix(c->UnknownDim(), 2));
      c->set_output(1, c->Vector(c->UnknownDim()));
      c->set_output(2, c->Vector(2));
      return Status::OK();
    });

REGISTER_OP("SparseCrossHashed")
    .Input("indices: N * int64")
    .Input("values: sparse_types")
    .Input("shapes: N * int64")
    .Input("dense_inputs: dense_types")
    .Input("num_buckets: int64")
    .Input("strong_hash: bool")
    .Input("salt: int64")
    .Output("output_indices: int64")
    .Output("output_values: int64")
    .Output("output_shape: int64")
    .Attr("N: int >= 0")
    .Attr("sparse_types: list({int64, string}) >= 0")
    .Attr("dense_types: list({int64, string}) >= 0")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Matrix(c->UnknownDim(), 2));
      c->set_output(1, c->Vector(c->UnknownDim()));
      c->set_output(2, c->Vector(2));
      return Status::OK();
    });

REGISTER_OP("SparseSplit")
    .Input("split_dim: int64")
    .Input("indices: int64")
    .Input("values: T")
    .Input("shape: int64")
    .Output("output_indices: num_split * int64")
    .Output("output_values:  num_split * T")
    .Output("output_shape:   num_split * int64")
    .Attr("num_split: int >= 1")
    .Attr("T: type")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle input_shape = c->input(3);
      ShapeHandle output_indices =
          c->Matrix(InferenceContext::kUnknownDim, c->NumElements(input_shape));
      ShapeHandle output_values = c->Vector(InferenceContext::kUnknownDim);
      ShapeHandle output_shape = input_shape;

      // Copy the outputs into the output ranges.
      int num_splits = c->num_outputs() / 3;
      int out_idx = 0;
      for (int i = 0; i < num_splits; ++i)
        c->set_output(out_idx++, output_indices);
      for (int i = 0; i < num_splits; ++i)
        c->set_output(out_idx++, output_values);
      for (int i = 0; i < num_splits; ++i)
        c->set_output(out_idx++, output_shape);
      return Status::OK();
    });

REGISTER_OP("SparseSliceGrad")
    .Input("backprop_val_grad: T")
    .Input("input_indices: int64")
    .Input("input_start: int64")
    .Input("output_indices: int64")
    .Output("val_grad: T")
    .Attr("T: numbertype")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle indices;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &indices));
      c->set_output(0, c->Vector(c->Dim(indices, 0)));
      return Status::OK();
    });

REGISTER_OP("SparseSlice")
    .Input("indices: int64")
    .Input("values: T")
    .Input("shape: int64")
    .Input("start: int64")
    .Input("size: int64")
    .Output("output_indices: int64")
    .Output("output_values: T")
    .Output("output_shape: int64")
    .Attr("T: type")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle input_shape = c->input(2);
      ShapeHandle output_indices =
          c->Matrix(InferenceContext::kUnknownDim, c->NumElements(input_shape));
      ShapeHandle output_values = c->Vector(InferenceContext::kUnknownDim);
      ShapeHandle output_shape = input_shape;

      c->set_output(0, output_indices);
      c->set_output(1, output_values);
      c->set_output(2, output_shape);
      return Status::OK();
    });

REGISTER_OP("SparseReorder")
    .Input("input_indices: int64")
    .Input("input_values: T")
    .Input("input_shape: int64")
    .Output("output_indices: int64")
    .Output("output_values: T")
    .Attr("T: type")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle indices;
      ShapeHandle values;
      ShapeHandle unused;

      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &indices));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &values));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &unused));

      c->set_output(0, indices);
      c->set_output(1, values);
      return Status::OK();
    });

REGISTER_OP("SparseReshape")
    .Input("input_indices: int64")
    .Input("input_shape: int64")
    .Input("new_shape: int64")
    .Output("output_indices: int64")
    .Output("output_shape: int64")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle indices;
      ShapeHandle unused;
      ShapeHandle new_shape;

      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &indices));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &new_shape));

      c->set_output(0, c->Matrix(c->Dim(indices, 0), c->Dim(new_shape, 0)));
      c->set_output(1, new_shape);
      return Status::OK();
    });

REGISTER_OP("SparseTensorDenseAdd")
    .Input("a_indices: Tindices")
    .Input("a_values: T")
    .Input("a_shape: Tindices")
    .Input("b: T")
    .Output("output: T")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64}")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->input(3));
      return Status::OK();
    });

REGISTER_OP("SparseReduceMax")
    .Input("input_indices: int64")
    .Input("input_values: T")
    .Input("input_shape: int64")
    .Input("reduction_axes: int32")
    .Attr("keep_dims: bool = False")
    .Output("output: T")
    .Attr("T: realnumbertype")
    .SetShapeFn(shape_inference::SparseReduceShapeFn);

REGISTER_OP("SparseReduceMaxSparse")
    .Input("input_indices: int64")
    .Input("input_values: T")
    .Input("input_shape: int64")
    .Input("reduction_axes: int32")
    .Attr("keep_dims: bool = False")
    .Output("output_indices: int64")
    .Output("output_values: T")
    .Output("output_shape: int64")
    .Attr("T: realnumbertype")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("SparseReduceSum")
    .Input("input_indices: int64")
    .Input("input_values: T")
    .Input("input_shape: int64")
    .Input("reduction_axes: int32")
    .Attr("keep_dims: bool = False")
    .Output("output: T")
    .Attr("T: numbertype")
    .SetShapeFn(shape_inference::SparseReduceShapeFn);

REGISTER_OP("SparseReduceSumSparse")
    .Input("input_indices: int64")
    .Input("input_values: T")
    .Input("input_shape: int64")
    .Input("reduction_axes: int32")
    .Attr("keep_dims: bool = False")
    .Output("output_indices: int64")
    .Output("output_values: T")
    .Output("output_shape: int64")
    .Attr("T: numbertype")
    .SetShapeFn(shape_inference::UnknownShape);

#define SPARSE_DENSE_CWISE_SIGNATURE()                           \
  Input("sp_indices: int64")                                     \
      .Input("sp_values: T")                                     \
      .Input("sp_shape: int64")                                  \
      .Input("dense: T")                                         \
      .Output("output: T")                                       \
      .Attr("T: numbertype")                                     \
      .SetShapeFn([](InferenceContext* c) {                      \
        ShapeHandle input;                                       \
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input)); \
        c->set_output(0, c->Vector(c->Dim(input, 0)));           \
        return Status::OK();                                     \
      })

REGISTER_OP("SparseDenseCwiseMul").SPARSE_DENSE_CWISE_SIGNATURE();

REGISTER_OP("SparseDenseCwiseDiv").SPARSE_DENSE_CWISE_SIGNATURE();

REGISTER_OP("SparseDenseCwiseAdd").SPARSE_DENSE_CWISE_SIGNATURE();

#undef SPARSE_DENSE_CWISE_SIGNATURE

REGISTER_OP("SparseSoftmax")
    .Input("sp_indices: int64")
    .Input("sp_values: T")
    .Input("sp_shape: int64")
    .Output("output: T")
    .Attr("T: {float, double}")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      ShapeHandle values;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &unused));  // sp_indices
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &values));  // sp_values
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &unused));
      c->set_output(0, values);
      return Status::OK();
    });

REGISTER_OP("SparseSparseMaximum")
    .Input("a_indices: int64")
    .Input("a_values: T")
    .Input("a_shape: int64")
    .Input("b_indices: int64")
    .Input("b_values: T")
    .Input("b_shape: int64")
    .Output("output_indices: int64")
    .Output("output_values: T")
    .Attr("T: realnumbertype")
    .SetShapeFn(SparseSparseMinOrMaxShapeFn);

REGISTER_OP("SparseSparseMinimum")
    .Input("a_indices: int64")
    .Input("a_values: T")
    .Input("a_shape: int64")
    .Input("b_indices: int64")
    .Input("b_values: T")
    .Input("b_shape: int64")
    .Output("output_indices: int64")
    .Output("output_values: T")
    .Attr("T: numbertype")
    .SetShapeFn(SparseSparseMinOrMaxShapeFn);

REGISTER_OP("AddSparseToTensorsMap")
    .Input("sparse_indices: int64")
    .Input("sparse_values: T")
    .Input("sparse_shape: int64")
    .Output("sparse_handle: int64")
    .Attr("T: type")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &unused));
      c->set_output(0, c->Scalar());
      return Status::OK();
    });

REGISTER_OP("AddManySparseToTensorsMap")
    .Input("sparse_indices: int64")
    .Input("sparse_values: T")
    .Input("sparse_shape: int64")
    .Output("sparse_handles: int64")
    .Attr("T: type")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &unused));
      c->set_output(0, c->Vector(InferenceContext::kUnknownDim));
      return Status::OK();
    });

REGISTER_OP("TakeManySparseFromTensorsMap")
    .Input("sparse_handles: int64")
    .Output("sparse_indices: int64")
    .Output("sparse_values: dtype")
    .Output("sparse_shape: int64")
    .Attr("dtype: type")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      // serialized sparse is [?,1] matrix.
      ShapeHandle sparse_handles;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &sparse_handles));

      c->set_output(0, c->Matrix(InferenceContext::kUnknownDim,
                                 InferenceContext::kUnknownDim));
      c->set_output(1, c->Vector(InferenceContext::kUnknownDim));
      c->set_output(2, c->Vector(InferenceContext::kUnknownDim));
      return Status::OK();
    });

REGISTER_OP("SparseFillEmptyRows")
    .Input("indices: int64")
    .Input("values: T")
    .Input("dense_shape: int64")
    .Input("default_value: T")
    .Output("output_indices: int64")
    .Output("output_values: T")
    .Output("empty_row_indicator: bool")
    .Output("reverse_index_map: int64")
    .Attr("T: type")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle input_indices = c->input(0);
      TF_RETURN_IF_ERROR(c->WithRank(input_indices, 2, &input_indices));
      ShapeHandle input_values = c->input(1);
      TF_RETURN_IF_ERROR(c->WithRank(input_values, 1, &input_values));
      ShapeHandle input_shape = c->input(2);
      TF_RETURN_IF_ERROR(c->WithRank(input_shape, 1, &input_shape));
      ShapeHandle default_value = c->input(3);
      TF_RETURN_IF_ERROR(c->WithRank(default_value, 0, &default_value));
      DimensionHandle N = c->Dim(input_indices, 0);
      TF_RETURN_IF_ERROR(c->Merge(N, c->Dim(input_values, 0), &N));
      DimensionHandle unused_dim;
      TF_RETURN_IF_ERROR(c->Merge(c->Dim(input_indices, 1),
                                  c->Dim(input_shape, 0), &unused_dim));
      if (c->Value(c->NumElements(input_shape)) == 0)
        return errors::InvalidArgument("dense_shape must not be empty");
      ShapeHandle output_indices =
          c->Matrix(InferenceContext::kUnknownDim, c->NumElements(input_shape));
      ShapeHandle output_values = c->Vector(InferenceContext::kUnknownDim);
      ShapeHandle constant_input_shape;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(2, &constant_input_shape));
      ShapeHandle empty_row_indicator =
          c->Vector(c->Dim(constant_input_shape, 0));
      ShapeHandle reverse_index_map = c->Vector(N);
      c->set_output(0, output_indices);
      c->set_output(1, output_values);
      c->set_output(2, empty_row_indicator);
      c->set_output(3, reverse_index_map);
      return Status::OK();
    });

REGISTER_OP("SparseFillEmptyRowsGrad")
    .Input("reverse_index_map: int64")
    .Input("grad_values: T")
    .Output("d_values: T")
    .Output("d_default_value: T")
    .Attr("T: type")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle reverse_index_map = c->input(0);
      TF_RETURN_IF_ERROR(c->WithRank(reverse_index_map, 1, &reverse_index_map));
      ShapeHandle grad_values = c->input(1);
      TF_RETURN_IF_ERROR(c->WithRank(grad_values, 1, &grad_values));
      c->set_output(0, reverse_index_map);
      c->set_output(1, c->Scalar());
      return Status::OK();
    });

}  // namespace tensorflow
