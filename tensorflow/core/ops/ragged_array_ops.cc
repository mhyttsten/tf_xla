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
class MHTracer_DTPStensorflowPScorePSopsPSragged_array_opsDTcc {
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
   MHTracer_DTPStensorflowPScorePSopsPSragged_array_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSopsPSragged_array_opsDTcc() {
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

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

Status RaggedGatherShapeFn(InferenceContext* c);

//==============================================================================
// Registered Ops
//==============================================================================

REGISTER_OP("RaggedGather")
    .Input("params_nested_splits: PARAMS_RAGGED_RANK * Tsplits")
    .Input("params_dense_values: Tvalues")
    .Input("indices: Tindices")
    .Output("output_nested_splits: OUTPUT_RAGGED_RANK * Tsplits")
    .Output("output_dense_values: Tvalues")
    .Attr("Tvalues: type")
    .Attr("Tindices: {int32, int64}")
    .Attr("Tsplits: {int32, int64} = DT_INT64")
    .Attr("PARAMS_RAGGED_RANK: int >= 1")
    .Attr("OUTPUT_RAGGED_RANK: int >= 0")
    .SetShapeFn(RaggedGatherShapeFn);

REGISTER_OP("RaggedCross")
    .Input("ragged_values: ragged_values_types")
    .Input("ragged_row_splits: ragged_splits_types")
    .Input("sparse_indices: Nsparse * int64")
    .Input("sparse_values: sparse_values_types")
    .Input("sparse_shape: Nsparse * int64")
    .Input("dense_inputs: dense_types")
    .Output("output_values: out_values_type")
    .Output("output_row_splits: out_row_splits_type")
    .Attr("Nsparse: int >= 0")
    .Attr("input_order: string")
    .Attr("hashed_output: bool")
    .Attr("num_buckets: int >= 0")
    .Attr("hash_key: int")
    .Attr("ragged_values_types: list({int64, string}) >= 0")
    .Attr("ragged_splits_types: list({int32, int64}) >= 0")
    .Attr("sparse_values_types: list({int64, string}) >= 0")
    .Attr("dense_types: list({int64, string}) >= 0")
    .Attr("out_values_type: {int64, string}")
    .Attr("out_row_splits_type: {int32, int64}")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      std::vector<DataType> ragged_values_types;
      std::vector<DataType> ragged_splits_types;
      std::vector<DataType> sparse_values_types;
      std::vector<DataType> dense_types;

      TF_RETURN_IF_ERROR(
          c->GetAttr("ragged_values_types", &ragged_values_types));
      TF_RETURN_IF_ERROR(
          c->GetAttr("ragged_splits_types", &ragged_splits_types));
      TF_RETURN_IF_ERROR(c->GetAttr("dense_types", &dense_types));
      TF_RETURN_IF_ERROR(
          c->GetAttr("sparse_values_types", &sparse_values_types));

      int num_ragged = ragged_values_types.size();
      if (num_ragged != ragged_splits_types.size()) {
        return errors::InvalidArgument(
            "ragged values and splits must have the same length.");
      }

      int num_sparse;
      TF_RETURN_IF_ERROR(c->GetAttr("Nsparse", &num_sparse));
      if (num_sparse != sparse_values_types.size()) {
        return errors::InvalidArgument(
            "sparse indices and values must have the same length");
      }

      ShapeHandle out_values = c->UnknownShapeOfRank(1);
      ShapeHandle out_splits = c->UnknownShapeOfRank(1);

      // Merge the shapes of row_splits from ragged inputs.  (This is one plus
      // the batch size.)
      int ragged_splits_start = num_ragged;
      for (int i = 0; i < ragged_splits_types.size(); ++i) {
        ShapeHandle row_splits = c->input(i + ragged_splits_start);
        if (!c->Merge(out_splits, row_splits, &out_splits).ok()) {
          return errors::InvalidArgument(
              "inputs must all have the same batch dimension size.");
        }
      }

      // Merge the batch size of each dense input into out_splits.
      int dense_start = num_ragged * 2 + num_sparse * 3;
      for (int i = 0; i < dense_types.size(); ++i) {
        ShapeHandle dense_input = c->input(i + dense_start);
        int32 rank = c->Rank(dense_input);
        if (rank == InferenceContext::kUnknownRank) {
          continue;
        } else if (rank != 2) {
          return errors::InvalidArgument(
              "tf.ragged.cross only supports inputs with rank=2");
        }
        int64_t batch_size = c->Value(c->Dim(dense_input, 0));
        if (batch_size != InferenceContext::kUnknownDim) {
          ShapeHandle row_splits = c->Vector(batch_size + 1);
          if (!c->Merge(out_splits, row_splits, &out_splits).ok()) {
            return errors::InvalidArgument(
                "inputs must all have the same batch dimension size.");
          }
        }
      }

      c->set_output(0, out_values);
      c->set_output(1, out_splits);
      return Status::OK();
    });

//==============================================================================
// Shape Functions
//==============================================================================

Status RaggedGatherShapeFn(InferenceContext* c) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSopsPSragged_array_opsDTcc mht_0(mht_0_v, 304, "", "./tensorflow/core/ops/ragged_array_ops.cc", "RaggedGatherShapeFn");

  int num_splits;
  int64_t PARAMS_RAGGED_RANK;
  TF_RETURN_IF_ERROR(
      c->GetAttr<int64_t>("PARAMS_RAGGED_RANK", &PARAMS_RAGGED_RANK));
  TF_RETURN_IF_ERROR(c->GetAttr<int>("OUTPUT_RAGGED_RANK", &num_splits));

  // Check rank of `indices`.
  ShapeHandle indices = c->input(PARAMS_RAGGED_RANK + 1);
  TF_RETURN_IF_ERROR(
      c->WithRank(indices, num_splits - PARAMS_RAGGED_RANK + 1, &indices));

  // Check that all params_nested_splits have rank 1.
  for (int64_t i = 0; i < PARAMS_RAGGED_RANK; ++i) {
    ShapeHandle splits = c->input(i);
    TF_RETURN_IF_ERROR(c->WithRank(splits, 1, &splits));
  }

  // Check that `params_dense_values` has rank>=1.
  ShapeHandle params_dense_values = c->input(PARAMS_RAGGED_RANK);
  TF_RETURN_IF_ERROR(
      c->WithRankAtLeast(params_dense_values, 1, &params_dense_values));

  // Set the rank for the `splits` outputs.
  for (int i = 0; i < num_splits; ++i) {
    c->set_output(i, c->UnknownShapeOfRank(1));
  }

  // Calculate the `values` shape.
  ShapeHandle value = c->UnknownShape();
  ShapeHandle values = c->UnknownShape();
  TF_RETURN_IF_ERROR(c->Subshape(params_dense_values, 1, &value));
  TF_RETURN_IF_ERROR(c->Concatenate(c->UnknownShapeOfRank(1), value, &values));
  c->set_output(num_splits, values);

  return Status::OK();
}

}  // namespace tensorflow
