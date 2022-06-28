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
class MHTracer_DTPStensorflowPScorePSopsPSragged_conversion_opsDTcc {
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
   MHTracer_DTPStensorflowPScorePSopsPSragged_conversion_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSopsPSragged_conversion_opsDTcc() {
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
#include "tensorflow/core/util/ragged_to_dense_util.h"

namespace tensorflow {

using errors::InvalidArgument;
using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

namespace {
tensorflow::Status ValidateRowPartitionTypesAndShapes(
    const std::vector<RowPartitionType>& row_partition_types,
    InferenceContext* c) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSopsPSragged_conversion_opsDTcc mht_0(mht_0_v, 199, "", "./tensorflow/core/ops/ragged_conversion_ops.cc", "ValidateRowPartitionTypesAndShapes");

  // Note: the allowed types may be extended in the future.
  for (RowPartitionType row_partition_type : row_partition_types) {
    switch (row_partition_type) {
      case RowPartitionType::FIRST_DIM_SIZE:
      case RowPartitionType::VALUE_ROWIDS:
      case RowPartitionType::ROW_SPLITS:
        break;
      default:
        return InvalidArgument("Unsupported partition type: ",
                               RowPartitionTypeToString(row_partition_type));
    }
  }

  if (row_partition_types.empty()) {
    return InvalidArgument("Partition info types should not be empty");
  }
  for (int i = 1; i < row_partition_types.size(); ++i) {
    if (row_partition_types[i] == RowPartitionType::FIRST_DIM_SIZE) {
      return InvalidArgument("FIRST_DIM_SIZE must be first");
    }
  }
  if (row_partition_types[0] == RowPartitionType::FIRST_DIM_SIZE &&
      (row_partition_types.size() < 2 ||
       row_partition_types[1] != RowPartitionType::VALUE_ROWIDS)) {
    return InvalidArgument("FIRST_DIM_SIZE must be followed by VALUE_ROWIDS");
  }
  if (row_partition_types[0] == RowPartitionType::VALUE_ROWIDS) {
    return InvalidArgument("VALUE_ROWIDS cannot be first");
  }

  int num_row_partition_tensors;
  TF_RETURN_IF_ERROR(
      c->GetAttr("num_row_partition_tensors", &num_row_partition_tensors));
  if (num_row_partition_tensors != row_partition_types.size()) {
    return InvalidArgument(
        "Number of row partition tensors (", num_row_partition_tensors,
        ") does not equal the number of row partition types(",
        row_partition_types.size(), ").");
  }

  for (int i = 0; i < num_row_partition_tensors; ++i) {
    TensorShapeProto partition_shape;
    c->ShapeHandleToProto(c->input(3 + i), &partition_shape);
    if (partition_shape.unknown_rank()) {
      continue;
    }
    if (row_partition_types[i] == RowPartitionType::FIRST_DIM_SIZE) {
      if (partition_shape.dim_size() != 0) {
        return InvalidArgument("FIRST_DIM_SIZE must be a scalar.");
      }
    } else {
      if (partition_shape.dim_size() != 1) {
        return InvalidArgument("Row partition must be a vector.");
      }
    }
  }
  return tensorflow::Status::OK();
}

}  // namespace

Status RaggedTensorToSparseShapeFn(InferenceContext* c);
Status RaggedTensorToVariantShapeFn(InferenceContext* c);
Status RaggedTensorFromVariantShapeFn(InferenceContext* c);
Status RaggedTensorToVariantGradientShapeFn(InferenceContext* c);
Status RaggedTensorToTensorShapeFn(InferenceContext* c);

//==============================================================================
// Registered Ops
//==============================================================================

REGISTER_OP("RaggedTensorToSparse")
    .Input("rt_nested_splits: RAGGED_RANK * Tsplits")
    .Input("rt_dense_values: T")
    .Output("sparse_indices: int64")
    .Output("sparse_values: T")
    .Output("sparse_dense_shape: int64")
    .Attr("RAGGED_RANK: int >= 1")
    .Attr("T: type")
    .Attr("Tsplits: {int32, int64} = DT_INT64")
    .SetShapeFn(RaggedTensorToSparseShapeFn);

REGISTER_OP("RaggedTensorToVariant")
    .Input("rt_nested_splits: RAGGED_RANK * Tsplits")
    .Input("rt_dense_values: Tvalues")
    .Output("encoded_ragged: variant")
    .Attr("RAGGED_RANK: int >= 0")
    .Attr("Tvalues: type")
    .Attr("Tsplits: {int32, int64} = DT_INT64")
    .Attr("batched_input: bool")
    .SetTypeConstructor(full_type::Unary(TFT_RAGGED, "Tvalues"))
    .SetShapeFn(RaggedTensorToVariantShapeFn);

REGISTER_OP("RaggedTensorFromVariant")
    .Input("encoded_ragged: variant")
    .Output("output_nested_splits: output_ragged_rank * Tsplits")
    .Output("output_dense_values: Tvalues")
    .Attr("input_ragged_rank: int >= -1")
    .Attr("output_ragged_rank: int >= 0")
    .Attr("Tvalues: type")
    .Attr("Tsplits: {int32, int64} = DT_INT64")
    .SetShapeFn(RaggedTensorFromVariantShapeFn);

REGISTER_OP("RaggedTensorToVariantGradient")
    .Input("encoded_ragged_grad: variant")
    .Input("row_splits: Tsplits")
    .Input("dense_values_shape: int32")
    .Output("dense_values_grad: Tvalues")
    .Attr("Tvalues: type")
    .Attr("Tsplits: {int32, int64} = DT_INT64")
    .SetShapeFn(RaggedTensorToVariantGradientShapeFn);

REGISTER_OP("RaggedTensorToTensor")
    .Attr("T: type")
    .Attr("Tindex: {int64, int32}")
    .Attr("Tshape: {int64, int32}")
    .Attr("num_row_partition_tensors: int")
    .Attr("row_partition_types: list(string)")
    .Input("shape: Tshape")
    .Input("values: T")
    .Input("default_value: T")
    .Input("row_partition_tensors: num_row_partition_tensors * Tindex")
    .Output("result: T")
    .SetShapeFn(RaggedTensorToTensorShapeFn);

//==============================================================================
// Shape Functions
//==============================================================================

Status RaggedTensorToSparseShapeFn(InferenceContext* c) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSopsPSragged_conversion_opsDTcc mht_1(mht_1_v, 332, "", "./tensorflow/core/ops/ragged_conversion_ops.cc", "RaggedTensorToSparseShapeFn");

  int64_t num_splits;
  TF_RETURN_IF_ERROR(c->GetAttr<int64_t>("RAGGED_RANK", &num_splits));
  // TODO(b/112274756): Allow ragged_rank to be 0.
  if (num_splits < 1) {
    return errors::InvalidArgument("Requires RAGGED_RANK>0");
  }
  ShapeHandle rt_dense_values = c->input(num_splits);
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(rt_dense_values, 1, &rt_dense_values));

  // Check that all rt_nested_splits have rank 1.
  for (int64_t i = 0; i < num_splits; ++i) {
    ShapeHandle splits = c->input(i);
    TF_RETURN_IF_ERROR(c->WithRank(splits, 1, &splits));
  }

  DimensionHandle dense_dims =
      c->RankKnown(rt_dense_values)
          ? c->MakeDim(c->Rank(rt_dense_values) + num_splits)
          : c->UnknownDim();
  DimensionHandle num_values = c->NumElements(rt_dense_values);

  c->set_output(0, c->Matrix(num_values, dense_dims));  // indices
  c->set_output(1, c->Vector(num_values));              // values
  c->set_output(2, c->Vector(dense_dims));              // dense_shape

  return Status::OK();
}

Status RaggedTensorToVariantShapeFn(InferenceContext* c) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSopsPSragged_conversion_opsDTcc mht_2(mht_2_v, 364, "", "./tensorflow/core/ops/ragged_conversion_ops.cc", "RaggedTensorToVariantShapeFn");

  int64_t num_splits;
  TF_RETURN_IF_ERROR(c->GetAttr<int64_t>("RAGGED_RANK", &num_splits));
  bool batched;
  TF_RETURN_IF_ERROR(c->GetAttr<bool>("batched_input", &batched));
  shape_inference::ShapeHandle rt_dense_values = c->input(num_splits);
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(rt_dense_values, 1, &rt_dense_values));
  for (int64_t i = 0; i < num_splits; ++i) {
    shape_inference::ShapeHandle splits = c->input(i);
    TF_RETURN_IF_ERROR(c->WithRank(splits, 1, &splits));
  }
  if (batched) {
    auto num_first_splits = c->Dim(c->input(0), 0);
    shape_inference::DimensionHandle num_rows;
    TF_RETURN_IF_ERROR(c->Subtract(num_first_splits, 1, &num_rows));
    c->set_output(0, c->Vector(num_rows));
  } else {
    c->set_output(0, c->Scalar());
  }
  return Status::OK();
}

Status RaggedTensorToVariantGradientShapeFn(InferenceContext* c) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSopsPSragged_conversion_opsDTcc mht_3(mht_3_v, 389, "", "./tensorflow/core/ops/ragged_conversion_ops.cc", "RaggedTensorToVariantGradientShapeFn");

  ShapeHandle shape;
  TF_RETURN_IF_ERROR(
      c->MakeShapeFromShapeTensorTreatScalarAsUnknownShape(2, &shape));
  c->set_output(0, shape);
  return Status::OK();
}

Status RaggedTensorFromVariantShapeFn(InferenceContext* c) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSopsPSragged_conversion_opsDTcc mht_4(mht_4_v, 400, "", "./tensorflow/core/ops/ragged_conversion_ops.cc", "RaggedTensorFromVariantShapeFn");

  int64_t input_ragged_rank;
  TF_RETURN_IF_ERROR(
      c->GetAttr<int64_t>("input_ragged_rank", &input_ragged_rank));
  int64_t output_ragged_rank;
  TF_RETURN_IF_ERROR(
      c->GetAttr<int64_t>("output_ragged_rank", &output_ragged_rank));
  shape_inference::ShapeHandle encoded_ragged = c->input(0);
  if (c->RankKnown(encoded_ragged) && input_ragged_rank >= 0) {
    shape_inference::ShapeHandle unused;
    TF_RETURN_IF_ERROR(c->WithRank(
        encoded_ragged, output_ragged_rank - input_ragged_rank, &unused));
  }
  for (int64_t i = 0; i < output_ragged_rank; i++) {
    c->set_output(i, c->UnknownShapeOfRank(1));
  }
  c->set_output(output_ragged_rank, c->UnknownShape());
  return Status::OK();
}

tensorflow::Status RaggedTensorToTensorShapeFn(InferenceContext* c) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSopsPSragged_conversion_opsDTcc mht_5(mht_5_v, 423, "", "./tensorflow/core/ops/ragged_conversion_ops.cc", "RaggedTensorToTensorShapeFn");

  TensorShapeProto shape;
  {
    ShapeHandle shape_handle;
    TF_RETURN_IF_ERROR(
        c->MakeShapeFromShapeTensorTreatScalarAsUnknownShape(0, &shape_handle));
    c->ShapeHandleToProto(shape_handle, &shape);
  }

  std::vector<RowPartitionType> row_partition_types;
  TF_RETURN_IF_ERROR(GetRowPartitionTypes(c, &row_partition_types));
  int ragged_rank = GetRaggedRank(row_partition_types);
  TF_RETURN_IF_ERROR(
      ValidateRowPartitionTypesAndShapes(row_partition_types, c));

  TensorShapeProto value_shape;
  c->ShapeHandleToProto(c->input(1), &value_shape);

  TensorShapeProto default_value_shape;
  c->ShapeHandleToProto(c->input(2), &default_value_shape);

  TF_RETURN_IF_ERROR(
      ValidateDefaultValueShape(default_value_shape, value_shape));

  // TODO(martinz): Theoretically, we could check the first dimension of
  // value_shape against the first dimension of the last row_partition_tensor
  // assuming it is a VALUE_ROWIDS type.
  // TODO(martinz): Although we normally don't know the first dimension of the
  // output, we could infer it from the first dimension of the first
  // row_partition_tensor if it is ROW_SPLITS type.
  // TODO(martinz): If the shape is provided, but the value_shape has missing
  // dimensions, we can check the default_value_shape against the shape.
  TensorShapeProto output_shape;
  TF_RETURN_IF_ERROR(CombineRaggedTensorToTensorShapes(
      ragged_rank, shape, value_shape, &output_shape));

  ShapeHandle output_shape_handle;
  TF_RETURN_IF_ERROR(
      c->MakeShapeFromShapeProto(output_shape, &output_shape_handle));
  c->set_output(0, output_shape_handle);
  return Status::OK();
}

}  // namespace tensorflow
