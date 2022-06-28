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
class MHTracer_DTPStensorflowPScorePSopsPSparsing_opsDTcc {
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
   MHTracer_DTPStensorflowPScorePSopsPSparsing_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSopsPSparsing_opsDTcc() {
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
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/example_proto_helper.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

namespace {

// Adds output shapes for dense tensors in Parse*Example ops.
template <typename TensorShapeType>  // TensorShape or PartialTensorShape
Status AddDenseOutputShapes(const std::vector<TensorShapeType>& dense_shapes,
                            const ShapeHandle& prefix, InferenceContext* c,
                            int* output_idx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSopsPSparsing_opsDTcc mht_0(mht_0_v, 203, "", "./tensorflow/core/ops/parsing_ops.cc", "AddDenseOutputShapes");

  for (const auto& dense_shape : dense_shapes) {
    ShapeHandle s;
    TF_RETURN_IF_ERROR(c->MakeShapeFromPartialTensorShape(dense_shape, &s));
    TF_RETURN_IF_ERROR(c->Concatenate(prefix, s, &s));
    c->set_output((*output_idx)++, s);
  }
  return Status::OK();
}

// Adds output shapes for sparse tensors in Parse*Example ops.
void AddSparseOutputShapes(int num_sparse, const ShapeHandle input_shape,
                           int64_t rank_delta, InferenceContext* c,
                           int* output_idx) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSopsPSparsing_opsDTcc mht_1(mht_1_v, 219, "", "./tensorflow/core/ops/parsing_ops.cc", "AddSparseOutputShapes");

  // Rank of SparseTensor is rank of input tensor plus rank_delta.
  shape_inference::DimensionOrConstant rank(c->UnknownDim());
  if (c->RankKnown(input_shape)) {
    rank = c->Rank(input_shape) + rank_delta;
  }
  for (int i = 0; i < num_sparse; ++i) {  // sparse_indices
    c->set_output((*output_idx)++, c->Matrix(c->UnknownDim(), rank));
  }
  for (int i = 0; i < num_sparse; ++i) {  // sparse_values
    c->set_output((*output_idx)++, c->Vector(c->UnknownDim()));
  }
  for (int i = 0; i < num_sparse; ++i) {  // sparse_dense_shapes
    c->set_output((*output_idx)++, c->Vector(rank));
  }
}

// Adds output shapes for ragged tensors in Parse*Example ops.
Status AddRaggedOutputShapes(int num_ragged, bool ragged_rank_2,
                             const DimensionHandle& num_examples,
                             InferenceContext* c, int* output_idx) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSopsPSparsing_opsDTcc mht_2(mht_2_v, 242, "", "./tensorflow/core/ops/parsing_ops.cc", "AddRaggedOutputShapes");

  DimensionHandle num_splits;
  TF_RETURN_IF_ERROR(c->Add(num_examples, 1, &num_splits));
  // Values
  for (int i = 0; i < num_ragged; ++i) {
    c->set_output((*output_idx)++, c->Vector(c->UnknownDim()));
  }
  // Outer row_splits.
  for (int i = 0; i < num_ragged; ++i) {
    c->set_output((*output_idx)++, c->Vector(num_splits));
  }
  // Inner row_splits  (for ParseSequenceExample feature_list features)
  if (ragged_rank_2) {
    for (int i = 0; i < num_ragged; ++i) {
      c->set_output((*output_idx)++, c->Vector(c->UnknownDim()));
    }
  }
  return Status::OK();
}

// Adds output shapes for dense_lengths tensors in Parse*Example ops.
void AddDenseLengthsShapes(int num_dense, const ShapeHandle& shape,
                           InferenceContext* c, int* output_idx) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSopsPSparsing_opsDTcc mht_3(mht_3_v, 267, "", "./tensorflow/core/ops/parsing_ops.cc", "AddDenseLengthsShapes");

  for (int i = 0; i < num_dense; ++i) {
    c->set_output((*output_idx)++, shape);
  }
}

}  // namespace

REGISTER_OP("DecodeRaw")
    .Input("bytes: string")
    .Output("output: out_type")
    .Attr(
        "out_type: "
        "{half,float,double,int32,uint16,uint8,int16,int8,int64,complex64,"
        "complex128,bool,bfloat16}")
    .Attr("little_endian: bool = true")
    .SetShapeFn([](InferenceContext* c) {
      // Note: last dimension is data dependent.
      ShapeHandle out;
      TF_RETURN_IF_ERROR(c->Concatenate(
          c->input(0), c->Vector(InferenceContext::kUnknownDim), &out));
      c->set_output(0, out);
      return Status::OK();
    });

REGISTER_OP("DecodePaddedRaw")
    .Input("input_bytes: string")
    .Input("fixed_length: int32")
    .Output("output: out_type")
    .Attr(
        "out_type: {half,float,double,int32,uint16,uint8,int16,int8,int64,"
        "bfloat16}")
    .Attr("little_endian: bool = true")
    .SetShapeFn([](InferenceContext* c) {
      DimensionHandle fixed_length;
      TF_RETURN_IF_ERROR(c->MakeDimForScalarInput(1, &fixed_length));

      DataType out_type;
      TF_RETURN_IF_ERROR(c->GetAttr("out_type", &out_type));

      int32_t data_type_size = DataTypeSize(out_type);

      DimensionHandle width;
      TF_RETURN_IF_ERROR(c->Divide(fixed_length, data_type_size, true, &width));

      ShapeHandle out;
      TF_RETURN_IF_ERROR(c->Concatenate(c->input(0), c->Vector(width), &out));

      c->set_output(0, out);
      return Status::OK();
    });

REGISTER_OP("DecodeCompressed")
    .Input("bytes: string")
    .Output("output: string")
    .Attr("compression_type: string = ''")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("ParseExample")
    .Input("serialized: string")
    .Input("names: string")
    .Input("sparse_keys: Nsparse * string")
    .Input("dense_keys: Ndense * string")
    .Input("dense_defaults: Tdense")
    .Output("sparse_indices: Nsparse * int64")
    .Output("sparse_values: sparse_types")
    .Output("sparse_shapes: Nsparse * int64")
    .Output("dense_values: Tdense")
    .Attr("Nsparse: int >= 0")  // Inferred from sparse_keys
    .Attr("Ndense: int >= 0")   // Inferred from dense_keys
    .Attr("sparse_types: list({float,int64,string}) >= 0")
    .Attr("Tdense: list({float,int64,string}) >= 0")
    .Attr("dense_shapes: list(shape) >= 0")
    .SetShapeFn([](InferenceContext* c) {
      ParseExampleAttrs attrs;
      TF_RETURN_IF_ERROR(attrs.Init(c, /*op_version=*/1));

      ShapeHandle input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &input));
      ShapeHandle names;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &names));

      int output_idx = 0;
      AddSparseOutputShapes(attrs.num_sparse, input, 1, c, &output_idx);
      TF_RETURN_IF_ERROR(
          AddDenseOutputShapes(attrs.dense_shapes, input, c, &output_idx));
      return Status::OK();
    });

// Differences between ParseExample and ParseExampleV2:
//   * Supports ragged features.
//   * `serialized` may be a vector or a scalar.  (With v1, `serialized` could
//      only be a vector).
//   * Each set of keys is passed with a vector instead of a list of scalars.
//   * No Ndense attribute (not needed).
//   * num_sparse (formerly Nsparse) is no longer inferred; you must specify it
//     explicitly.
REGISTER_OP("ParseExampleV2")
    .Input("serialized: string")
    .Input("names: string")
    .Input("sparse_keys: string")
    .Input("dense_keys: string")
    .Input("ragged_keys: string")
    .Input("dense_defaults: Tdense")
    .Output("sparse_indices: num_sparse * int64")
    .Output("sparse_values: sparse_types")
    .Output("sparse_shapes: num_sparse * int64")
    .Output("dense_values: Tdense")
    .Output("ragged_values: ragged_value_types")
    .Output("ragged_row_splits: ragged_split_types")
    .Attr("Tdense: list({float,int64,string}) >= 0")  // Inferred
    .Attr("num_sparse: int >= 0")
    .Attr("sparse_types: list({float,int64,string}) >= 0")
    .Attr("ragged_value_types: list({float,int64,string}) >= 0")
    .Attr("ragged_split_types: list({int32,int64}) >= 0")
    .Attr("dense_shapes: list(shape) >= 0")

    .SetShapeFn([](InferenceContext* c) {
      ParseExampleAttrs attrs;
      TF_RETURN_IF_ERROR(attrs.Init(c, /*op_version=*/2));

      ShapeHandle input;
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(0), 1, &input));
      ShapeHandle names;
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(1), 1, &names));
      DimensionHandle num_examples = c->UnknownDim();
      if (c->RankKnown(input) && c->Rank(input) == 1) {
        num_examples = c->Dim(input, 0);
      }

      int output_idx = 0;
      AddSparseOutputShapes(attrs.num_sparse, input, 1, c, &output_idx);
      TF_RETURN_IF_ERROR(
          AddDenseOutputShapes(attrs.dense_shapes, input, c, &output_idx));
      TF_RETURN_IF_ERROR(AddRaggedOutputShapes(attrs.num_ragged, false,
                                               num_examples, c, &output_idx));

      return Status::OK();
    });

REGISTER_OP("ParseSingleExample")
    .Input("serialized: string")
    .Input("dense_defaults: Tdense")
    .Output("sparse_indices: num_sparse * int64")
    .Output("sparse_values: sparse_types")
    .Output("sparse_shapes: num_sparse * int64")
    .Output("dense_values: Tdense")
    .Attr("num_sparse: int >= 0")
    .Attr("sparse_keys: list(string) >= 0")
    .Attr("dense_keys: list(string) >= 0")
    .Attr("sparse_types: list({float,int64,string}) >= 0")
    .Attr("Tdense: list({float,int64,string}) >= 0")
    .Attr("dense_shapes: list(shape) >= 0")
    .SetShapeFn([](InferenceContext* c) {
      ParseSingleExampleAttrs attrs;
      TF_RETURN_IF_ERROR(attrs.Init(c));

      ShapeHandle input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &input));

      int output_idx = 0;
      AddSparseOutputShapes(attrs.sparse_keys.size(), input, 1, c, &output_idx);
      TF_RETURN_IF_ERROR(
          AddDenseOutputShapes(attrs.dense_shapes, input, c, &output_idx));
      return Status::OK();
    });

REGISTER_OP("ParseSequenceExample")
    .Input("serialized: string")
    .Input("debug_name: string")
    .Input("context_dense_defaults: Tcontext_dense")
    .Output("context_sparse_indices: Ncontext_sparse * int64")
    .Output("context_sparse_values: context_sparse_types")
    .Output("context_sparse_shapes: Ncontext_sparse * int64")
    .Output("context_dense_values: Tcontext_dense")
    .Output("feature_list_sparse_indices: Nfeature_list_sparse * int64")
    .Output("feature_list_sparse_values: feature_list_sparse_types")
    .Output("feature_list_sparse_shapes: Nfeature_list_sparse * int64")
    .Output("feature_list_dense_values: feature_list_dense_types")
    .Output("feature_list_dense_lengths: Nfeature_list_dense * int64")
    .Attr("feature_list_dense_missing_assumed_empty: list(string) >= 0")
    .Attr("context_sparse_keys: list(string) >= 0")
    .Attr("context_dense_keys: list(string) >= 0")
    .Attr("feature_list_sparse_keys: list(string) >= 0")
    .Attr("feature_list_dense_keys: list(string) >= 0")
    .Attr("Ncontext_sparse: int >= 0 = 0")
    .Attr("Ncontext_dense: int >= 0 = 0")
    .Attr("Nfeature_list_sparse: int >= 0 = 0")
    .Attr("Nfeature_list_dense: int >= 0 = 0")
    .Attr("context_sparse_types: list({float,int64,string}) >= 0 = []")
    .Attr("Tcontext_dense: list({float,int64,string}) >= 0 = []")
    .Attr("feature_list_dense_types: list({float,int64,string}) >= 0 = []")
    .Attr("context_dense_shapes: list(shape) >= 0 = []")
    .Attr("feature_list_sparse_types: list({float,int64,string}) >= 0 = []")
    .Attr("feature_list_dense_shapes: list(shape) >= 0 = []")
    .SetShapeFn([](InferenceContext* c) {
      ParseSequenceExampleAttrs attrs;
      TF_RETURN_IF_ERROR(attrs.Init(c));

      // Verify that the input is a vector, and carry the shape if known.
      ShapeHandle input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &input));
      ShapeHandle names;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &names));
      DimensionHandle num_examples = c->Dim(input, 0);
      ShapeHandle feature_list_dense_prefix =
          c->Matrix(num_examples, c->UnknownDim());

      int output_idx = 0;
      AddSparseOutputShapes(attrs.num_context_sparse, input, 1, c, &output_idx);
      TF_RETURN_IF_ERROR(AddDenseOutputShapes(attrs.context_dense_shapes, input,
                                              c, &output_idx));
      AddSparseOutputShapes(attrs.num_feature_list_sparse, input, 2, c,
                            &output_idx);
      TF_RETURN_IF_ERROR(AddDenseOutputShapes(attrs.feature_list_dense_shapes,
                                              feature_list_dense_prefix, c,
                                              &output_idx));
      AddDenseLengthsShapes(attrs.num_feature_list_dense, input, c,
                            &output_idx);

      return Status::OK();
    });

// Differences between ParseSequenceExample and ParseSequenceExampleV2:
//   * Supports ragged features.
//   * `serialized` may be a vector or a scalar.  (With v1, `serialized` could
//      only be a vector).
//   * Each set of keys is passed with a vector instead of an attr list.
//   * feature_list_dense_missing_assumed_empty is passed with as a boolean
//     vector (aligned 1:1 w/ feature_list_dense_kyes) rather than an attrib
//     containing a list of strings.
//   * No Ncontext_dense attribute (not needed).
REGISTER_OP("ParseSequenceExampleV2")
    .Input("serialized: string")
    .Input("debug_name: string")
    // Inputs: context features
    .Input("context_sparse_keys: string")
    .Input("context_dense_keys:  string")
    .Input("context_ragged_keys: string")
    // Inputs: feature lists
    .Input("feature_list_sparse_keys: string")
    .Input("feature_list_dense_keys: string")
    .Input("feature_list_ragged_keys: string")
    .Input("feature_list_dense_missing_assumed_empty: bool")
    .Input("context_dense_defaults: Tcontext_dense")
    // Outputs: context features
    .Output("context_sparse_indices: Ncontext_sparse * int64")
    .Output("context_sparse_values: context_sparse_types")
    .Output("context_sparse_shapes: Ncontext_sparse * int64")
    .Output("context_dense_values: Tcontext_dense")
    .Output("context_ragged_values: context_ragged_value_types")
    .Output("context_ragged_row_splits: context_ragged_split_types")
    // Outputs: feature lists
    .Output("feature_list_sparse_indices: Nfeature_list_sparse * int64")
    .Output("feature_list_sparse_values: feature_list_sparse_types")
    .Output("feature_list_sparse_shapes: Nfeature_list_sparse * int64")
    .Output("feature_list_dense_values: feature_list_dense_types")
    .Output("feature_list_dense_lengths: Nfeature_list_dense * int64")
    .Output("feature_list_ragged_values: feature_list_ragged_value_types")
    .Output("feature_list_ragged_outer_splits: feature_list_ragged_split_types")
    .Output("feature_list_ragged_inner_splits: feature_list_ragged_split_types")
    // Attribs: context features
    .Attr("Ncontext_sparse: int >= 0 = 0")
    .Attr("Tcontext_dense: list({float,int64,string}) >= 0 = []")  // inferred
    .Attr("context_sparse_types: list({float,int64,string}) >= 0 = []")
    .Attr("context_ragged_value_types: list({float,int64,string}) >= 0 = []")
    .Attr("context_ragged_split_types: list({int32,int64}) >= 0 = []")
    .Attr("context_dense_shapes: list(shape) >= 0 = []")
    // Attribs: feature lists
    .Attr("Nfeature_list_sparse: int >= 0 = 0")
    .Attr("Nfeature_list_dense: int >= 0 = 0")
    .Attr("feature_list_dense_types: list({float,int64,string}) >= 0 = []")
    .Attr("feature_list_sparse_types: list({float,int64,string}) >= 0 = []")
    .Attr(
        "feature_list_ragged_value_types: list({float,int64,string}) >= 0 = []")
    .Attr("feature_list_ragged_split_types: list({int32,int64}) >= 0 = []")
    .Attr("feature_list_dense_shapes: list(shape) >= 0 = []")
    .SetShapeFn([](InferenceContext* c) {
      ParseSequenceExampleAttrs attrs;
      TF_RETURN_IF_ERROR(attrs.Init(c, /*op_version=*/2));
      ShapeHandle input;
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(0), 1, &input));
      ShapeHandle names;
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(1), 1, &names));
      ShapeHandle feature_list_dense_prefix;
      TF_RETURN_IF_ERROR(c->Concatenate(input, c->UnknownShapeOfRank(1),
                                        &feature_list_dense_prefix));
      DimensionHandle num_examples = c->UnknownDim();
      if (c->RankKnown(input) && c->Rank(input) == 1) {
        num_examples = c->Dim(input, 0);
      }

      int output_idx = 0;
      // Context outputs.
      AddSparseOutputShapes(attrs.num_context_sparse, input, 1, c, &output_idx);
      TF_RETURN_IF_ERROR(AddDenseOutputShapes(attrs.context_dense_shapes, input,
                                              c, &output_idx));
      TF_RETURN_IF_ERROR(AddRaggedOutputShapes(attrs.num_context_ragged, false,
                                               num_examples, c, &output_idx));
      // FeatureList outputs.
      AddSparseOutputShapes(attrs.num_feature_list_sparse, input, 2, c,
                            &output_idx);
      TF_RETURN_IF_ERROR(AddDenseOutputShapes(attrs.feature_list_dense_shapes,
                                              feature_list_dense_prefix, c,
                                              &output_idx));
      AddDenseLengthsShapes(attrs.num_feature_list_dense, input, c,
                            &output_idx);
      TF_RETURN_IF_ERROR(AddRaggedOutputShapes(
          attrs.num_feature_list_ragged, true, num_examples, c, &output_idx));
      return Status::OK();
    });

REGISTER_OP("ParseSingleSequenceExample")
    .Input("serialized: string")
    .Input("feature_list_dense_missing_assumed_empty: string")
    .Input("context_sparse_keys: Ncontext_sparse * string")
    .Input("context_dense_keys: Ncontext_dense * string")
    .Input("feature_list_sparse_keys: Nfeature_list_sparse * string")
    .Input("feature_list_dense_keys: Nfeature_list_dense * string")
    .Input("context_dense_defaults: Tcontext_dense")
    .Input("debug_name: string")
    .Output("context_sparse_indices: Ncontext_sparse * int64")
    .Output("context_sparse_values: context_sparse_types")
    .Output("context_sparse_shapes: Ncontext_sparse * int64")
    .Output("context_dense_values: Tcontext_dense")
    .Output("feature_list_sparse_indices: Nfeature_list_sparse * int64")
    .Output("feature_list_sparse_values: feature_list_sparse_types")
    .Output("feature_list_sparse_shapes: Nfeature_list_sparse * int64")
    .Output("feature_list_dense_values: feature_list_dense_types")
    // Infer from context_sparse_keys
    .Attr("Ncontext_sparse: int >= 0 = 0")
    // Infer from context_dense_keys
    .Attr("Ncontext_dense: int >= 0 = 0")
    // Infer from feature_list_sparse_keys
    .Attr("Nfeature_list_sparse: int >= 0 = 0")
    // Infer from feature_list_dense_keys
    .Attr("Nfeature_list_dense: int >= 0 = 0")
    .Attr("context_sparse_types: list({float,int64,string}) >= 0 = []")
    .Attr("Tcontext_dense: list({float,int64,string}) >= 0 = []")
    .Attr("feature_list_dense_types: list({float,int64,string}) >= 0 = []")
    .Attr("context_dense_shapes: list(shape) >= 0 = []")
    .Attr("feature_list_sparse_types: list({float,int64,string}) >= 0 = []")
    .Attr("feature_list_dense_shapes: list(shape) >= 0 = []")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      ParseSingleSequenceExampleAttrs attrs;
      TF_RETURN_IF_ERROR(attrs.Init(c));

      ShapeHandle input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &input));

      // feature_list_dense_missing_assumed_empty
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &unused));

      int output_idx = 0;
      AddSparseOutputShapes(attrs.num_context_sparse, input, 1, c, &output_idx);
      TF_RETURN_IF_ERROR(AddDenseOutputShapes(attrs.context_dense_shapes, input,
                                              c, &output_idx));
      AddSparseOutputShapes(attrs.num_feature_list_sparse, input, 2, c,
                            &output_idx);
      TF_RETURN_IF_ERROR(AddDenseOutputShapes(attrs.feature_list_dense_shapes,
                                              c->UnknownShapeOfRank(1), c,
                                              &output_idx));
      return Status::OK();
    });

REGISTER_OP("ParseTensor")
    .Input("serialized: string")
    .Output("output: out_type")
    .Attr("out_type: type")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("SerializeTensor")
    .Input("tensor: T")
    .Output("serialized: string")
    .Attr("T: type")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("DecodeJSONExample")
    .Input("json_examples: string")
    .Output("binary_examples: string")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("DecodeCSV")
    .Input("records: string")
    .Input("record_defaults: OUT_TYPE")
    .Output("output: OUT_TYPE")
    .Attr("OUT_TYPE: list({float,double,int32,int64,string})")
    .Attr("field_delim: string = ','")
    .Attr("use_quote_delim: bool = true")
    .Attr("na_value: string = ''")
    .Attr("select_cols: list(int) = []")
    .SetShapeFn([](InferenceContext* c) {
      // Validate the record_defaults inputs.
      for (int i = 1; i < c->num_inputs(); ++i) {
        ShapeHandle v;
        TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(i), 1, &v));
        if (c->Rank(c->input(i)) == 1 && c->Value(c->Dim(v, 0)) > 1) {
          return errors::InvalidArgument(
              "Shape of a default must be a length-0 or length-1 vector, or a "
              "scalar.");
        }
      }

      // Propagate shape of the records input.
      for (int i = 0; i < c->num_outputs(); ++i) c->set_output(i, c->input(0));
      return Status::OK();
    });

REGISTER_OP("StringToNumber")
    .Input("string_tensor: string")
    .Output("output: out_type")
    .Attr("out_type: {float, double, int32, int64} = DT_FLOAT")
    .SetShapeFn(shape_inference::UnchangedShape);

}  // namespace tensorflow
