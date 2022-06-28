/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_UTIL_EXAMPLE_PROTO_HELPER_H_
#define TENSORFLOW_CORE_UTIL_EXAMPLE_PROTO_HELPER_H_
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
class MHTracer_DTPStensorflowPScorePSutilPSexample_proto_helperDTh {
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
   MHTracer_DTPStensorflowPScorePSutilPSexample_proto_helperDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSutilPSexample_proto_helperDTh() {
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


#include <string>
#include <unordered_set>
#include <vector>

#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/example/feature.pb.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"

// This is a set of helper methods that will make it possible to share
// tensorflow::Example proto Tensor conversion code inside the ExampleParserOp
// OpKernel as well as in external code.
namespace tensorflow {

// "Dense" feature configuration.
struct FixedLenFeature {
  string key;
  DataType dtype;
  TensorShape shape;
  Tensor default_value;
  string values_output_tensor_name;
};

// "Sparse" feature configuration.
struct VarLenFeature {
  string key;
  DataType dtype;
  string values_output_tensor_name;
  string indices_output_tensor_name;
  string shapes_output_tensor_name;
};

// Given a single tensorflow::Example, with an optional example name
// at a particular index within a batch, and dense and sparse feature
// configurations from fixed_len_features, var_len_features, this method
// updates the dense value tensor and the sparse values temporary vector
// of tensors. The indexing of the output vectors correspond 1:1 to the
// indexing of the feature configuration vectors.
//
// The fixed_len_features and var_len_features maps are assume to be
// have disjoint key fields from the Feature map in the tensorflow.Example
// proto.
//
// For each sparse feature, the sparse values temporary vector holds a
// tensor for each Example. Each tensor is either empty or filled, depending
// on if the sparse feature value is set for the Example. This
// temporary structure is needed because we need to know the total number
// of filled elements in the batch to get the proper final sparse tensor
// shapes allocated.  After the entire batch is processed,
// GetSparseTensorShape can be used to calculate the final shapes and
// CopyIntoSparseTensor can be used to copy from the temporary vector
// into the final allocated tensors.
Status SingleExampleProtoToTensors(
    const Example& example, const string& name, const int batch_index,
    const std::vector<FixedLenFeature>& fixed_len_features,
    const std::vector<VarLenFeature>& var_len_features,
    std::vector<Tensor*>* dense_values,
    std::vector<std::vector<Tensor>>* sparse_values_temporary_vector);

// The shape of the indices and values tensors associated with a SparseTensor
// are dependent on the contents of the batch.
struct VarLenFeatureBatchShapes {
  TensorShape indices_shape;
  TensorShape values_shape;
  int max_num_features;
};

// Get the shape of the sparse values and indices tensors for the batch,
// given how many of the tensors in the temporary sparse values vector
// are actually filled.
Status GetSparseTensorShapes(const VarLenFeature& var_len_feature,
                             const std::vector<Tensor>& sparse_values_tmp,
                             const int batch_size,
                             VarLenFeatureBatchShapes* output_shapes);

// A method to convert a batch of tensorflow::Example protos into output
// tensors. This method is useful if there already is a batch of deserialized
// Example protos in memory (such as a serving use-case) and we do not wish
// to incur an extraneous serialize/deserialize.  It is intended
// as an outside of OpKernel compatible replacement for the functionality of
// ExampleParserOp. In a serving setting, this method could be used to produce
// a feed_dict of Tensors that could bypass the ExampleParserOp.
//
// Note that unlike SingleExampleProtoToTensors, output tensors are
// allocated using a provided Allocator within this method.
Status BatchExampleProtoToTensors(
    const std::vector<const Example*>& examples,
    const std::vector<string>& names,
    const std::vector<FixedLenFeature>& fixed_len_features,
    const std::vector<VarLenFeature>& var_len_features, Allocator* allocator,
    std::vector<Tensor>* output_dense_values_tensor,
    std::vector<Tensor>* output_sparse_indices_tensor,
    std::vector<Tensor>* output_sparse_values_tensor,
    std::vector<Tensor>* output_sparse_shapes_tensor);

// Check that the given dtype is one that is compatible with
// tensorflow::Example protocol buffer feature values.
Status CheckValidType(const DataType& dtype);

// Check that the provided Feature proto message's oneof value
// matches that of the provided dtype.
Status CheckTypesMatch(const Feature& feature, const DataType& dtype,
                       bool* match);

// For a single Example, copy a dense feature value into an output
// dense value tensor Out at the provided out_index offset.
Status FeatureDenseCopy(const std::size_t out_index, const string& name,
                        const string& key, const DataType& dtype,
                        const TensorShape& shape, const Feature& feature,
                        Tensor* out);

// Copy the value a provided Tensor into an output dense_value tensor Out
// at the provided out_index offset.
void RowDenseCopy(const std::size_t& out_index, const DataType& dtype,
                  const Tensor& in, Tensor* out);

// For a single Example, and given sparse feature return a temporary output
// Tensor suitable for being collected in the temporary sparse value vector.
Tensor FeatureSparseCopy(const std::size_t batch, const string& key,
                         const DataType& dtype, const Feature& feature);

// Copy a temporary Tensor into the final sparse indices and values
// tensor at a given batch index and element offset. This method
// assumes that the indices/values Tensors have been properly allocated
// for the batch.
int64_t CopyIntoSparseTensor(const Tensor& in, const int batch,
                             const int64_t offset, Tensor* indices,
                             Tensor* values);

// Check that each dense_shape has known rank and inner dimensions; and
// update variable_length (whether the outer dimension is None) and
// elements_per_stride for each denes_shape.
Status GetDenseShapes(const std::vector<PartialTensorShape>& dense_shapes,
                      std::vector<bool>* variable_length,
                      std::vector<std::size_t>* elements_per_stride);

// Parses the attributes passed to ParseExample.
// REQUIRES: Init must be called after construction.
struct ParseExampleAttrs {
 public:
  template <typename ContextType>
  Status Init(ContextType* ctx, int op_version = 1) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSutilPSexample_proto_helperDTh mht_0(mht_0_v, 335, "", "./tensorflow/core/util/example_proto_helper.h", "Init");

    TF_RETURN_IF_ERROR(ctx->GetAttr("sparse_types", &sparse_types));
    TF_RETURN_IF_ERROR(ctx->GetAttr("Tdense", &dense_types));
    TF_RETURN_IF_ERROR(ctx->GetAttr("dense_shapes", &dense_shapes));
    TF_RETURN_IF_ERROR(
        GetDenseShapes(dense_shapes, &variable_length, &elements_per_stride));
    switch (op_version) {
      case 1:
        TF_RETURN_IF_ERROR(ctx->GetAttr("Nsparse", &num_sparse));
        TF_RETURN_IF_ERROR(ctx->GetAttr("Ndense", &num_dense));
        break;
      case 2:
        TF_RETURN_IF_ERROR(
            ctx->GetAttr("ragged_value_types", &ragged_value_types));
        TF_RETURN_IF_ERROR(ctx->GetAttr("num_sparse", &num_sparse));
        TF_RETURN_IF_ERROR(
            ctx->GetAttr("ragged_split_types", &ragged_split_types));
        break;
      default:
        return errors::InvalidArgument("Unexpected op_version", op_version);
    }
    return FinishInit(op_version);
  }

  int64_t num_sparse;
  int64_t num_dense;
  int64_t num_ragged;
  std::vector<DataType> sparse_types;
  std::vector<DataType> dense_types;
  std::vector<DataType> ragged_value_types;
  std::vector<DataType> ragged_split_types;
  std::vector<PartialTensorShape> dense_shapes;
  std::vector<bool> variable_length;
  std::vector<std::size_t> elements_per_stride;

 private:
  Status FinishInit(int op_version);  // for context-independent parts of Init.
};

// Parses the attributes passed to ParseSingleExample.
// REQUIRES: Init must be called after construction.
struct ParseSingleExampleAttrs {
 public:
  template <typename ContextType>
  Status Init(ContextType* ctx) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSutilPSexample_proto_helperDTh mht_1(mht_1_v, 382, "", "./tensorflow/core/util/example_proto_helper.h", "Init");

    TF_RETURN_IF_ERROR(ctx->GetAttr("sparse_keys", &sparse_keys));
    TF_RETURN_IF_ERROR(ctx->GetAttr("sparse_types", &sparse_types));
    TF_RETURN_IF_ERROR(ctx->GetAttr("dense_keys", &dense_keys));
    TF_RETURN_IF_ERROR(ctx->GetAttr("Tdense", &dense_types));
    TF_RETURN_IF_ERROR(ctx->GetAttr("dense_shapes", &dense_shapes));

    int num_sparse;
    TF_RETURN_IF_ERROR(ctx->GetAttr("num_sparse", &num_sparse));
    if (num_sparse != sparse_keys.size() || num_sparse != sparse_types.size()) {
      return errors::InvalidArgument(
          "num_sparse (", num_sparse, ") must match the size of sparse_keys (",
          sparse_keys.size(), ") and sparse_types (", sparse_types.size(), ")");
    }

    TF_RETURN_IF_ERROR(
        GetDenseShapes(dense_shapes, &variable_length, &elements_per_stride));
    return FinishInit();
  }

  std::vector<tstring> sparse_keys;
  std::vector<DataType> sparse_types;
  std::vector<tstring> dense_keys;
  std::vector<DataType> dense_types;
  std::vector<PartialTensorShape> dense_shapes;
  std::vector<bool> variable_length;
  std::vector<std::size_t> elements_per_stride;

 private:
  Status FinishInit();  // for context-independent parts of Init.
};

// Parses the attributes passed to ParseSequenceExample.
// REQUIRES: Init must be called after construction.
struct ParseSequenceExampleAttrs {
 public:
  template <typename ContextType>
  Status Init(ContextType* ctx, int op_version = 1) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSutilPSexample_proto_helperDTh mht_2(mht_2_v, 422, "", "./tensorflow/core/util/example_proto_helper.h", "Init");

    switch (op_version) {
      case 1: {
        std::vector<string> missing_empty_vector;
        TF_RETURN_IF_ERROR(ctx->GetAttr(
            "feature_list_dense_missing_assumed_empty", &missing_empty_vector));
        for (const string& feature : missing_empty_vector) {
          feature_list_dense_missing_assumed_empty.insert(feature);
        }
      }
        TF_RETURN_IF_ERROR(
            ctx->GetAttr("context_sparse_keys", &context_sparse_keys));
        TF_RETURN_IF_ERROR(
            ctx->GetAttr("context_dense_keys", &context_dense_keys));
        TF_RETURN_IF_ERROR(ctx->GetAttr("feature_list_sparse_keys",
                                        &feature_list_sparse_keys));
        TF_RETURN_IF_ERROR(
            ctx->GetAttr("feature_list_dense_keys", &feature_list_dense_keys));
        TF_RETURN_IF_ERROR(ctx->GetAttr("Ncontext_dense", &num_context_dense));
        break;
      case 2:
        TF_RETURN_IF_ERROR(ctx->GetAttr("context_ragged_value_types",
                                        &context_ragged_value_types));
        TF_RETURN_IF_ERROR(ctx->GetAttr("context_ragged_split_types",
                                        &context_ragged_split_types));
        TF_RETURN_IF_ERROR(ctx->GetAttr("feature_list_ragged_value_types",
                                        &feature_list_ragged_value_types));
        TF_RETURN_IF_ERROR(ctx->GetAttr("feature_list_ragged_split_types",
                                        &feature_list_ragged_split_types));
        break;
      default:
        return errors::InvalidArgument("Unexpected op_version", op_version);
    }
    TF_RETURN_IF_ERROR(
        ctx->GetAttr("context_sparse_types", &context_sparse_types));
    TF_RETURN_IF_ERROR(
        ctx->GetAttr("Nfeature_list_dense", &num_feature_list_dense));
    TF_RETURN_IF_ERROR(ctx->GetAttr("Ncontext_sparse", &num_context_sparse));
    TF_RETURN_IF_ERROR(ctx->GetAttr("Tcontext_dense", &context_dense_types));
    TF_RETURN_IF_ERROR(
        ctx->GetAttr("feature_list_sparse_types", &feature_list_sparse_types));
    TF_RETURN_IF_ERROR(
        ctx->GetAttr("feature_list_dense_types", &feature_list_dense_types));
    TF_RETURN_IF_ERROR(
        ctx->GetAttr("Nfeature_list_sparse", &num_feature_list_sparse));
    TF_RETURN_IF_ERROR(
        ctx->GetAttr("context_dense_shapes", &context_dense_shapes));
    TF_RETURN_IF_ERROR(
        ctx->GetAttr("feature_list_dense_shapes", &feature_list_dense_shapes));
    return FinishInit(op_version);
  }

  std::unordered_set<string> feature_list_dense_missing_assumed_empty;
  int64_t num_context_sparse;
  int64_t num_context_dense;
  int64_t num_context_ragged;
  int64_t num_feature_list_sparse;
  int64_t num_feature_list_dense;
  int64_t num_feature_list_ragged;
  std::vector<tstring> context_sparse_keys;
  std::vector<tstring> context_dense_keys;
  std::vector<tstring> feature_list_sparse_keys;
  std::vector<tstring> feature_list_dense_keys;
  std::vector<DataType> context_sparse_types;
  std::vector<DataType> context_dense_types;
  std::vector<TensorShape> context_dense_shapes;
  std::vector<DataType> feature_list_sparse_types;
  std::vector<DataType> feature_list_dense_types;
  std::vector<TensorShape> feature_list_dense_shapes;
  std::vector<DataType> context_ragged_value_types;
  std::vector<DataType> context_ragged_split_types;
  std::vector<DataType> feature_list_ragged_value_types;
  std::vector<DataType> feature_list_ragged_split_types;

 private:
  Status FinishInit(int op_version);  // for context-independent parts of Init.
};

// Parses the attributes passed to ParseSingleSequenceExample.
// REQUIRES: Init must be called after construction.
struct ParseSingleSequenceExampleAttrs {
 public:
  template <typename ContextType>
  Status Init(ContextType* ctx) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSutilPSexample_proto_helperDTh mht_3(mht_3_v, 508, "", "./tensorflow/core/util/example_proto_helper.h", "Init");

    TF_RETURN_IF_ERROR(
        ctx->GetAttr("context_sparse_types", &context_sparse_types));
    TF_RETURN_IF_ERROR(ctx->GetAttr("Ncontext_dense", &num_context_dense));
    TF_RETURN_IF_ERROR(
        ctx->GetAttr("Nfeature_list_dense", &num_feature_list_dense));
    TF_RETURN_IF_ERROR(ctx->GetAttr("Ncontext_sparse", &num_context_sparse));
    TF_RETURN_IF_ERROR(ctx->GetAttr("Tcontext_dense", &context_dense_types));
    TF_RETURN_IF_ERROR(
        ctx->GetAttr("feature_list_sparse_types", &feature_list_sparse_types));
    TF_RETURN_IF_ERROR(
        ctx->GetAttr("feature_list_dense_types", &feature_list_dense_types));
    TF_RETURN_IF_ERROR(
        ctx->GetAttr("Nfeature_list_sparse", &num_feature_list_sparse));
    TF_RETURN_IF_ERROR(
        ctx->GetAttr("context_dense_shapes", &context_dense_shapes));
    TF_RETURN_IF_ERROR(
        ctx->GetAttr("feature_list_dense_shapes", &feature_list_dense_shapes));
    return FinishInit();
  }

  int64_t num_context_sparse;
  int64_t num_context_dense;
  int64_t num_feature_list_sparse;
  int64_t num_feature_list_dense;
  std::vector<DataType> context_sparse_types;
  std::vector<DataType> context_dense_types;
  std::vector<TensorShape> context_dense_shapes;
  std::vector<DataType> feature_list_sparse_types;
  std::vector<DataType> feature_list_dense_types;
  std::vector<TensorShape> feature_list_dense_shapes;

 private:
  Status FinishInit();  // for context-independent parts of Init.
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_EXAMPLE_PROTO_HELPER_H_
