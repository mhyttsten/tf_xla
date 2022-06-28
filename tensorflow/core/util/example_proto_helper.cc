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
class MHTracer_DTPStensorflowPScorePSutilPSexample_proto_helperDTcc {
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
   MHTracer_DTPStensorflowPScorePSutilPSexample_proto_helperDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSutilPSexample_proto_helperDTcc() {
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
#include "tensorflow/core/util/example_proto_helper.h"

#include <vector>

#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/example/feature.pb.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"

namespace tensorflow {

Status CheckValidType(const DataType& dtype) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSutilPSexample_proto_helperDTcc mht_0(mht_0_v, 199, "", "./tensorflow/core/util/example_proto_helper.cc", "CheckValidType");

  switch (dtype) {
    case DT_INT64:
    case DT_FLOAT:
    case DT_STRING:
      return Status::OK();
    default:
      return errors::InvalidArgument("Received input dtype: ",
                                     DataTypeString(dtype));
  }
}

Status CheckTypesMatch(const Feature& feature, const DataType& dtype,
                       bool* match) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSutilPSexample_proto_helperDTcc mht_1(mht_1_v, 215, "", "./tensorflow/core/util/example_proto_helper.cc", "CheckTypesMatch");

  switch (dtype) {
    case DT_INT64:
      *match = (feature.kind_case() == Feature::kInt64List);
      break;
    case DT_FLOAT:
      *match = (feature.kind_case() == Feature::kFloatList);
      break;
    case DT_STRING:
      *match = (feature.kind_case() == Feature::kBytesList);
      break;
    default:
      return errors::InvalidArgument("Invalid input dtype: ",
                                     DataTypeString(dtype));
  }
  return Status::OK();
}

Status FeatureDenseCopy(const std::size_t out_index, const string& name,
                        const string& key, const DataType& dtype,
                        const TensorShape& shape, const Feature& feature,
                        Tensor* out) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("name: \"" + name + "\"");
   mht_2_v.push_back("key: \"" + key + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSexample_proto_helperDTcc mht_2(mht_2_v, 241, "", "./tensorflow/core/util/example_proto_helper.cc", "FeatureDenseCopy");

  const std::size_t num_elements = shape.num_elements();
  const std::size_t offset = out_index * num_elements;

  switch (dtype) {
    case DT_INT64: {
      const Int64List& values = feature.int64_list();
      if (static_cast<size_t>(values.value_size()) != num_elements) {
        return errors::InvalidArgument(
            "Name: ", name, ", Key: ", key, ", Index: ", out_index,
            ".  Number of int64 values != expected.  "
            "values size: ",
            values.value_size(), " but output shape: ", shape.DebugString());
      }
      auto out_p = out->flat<int64_t>().data() + offset;
      std::copy_n(values.value().data(), num_elements, out_p);
      return Status::OK();
    }
    case DT_FLOAT: {
      const FloatList& values = feature.float_list();
      if (static_cast<size_t>(values.value_size()) != num_elements) {
        return errors::InvalidArgument(
            "Name: ", name, ", Key: ", key, ", Index: ", out_index,
            ".  Number of float values != expected.  "
            "values size: ",
            values.value_size(), " but output shape: ", shape.DebugString());
      }
      auto out_p = out->flat<float>().data() + offset;
      std::copy_n(values.value().data(), num_elements, out_p);
      return Status::OK();
    }
    case DT_STRING: {
      const BytesList& values = feature.bytes_list();
      if (static_cast<size_t>(values.value_size()) != num_elements) {
        return errors::InvalidArgument(
            "Name: ", name, ", Key ", key, ", Index: ", out_index,
            ".  Number of bytes values != expected.  "
            "Values size: ",
            values.value_size(), " but output shape: ", shape.DebugString());
      }
      auto out_p = out->flat<tstring>().data() + offset;
      std::transform(values.value().data(),
                     values.value().data() + num_elements, out_p,
                     [](const string* s) { return *s; });
      return Status::OK();
    }
    default:
      return errors::InvalidArgument("Invalid input dtype: ",
                                     DataTypeString(dtype));
  }
}

Tensor FeatureSparseCopy(const std::size_t batch, const string& key,
                         const DataType& dtype, const Feature& feature) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("key: \"" + key + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSexample_proto_helperDTcc mht_3(mht_3_v, 298, "", "./tensorflow/core/util/example_proto_helper.cc", "FeatureSparseCopy");

  switch (dtype) {
    case DT_INT64: {
      const Int64List& values = feature.int64_list();
      const int64_t num_elements = values.value_size();
      Tensor out(dtype, TensorShape({num_elements}));
      auto out_p = out.flat<int64_t>().data();
      std::copy_n(values.value().data(), num_elements, out_p);
      return out;
    }
    case DT_FLOAT: {
      const FloatList& values = feature.float_list();
      const int64_t num_elements = values.value_size();
      Tensor out(dtype, TensorShape({num_elements}));
      auto out_p = out.flat<float>().data();
      std::copy_n(values.value().data(), num_elements, out_p);
      return out;
    }
    case DT_STRING: {
      const BytesList& values = feature.bytes_list();
      const int64_t num_elements = values.value_size();
      Tensor out(dtype, TensorShape({num_elements}));
      auto out_p = out.flat<tstring>().data();
      std::transform(values.value().data(),
                     values.value().data() + num_elements, out_p,
                     [](const string* s) { return *s; });
      return out;
    }
    default:
      LOG(FATAL) << "not supposed to be here.  dtype requested: " << dtype;
  }
}

int64_t CopyIntoSparseTensor(const Tensor& in, const int batch,
                             const int64_t offset, Tensor* indices,
                             Tensor* values) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSutilPSexample_proto_helperDTcc mht_4(mht_4_v, 336, "", "./tensorflow/core/util/example_proto_helper.cc", "CopyIntoSparseTensor");

  const int64_t num_elements = in.shape().num_elements();
  const DataType& dtype = in.dtype();
  CHECK_EQ(dtype, values->dtype());

  // Update indices.
  if (num_elements > 0) {
    auto ix_t = indices->matrix<int64_t>();
    int64_t* ix_p = &ix_t(offset, 0);
    for (int64_t i = 0; i < num_elements; ++i, ix_p += 2) {
      *ix_p = batch;    // Column 0 stores the batch entry
      *(ix_p + 1) = i;  // Column 1 stores the index in the batch
    }
  }

  // Copy values over.
  switch (dtype) {
    case DT_INT64: {
      std::copy_n(in.flat<int64_t>().data(), num_elements,
                  values->flat<int64_t>().data() + offset);
      break;
    }
    case DT_FLOAT: {
      std::copy_n(in.flat<float>().data(), num_elements,
                  values->flat<float>().data() + offset);
      break;
    }
    case DT_STRING: {
      std::copy_n(in.flat<tstring>().data(), num_elements,
                  values->flat<tstring>().data() + offset);
      break;
    }
    default:
      LOG(FATAL) << "Not supposed to be here.  Saw dtype: " << dtype;
  }

  return num_elements;
}

void RowDenseCopy(const std::size_t& out_index, const DataType& dtype,
                  const Tensor& in, Tensor* out) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSutilPSexample_proto_helperDTcc mht_5(mht_5_v, 379, "", "./tensorflow/core/util/example_proto_helper.cc", "RowDenseCopy");

  const std::size_t num_elements = in.shape().num_elements();
  const std::size_t offset = out_index * num_elements;

  switch (dtype) {
    case DT_INT64: {
      std::copy_n(in.flat<int64_t>().data(), num_elements,
                  out->flat<int64_t>().data() + offset);
      break;
    }
    case DT_FLOAT: {
      std::copy_n(in.flat<float>().data(), num_elements,
                  out->flat<float>().data() + offset);
      break;
    }
    case DT_STRING: {
      // TODO(dero): verify.
      std::copy_n(in.flat<tstring>().data(), num_elements,
                  out->flat<tstring>().data() + offset);
      break;
    }
    default:
      LOG(FATAL) << "Not supposed to be here.  Saw dtype: " << dtype;
  }
}

Status SingleExampleProtoToTensors(
    const Example& example, const string& example_name, const int batch_index,
    const std::vector<FixedLenFeature>& fixed_len_features,
    const std::vector<VarLenFeature>& var_len_features,
    std::vector<Tensor*>* output_dense_values_tensor,
    std::vector<std::vector<Tensor>>* output_sparse_values_tmp) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("example_name: \"" + example_name + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSexample_proto_helperDTcc mht_6(mht_6_v, 414, "", "./tensorflow/core/util/example_proto_helper.cc", "SingleExampleProtoToTensors");

  const Features& features = example.features();
  const auto& feature_dict = features.feature();

  // Handle dense features.
  for (size_t d = 0; d < fixed_len_features.size(); ++d) {
    const FixedLenFeature& feature_config = fixed_len_features[d];
    const string& key = feature_config.key;
    const DataType& dtype = feature_config.dtype;
    const TensorShape& shape = feature_config.shape;
    const Tensor& default_value = feature_config.default_value;
    bool required = (default_value.NumElements() == 0);
    const auto& feature_found = feature_dict.find(key);
    const bool feature_has_data =  // Found key & data type is set
        (feature_found != feature_dict.end() &&
         (feature_found->second.kind_case() != Feature::KIND_NOT_SET));

    const bool required_ok = feature_has_data || !required;
    if (!required_ok) {
      return errors::InvalidArgument("Name: ", example_name, ", Feature: ", key,
                                     " is required but could not be found.");
    }

    // Perform the FeatureDenseCopy into the output dense_values tensor (if
    // the value is present).
    if (feature_has_data) {
      const Feature& f = feature_found->second;
      bool types_match;
      TF_RETURN_IF_ERROR(CheckTypesMatch(f, dtype, &types_match));
      if (!types_match) {
        return errors::InvalidArgument("Name: ", example_name,
                                       ", Feature: ", key,
                                       ".  Data types don't match. ",
                                       "Expected type: ", DataTypeString(dtype),
                                       "  Feature is: ", f.DebugString());
      }
      TF_RETURN_IF_ERROR(FeatureDenseCopy(batch_index, example_name, key, dtype,
                                          shape, f,
                                          (*output_dense_values_tensor)[d]));
    } else {
      // If the value is missing, RowDenseCopy the default value.
      RowDenseCopy(batch_index, dtype, default_value,
                   (*output_dense_values_tensor)[d]);
    }
  }

  // Handle sparse features.
  for (size_t d = 0; d < var_len_features.size(); ++d) {
    const VarLenFeature& feature_config = var_len_features[d];
    const string& key = feature_config.key;
    const DataType& dtype = feature_config.dtype;
    const auto& feature_found = feature_dict.find(key);

    const bool feature_has_data =  // Found key & data type is set
        (feature_found != feature_dict.end() &&
         (feature_found->second.kind_case() != Feature::KIND_NOT_SET));

    if (feature_has_data) {
      const Feature& f = feature_found->second;
      bool types_match;
      TF_RETURN_IF_ERROR(CheckTypesMatch(f, dtype, &types_match));
      if (!types_match) {
        return errors::InvalidArgument("Name: ", example_name,
                                       ", Feature: ", key,
                                       ".  Data types don't match. ",
                                       "Expected type: ", DataTypeString(dtype),
                                       "  Feature is: ", f.DebugString());
      }
      (*output_sparse_values_tmp)[d][batch_index] =
          FeatureSparseCopy(batch_index, key, dtype, f);
    } else {
      (*output_sparse_values_tmp)[d][batch_index] =
          Tensor(dtype, TensorShape({0}));
    }
  }
  return Status::OK();
}

Status GetSparseTensorShapes(const VarLenFeature& var_len_feature,
                             const std::vector<Tensor>& sparse_values_tmp,
                             const int batch_size,
                             VarLenFeatureBatchShapes* output_shapes) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSutilPSexample_proto_helperDTcc mht_7(mht_7_v, 498, "", "./tensorflow/core/util/example_proto_helper.cc", "GetSparseTensorShapes");

  int64_t total_num_features = 0;
  int64_t max_num_features = 0;
  for (int b = 0; b < batch_size; ++b) {
    const Tensor& t = sparse_values_tmp[b];
    const int64_t num_elements = t.shape().num_elements();
    total_num_features += num_elements;
    max_num_features = std::max(max_num_features, num_elements);
  }
  output_shapes->indices_shape.AddDim(total_num_features);
  output_shapes->indices_shape.AddDim(2);
  output_shapes->values_shape.AddDim(total_num_features);
  output_shapes->max_num_features = max_num_features;
  return Status::OK();
}

Status BatchExampleProtoToTensors(
    const std::vector<const Example*>& examples,
    const std::vector<string>& names,
    const std::vector<FixedLenFeature>& fixed_len_features,
    const std::vector<VarLenFeature>& var_len_features, Allocator* allocator,
    std::vector<Tensor>* output_dense_values_tensor,
    std::vector<Tensor>* output_sparse_indices_tensor,
    std::vector<Tensor>* output_sparse_values_tensor,
    std::vector<Tensor>* output_sparse_shapes_tensor) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSutilPSexample_proto_helperDTcc mht_8(mht_8_v, 525, "", "./tensorflow/core/util/example_proto_helper.cc", "BatchExampleProtoToTensors");

  const int batch_size = examples.size();

  const bool has_names = (!names.empty());
  if (has_names) {
    if (names.size() != examples.size()) {
      return errors::InvalidArgument(
          "Expected len(names) == len(examples), but got: ", names.size(),
          " vs. ", examples.size());
    }
  }

  // We also need a map of Tensor pointers for the SingleExampleProtoToTensors
  // call. (Is there a better solution here?)
  std::vector<Tensor*> output_dense_values_tensor_ptrs(
      fixed_len_features.size());

  // Preallocate dense_values, since we know their sizes.
  for (size_t d = 0; d < fixed_len_features.size(); ++d) {
    const FixedLenFeature& config = fixed_len_features[d];
    TensorShape out_shape;
    out_shape.AddDim(batch_size);
    const TensorShape& shape = config.shape;
    const DataType& dtype = config.dtype;
    for (const int dim : shape.dim_sizes()) out_shape.AddDim(dim);
    (*output_dense_values_tensor)[d] = Tensor(allocator, dtype, out_shape);
    output_dense_values_tensor_ptrs[d] = &(*output_dense_values_tensor)[d];
  }

  // Temporary vector to hold sparse values.
  std::vector<std::vector<Tensor>> sparse_values_tmp(var_len_features.size());

  for (size_t d = 0; d < var_len_features.size(); ++d) {
    sparse_values_tmp[d] = std::vector<Tensor>(batch_size);
  }

  for (size_t b = 0; b < examples.size(); ++b) {
    const Example& ex = *(examples[b]);
    const string& example_name = (has_names) ? names[b] : "<unknown>";
    TF_RETURN_IF_ERROR(SingleExampleProtoToTensors(
        ex, example_name, b, fixed_len_features, var_len_features,
        &output_dense_values_tensor_ptrs, &sparse_values_tmp));
  }

  for (size_t d = 0; d < var_len_features.size(); ++d) {
    const VarLenFeature& feature_config = var_len_features[d];
    const DataType& dtype = feature_config.dtype;
    const std::vector<Tensor>& sparse_values_tensor = sparse_values_tmp[d];

    VarLenFeatureBatchShapes sparse_tensor_batch_shapes;
    TF_RETURN_IF_ERROR(GetSparseTensorShapes(feature_config,
                                             sparse_values_tensor, batch_size,
                                             &sparse_tensor_batch_shapes));
    const TensorShape& indices_shape = sparse_tensor_batch_shapes.indices_shape;
    const TensorShape& values_shape = sparse_tensor_batch_shapes.values_shape;

    // Allocate the sparse indices here.
    (*output_sparse_indices_tensor)[d] =
        Tensor(allocator, DT_INT64, indices_shape);
    (*output_sparse_values_tensor)[d] = Tensor(allocator, dtype, values_shape);
    (*output_sparse_shapes_tensor)[d] =
        Tensor(allocator, DT_INT64, TensorShape({2}));

    auto shape_t = (*output_sparse_shapes_tensor)[d].vec<int64_t>();
    shape_t(0) = batch_size;
    shape_t(1) = sparse_tensor_batch_shapes.max_num_features;

    Tensor* sp_indices_d = &(*output_sparse_indices_tensor)[d];
    Tensor* sp_values_d = &(*output_sparse_values_tensor)[d];

    int64_t offset = 0;
    for (int b = 0; b < batch_size; ++b) {
      const int64_t num_elements = CopyIntoSparseTensor(
          sparse_values_tensor[b], b, offset, sp_indices_d, sp_values_d);
      offset += num_elements;
    }
  }
  return Status::OK();
}

Status ParseExampleAttrs::FinishInit(int op_version) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSutilPSexample_proto_helperDTcc mht_9(mht_9_v, 608, "", "./tensorflow/core/util/example_proto_helper.cc", "ParseExampleAttrs::FinishInit");

  switch (op_version) {
    case 1:
      num_ragged = 0;
      break;
    case 2:
      num_dense = dense_types.size();
      num_ragged = ragged_value_types.size();
      break;
    default:
      return errors::InvalidArgument("Unexpected op_version", op_version);
  }
  if (static_cast<size_t>(num_sparse) != sparse_types.size()) {
    return errors::InvalidArgument("len(sparse_keys) != len(sparse_types)");
  }
  if (static_cast<size_t>(num_dense) != dense_types.size()) {
    return errors::InvalidArgument("len(dense_keys) != len(dense_types)");
  }
  if (static_cast<size_t>(num_dense) != dense_shapes.size()) {
    return errors::InvalidArgument("len(dense_keys) != len(dense_shapes)");
  }
  if (static_cast<size_t>(num_ragged) != ragged_value_types.size()) {
    return errors::InvalidArgument(
        "len(ragged_keys) != len(ragged_value_types)");
  }
  if (static_cast<size_t>(num_ragged) != ragged_split_types.size()) {
    return errors::InvalidArgument(
        "len(ragged_keys) != len(ragged_split_types)");
  }
  if (num_dense > std::numeric_limits<int32>::max()) {
    return errors::InvalidArgument("num_dense_ too large");
  }
  for (const DataType& type : dense_types) {
    TF_RETURN_IF_ERROR(CheckValidType(type));
  }
  for (const DataType& type : sparse_types) {
    TF_RETURN_IF_ERROR(CheckValidType(type));
  }
  for (const DataType& type : ragged_value_types) {
    TF_RETURN_IF_ERROR(CheckValidType(type));
  }
  for (const DataType& type : ragged_split_types) {
    if (!(type == DT_INT64 || type == DT_INT32)) {
      return errors::InvalidArgument("Invalid ragged_split_type: ",
                                     DataTypeString(type));
    }
  }
  return Status::OK();
}

Status ParseSingleExampleAttrs::FinishInit() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSutilPSexample_proto_helperDTcc mht_10(mht_10_v, 661, "", "./tensorflow/core/util/example_proto_helper.cc", "ParseSingleExampleAttrs::FinishInit");

  if (sparse_keys.size() != sparse_types.size()) {
    return errors::InvalidArgument("len(sparse_keys) != len(sparse_types)");
  }
  if (dense_keys.size() != dense_types.size()) {
    return errors::InvalidArgument("len(dense_keys) != len(dense_types)");
  }
  if (dense_keys.size() != dense_shapes.size()) {
    return errors::InvalidArgument("len(dense_keys) != len(dense_shapes)");
  }
  for (const DataType& type : dense_types) {
    TF_RETURN_IF_ERROR(CheckValidType(type));
  }
  for (const DataType& type : sparse_types) {
    TF_RETURN_IF_ERROR(CheckValidType(type));
  }
  return Status::OK();
}

Status ParseSequenceExampleAttrs::FinishInit(int op_version) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSutilPSexample_proto_helperDTcc mht_11(mht_11_v, 683, "", "./tensorflow/core/util/example_proto_helper.cc", "ParseSequenceExampleAttrs::FinishInit");

  switch (op_version) {
    case 1:
      num_context_ragged = 0;
      num_feature_list_ragged = 0;
      if (num_context_sparse != context_sparse_keys.size()) {
        return errors::InvalidArgument(
            "num_context_sparse (", num_context_sparse,
            ") must match the size of context_sparse_keys (",
            context_sparse_keys.size(), ")");
      }
      if (num_context_dense != context_dense_keys.size()) {
        return errors::InvalidArgument(
            "num_context_dense (", num_context_dense,
            ") must match the size of context_dense_keys (",
            context_dense_keys.size(), ")");
      }
      if (num_feature_list_sparse != feature_list_sparse_keys.size()) {
        return errors::InvalidArgument(
            "num_feature_list_sparse (", num_feature_list_sparse,
            ") must match the size of feature_list_sparse_keys (",
            feature_list_sparse_keys.size(), ")");
      }
      if (num_feature_list_dense != feature_list_dense_keys.size()) {
        return errors::InvalidArgument(
            "num_feature_list_dense (", num_feature_list_dense,
            ") must match the size of feature_list_dense_keys (",
            feature_list_dense_keys.size(), ")");
      }
      break;
    case 2:
      num_context_dense = context_dense_types.size();
      num_context_ragged = context_ragged_value_types.size();
      num_feature_list_ragged = feature_list_ragged_value_types.size();
      break;
    default:
      return errors::InvalidArgument("Unexpected op_version", op_version);
  }
  if (num_context_sparse != context_sparse_types.size()) {
    return errors::InvalidArgument(
        "num_context_sparse (", num_context_sparse,
        ") must match the size of context_sparse_types (",
        context_sparse_types.size(), ")");
  }
  if (num_context_dense != context_dense_types.size() ||
      num_context_dense != context_dense_shapes.size()) {
    return errors::InvalidArgument(
        "num_context_dense (", num_context_dense,
        ") must match the size of context_dense_types (",
        context_dense_types.size(), ") and context_dense_shapes (",
        context_dense_shapes.size(), ")");
  }
  if ((num_context_ragged != context_ragged_value_types.size()) ||
      (num_context_ragged != context_ragged_split_types.size())) {
    return errors::InvalidArgument(
        "num_context_ragged (", num_context_ragged,
        ") must match the size of context_ragged_value_types (",
        context_ragged_value_types.size(), ") and context_ragged_split_types (",
        context_ragged_split_types.size(), ")");
  }
  if (num_feature_list_sparse != feature_list_sparse_types.size()) {
    return errors::InvalidArgument(
        "num_feature_list_sparse (", num_feature_list_sparse,
        ") must match the size of feature_list_sparse_types (",
        feature_list_sparse_types.size(), ")");
  }
  if (num_feature_list_dense != feature_list_dense_types.size() ||
      num_feature_list_dense != feature_list_dense_shapes.size()) {
    return errors::InvalidArgument(
        "num_feature_list_dense (", num_feature_list_dense,
        ") must match the size of feature_list_dense_types (",
        feature_list_dense_types.size(), ") and feature_list_dense_shapes (",
        feature_list_dense_shapes.size(), ")");
  }
  if ((num_feature_list_ragged != feature_list_ragged_value_types.size()) ||
      (num_feature_list_ragged != feature_list_ragged_split_types.size())) {
    return errors::InvalidArgument(
        "num_feature_list_ragged (", num_feature_list_ragged,
        ") must match the size of feature_list_ragged_value_types (",
        feature_list_ragged_value_types.size(),
        ") and feature_list_ragged_split_types (",
        feature_list_ragged_split_types.size(), ")");
  }
  for (const DataType& type : context_dense_types) {
    TF_RETURN_IF_ERROR(CheckValidType(type));
  }
  for (const DataType& type : context_sparse_types) {
    TF_RETURN_IF_ERROR(CheckValidType(type));
  }
  for (const DataType& type : feature_list_dense_types) {
    TF_RETURN_IF_ERROR(CheckValidType(type));
  }
  for (const DataType& type : feature_list_sparse_types) {
    TF_RETURN_IF_ERROR(CheckValidType(type));
  }
  for (const DataType& type : context_ragged_value_types) {
    TF_RETURN_IF_ERROR(CheckValidType(type));
  }
  for (const DataType& type : context_ragged_split_types) {
    if (!(type == DT_INT64 || type == DT_INT32)) {
      return errors::InvalidArgument("Invalid context_ragged_split_type: ",
                                     DataTypeString(type));
    }
  }
  for (const DataType& type : feature_list_ragged_value_types) {
    TF_RETURN_IF_ERROR(CheckValidType(type));
  }
  for (const DataType& type : feature_list_ragged_split_types) {
    if (!(type == DT_INT64 || type == DT_INT32)) {
      return errors::InvalidArgument("Invalid feature_list_ragged_split_type: ",
                                     DataTypeString(type));
    }
  }

  return Status::OK();
}

Status ParseSingleSequenceExampleAttrs::FinishInit() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSutilPSexample_proto_helperDTcc mht_12(mht_12_v, 803, "", "./tensorflow/core/util/example_proto_helper.cc", "ParseSingleSequenceExampleAttrs::FinishInit");

  if (static_cast<size_t>(num_context_sparse) != context_sparse_types.size()) {
    return errors::InvalidArgument(
        "len(context_sparse_keys) != len(context_sparse_types)");
  }
  if (static_cast<size_t>(num_context_dense) != context_dense_types.size()) {
    return errors::InvalidArgument(
        "len(context_dense_keys) != len(context_dense_types)");
  }
  if (static_cast<size_t>(num_context_dense) != context_dense_shapes.size()) {
    return errors::InvalidArgument(
        "len(context_dense_keys) != len(context_dense_shapes)");
  }
  if (static_cast<size_t>(num_feature_list_sparse) !=
      feature_list_sparse_types.size()) {
    return errors::InvalidArgument(
        "len(feature_list_sparse_keys) != len(feature_list_sparse_types)");
  }
  if (static_cast<size_t>(num_feature_list_dense) !=
      feature_list_dense_types.size()) {
    return errors::InvalidArgument(
        "len(feature_list_dense_keys) != "
        "len(feature_list_dense_types)");
  }
  for (const DataType& type : context_dense_types) {
    TF_RETURN_IF_ERROR(CheckValidType(type));
  }
  for (const DataType& type : context_sparse_types) {
    TF_RETURN_IF_ERROR(CheckValidType(type));
  }
  for (const DataType& type : feature_list_dense_types) {
    TF_RETURN_IF_ERROR(CheckValidType(type));
  }
  for (const DataType& type : feature_list_sparse_types) {
    TF_RETURN_IF_ERROR(CheckValidType(type));
  }
  return Status::OK();
}

Status GetDenseShapes(const std::vector<PartialTensorShape>& dense_shapes,
                      std::vector<bool>* variable_length,
                      std::vector<std::size_t>* elements_per_stride) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSutilPSexample_proto_helperDTcc mht_13(mht_13_v, 847, "", "./tensorflow/core/util/example_proto_helper.cc", "GetDenseShapes");

  // Temporary check until we start allowing a variable length outer
  // dimension.
  for (int i = 0; i < dense_shapes.size(); ++i) {
    bool shape_ok = true;
    if (dense_shapes[i].dims() == -1) {
      shape_ok = false;
    } else {
      for (int d = 1; d < dense_shapes[i].dims(); ++d) {
        if (dense_shapes[i].dim_size(d) == -1) {
          shape_ok = false;
        }
      }
    }
    if (!shape_ok) {
      return errors::InvalidArgument(
          "dense_shapes[", i,
          "] has unknown rank or unknown inner dimensions: ",
          dense_shapes[i].DebugString());
    }
    TensorShape dense_shape;
    if (dense_shapes[i].dims() > 0 && dense_shapes[i].dim_size(0) == -1) {
      variable_length->push_back(true);
      for (int d = 1; d < dense_shapes[i].dims(); ++d) {
        dense_shape.AddDim(dense_shapes[i].dim_size(d));
      }
    } else {
      variable_length->push_back(false);
      dense_shapes[i].AsTensorShape(&dense_shape);
    }
    elements_per_stride->push_back(dense_shape.num_elements());
  }
  return Status::OK();
}

}  // namespace tensorflow
