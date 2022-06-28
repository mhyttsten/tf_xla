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
class MHTracer_DTPStensorflowPSlitePSkernelsPSparse_examplePSparse_exampleDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSparse_examplePSparse_exampleDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSparse_examplePSparse_exampleDTcc() {
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
#include "tensorflow/lite/kernels/parse_example/parse_example.h"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <unordered_map>

#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "tensorflow/core/example/feature.pb.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/blocking_counter.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/fingerprint.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/util/example_proto_fast_parsing.h"
#include "tensorflow/core/util/presized_cuckoo_map.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/parse_example/example_proto_fast_parsing.h"
#include "tensorflow/lite/mutable_op_resolver.h"
#include "tensorflow/lite/string_util.h"

namespace tflite {
namespace ops {
namespace custom {
namespace parse_example {
namespace {

namespace tf = ::tensorflow;
using tf::Status;
using tf::StringPiece;
using tf::tstring;
using tf::example::CopyOrMoveBlock;
using tf::example::FastParseExampleConfig;
using tf::example::GetListFromBuffer;
using tf::example::LimitedArraySlice;
using tf::example::ParseExample;
using tf::example::SeededHasher;
using tf::example::SmallVector;
using tf::example::SparseBuffer;
using tf::example::Type;
using tf::example::parsed::Example;

using ConfigIndex = tf::PresizedCuckooMap<std::pair<int32_t, Type>>;

struct TfLiteResult {
  std::vector<TfLiteTensor*> dense_values;
  std::vector<TfLiteTensor*> sparse_values;
  std::vector<TfLiteTensor*> sparse_indices;
  std::vector<TfLiteTensor*> sparse_shapes;
  std::map<int, tf::Tensor> dense_tensors;
};

template <typename T>
void FillAndCopyVarLen(const int d, const size_t num_elements,
                       const size_t num_elements_per_minibatch,
                       const FastParseExampleConfig& config,
                       std::vector<SparseBuffer>& varlen_dense_buffers,
                       TfLiteTensor* values) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSparse_examplePSparse_exampleDTcc mht_0(mht_0_v, 245, "", "./tensorflow/lite/kernels/parse_example/parse_example.cc", "FillAndCopyVarLen");

  const tf::Tensor& default_value = config.dense[d].default_value;

  // Copy-fill the tensors (creating the zero/fill-padding)
  std::fill(reinterpret_cast<T*>(values->data.raw),
            reinterpret_cast<T*>(values->data.raw) + num_elements,
            default_value.flat<T>()(0));

  auto data = reinterpret_cast<T*>(values->data.raw);

  const SparseBuffer& buffer = varlen_dense_buffers[d];
  // Number of examples being stored in this buffer
  const auto& end_indices = buffer.example_end_indices;
  const size_t examples_in_buffer = end_indices.size();

  const auto& list = GetListFromBuffer<T>(buffer);
  auto list_ptr = list.begin();

  size_t elements_tally = 0;
  // Iterate through all the examples stored in this buffer.
  for (size_t j = 0; j < examples_in_buffer; ++j) {
    // Number of elements stored for this example.
    const size_t num_elems = end_indices[j] - elements_tally;
    CopyOrMoveBlock(list_ptr, list_ptr + num_elems, data);
    // Move forward this many elements in the varlen buffer.
    list_ptr += num_elems;
    // Move forward to the next minibatch entry in the values output.
    data += num_elements_per_minibatch;
    elements_tally = end_indices[j];
  }
  DCHECK(elements_tally == list.size());
}

bool ParseExample(StringRef serialized, Example* example) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSparse_examplePSparse_exampleDTcc mht_1(mht_1_v, 281, "", "./tensorflow/lite/kernels/parse_example/parse_example.cc", "ParseExample");

  DCHECK(example != nullptr);
  tf::protobuf::io::CodedInputStream stream(
      reinterpret_cast<const uint8*>(serialized.str), serialized.len);
  tensorflow::example::EnableAliasing(&stream);
  return ParseExample(&stream, example);
}

Status FastParseSerializedExample(
    StringRef serialized_example, const tstring& example_name,
    const size_t example_index, const FastParseExampleConfig& config,
    bool* quick_filter, int quick_filter_size,
    const std::unique_ptr<ConfigIndex>& config_index, int config_index_size,
    SeededHasher* hasher, std::vector<TfLiteTensor*>* output_dense,
    std::vector<SparseBuffer>* output_varlen_dense,
    std::vector<SparseBuffer>* output_sparse,
    std::map<absl::string_view, int>& stats, TfLiteResult* result) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("example_name: \"" + (std::string)example_name + "\"");
   MHTracer_DTPStensorflowPSlitePSkernelsPSparse_examplePSparse_exampleDTcc mht_2(mht_2_v, 301, "", "./tensorflow/lite/kernels/parse_example/parse_example.cc", "FastParseSerializedExample");

  DCHECK(output_dense != nullptr);
  tensorflow::example::parsed::Example parsed_example;
  if (!ParseExample(serialized_example, &parsed_example)) {
    return tf::errors::Internal("Failed to parse example");
  }
  std::vector<int64_t> dense_feature_last_example(config.dense.size(), -1);
  std::vector<int64_t> sparse_feature_last_example(config.sparse.size(), -1);
  // Handle features present in the example.
  const size_t parsed_example_size = parsed_example.size();
  for (size_t i = 0; i < parsed_example_size; ++i) {
    // This is a logic that standard protobuf parsing is implementing.
    // I.e. last entry in the map overwrites all the previous ones.
    tensorflow::example::parsed::FeatureMapEntry& name_and_feature =
        parsed_example[parsed_example_size - i - 1];
    const StringPiece feature_name = name_and_feature.first;
    tensorflow::example::parsed::Feature& feature = name_and_feature.second;
    if (feature_name.length() >= quick_filter_size ||
        !quick_filter[feature_name.length()]) {
      continue;
    }
    const uint64_t h = (*hasher)(feature_name);
    std::pair<int32_t, Type> d_and_type;
    if (!config_index->Find(h, &d_and_type)) {
      continue;
    }
    size_t d = d_and_type.first;
    bool is_dense = d_and_type.second == Type::Dense;

    auto example_error = [&](StringPiece suffix) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSparse_examplePSparse_exampleDTcc mht_3(mht_3_v, 333, "", "./tensorflow/lite/kernels/parse_example/parse_example.cc", "lambda");

      return tf::errors::Internal("Name: ", example_name,
                                  ", Key: ", feature_name,
                                  ", Index: ", example_index, ".  ", suffix);
    };

    auto parse_error = [&] {
      return example_error("Can't parse serialized Example.");
    };

    tf::DataType example_dtype;
    if (feature.ParseDataType(&example_dtype) != Status::OK()) {
      return parse_error();
    }
    if (is_dense) {
      if (example_dtype == tf::DT_INVALID) continue;

      dense_feature_last_example[d] = example_index;

      if (example_dtype != config.dense[d].dtype) {
        return example_error(absl::StrCat(
            "Data types don't match. Data type: ",
            DataTypeString(example_dtype),
            " but expected type: ", DataTypeString(config.dense[d].dtype)));
      }
      if (!config.dense[d].variable_length) {
        TfLiteTensor* out = (*output_dense)[d];

        const std::size_t num_elements = config.dense[d].elements_per_stride;
        const std::size_t offset = example_index * num_elements;

        auto shape_error = [&](size_t size, StringPiece type_str) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSparse_examplePSparse_exampleDTcc mht_4(mht_4_v, 367, "", "./tensorflow/lite/kernels/parse_example/parse_example.cc", "lambda");

          return example_error(absl::StrCat(
              "Number of ", type_str,
              " values != expected.  "
              "Values size:",
              size,
              " but output shape: ", config.dense[d].shape.DebugString()));
        };

        switch (config.dense[d].dtype) {
          case tf::DT_INT64: {
            auto out_p = reinterpret_cast<int64_t*>(out->data.raw) + offset;
            LimitedArraySlice<int64_t> slice(out_p, num_elements);
            if (!feature.ParseInt64List(&slice)) return parse_error();
            if (slice.EndDistance() != 0) {
              return shape_error(num_elements - slice.EndDistance(), "int64");
            }
            break;
          }
          case tf::DT_FLOAT: {
            auto out_p = reinterpret_cast<float*>(out->data.raw) + offset;
            LimitedArraySlice<float> slice(out_p, num_elements);
            if (!feature.ParseFloatList(&slice)) return parse_error();
            if (slice.EndDistance() != 0) {
              return shape_error(num_elements - slice.EndDistance(), "float");
            }
            break;
          }
          case tf::DT_STRING: {
            auto& out_tensor = result->dense_tensors[d];
            auto out_p = out_tensor.flat<tstring>().data() + offset;
            LimitedArraySlice<tstring> slice(out_p, num_elements);
            if (!feature.ParseBytesList(&slice)) return parse_error();
            if (slice.EndDistance() != 0) {
              return shape_error(num_elements - slice.EndDistance(), "bytes");
            }
            break;
          }
          default:
            return tf::errors::Internal("Unrecognized dense type: ",
                                        config.dense[d].dtype);
        }
      } else {  // if dense variable length
        SparseBuffer& out = (*output_varlen_dense)[d];

        const std::size_t num_elements = config.dense[d].elements_per_stride;

        if (example_dtype != tf::DT_INVALID &&
            example_dtype != config.dense[d].dtype) {
          return example_error(absl::StrCat(
              "Data types don't match. ",
              "Expected type: ", DataTypeString(config.dense[d].dtype)));
        }

        auto shape_error = [&](size_t size, StringPiece type_str) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSparse_examplePSparse_exampleDTcc mht_5(mht_5_v, 424, "", "./tensorflow/lite/kernels/parse_example/parse_example.cc", "lambda");

          return example_error(
              absl::StrCat("Number of ", type_str,
                           " values is not a multiple of stride length. Saw ",
                           size, " values but output shape is: ",
                           config.dense[d].shape.DebugString()));
        };

        switch (config.dense[d].dtype) {
          case tf::DT_INT64: {
            if (example_dtype != tf::DT_INVALID) {
              if (!feature.ParseInt64List(&out.int64_list)) {
                return parse_error();
              }
              if (out.int64_list.size() % num_elements != 0) {
                return shape_error(out.int64_list.size(), "int64");
              }
            }
            out.example_end_indices.push_back(out.int64_list.size());
            break;
          }
          case tf::DT_FLOAT: {
            if (example_dtype != tf::DT_INVALID) {
              if (!feature.ParseFloatList(&out.float_list)) {
                return parse_error();
              }
              if (out.float_list.size() % num_elements != 0) {
                return shape_error(out.float_list.size(), "float");
              }
            }
            out.example_end_indices.push_back(out.float_list.size());
            break;
          }
          case tf::DT_STRING: {
            if (example_dtype != tf::DT_INVALID) {
              if (!feature.ParseBytesList(&out.bytes_list)) {
                return parse_error();
              }
              if (out.bytes_list.size() % num_elements != 0) {
                return shape_error(out.bytes_list.size(), "byte");
              }
            }
            out.example_end_indices.push_back(out.bytes_list.size());
            break;
          }
          default:
            return tf::errors::Internal("Should not happen: ",
                                        config.dense[d].dtype);
        }
      }
    } else {
      // is sparse or ragged
      auto& last_example = sparse_feature_last_example;
      if (last_example[d] == example_index) {
        continue;
      }
      last_example[d] = example_index;
      SparseBuffer& out = (*output_sparse)[d];
      tf::DataType feature_dtype = config.sparse[d].dtype;
      if (example_dtype != tf::DT_INVALID && example_dtype != feature_dtype) {
        return tf::errors::Internal("Data types don't match:", example_dtype,
                                    " != ", feature_dtype);
      }
      switch (feature_dtype) {
        case tf::DT_INT64: {
          if (example_dtype != tf::DT_INVALID) {
            if (!feature.ParseInt64List(&out.int64_list)) {
              return parse_error();
            }
          }
          out.example_end_indices.push_back(out.int64_list.size());
          break;
        }
        case tf::DT_FLOAT: {
          if (example_dtype != tf::DT_INVALID) {
            if (!feature.ParseFloatList(&out.float_list)) {
              return parse_error();
            }
          }
          out.example_end_indices.push_back(out.float_list.size());
          break;
        }
        case tf::DT_STRING: {
          if (example_dtype != tf::DT_INVALID) {
            if (!feature.ParseBytesList(&out.bytes_list)) {
              return parse_error();
            }
          }
          out.example_end_indices.push_back(out.bytes_list.size());
          break;
        }
        default:
          return tf::errors::Internal("Should not happen: ", feature_dtype);
      }
    }
  }
  // Handle missing dense features for fixed strides.
  for (size_t d = 0; d < config.dense.size(); ++d) {
    if (config.dense[d].variable_length) continue;
    if (dense_feature_last_example[d] == example_index) continue;
    if (config.dense[d].default_value.NumElements() == 0) {
      return tf::errors::Internal(
          "Name: ", example_name, ", Feature: ", config.dense[d].feature_name,
          " (data type: ", DataTypeString(config.dense[d].dtype), ")",
          " is required but could not be found.");
    }
    const tf::Tensor& in = config.dense[d].default_value;
    TfLiteTensor* out = result->dense_values[d];
    const std::size_t num_elements = in.shape().num_elements();
    const std::size_t offset = example_index * num_elements;
    switch (config.dense[d].dtype) {
      case tf::DT_INT64: {
        std::copy_n(in.flat<int64_t>().data(), num_elements,
                    out->data.i64 + offset);
        break;
      }
      case tf::DT_FLOAT: {
        std::copy_n(in.flat<float>().data(), num_elements,
                    out->data.f + offset);
        break;
      }
      case tf::DT_STRING: {
        auto& out_tensor = result->dense_tensors[d];
        std::copy_n(in.flat<tstring>().data(), num_elements,
                    out_tensor.flat<tstring>().data() + offset);
        break;
      }
      default:
        return tf::errors::Internal("Should not happen: ",
                                    config.dense[d].dtype);
    }
  }
  for (size_t d = 0; d < config.dense.size(); ++d) {
    if (!config.dense[d].variable_length) continue;
    if (dense_feature_last_example[d] == example_index) continue;
    SparseBuffer& out = (*output_varlen_dense)[d];
    size_t prev_example_end_index =
        out.example_end_indices.empty() ? 0 : out.example_end_indices.back();
    out.example_end_indices.push_back(prev_example_end_index);
  }

  for (size_t d = 0; d < config.sparse.size(); ++d) {
    if (sparse_feature_last_example[d] == example_index) continue;
    SparseBuffer& out = (*output_sparse)[d];
    size_t prev_example_end_index =
        out.example_end_indices.empty() ? 0 : out.example_end_indices.back();
    out.example_end_indices.push_back(prev_example_end_index);
  }

  return Status::OK();
}

void CountSparseFeatures(const SparseBuffer& sparse_buffer,
                         size_t* total_num_features, size_t* max_num_features) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSparse_examplePSparse_exampleDTcc mht_6(mht_6_v, 580, "", "./tensorflow/lite/kernels/parse_example/parse_example.cc", "CountSparseFeatures");

  const std::vector<size_t>& end_indices = sparse_buffer.example_end_indices;
  *total_num_features += end_indices.back();
  *max_num_features = std::max(*max_num_features, end_indices[0]);
  for (size_t i = 1; i < end_indices.size(); ++i) {
    size_t example_size = end_indices[i] - end_indices[i - 1];
    *max_num_features = std::max(*max_num_features, example_size);
  }
}

void CopySparseBufferToTensor(tf::DataType dtype, size_t offset,
                              SparseBuffer* src, TfLiteTensor* dst) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSparse_examplePSparse_exampleDTcc mht_7(mht_7_v, 594, "", "./tensorflow/lite/kernels/parse_example/parse_example.cc", "CopySparseBufferToTensor");

  switch (dtype) {
    case tf::DT_INT64: {
      std::copy(src->int64_list.begin(), src->int64_list.end(),
                reinterpret_cast<int64_t*>(dst->data.raw) + offset);
      break;
    }
    case tf::DT_FLOAT: {
      std::copy(src->float_list.begin(), src->float_list.end(),
                reinterpret_cast<float*>(dst->data.raw) + offset);
      break;
    }
    case tf::DT_STRING: {
      DynamicBuffer buffer;
      for (auto* begin = src->bytes_list.begin();
           begin != src->bytes_list.end(); begin++) {
        buffer.AddString(begin->c_str(), begin->size());
      }
      buffer.WriteToTensor(dst, nullptr);
      break;
    }
    default:
      DCHECK(false) << "Encountered unexpected DataType "
                    << DataTypeString(dtype)
                    << "in variable that should have been checked.";
  }
}

inline void CopyToBuffer(tf::gtl::ArraySlice<tstring> vec, char* tensor_buffer,
                         int num_examples, int batch_size,
                         int elements_per_stride) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("tensor_buffer: \"" + (tensor_buffer == nullptr ? std::string("nullptr") : std::string((char*)tensor_buffer)) + "\"");
   MHTracer_DTPStensorflowPSlitePSkernelsPSparse_examplePSparse_exampleDTcc mht_8(mht_8_v, 628, "", "./tensorflow/lite/kernels/parse_example/parse_example.cc", "CopyToBuffer");

  int i = 0, k = 0;
  int start = 0;
  for (; i < num_examples; ++i) {
    for (int j = 0; j < elements_per_stride; ++j) {
      memcpy(tensor_buffer + start, vec[k].c_str(), vec[k].size());
      start += vec[k].size();
      k++;
    }
  }
  // Will happen if the number of examples is less than the desired batch size.
  for (; i < batch_size; ++i) {
    for (int j = 0; j < elements_per_stride; ++j) {
      memcpy(tensor_buffer + start, vec[k].c_str(), vec[k].size());
      start += vec[k].size();
      k++;
    }
  }
}

Status FastParseExampleLite(
    const FastParseExampleConfig& config, const TfLiteTensor* serialized,
    tf::gtl::ArraySlice<tstring> example_names, bool* quick_filter,
    int quick_filter_size, const std::unique_ptr<ConfigIndex>& config_index,
    int config_index_size, SeededHasher* hasher, TfLiteResult* result,
    std::map<absl::string_view, int>& stats, TfLiteContext* context) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSparse_examplePSparse_exampleDTcc mht_9(mht_9_v, 656, "", "./tensorflow/lite/kernels/parse_example/parse_example.cc", "FastParseExampleLite");

  if (result == nullptr) {
    return tf::errors::Internal("Result is null");
  }
  const int count = GetStringCount(serialized);
  std::vector<tf::Tensor> fixed_dense_values(config.dense.size());
  std::vector<SparseBuffer> sparse_buffers(config.sparse.size());
  std::vector<SparseBuffer> varlen_dense_buffers(config.dense.size());
  Status status_of_minibatch;
  for (size_t e = 0; e < count; ++e) {
    Status status_of_minibatch = FastParseSerializedExample(
        GetString(serialized, e),
        (!example_names.empty() ? example_names[e] : "<unknown>"), e, config,
        quick_filter, quick_filter_size, config_index, config_index_size,
        hasher, &result->dense_values, &varlen_dense_buffers, &sparse_buffers,
        /*arena,*/ stats, result);
    if (!status_of_minibatch.ok()) break;
  }
  if (!status_of_minibatch.ok()) {
    return status_of_minibatch;
  }
  // Merge SparseBuffers from all minibatches for every config.sparse.
  // auto MergeSparseMinibatches = [&](size_t d) {
  // Loop over minibatches
  for (size_t d = 0; d < config.sparse.size(); ++d) {
    size_t total_num_features = 0;
    size_t max_num_features = 0;
    CountSparseFeatures(sparse_buffers[d], &total_num_features,
                        &max_num_features);
    tf::TensorShape indices_shape;
    TfLiteTensor* indices = result->sparse_indices[d];
    TfLiteTensor* values = result->sparse_values[d];

    TfLiteTensor* dense_shape = result->sparse_shapes[d];
    auto* dense_shape_ptr = reinterpret_cast<int64_t*>(dense_shape->data.raw);
    dense_shape_ptr[1] = max_num_features;

    TfLiteIntArray* index_shape = TfLiteIntArrayCreate(2);
    index_shape->data[0] = total_num_features;
    index_shape->data[1] = 2;
    context->ResizeTensor(context, indices, index_shape);

    TfLiteIntArray* output_shape = TfLiteIntArrayCreate(1);
    output_shape->data[0] = total_num_features;
    context->ResizeTensor(context, values, output_shape);

    SparseBuffer& buffer = sparse_buffers[d];

    // Update indices.
    auto* indices_p = reinterpret_cast<int64_t*>(indices->data.raw);
    if (!indices_p) {
      return tf::errors::Internal("Indices tensor not allocated!");
    }

    if (total_num_features > 0) {
      int64_t* ix_p = indices_p;
      size_t example_index = 0;
      int idx0 = 0;
      size_t delta = 0;
      for (size_t example_end_index : buffer.example_end_indices) {
        size_t feature_index = 0;
        for (; delta < example_end_index; ++delta) {
          // Column 0: example index
          if (idx0 < total_num_features) {
            *ix_p = example_index;
            // Column 1: the feature index buffer example
            *(ix_p + 1) = feature_index;
            ix_p += 2;
          }
          ++feature_index;
          ++idx0;
        }
        ++example_index;
      }
      CopySparseBufferToTensor(config.sparse[d].dtype, 0, &buffer, values);
    }
  }

  // Merge SparseBuffers from all minibatches for every config.dense having
  // variable_length.
  for (size_t d = 0; d < config.dense.size(); ++d) {
    if (!config.dense[d].variable_length) {
      continue;
    }
    size_t max_num_features = 0;
    std::vector<size_t>& end_indices =
        varlen_dense_buffers[d].example_end_indices;
    max_num_features = std::max(max_num_features, end_indices[0]);
    for (size_t i = 1; i < end_indices.size(); ++i) {
      size_t example_size = end_indices[i] - end_indices[i - 1];
      max_num_features = std::max(max_num_features, example_size);
    }

    const size_t stride_size = config.dense[d].elements_per_stride;
    const size_t max_num_elements = max_num_features / stride_size;
    tf::TensorShape values_shape;
    DCHECK_EQ(max_num_features % config.dense[d].elements_per_stride, 0);
    const size_t batch_size = GetStringCount(serialized);
    values_shape.AddDim(batch_size);
    values_shape.AddDim(max_num_elements);
    for (int i = 1; i < config.dense[d].shape.dims(); ++i) {
      values_shape.AddDim(config.dense[d].shape.dim_size(i));
    }
    TfLiteTensor* values = result->dense_values[d];
    const size_t num_elements = GetTensorShape(values).FlatSize();

    // Nothing to write, exit early.
    if (num_elements == 0) {
      continue;
    }

    const size_t num_elements_per_minibatch = num_elements / batch_size;
    switch (config.dense[d].dtype) {
      case tf::DT_INT64: {
        FillAndCopyVarLen<int64_t>(d, num_elements, num_elements_per_minibatch,
                                   config, varlen_dense_buffers, values);
        break;
      }
      case tf::DT_FLOAT: {
        FillAndCopyVarLen<float>(d, num_elements, num_elements_per_minibatch,
                                 config, varlen_dense_buffers, values);
        break;
      }
      default:
        DCHECK(false) << "Encountered unexpected DataType "
                      << config.dense[d].dtype
                      << "in variable that should have been checked";
    }
  }

  // Merge tflite string buffers if necessary.
  for (size_t d = 0; d < config.dense.size(); ++d) {
    if (config.dense[d].variable_length) {
      continue;
    }
    if (result->dense_values[d]->type == kTfLiteString) {
      auto& in = result->dense_tensors[d];
      auto vec = in.vec<tstring>();
      const int batch_size = result->dense_values[d]->dims->data[0];
      const int elements_per_stride = config.dense[d].elements_per_stride;
      int total_size = 0;
      std::vector<int32_t> offsets;
      offsets.reserve(vec.size() + 1);
      offsets.push_back(0);
      int k = 0;
      for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < elements_per_stride; ++j) {
          if (i < count) {
            total_size += vec(k++).size();
            offsets.push_back(total_size);
          } else {
            offsets.push_back(total_size);
          }
        }
      }
      const int32_t num_strings = offsets.size() - 1;
      const size_t required_bytes = sizeof(int32_t) * (num_strings + 2) +
          total_size;
      char* tensor_buffer =
          reinterpret_cast<char*>(result->dense_values[d]->data.raw);
      if (result->dense_values[d]->bytes < required_bytes) {
        if (result->dense_values[d]->data.raw) {
          free(result->dense_values[d]->data.raw);
        }
        tensor_buffer = reinterpret_cast<char*>(malloc(required_bytes));
        result->dense_values[d]->data.raw = tensor_buffer;
        result->dense_values[d]->bytes = required_bytes;
      }
      const int32_t start = sizeof(int32_t) * (num_strings + 2);
      memcpy(tensor_buffer, &num_strings, sizeof(int32_t));
      for (size_t i = 0; i < offsets.size(); i++) {
        int32_t offset_i = start + offsets[i];
        memcpy(tensor_buffer + sizeof(int32_t) * (i + 1), &offset_i,
               sizeof(int32_t));
      }
      tf::gtl::ArraySlice<tstring> slice(vec.data(), vec.size());
      CopyToBuffer(slice, tensor_buffer + start, count, batch_size,
                   elements_per_stride);
    }
  }
  return Status::OK();
}

}  // namespace

enum InputTensor {
  kExampleTensor = 0,
  kNamesTensor = 1,
  kSparseKeysTensor = 2,
  kDenseKeysTensor = 3,
  kRaggedKeysTensor = 4,
};

struct OpData {
  FastParseExampleConfig config;
  std::vector<tf::TensorShape> dense_shapes;
  int dense_size = 0;
  int sparse_size = 0;
  std::unique_ptr<ConfigIndex> config_index;
  int config_index_size;
  SeededHasher hasher;
  TfLiteResult got;
  bool* quick_filter = nullptr;
  int quick_filter_size;
  bool created = false;
  ~OpData() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSparse_examplePSparse_exampleDTcc mht_10(mht_10_v, 864, "", "./tensorflow/lite/kernels/parse_example/parse_example.cc", "~OpData");

    if (quick_filter) {
      free(quick_filter);
    }
  }
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
   std::vector<std::string> mht_11_v;
   mht_11_v.push_back("buffer: \"" + (buffer == nullptr ? std::string("nullptr") : std::string((char*)buffer)) + "\"");
   MHTracer_DTPStensorflowPSlitePSkernelsPSparse_examplePSparse_exampleDTcc mht_11(mht_11_v, 875, "", "./tensorflow/lite/kernels/parse_example/parse_example.cc", "Init");

  return new OpData;
}

template <typename T>
tf::Tensor AsTensor(const std::vector<T>& val) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSparse_examplePSparse_exampleDTcc mht_12(mht_12_v, 883, "", "./tensorflow/lite/kernels/parse_example/parse_example.cc", "AsTensor");

  tf::Tensor ret(tf::DataTypeToEnum<T>::value,
                 {static_cast<int64_t>(val.size())});
  std::copy_n(val.begin(), val.size(), ret.flat<T>().data());
  return ret;
}

enum Version {
  V1,
  V2,
};

tf::TensorShape TfLiteToTfShape(TfLiteIntArray* array) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSparse_examplePSparse_exampleDTcc mht_13(mht_13_v, 898, "", "./tensorflow/lite/kernels/parse_example/parse_example.cc", "TfLiteToTfShape");

  tf::TensorShape shape;
  for (int i = 0; i < array->size; i++) {
    shape.AddDim(array->data[i]);
  }
  return shape;
}

template <Version version>
TfLiteStatus PrepareParseExample(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSparse_examplePSparse_exampleDTcc mht_14(mht_14_v, 910, "", "./tensorflow/lite/kernels/parse_example/parse_example.cc", "PrepareParseExample");

  OpData* data = reinterpret_cast<OpData*>(node->user_data);
  TF_LITE_ENSURE(context, node->custom_initial_data);
  data->config.dense.clear();
  data->config.sparse.clear();
  data->got.dense_values.clear();
  const flexbuffers::Vector& v =
      flexbuffers::GetRoot(
          reinterpret_cast<const uint8_t*>(node->custom_initial_data),
          node->custom_initial_data_size)
          .AsVector();
  if (v.size() == 2) {
    tf::NodeDef nodedef;
    TF_LITE_ENSURE_EQ(context, nodedef.ParseFromString(v[1].AsString().str()),
                      true);
    if (version == V1) {
      data->dense_size = nodedef.attr().at("Ndense").i();
      data->sparse_size = nodedef.attr().at("Nsparse").i();
    } else if (version == V2) {
      data->dense_size = nodedef.attr().at("Tdense").list().type_size();
      data->sparse_size = nodedef.attr().at("num_sparse").i();
    }
    auto dense_shapes = nodedef.attr().at("dense_shapes").list();
    if (data->dense_shapes.empty()) {
      for (int i = 0; i < dense_shapes.shape_size(); ++i) {
        data->dense_shapes.push_back(dense_shapes.shape(i));
      }
    }
  } else {
    const flexbuffers::Map& m =
        flexbuffers::GetRoot(
            reinterpret_cast<const uint8_t*>(node->custom_initial_data),
            node->custom_initial_data_size)
            .AsMap();
    const flexbuffers::TypedVector keys = m.Keys();
    int num_sparse = 0;
    int num_dense = 0;
    for (int k = 0; k < keys.size(); ++k) {
      const std::string key = keys[k].ToString();
      const auto value = m[key];
      if (key == "Nsparse" || key == "num_sparse") {
        num_sparse = value.AsInt32();
      }
      if (key == "Ndense") {
        num_dense = value.AsInt32();
      }
    }
    data->sparse_size = num_sparse;
    data->dense_size = num_dense;
    if (version == V2) {
      const TfLiteTensor* dense_key_tensor =
          GetInput(context, node, kDenseKeysTensor);
      data->dense_size = GetTensorShape(dense_key_tensor).FlatSize();
    }
  }

  data->config.dense.reserve(data->dense_size);
  data->config.sparse.reserve(data->sparse_size);
  data->dense_shapes.reserve(data->dense_size);
  const auto* serialized = GetInput(context, node, 0);
  const int batch_size =
      serialized->dims->size > 0 ? serialized->dims->data[0] : 1;
  const bool missing_shape_info = data->dense_shapes.empty();
  for (int i = 0; i < data->dense_size; i++) {
    TfLiteTensor* dense_key_tensor =
        GetOutput(context, node, data->sparse_size * 3 + i);
    TfLiteIntArray* output_size = TfLiteIntArrayCopy(dense_key_tensor->dims);
    if (missing_shape_info) {
      data->dense_shapes.push_back(TfLiteToTfShape(output_size));
    }
    // use original tflite tensor size if inputs are resized.
    const int original_size = data->dense_shapes[i].dims() > 0
                                  ? data->dense_shapes[i].dim_size(0)
                                  : 1;
    output_size->data[0] = batch_size * original_size;
    context->ResizeTensor(context, dense_key_tensor, output_size);
  }

  size_t offset = 0;
  for (int i = 0; i < data->sparse_size; i++) {
    auto* parse_output = GetOutput(context, node, i + offset);
    SetTensorToDynamic(parse_output);
    TfLiteIntArray* sparse_size = TfLiteIntArrayCreate(2);
    sparse_size->data[0] = batch_size;
    sparse_size->data[1] = 2;
    context->ResizeTensor(context, parse_output, sparse_size);
    data->got.sparse_indices.push_back(parse_output);
  }
  offset += data->sparse_size;
  for (int i = 0; i < data->sparse_size; i++) {
    auto* parse_output = GetOutput(context, node, i + offset);
    SetTensorToDynamic(parse_output);
    TfLiteIntArray* sparse_size = TfLiteIntArrayCreate(1);
    sparse_size->data[0] = 0;
    context->ResizeTensor(context, parse_output, sparse_size);
    data->got.sparse_values.push_back(parse_output);
  }
  offset += data->sparse_size;
  for (int i = 0; i < data->sparse_size; i++) {
    TfLiteTensor* parse_output = GetOutput(context, node, i + offset);
    SetTensorToDynamic(parse_output);
    TfLiteIntArray* sparse_size = TfLiteIntArrayCreate(1);
    sparse_size->data[0] = 2;
    context->ResizeTensor(context, parse_output, sparse_size);
    auto* shapes_shape_t = reinterpret_cast<int64_t*>(parse_output->data.i64);
    shapes_shape_t[0] = batch_size;
    shapes_shape_t[1] = 1;
    data->got.sparse_shapes.push_back(parse_output);
  }
  data->created = false;
  return kTfLiteOk;
}

template <Version version>
TfLiteStatus EvalParseExample(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSparse_examplePSparse_exampleDTcc mht_15(mht_15_v, 1027, "", "./tensorflow/lite/kernels/parse_example/parse_example.cc", "EvalParseExample");

  OpData* data = reinterpret_cast<OpData*>(node->user_data);
  if (!data->created) {
    for (int i = 0; i < data->sparse_size; i++) {
      int input_index =
          version == V1 ? kSparseKeysTensor + i : kSparseKeysTensor;
      int string_index = version == V1 ? 0 : i;
      const TfLiteTensor* sparse_key_tensor =
          GetInput(context, node, input_index);
      const auto key = GetString(sparse_key_tensor, string_index);
      const auto* sparse_output =
          GetOutput(context, node, i + data->sparse_size);
      std::string k(key.str, key.len);
      switch (sparse_output->type) {
        case kTfLiteInt64:
          data->config.sparse.emplace_back(k,
                                           tf::DataTypeToEnum<int64_t>::value);
          break;
        case kTfLiteFloat32:
          data->config.sparse.emplace_back(k, tf::DataTypeToEnum<float>::value);
          break;
        case kTfLiteString:
          data->config.sparse.emplace_back(k,
                                           tf::DataTypeToEnum<tstring>::value);
          break;
        default:
          return kTfLiteError;
      }
    }

    const auto& dense_shapes = data->dense_shapes;
    for (int i = 0; i < data->dense_size; i++) {
      const int input_index = version == V1
                                  ? kSparseKeysTensor + data->sparse_size + i
                                  : kDenseKeysTensor;
      const int dense_defaults_index =
          version == V1
              ? kSparseKeysTensor + data->sparse_size + data->dense_size + i
              : kRaggedKeysTensor + i + 1;
      int string_index = version == V1 ? 0 : i;
      const TfLiteTensor* dense_key_tensor =
          GetInput(context, node, input_index);
      const auto* dense_output =
          GetOutput(context, node, i + data->sparse_size * 3);
      const auto* dense_defaults =
          GetInput(context, node, dense_defaults_index);
      const auto key = GetString(dense_key_tensor, string_index);
      std::string k(key.str, key.len);
      const int elements_per_stride =
          dense_shapes[i].dims() ? dense_shapes[i].num_elements() : 1;
      switch (dense_output->type) {
        case kTfLiteInt64:
          data->config.dense.emplace_back(
              k, tf::DataTypeToEnum<int64_t>::value, dense_shapes[i],
              AsTensor<int64_t>(std::vector<int64_t>(
                  dense_defaults->data.i64,
                  dense_defaults->data.i64 + elements_per_stride)),
              false, elements_per_stride);
          break;
        case kTfLiteFloat32:
          data->config.dense.emplace_back(
              k, tf::DataTypeToEnum<float>::value, dense_shapes[i],
              AsTensor<float>(std::vector<float>(
                  dense_defaults->data.f,
                  dense_defaults->data.f + elements_per_stride)),
              false, elements_per_stride);
          break;
        case kTfLiteString: {
          const int num_strings = GetStringCount(dense_defaults);
          std::vector<tstring> values;
          for (int i = 0; i < num_strings; ++i) {
            auto ref = GetString(dense_defaults, i);
            values.emplace_back(ref.str, ref.len);
          }
          data->config.dense.emplace_back(
              k, tf::DataTypeToEnum<tstring>::value, dense_shapes[i],
              AsTensor<tstring>(values), false, elements_per_stride);
          break;
        }
        default:
          return kTfLiteError;
      }
    }

    int offset = 3 * data->sparse_size;
    for (int i = 0; i < data->dense_size; i++) {
      auto* parse_output = GetOutput(context, node, i + offset);
      data->got.dense_values.push_back(parse_output);
      if (parse_output->type == kTfLiteString) {
        tf::TensorShape shape;
        if (parse_output->dims->size == 1) {
          shape.AddDim(parse_output->dims->data[0]);
        } else {
          shape.AddDim(GetTensorShape(parse_output).FlatSize());
        }
        data->got.dense_tensors[i] =
            tf::Tensor(tf::DataTypeToEnum<tstring>::value, shape);
      }
    }

    size_t config_size = data->config.dense.size();
    config_size += data->config.sparse.size();
    data->config_index_size = config_size;
    auto config_index = std::make_unique<ConfigIndex>(config_size);
    bool ok = true;
    int max_length = 0;
    for (size_t d = 0; d < data->config.dense.size(); ++d) {
      auto s = data->config.dense[d].feature_name;
      max_length = s.length() > max_length ? s.length() : max_length;
    }
    for (size_t d = 0; d < data->config.sparse.size(); ++d) {
      auto s = data->config.sparse[d].feature_name;
      max_length = s.length() > max_length ? s.length() : max_length;
    }
    if (data->quick_filter) {
      free(data->quick_filter);
    }
    data->quick_filter =
        static_cast<bool*>(malloc(++max_length * sizeof(bool)));
    memset(data->quick_filter, 0, max_length * sizeof(bool));
    data->quick_filter_size = max_length;
    for (size_t d = 0; d < data->config.dense.size(); ++d) {
      const auto& s = data->config.dense[d].feature_name;
      data->quick_filter[s.length()] = true;
    }
    for (size_t d = 0; d < data->config.sparse.size(); ++d) {
      const auto& s = data->config.sparse[d].feature_name;
      data->quick_filter[s.length()] = true;
    }

    for (int i = 0; i < 1000; ++i) {
      for (size_t d = 0; d < data->config.dense.size(); ++d) {
        ok &= config_index->InsertUnique(
            data->hasher(data->config.dense[d].feature_name), {d, Type::Dense});
      }
      for (size_t d = 0; d < data->config.sparse.size(); ++d) {
        ok &= config_index->InsertUnique(
            data->hasher(data->config.sparse[d].feature_name),
            {d, Type::Sparse});
      }
      if (ok) {
        break;
      }
      data->hasher.seed++;
      config_index->Clear(config_size);
      ok = true;
    }
    if (!ok) {
      return kTfLiteError;
    }
    data->config_index = std::move(config_index);
    data->created = true;
  }

  const TfLiteTensor* serialized = GetInput(context, node, kExampleTensor);

  std::map<absl::string_view, int> stats;
  const auto status = FastParseExampleLite(
      data->config, serialized, {}, data->quick_filter, data->quick_filter_size,
      data->config_index, data->config_index_size, &data->hasher, &data->got,
      stats, context);
  if (status != tf::Status::OK()) {
    TF_LITE_KERNEL_LOG(context, status.ToString().c_str());
    return kTfLiteError;
  }
  return kTfLiteOk;
}

void Free(TfLiteContext* context, void* buffer) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSparse_examplePSparse_exampleDTcc mht_16(mht_16_v, 1198, "", "./tensorflow/lite/kernels/parse_example/parse_example.cc", "Free");

  auto* obj = reinterpret_cast<OpData*>(buffer);
  delete obj;
}

}  // namespace parse_example

TfLiteRegistration* Register_PARSE_EXAMPLE() {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSparse_examplePSparse_exampleDTcc mht_17(mht_17_v, 1208, "", "./tensorflow/lite/kernels/parse_example/parse_example.cc", "Register_PARSE_EXAMPLE");

  static TfLiteRegistration r = {
      parse_example::Init, parse_example::Free,
      parse_example::PrepareParseExample<parse_example::V1>,
      parse_example::EvalParseExample<parse_example::V1>};
  return &r;
}

TfLiteRegistration* Register_PARSE_EXAMPLE_V2() {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSparse_examplePSparse_exampleDTcc mht_18(mht_18_v, 1219, "", "./tensorflow/lite/kernels/parse_example/parse_example.cc", "Register_PARSE_EXAMPLE_V2");

  static TfLiteRegistration r = {
      parse_example::Init, parse_example::Free,
      parse_example::PrepareParseExample<parse_example::V2>,
      parse_example::EvalParseExample<parse_example::V2>};
  return &r;
}

extern "C" void AddParseExampleOp(::tflite::MutableOpResolver* resolver) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSparse_examplePSparse_exampleDTcc mht_19(mht_19_v, 1230, "", "./tensorflow/lite/kernels/parse_example/parse_example.cc", "AddParseExampleOp");

  resolver->AddCustom("ParseExample", Register_PARSE_EXAMPLE());
  resolver->AddCustom("ParseExampleV2", Register_PARSE_EXAMPLE_V2());
}

}  // namespace custom
}  // namespace ops
}  // namespace tflite
