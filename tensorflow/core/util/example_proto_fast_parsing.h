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

#ifndef TENSORFLOW_CORE_UTIL_EXAMPLE_PROTO_FAST_PARSING_H_
#define TENSORFLOW_CORE_UTIL_EXAMPLE_PROTO_FAST_PARSING_H_
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
class MHTracer_DTPStensorflowPScorePSutilPSexample_proto_fast_parsingDTh {
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
   MHTracer_DTPStensorflowPScorePSutilPSexample_proto_fast_parsingDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSutilPSexample_proto_fast_parsingDTh() {
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
#include <unordered_map>
#include <vector>

#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"

namespace tensorflow {
namespace example {

// FastParseExampleConfig defines how to parse features in Example.
// Each sub-config is responsible for one feature identified with feature_name.
// FastParseExampleConfig can't have two sub-configs with the same feature_name.
// dtype identifies the type of output vector and the kind of Feature expected
// in Example.
struct FastParseExampleConfig {
  struct Dense {
    Dense(StringPiece feature_name, DataType dtype, PartialTensorShape shape,
          Tensor default_value, bool variable_length,
          std::size_t elements_per_stride)
        : feature_name(feature_name),  // TODO(mrry): Switch to preallocated
                                       // tstring when this is available.
          dtype(dtype),
          shape(std::move(shape)),
          default_value(std::move(default_value)),
          variable_length(variable_length),
          elements_per_stride(elements_per_stride) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSutilPSexample_proto_fast_parsingDTh mht_0(mht_0_v, 222, "", "./tensorflow/core/util/example_proto_fast_parsing.h", "Dense");
}
    Dense() = default;

    tstring feature_name;
    DataType dtype;
    // These 2 fields correspond exactly to dense_shapes and dense_defaults in
    // ParseExample op.
    // Documentation is available in: tensorflow/core/ops/parsing_ops.cc
    PartialTensorShape shape;
    Tensor default_value;
    bool variable_length;
    std::size_t elements_per_stride;
  };

  struct Sparse {
    Sparse(StringPiece feature_name, DataType dtype)
        : feature_name(feature_name),  // TODO(mrry): Switch to preallocated
                                       // tstring when this is available.
          dtype(dtype) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSutilPSexample_proto_fast_parsingDTh mht_1(mht_1_v, 243, "", "./tensorflow/core/util/example_proto_fast_parsing.h", "Sparse");
}
    Sparse() = default;

    tstring feature_name;
    DataType dtype;
  };

  struct Ragged {
    Ragged(StringPiece feature_name, DataType dtype, DataType splits_dtype)
        : feature_name(feature_name),  // TODO(mrry): Switch to preallocated
                                       // tstring when this is available.
          dtype(dtype),
          splits_dtype(splits_dtype) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSutilPSexample_proto_fast_parsingDTh mht_2(mht_2_v, 258, "", "./tensorflow/core/util/example_proto_fast_parsing.h", "Ragged");
}
    Ragged() = default;

    tstring feature_name;
    DataType dtype;
    DataType splits_dtype;
  };

  std::vector<Dense> dense;
  std::vector<Sparse> sparse;
  std::vector<Ragged> ragged;

  // If `true`, `Result::feature_stats` will contain one
  // `PerExampleFeatureStats` for each serialized example in the input.
  bool collect_feature_stats = false;
};

// Statistics about the features in each example passed to
// `FastParse[Single]Example()`.
//
// TODO(b/111553342): The gathered statistics currently have two limitations:
// * Feature names that appear more than once will be counted multiple times.
// * The feature values count only represents the counts for features that were
//   requested in the `FastParseExampleConfig`.
// These could be addressed with additional work at runtime.
struct PerExampleFeatureStats {
  // The number of feature names in an example.
  size_t features_count = 0;

  // The sum of the number of values in each feature that is parsed.
  size_t feature_values_count = 0;
};

// This is exactly the output of TF's ParseExample Op.
// Documentation is available in: tensorflow/core/ops/parsing_ops.cc
struct Result {
  std::vector<Tensor> sparse_indices;
  std::vector<Tensor> sparse_values;
  std::vector<Tensor> sparse_shapes;
  std::vector<Tensor> dense_values;
  std::vector<Tensor> ragged_values;
  std::vector<Tensor> ragged_splits;
  std::vector<Tensor> ragged_outer_splits;  // For SequenceExamples

  // This vector will be populated with one element per example if
  // `FastParseExampleConfig::collect_feature_stats` is set to `true`.
  std::vector<PerExampleFeatureStats> feature_stats;
};

// Parses a batch of serialized Example protos and converts them into result
// according to given config.
// Given example names have to either be empty or the same size as serialized.
// example_names are used only for error messages.
Status FastParseExample(const FastParseExampleConfig& config,
                        gtl::ArraySlice<tstring> serialized,
                        gtl::ArraySlice<tstring> example_names,
                        thread::ThreadPool* thread_pool, Result* result);

// TODO(mrry): Move the hash table construction into the config object.
typedef FastParseExampleConfig FastParseSingleExampleConfig;

Status FastParseSingleExample(const FastParseSingleExampleConfig& config,
                              StringPiece serialized, Result* result);

// Parses a batch of serialized SequenceExample protos and converts them into
// result according to given config.
// Given example names have to either be empty or the same size as serialized.
// example_names are used only for error messages.
// (If batch=true, then this parses a single SequenceExample.)
Status FastParseSequenceExample(
    const example::FastParseExampleConfig& context_config,
    const example::FastParseExampleConfig& feature_list_config,
    gtl::ArraySlice<tstring> serialized, gtl::ArraySlice<tstring> example_names,
    thread::ThreadPool* thread_pool, example::Result* context_result,
    example::Result* feature_list_result,
    std::vector<Tensor>* dense_feature_lengths, bool is_batch = true);

// This function parses serialized Example and populates given example.
// It uses the same specialized parser as FastParseExample which is efficient.
// But then constructs Example which is relatively slow.
// It is exported here as a convenient API to test parser part separately.
bool TestFastParse(const string& serialized, Example* example);

}  // namespace example
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_EXAMPLE_PROTO_FAST_PARSING_H_
