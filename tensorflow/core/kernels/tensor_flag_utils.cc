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
class MHTracer_DTPStensorflowPScorePSkernelsPStensor_flag_utilsDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_flag_utilsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPStensor_flag_utilsDTcc() {
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

/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/kernels/tensor_flag_utils.h"

#include "absl/strings/str_cat.h"
#include "tensorflow/core/framework/tensor_shape.h"

namespace tensorflow {
namespace tensor_flag_utils {

Status ValidateSparseMatrixShardingConfig(const Tensor& config) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_flag_utilsDTcc mht_0(mht_0_v, 193, "", "./tensorflow/core/kernels/tensor_flag_utils.cc", "ValidateSparseMatrixShardingConfig");

  if (TensorShapeUtils::IsScalar(config.shape())) {
    const float scalar_config = config.template scalar<float>()();
    if (0 < scalar_config && scalar_config <= 1.0) {
      return Status::OK();
    }
    return Status(
        error::INVALID_ARGUMENT,
        absl::StrCat("Expected config to be in range (0, 1] but instead found ",
                     scalar_config));
  }
  if (!TensorShapeUtils::IsMatrix(config.shape())) {
    return Status(error::INVALID_ARGUMENT,
                  absl::StrCat("Expected config to be either scalar or matrix "
                               "but instead found tensor of rank ",
                               config.dims()));
  }
  if (config.dim_size(1) != 3) {
    return Status(
        error::INVALID_ARGUMENT,
        absl::StrCat(
            "Expected config matrix to have dim(1) = 3 but instead found ",
            config.dim_size(1)));
  }

  auto config_matrix = config.matrix<float>();
  for (int i = 0; i < config.dim_size(0); ++i) {
    if (0 > config_matrix(i, 0)) {
      return errors::InvalidArgument(
          "First column of fraction_rows_per_thread_config "
          "should "
          "have non-negative values but found ",
          config_matrix(i, 0), " in row ", i);
    }
    if (0 > config_matrix(i, 1)) {
      return errors::InvalidArgument(
          "Second column of fraction_rows_per_thread_config "
          "should "
          "have non-negative values but found ",
          config_matrix(i, 1), " in row ", i);
    }
    if (!(0 < config_matrix(i, 2) && config_matrix(i, 2) <= 1)) {
      return errors::InvalidArgument(
          "Last column of fraction_rows_per_thread_config should "
          "have values in the range (0, 1] but found ",
          config_matrix(i, 2), " in row ", i);
    }
  }
  return Status::OK();
}

template <typename MatrixType, typename K>
MatrixType FindConfigValueForKey(
    const typename TTypes<MatrixType>::ConstMatrix& config_mat,
    const std::pair<K, K>& key) {
  const int last_row_index = config_mat.dimension(0) - 1;
  for (int i = 0; i < last_row_index; ++i) {
    if (key.first >= config_mat(i, 0) && key.second >= config_mat(i, 1)) {
      return config_mat(i, 2);
    }
  }
  return config_mat(last_row_index, 2);
}

Status ValidateScalarQuantityShardingConfig(const Tensor& config) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_flag_utilsDTcc mht_1(mht_1_v, 260, "", "./tensorflow/core/kernels/tensor_flag_utils.cc", "ValidateScalarQuantityShardingConfig");

  if (TensorShapeUtils::IsScalar(config.shape())) {
    const float scalar_config = config.template scalar<float>()();
    if (0 < scalar_config && scalar_config <= 1.0) {
      return Status::OK();
    }
    return Status(
        error::INVALID_ARGUMENT,
        absl::StrCat("Expected config to be in range (0, 1] but instead found ",
                     scalar_config));
  }
  if (!TensorShapeUtils::IsMatrix(config.shape())) {
    return Status(error::INVALID_ARGUMENT,
                  absl::StrCat("Expected config to be either scalar or matrix "
                               "but instead found tensor of rank ",
                               config.dims()));
  }
  if (config.dim_size(1) != 2) {
    return Status(
        error::INVALID_ARGUMENT,
        absl::StrCat(
            "Expected config matrix to have dim(1) = 2 but instead found ",
            config.dim_size(1)));
  }

  auto config_matrix = config.matrix<float>();
  for (int i = 0; i < config.dim_size(0); ++i) {
    if (0 > config_matrix(i, 0)) {
      return errors::InvalidArgument(
          "First column of fraction_rows_per_thread_config "
          "should "
          "have non-negative values but found ",
          config_matrix(i, 0), " in row ", i);
    }
    if (!(0 < config_matrix(i, 1) && config_matrix(i, 1) <= 1)) {
      return errors::InvalidArgument(
          "Last column of fraction_rows_per_thread_config should "
          "have values in the range (0, 1] but found ",
          config_matrix(i, 1), " in row ", i);
    }
  }
  return Status::OK();
}

template <typename MatrixType, typename K>
MatrixType FindConfigValueForKey(
    const typename TTypes<MatrixType>::ConstMatrix& config_mat, const K key) {
  const int last_row_index = config_mat.dimension(0) - 1;
  for (int i = 0; i < last_row_index; ++i) {
    if (key >= config_mat(i, 0)) {
      return config_mat(i, 1);
    }
  }
  return config_mat(last_row_index, 1);
}

template <typename Tindices>
Tindices GetLinearBucket(const Tindices value, const Tindices bucket_size) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_flag_utilsDTcc mht_2(mht_2_v, 320, "", "./tensorflow/core/kernels/tensor_flag_utils.cc", "GetLinearBucket");

  const Tindices next_multiple_of_bucket_size =
      (value + bucket_size - 1) / bucket_size * bucket_size;
  return next_multiple_of_bucket_size - (bucket_size - 1);
}

template <typename Tindices>
Tindices GetPowerBucket(const Tindices value, const Tindices bucket_size) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_flag_utilsDTcc mht_3(mht_3_v, 330, "", "./tensorflow/core/kernels/tensor_flag_utils.cc", "GetPowerBucket");

  if (bucket_size == 1) {
    return 1;
  }
  return std::pow(bucket_size, std::floor(std::log(bucket_size * (value - 1)) /
                                          std::log(bucket_size)) -
                                   1) +
         1;
}

#define REGISTER_SPARSE_UTIL_FUNCTIONS(TypeIndex)                         \
  template float FindConfigValueForKey<float, TypeIndex>(                 \
      const TTypes<float>::ConstMatrix& config_mat,                       \
      const std::pair<TypeIndex, TypeIndex>& key);                        \
  template float FindConfigValueForKey<float, TypeIndex>(                 \
      const TTypes<float>::ConstMatrix& config_mat, const TypeIndex key); \
  template int64 FindConfigValueForKey<int64, TypeIndex>(                 \
      const TTypes<int64_t>::ConstMatrix& config_mat, const TypeIndex key);

REGISTER_SPARSE_UTIL_FUNCTIONS(int32);
REGISTER_SPARSE_UTIL_FUNCTIONS(int64);
REGISTER_SPARSE_UTIL_FUNCTIONS(uint8);
REGISTER_SPARSE_UTIL_FUNCTIONS(uint16);
REGISTER_SPARSE_UTIL_FUNCTIONS(uint32);
REGISTER_SPARSE_UTIL_FUNCTIONS(uint64);

template int32 GetLinearBucket(const int32 value, const int32 bucket_size);

template int64 GetLinearBucket(const int64 value, const int64 bucket_size);

template int32 GetPowerBucket(const int32 value, const int32 bucket_size);

template int64 GetPowerBucket(const int64 value, const int64 bucket_size);

}  // namespace tensor_flag_utils
}  // namespace tensorflow
