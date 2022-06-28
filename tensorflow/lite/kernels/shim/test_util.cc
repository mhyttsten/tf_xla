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
class MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStest_utilDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStest_utilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStest_utilDTcc() {
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

/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/kernels/shim/test_util.h"

#include <string>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/string_util.h"

namespace tflite {

using std::size_t;

TfLiteTensor* UniqueTfLiteTensor::get() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStest_utilDTcc mht_0(mht_0_v, 197, "", "./tensorflow/lite/kernels/shim/test_util.cc", "UniqueTfLiteTensor::get");
 return tensor_; }

TfLiteTensor& UniqueTfLiteTensor::operator*() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStest_utilDTcc mht_1(mht_1_v, 202, "", "./tensorflow/lite/kernels/shim/test_util.cc", "*");
 return *tensor_; }

TfLiteTensor* UniqueTfLiteTensor::operator->() { return tensor_; }

const TfLiteTensor* UniqueTfLiteTensor::operator->() const { return tensor_; }

void UniqueTfLiteTensor::reset(TfLiteTensor* tensor) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStest_utilDTcc mht_2(mht_2_v, 211, "", "./tensorflow/lite/kernels/shim/test_util.cc", "UniqueTfLiteTensor::reset");
 tensor_ = tensor; }

UniqueTfLiteTensor::~UniqueTfLiteTensor() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStest_utilDTcc mht_3(mht_3_v, 216, "", "./tensorflow/lite/kernels/shim/test_util.cc", "UniqueTfLiteTensor::~UniqueTfLiteTensor");
 TfLiteTensorFree(tensor_); }

namespace {

template <typename T>
std::string TensorValueToString(const ::TfLiteTensor* tensor,
                                const size_t idx) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStest_utilDTcc mht_4(mht_4_v, 225, "", "./tensorflow/lite/kernels/shim/test_util.cc", "TensorValueToString");

  TFLITE_DCHECK_EQ(tensor->type, ::tflite::typeToTfLiteType<T>());
  const T* val_array = reinterpret_cast<const T*>(tensor->data.raw);
  return std::to_string(val_array[idx]);
}

template <>
std::string TensorValueToString<bool>(const ::TfLiteTensor* tensor,
                                      const size_t idx) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStest_utilDTcc mht_5(mht_5_v, 236, "", "./tensorflow/lite/kernels/shim/test_util.cc", "TensorValueToString<bool>");

  TFLITE_DCHECK_EQ(tensor->type, ::tflite::typeToTfLiteType<bool>());
  const bool* val_array = reinterpret_cast<const bool*>(tensor->data.raw);
  return val_array[idx] ? "1" : "0";
}

template <typename FloatType>
std::string TensorValueToStringFloat(const ::TfLiteTensor* tensor,
                                     const size_t idx) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStest_utilDTcc mht_6(mht_6_v, 247, "", "./tensorflow/lite/kernels/shim/test_util.cc", "TensorValueToStringFloat");

  TFLITE_DCHECK_EQ(tensor->type, ::tflite::typeToTfLiteType<FloatType>());
  const FloatType* val_array =
      reinterpret_cast<const FloatType*>(tensor->data.raw);
  std::stringstream ss;
  ss << val_array[idx];
  return std::string(ss.str().data(), ss.str().length());
}

template <>
std::string TensorValueToString<float>(const ::TfLiteTensor* tensor,
                                       const size_t idx) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStest_utilDTcc mht_7(mht_7_v, 261, "", "./tensorflow/lite/kernels/shim/test_util.cc", "TensorValueToString<float>");

  return TensorValueToStringFloat<float>(tensor, idx);
}

template <>
std::string TensorValueToString<double>(const ::TfLiteTensor* tensor,
                                        const size_t idx) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStest_utilDTcc mht_8(mht_8_v, 270, "", "./tensorflow/lite/kernels/shim/test_util.cc", "TensorValueToString<double>");

  return TensorValueToStringFloat<double>(tensor, idx);
}

template <>
std::string TensorValueToString<StringRef>(const ::TfLiteTensor* tensor,
                                           const size_t idx) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStest_utilDTcc mht_9(mht_9_v, 279, "", "./tensorflow/lite/kernels/shim/test_util.cc", "TensorValueToString<StringRef>");

  TFLITE_DCHECK_EQ(tensor->type, kTfLiteString);
  const auto ref = ::tflite::GetString(tensor, idx);
  return std::string(ref.str, ref.len);
}

std::string TfliteTensorDebugStringImpl(const ::TfLiteTensor* tensor,
                                        const size_t axis,
                                        const size_t max_values,
                                        size_t* start_idx) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStest_utilDTcc mht_10(mht_10_v, 291, "", "./tensorflow/lite/kernels/shim/test_util.cc", "TfliteTensorDebugStringImpl");

  const size_t dim_size = tensor->dims->data[axis];
  if (axis == tensor->dims->size - 1) {
    std::vector<std::string> ret_list;
    ret_list.reserve(dim_size);
    int idx = *start_idx;
    for (int i = 0; i < dim_size && idx < max_values; ++i, ++idx) {
      std::string val_str;
      switch (tensor->type) {
        case kTfLiteBool: {
          val_str = TensorValueToString<bool>(tensor, idx);
          break;
        }
        case kTfLiteUInt8: {
          val_str = TensorValueToString<uint8_t>(tensor, idx);
          break;
        }
        case kTfLiteInt8: {
          val_str = TensorValueToString<int8_t>(tensor, idx);
          break;
        }
        case kTfLiteInt16: {
          val_str = TensorValueToString<int16_t>(tensor, idx);
          break;
        }
        case kTfLiteInt32: {
          val_str = TensorValueToString<int32_t>(tensor, idx);
          break;
        }
        case kTfLiteInt64: {
          val_str = TensorValueToString<int64_t>(tensor, idx);
          break;
        }
        case kTfLiteString: {
          val_str = TensorValueToString<StringRef>(tensor, idx);
          break;
        }
        case kTfLiteFloat32: {
          val_str = TensorValueToString<float>(tensor, idx);
          break;
        }
        case kTfLiteFloat64: {
          val_str = TensorValueToString<double>(tensor, idx);
          break;
        }
        default: {
          val_str = "unsupported_type";
        }
      }
      ret_list.push_back(val_str);
    }
    *start_idx = idx;
    if (idx == max_values && ret_list.size() < dim_size) {
      ret_list.push_back("...");
    }
    return absl::StrCat("[", absl::StrJoin(ret_list, ", "), "]");
  } else {
    std::vector<std::string> ret_list;
    ret_list.reserve(dim_size);
    for (int i = 0; i < dim_size && *start_idx < max_values; ++i) {
      ret_list.push_back(
          TfliteTensorDebugStringImpl(tensor, axis + 1, max_values, start_idx));
    }
    return absl::StrCat("[", absl::StrJoin(ret_list, ", "), "]");
  }
}

}  // namespace

std::string TfliteTensorDebugString(const ::TfLiteTensor* tensor,
                                    const size_t max_values) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStest_utilDTcc mht_11(mht_11_v, 364, "", "./tensorflow/lite/kernels/shim/test_util.cc", "TfliteTensorDebugString");

  if (tensor->dims->size == 0) return "";
  size_t start_idx = 0;
  return TfliteTensorDebugStringImpl(tensor, 0, max_values, &start_idx);
}

size_t NumTotalFromShape(const std::initializer_list<int>& shape) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStest_utilDTcc mht_12(mht_12_v, 373, "", "./tensorflow/lite/kernels/shim/test_util.cc", "NumTotalFromShape");

  size_t num_total;
  if (shape.size() > 0)
    num_total = 1;
  else
    num_total = 0;
  for (const int dim : shape) num_total *= dim;
  return num_total;
}

template <>
void PopulateTfLiteTensorValue<std::string>(
    const std::initializer_list<std::string> values, TfLiteTensor* tensor) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStest_utilDTcc mht_13(mht_13_v, 388, "", "./tensorflow/lite/kernels/shim/test_util.cc", "PopulateTfLiteTensorValue<std::string>");

  tflite::DynamicBuffer buf;
  for (const std::string& s : values) {
    buf.AddString(s.data(), s.length());
  }
  buf.WriteToTensor(tensor, /*new_shape=*/nullptr);
}

}  // namespace tflite
