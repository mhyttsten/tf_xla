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
class MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStf_tensor_viewDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStf_tensor_viewDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStf_tensor_viewDTcc() {
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
#include "tensorflow/lite/kernels/shim/tf_tensor_view.h"

#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "tensorflow/core/framework/types.pb.h"

// Creates a case statement for the switch() clause given the dtype
#define CASE_FOR_DTYPE_GIVEN_CPP_DTYPE(TF_DTYPE, CPP_DTYPE) \
  case TF_DTYPE: {                                          \
    using DType = typename CPP_DTYPE;                       \
    return TfTensorView(wrapped_tensor, DType());           \
  }

#define CASE_FOR_DTYPE(TF_DTYPE)           \
  CASE_FOR_DTYPE_GIVEN_CPP_DTYPE(TF_DTYPE, \
                                 ::tensorflow::EnumToDataType<TF_DTYPE>::Type)

namespace tflite {
namespace shim {

// ctors

TfTensorView::TfTensorView(TfTensorView &&o) noexcept
    : TensorView(std::move(o)), shape_data_(std::move(o.shape_data_)) {
  shape_ = absl::Span<int>(shape_data_);
}

TfTensorView::TfTensorView(const TfTensorView &o)
    : TensorView(o), shape_data_(o.shape_data_) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStf_tensor_viewDTcc mht_0(mht_0_v, 214, "", "./tensorflow/lite/kernels/shim/tf_tensor_view.cc", "TfTensorView::TfTensorView");

  shape_ = absl::Span<int>(shape_data_);
}

TfTensorView &TfTensorView::operator=(TfTensorView &&o) noexcept {
  shape_data_ = std::move(o.shape_data_);
  TensorView::operator=(std::move(o));
  shape_ = absl::Span<int>(shape_data_);
  return *this;
}

TfTensorView &TfTensorView::operator=(const TfTensorView &o) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStf_tensor_viewDTcc mht_1(mht_1_v, 228, "", "./tensorflow/lite/kernels/shim/tf_tensor_view.cc", "=");

  if (&o == this) return *this;
  TensorView::operator=(o);
  shape_data_ = o.shape_data_;
  shape_ = absl::Span<int>(shape_data_);
  return *this;
}

template <typename TfTensorType>
absl::StatusOr<typename MatchConstNess<TfTensorType, TfTensorView>::Type>
TfTensorViewTemplatizedNew(TfTensorType *wrapped_tensor) {
  switch (wrapped_tensor->dtype()) {
    CASE_FOR_DTYPE(::tensorflow::DT_BOOL);
    CASE_FOR_DTYPE(::tensorflow::DT_UINT8);
    CASE_FOR_DTYPE(::tensorflow::DT_UINT64);
    CASE_FOR_DTYPE(::tensorflow::DT_INT8);
    CASE_FOR_DTYPE(::tensorflow::DT_INT16);
    CASE_FOR_DTYPE(::tensorflow::DT_INT32);
    // Map DT_INT64 to int64_t instead of int64 to have a single int64 datatype.
    CASE_FOR_DTYPE_GIVEN_CPP_DTYPE(::tensorflow::DT_INT64, std::int64_t);
    CASE_FOR_DTYPE(::tensorflow::DT_FLOAT);
    CASE_FOR_DTYPE(::tensorflow::DT_DOUBLE);
    CASE_FOR_DTYPE(::tensorflow::DT_STRING);
    default: {
      return absl::UnimplementedError(
          absl::StrCat("Unsupported data type: ", wrapped_tensor->dtype()));
    }
  }
}

template <>
absl::StatusOr<TfTensorView> TensorView::New<::tensorflow::Tensor>(
    ::tensorflow::Tensor *wrapped_tensor) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStf_tensor_viewDTcc mht_2(mht_2_v, 263, "", "./tensorflow/lite/kernels/shim/tf_tensor_view.cc", "TensorView::New<::tensorflow::Tensor>");

  return TfTensorViewTemplatizedNew(wrapped_tensor);
}

template <>
absl::StatusOr<const TfTensorView> TensorView::New<const ::tensorflow::Tensor>(
    const ::tensorflow::Tensor *wrapped_tensor) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStf_tensor_viewDTcc mht_3(mht_3_v, 272, "", "./tensorflow/lite/kernels/shim/tf_tensor_view.cc", "::tensorflow::Tensor>");

  return TfTensorViewTemplatizedNew(wrapped_tensor);
}

}  // namespace shim
}  // namespace tflite
