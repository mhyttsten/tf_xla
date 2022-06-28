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
class MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStflite_tensor_viewDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStflite_tensor_viewDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStflite_tensor_viewDTcc() {
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
#include "tensorflow/lite/kernels/shim/tflite_tensor_view.h"

#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/variant.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/shim/tensor_view.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/type_to_tflitetype.h"

// Creates a case statement for the switch() clause given the dtype
#define CASE_FOR_DTYPE_GIVEN_CPP_DTYPE(TFLITE_DTYPE, CPP_DTYPE) \
  case TFLITE_DTYPE: {                                          \
    using DType = typename CPP_DTYPE;                           \
    return TfLiteTensorView(wrapped_tensor, DType());           \
  }

#define CASE_FOR_DTYPE(TFLITE_DTYPE) \
  CASE_FOR_DTYPE_GIVEN_CPP_DTYPE(    \
      TFLITE_DTYPE, ::tflite::TfLiteTypeToType<TFLITE_DTYPE>::Type)

namespace tflite {
namespace shim {

TfLiteTensorView::TfLiteTensorView(::TfLiteTensor *wrapped_tensor,
                                   const ::tensorflow::tstring &dtype)
    : TensorView(absl::Span<int>(wrapped_tensor->dims->data,
                                 wrapped_tensor->dims->size),
                 nullptr, 0, dtype),
      wrapped_tensor_(wrapped_tensor),
      const_wrapped_tensor_(wrapped_tensor) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStflite_tensor_viewDTcc mht_0(mht_0_v, 216, "", "./tensorflow/lite/kernels/shim/tflite_tensor_view.cc", "TfLiteTensorView::TfLiteTensorView");

  InitForStringDType();
}

TfLiteTensorView::TfLiteTensorView(const ::TfLiteTensor *wrapped_tensor,
                                   const ::tensorflow::tstring &dtype)
    : TensorView(absl::Span<int>(wrapped_tensor->dims->data,
                                 wrapped_tensor->dims->size),
                 nullptr, 0, dtype),
      const_wrapped_tensor_(wrapped_tensor) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStflite_tensor_viewDTcc mht_1(mht_1_v, 228, "", "./tensorflow/lite/kernels/shim/tflite_tensor_view.cc", "TfLiteTensorView::TfLiteTensorView");

  InitForStringDType();
}

TfLiteTensorView::TfLiteTensorView(TfLiteTensorView &&o) noexcept
    : TensorView(std::move(o)),
      wrapped_tensor_(o.wrapped_tensor_),
      const_wrapped_tensor_(o.const_wrapped_tensor_),
      str_vec_(std::move(o.str_vec_)) {
}

TfLiteTensorView::TfLiteTensorView(const TfLiteTensorView &o)
    : TensorView(o),
      wrapped_tensor_(o.wrapped_tensor_),
      const_wrapped_tensor_(o.const_wrapped_tensor_),
      str_vec_(o.str_vec_) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStflite_tensor_viewDTcc mht_2(mht_2_v, 246, "", "./tensorflow/lite/kernels/shim/tflite_tensor_view.cc", "TfLiteTensorView::TfLiteTensorView");

}

TfLiteTensorView &TfLiteTensorView::operator=(TfLiteTensorView &&o) noexcept {
  wrapped_tensor_ = o.wrapped_tensor_;
  const_wrapped_tensor_ = o.const_wrapped_tensor_;
  str_vec_ = std::move(o.str_vec_);
  TensorView::operator=(std::move(o));
  return *this;
}

TfLiteTensorView &TfLiteTensorView::operator=(const TfLiteTensorView &o) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStflite_tensor_viewDTcc mht_3(mht_3_v, 260, "", "./tensorflow/lite/kernels/shim/tflite_tensor_view.cc", "=");

  if (&o == this) return *this;
  TensorView::operator=(o);
  wrapped_tensor_ = o.wrapped_tensor_;
  const_wrapped_tensor_ = o.const_wrapped_tensor_;
  str_vec_ = o.str_vec_;
  return *this;
}

void TfLiteTensorView::InitForStringDType() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStflite_tensor_viewDTcc mht_4(mht_4_v, 272, "", "./tensorflow/lite/kernels/shim/tflite_tensor_view.cc", "TfLiteTensorView::InitForStringDType");

  if (str_vec_ == nullptr) {
    str_vec_ = std::make_shared<StringBuffer>(this);
  }
  data_ = absl::Span<::tensorflow::tstring>(str_vec_->buffer);
}

TfLiteTensorView::StringBuffer::StringBuffer(TfLiteTensorView *t_view)
    : wrapped_tensor(t_view->wrapped_tensor_) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStflite_tensor_viewDTcc mht_5(mht_5_v, 283, "", "./tensorflow/lite/kernels/shim/tflite_tensor_view.cc", "TfLiteTensorView::StringBuffer::StringBuffer");

  buffer.resize(NumElements(t_view->shape_));
  // Read the TfLite string into the buffer
  const auto const_wrapped_tensor = t_view->const_wrapped_tensor_;
  std::size_t str_count;
  if (const_wrapped_tensor->data.raw == nullptr)
    str_count = 0;
  else
    str_count = ::tflite::GetStringCount(const_wrapped_tensor);
  for (int i = 0; i < str_count; ++i) {
    const auto str_ref = ::tflite::GetString(const_wrapped_tensor, i);
    buffer[i].assign_as_view(str_ref.str, str_ref.len);
  }
}

TfLiteTensorView::StringBuffer::~StringBuffer() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStflite_tensor_viewDTcc mht_6(mht_6_v, 301, "", "./tensorflow/lite/kernels/shim/tflite_tensor_view.cc", "TfLiteTensorView::StringBuffer::~StringBuffer");

  if (wrapped_tensor == nullptr) return;
  tflite::DynamicBuffer buf;
  for (const auto &s : buffer) buf.AddString(s.data(), s.length());
  buf.WriteToTensor(wrapped_tensor, /*new_shape=*/nullptr);
}

template <typename TfLiteTensorType>
absl::StatusOr<
    typename MatchConstNess<TfLiteTensorType, TfLiteTensorView>::Type>
TfLiteTensorViewTemplatizedNew(TfLiteTensorType *wrapped_tensor) {
  switch (wrapped_tensor->type) {
    CASE_FOR_DTYPE(kTfLiteBool);
    CASE_FOR_DTYPE(kTfLiteUInt8);
    CASE_FOR_DTYPE(kTfLiteUInt64);
    CASE_FOR_DTYPE(kTfLiteInt8);
    CASE_FOR_DTYPE(kTfLiteInt16);
    CASE_FOR_DTYPE(kTfLiteInt32);
    CASE_FOR_DTYPE(kTfLiteInt64);
    CASE_FOR_DTYPE(kTfLiteFloat32);
    CASE_FOR_DTYPE(kTfLiteFloat64);
    // The DType for kTfLiteString is slightly different as we need to use
    // tensorflow::tstring rather than std::string
    CASE_FOR_DTYPE_GIVEN_CPP_DTYPE(kTfLiteString, ::tensorflow::tstring);
    default: {
      return absl::UnimplementedError(
          absl::StrCat("Unsupported dtype: ", wrapped_tensor->type));
    }
  }
}

template <>
absl::StatusOr<TfLiteTensorView> TensorView::New<::TfLiteTensor>(
    ::TfLiteTensor *wrapped_tensor) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStflite_tensor_viewDTcc mht_7(mht_7_v, 337, "", "./tensorflow/lite/kernels/shim/tflite_tensor_view.cc", "TensorView::New<::TfLiteTensor>");

  return TfLiteTensorViewTemplatizedNew(wrapped_tensor);
}

template <>
absl::StatusOr<const TfLiteTensorView> TensorView::New<const ::TfLiteTensor>(
    const ::TfLiteTensor *wrapped_tensor) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStflite_tensor_viewDTcc mht_8(mht_8_v, 346, "", "./tensorflow/lite/kernels/shim/tflite_tensor_view.cc", "::TfLiteTensor>");

  return TfLiteTensorViewTemplatizedNew(wrapped_tensor);
}

}  // namespace shim
}  // namespace tflite
