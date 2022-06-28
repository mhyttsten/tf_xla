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
#ifndef TENSORFLOW_LITE_KERNELS_SHIM_TFLITE_TENSOR_VIEW_H_
#define TENSORFLOW_LITE_KERNELS_SHIM_TFLITE_TENSOR_VIEW_H_
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
class MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStflite_tensor_viewDTh {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStflite_tensor_viewDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStflite_tensor_viewDTh() {
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


#include <cstring>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/shim/tensor_view.h"
#include "tensorflow/lite/type_to_tflitetype.h"

namespace tflite {
namespace shim {

// A view over a TFLite tensor without taking ownership. It can either be
// mutable or immutable.
class TfLiteTensorView : public TensorView {
 public:
  // Move constructor
  TfLiteTensorView(TfLiteTensorView &&o) noexcept;
  // Copy constructor
  TfLiteTensorView(const TfLiteTensorView &o);
  // Move assignment operator
  TfLiteTensorView &operator=(TfLiteTensorView &&o) noexcept;
  // Copy assignment operator
  TfLiteTensorView &operator=(const TfLiteTensorView &);

 protected:
  // Templated constructor. Since it's not possible to specify the template
  // argument directly we place a dummy argument of that type so compiler can
  // deduce the right template parameter
  template <typename DType>
  TfLiteTensorView(::TfLiteTensor *wrapped_tensor, const DType &dtype)
      : TensorView(absl::Span<int>(wrapped_tensor->dims->data,
                                   wrapped_tensor->dims->size),
                   wrapped_tensor->data.raw, wrapped_tensor->bytes, dtype),
        wrapped_tensor_(wrapped_tensor),
        const_wrapped_tensor_(wrapped_tensor) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStflite_tensor_viewDTh mht_0(mht_0_v, 222, "", "./tensorflow/lite/kernels/shim/tflite_tensor_view.h", "TfLiteTensorView");
}

  // Specialization for string. (this take precedence over the above template)
  TfLiteTensorView(::TfLiteTensor *wrapped_tensor,
                   const ::tensorflow::tstring &dtype);

  // Templated constructor for const input.
  template <typename DType>
  TfLiteTensorView(const ::TfLiteTensor *wrapped_tensor, const DType &dtype)
      : TensorView(absl::Span<int>(wrapped_tensor->dims->data,
                                   wrapped_tensor->dims->size),
                   wrapped_tensor->data.raw, wrapped_tensor->bytes, dtype),
        const_wrapped_tensor_(wrapped_tensor) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStflite_tensor_viewDTh mht_1(mht_1_v, 237, "", "./tensorflow/lite/kernels/shim/tflite_tensor_view.h", "TfLiteTensorView");
}

  // Specialization for const string. (this take precedence over the above
  // template)
  TfLiteTensorView(const ::TfLiteTensor *wrapped_tensor,
                   const ::tensorflow::tstring &dtype);

  // Let the factory implementation use private constructors
  template <typename TfLiteTensorType>
  friend absl::StatusOr<
      typename MatchConstNess<TfLiteTensorType, TfLiteTensorView>::Type>
  TfLiteTensorViewTemplatizedNew(TfLiteTensorType *wrapped_tensor);

  struct StringBuffer {
    explicit StringBuffer(TfLiteTensorView *t_view);
    ~StringBuffer();

    // A vector of string as the intermediate shared buffer between
    // TensorViews
    std::vector<::tensorflow::tstring> buffer;
    // The TFLite tensor to which the contents of the buffer is flushed in
    // dtor
    ::TfLiteTensor *wrapped_tensor = nullptr;
  };

  // Initialize the data_ field for string tensors
  void InitForStringDType();

  // The wrapped TFLiteTensor
  ::TfLiteTensor *wrapped_tensor_ = nullptr;
  // A const version of the wrapped TFLiteTensor used when the input is const
  const ::TfLiteTensor *const_wrapped_tensor_ = nullptr;
  // A temporary buffer used to expose TfLite strings tensor as Span<tstring>.
  // This buffer will be flushed and serialized back to the underlying TfLite
  // string tensor once all the TensorViews over that tensor are destructed.
  std::shared_ptr<StringBuffer> str_vec_ = nullptr;
};

// Mapping of ::TfLiteTensor -> TfLiteTensorView
template <>
struct TensorViewSubType<::TfLiteTensor> {
  using Type = TfLiteTensorView;
};

// Mapping of const ::TfLiteTensor -> const TfLiteTensorView
template <>
struct TensorViewSubType<const ::TfLiteTensor> {
  using Type = const TfLiteTensorView;
};

// Specialization for TensorView::New()
template <>
absl::StatusOr<TfLiteTensorView> TensorView::New<::TfLiteTensor>(
    ::TfLiteTensor *wrapped_tensor);

// Specialization for TensorView::New()
template <>
absl::StatusOr<const TfLiteTensorView> TensorView::New<const ::TfLiteTensor>(
    const ::TfLiteTensor *wrapped_tensor);

}  // namespace shim
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_SHIM_TFLITE_TENSOR_VIEW_H_
