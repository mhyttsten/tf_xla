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
#ifndef TENSORFLOW_CC_EXPERIMENTAL_LIBTF_RUNTIME_H_
#define TENSORFLOW_CC_EXPERIMENTAL_LIBTF_RUNTIME_H_
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
class MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSruntimePSruntimeDTh {
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
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSruntimePSruntimeDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSruntimePSruntimeDTh() {
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


#include <sys/types.h>

#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/tfe_context_internal.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/c/tf_status_internal.h"
#include "tensorflow/cc/experimental/libtf/object.h"
#include "tensorflow/cc/experimental/libtf/value.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/statusor.h"

namespace tf {
namespace libtf {
namespace runtime {

/// @brief A runtime object capable of loading modules and executing functions.
///
/// It is the responsibility of the owner of the Runtime to keep it alive longer
/// than all imported modules.
class Runtime : public Object {
 public:
  // TODO(b/191264214): Remove need for AbstractContext
  explicit Runtime(tensorflow::AbstractContext* ctx);
  /// @brief Loads the module indicated by `name` and returns it.
  ///
  /// @param name The name of the module / file path to load
  /// @return An `Object` representing the module, if successful.  Otherwise, a
  /// non-ok `absl::Status`.
  tensorflow::StatusOr<Object> Load(const String& name);
  // TODO(b/186787000): Loading a module with identically-named functions as
  // a previously loaded module results in undefined behavior. This
  // functionality will be supported in the future.

  // Create a host tensor and copy data into it.
  //
  // Raises an error if shape or dtype are incompatible with T.
  // TODO(b/189458441): Update this when we decide on the representation of
  // shape and dtype in this API.
  // Disclaimer: This API is subject to change as we add support for creating
  // device tensors b/187222691 and enable buffer re-use b/187223179.
  // TODO(b/190715501): Make this available via a soft API as well.
  template <class T>
  tensorflow::StatusOr<Tensor> CreateHostTensor(absl::Span<const int64_t> shape,
                                                int dtype,
                                                absl::Span<const T> data);
};

template <class T>
tensorflow::StatusOr<Tensor> Runtime::CreateHostTensor(
    absl::Span<const int64_t> shape, int dtype, absl::Span<const T> data) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSruntimePSruntimeDTh mht_0(mht_0_v, 238, "", "./tensorflow/cc/experimental/libtf/runtime/runtime.h", "Runtime::CreateHostTensor");

  size_t num_elements = 1;
  for (int dim = 0; dim < shape.size(); dim++) {
    if (shape[dim] < 0) {
      return tensorflow::errors::InvalidArgument(absl::StrCat(
          "Shape must be fully-defined, got: shape[", dim, "] = ", shape[dim]));
    }
    num_elements *= shape[dim];
  }
  if (data.size() != num_elements) {
    return tensorflow::errors::InvalidArgument(absl::StrCat(
        "Mismatched shape and data size: \n", "Shape num_elements: ",
        num_elements, "\n", "Data size: ", data.size(), "\n"));
  }
  auto maybe_capsule = Get<internal::Capsule>(String("ctx"));
  if (!maybe_capsule.status().ok()) {
    return maybe_capsule.status();
  }
  auto capsule = maybe_capsule.ValueOrDie();
  auto ctx = capsule.cast<tensorflow::ImmediateExecutionContext*>();
  tensorflow::AbstractTensorPtr t(
      ctx->CreateTensor(static_cast<tensorflow::DataType>(dtype), shape));
  // TODO(srbs): This is still a weak check. Check that dtype and T are
  // compatible.
  if (t->ByteSize() != sizeof(T) * data.size()) {
    return tensorflow::errors::InvalidArgument(absl::StrCat(
        "Invalid number of bytes in data buffer\n", "Expected bytes: ",
        t->ByteSize(), "\n", "Actual bytes: ", sizeof(T) * data.size()));
  }
  memcpy(t->Data(), data.data(), t->ByteSize());
  return Tensor(Convert(TaggedValue(
      impl::TaggedValueTensor(ctx->CreateLocalHandle(t.get()), false))));
}

}  // namespace runtime
}  // namespace libtf
}  // namespace tf

#endif  // TENSORFLOW_CC_EXPERIMENTAL_LIBTF_RUNTIME_H_
