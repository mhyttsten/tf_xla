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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_CL_API_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_CL_API_H_
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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSapiDTh {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSapiDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSapiDTh() {
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


#ifdef CL_DELEGATE_NO_GL
#define EGL_NO_PROTOTYPES
#endif

#include <EGL/egl.h>

#include <cstdint>
#include <memory>

#include "absl/types/span.h"
#include "tensorflow/lite/delegates/gpu/api.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

// Usage example:
//
//   std::unique_ptr<InferenceEnvironment> env;
//   RETURN_IF_ERROR(NewInferenceEnvironment(option, &env));
//
//   InferenceOptions options;
//
//   std::unique_ptr<InferenceBuilder> builder;
//   RETURN_IF_ERROR(env->NewInferenceBuilder(options, model, &builder));
//   // now builder is ready to prepare inference runner.
//
// -----------------
// Supported formats
// -----------------
//
// OpenCL implementation uses 2D textures as the primary format.
// Tensor in HWDC4 layout is {TEXTURE_2D, RGBA, width := W*D, height := H}.
//

namespace tflite {
namespace gpu {
namespace cl {

struct InferenceOptions : public tflite::gpu::InferenceOptions {};

// Indicates environment
struct InferenceEnvironmentProperties {
  bool is_opencl_available = false;

  // GL objects (buffers and textures) could be shared with CL context.
  bool is_gl_sharing_supported = false;

  // Indicates whether fast GL->CL synchronization is supported.
  bool is_gl_to_cl_fast_sync_supported = false;

  // Indicates whether fast CL->GL synchronization is supported.
  bool is_cl_to_gl_fast_sync_supported = false;
};

// Environment manages all resources that need to stay until any inference is
// running using OpenCL backend.
class InferenceEnvironment {
 public:
  virtual ~InferenceEnvironment() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSapiDTh mht_0(mht_0_v, 245, "", "./tensorflow/lite/delegates/gpu/cl/api.h", "~InferenceEnvironment");
}

  // Converts GraphFloat32 into intermediate, device-specific representation.
  // This serialized_model specific for device and InferenceOptions.
  // serialized_model cannot be used with another device or InferenceOptions.
  // Loading serialized_model is much faster than loading GraphFloat32.
  // serialized_model must be used with appropriate NewInferenceBuilder
  // method (see below).
  // Normally BuildSerializedModel method need to be called whenever a model or
  // OS GPU driver is updated.
  virtual absl::Status BuildSerializedModel(
      const InferenceOptions& options, GraphFloat32 model,
      std::vector<uint8_t>* serialized_model) = 0;

  // Serialized model can became invalid when environment changes. In this case
  // this call will fail and model must be regenerated(with
  // BuildSerializedModel).
  virtual absl::Status NewInferenceBuilder(
      const absl::Span<const uint8_t> serialized_model,
      std::unique_ptr<InferenceBuilder>* builder) = 0;

  virtual absl::Status NewInferenceBuilder(
      const InferenceOptions& options, GraphFloat32 model,
      std::unique_ptr<InferenceBuilder>* builder) = 0;

  // Returns opaque binary blob that contains a collection of already compiled
  // OpenCL kernels present in a cache. Returned data could be re-used later
  // to speed up compilation time when new environment is created for the same
  // set of models.
  // Returned data is valid only if used on the same device, otherwise it will
  // not be compatible and will be discarded.
  virtual std::vector<uint8_t> GetSerializedBinaryCache() const = 0;
};

struct InferenceEnvironmentOptions {
  // If any of these objects are set, created environment will use them instead
  // of creating/choosing own instances.
  cl_device_id device = nullptr;
  cl_context context = nullptr;
  cl_command_queue command_queue = nullptr;

  // Whenever input and/or output is GL object, EGL display and context must be
  // set to create GL aware OpenCL context. Do not set these variables whenever
  // GL interoperability is not needed.
  // It is the error to set egl_display, egl_context AND context at the same
  // time. If egl_display and egl_context are set, they will be used to create
  // GL-aware CL context.
  EGLDisplay egl_display = EGL_NO_DISPLAY;
  EGLContext egl_context = EGL_NO_CONTEXT;

  // Should contain data returned from
  // InferenceEnvironment::GetSerializedBinaryCache method.
  // Invalid or incompatible data will be discarded. Compiled binary may become
  // incompatible when GPU driver is updated.
  absl::Span<const uint8_t> serialized_binary_cache;

  bool IsGlAware() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSapiDTh mht_1(mht_1_v, 304, "", "./tensorflow/lite/delegates/gpu/cl/api.h", "IsGlAware");

    return egl_context != EGL_NO_CONTEXT && egl_display != EGL_NO_DISPLAY;
  }
};

// Creates new OpenCL environment that needs to stay around until all inference
// runners are destroyed.
absl::Status NewInferenceEnvironment(
    const InferenceEnvironmentOptions& options,
    std::unique_ptr<InferenceEnvironment>* environment,
    InferenceEnvironmentProperties* properties /* optional */);

class CLInferenceRunner : public ::tflite::gpu::InferenceRunner {
 public:
  // The RunWithoutExternalBufferCopy provides a contract where the user of this
  // interface does not need
  //    a. Inputs to be copied to the internal GPU buffer from the external CPU
  //       input buffer
  //    b. Outputs to be copied from the internal GPU buffer to the
  //       external CPU buffer
  //
  // The user of this interface is responsible for copying the inputs prior to
  // running the GPU kernels and outputs post running with the other interfaces
  // provided here.
  virtual absl::Status RunWithoutExternalBufferCopy() = 0;

  // Copies from the external input tensor (normally CPU buffer) to the internal
  // OpenCL buffer.  The call only guarantees a queueing of the command. The
  // caller is expected to hold a copy of the queue and wait for completion if
  // the external buffer is a CPU buffer.
  virtual absl::Status CopyFromExternalInput(int index) = 0;

  // Copies from the internal output OpenCL buffer to the external output
  // tensor.  The call only guarantees a queueing of the command. The caller
  // is expected to hold a copy of the queue and wait for completion if the
  // external buffer is a CPU buffer.
  virtual absl::Status CopyToExternalOutput(int index) = 0;
};

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_API_H_
