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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_CL_GL_INTEROP_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_CL_GL_INTEROP_H_
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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSgl_interopDTh {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSgl_interopDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSgl_interopDTh() {
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


#include <EGL/egl.h>
#include <EGL/eglext.h>

#include <vector>

#include "tensorflow/lite/delegates/gpu/cl/cl_command_queue.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_context.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_device.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_event.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_memory.h"
#include "tensorflow/lite/delegates/gpu/cl/egl_sync.h"
#include "tensorflow/lite/delegates/gpu/cl/environment.h"
#include "tensorflow/lite/delegates/gpu/cl/opencl_wrapper.h"
#include "tensorflow/lite/delegates/gpu/common/access_type.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/gl/portable_gl31.h"
#include "tensorflow/lite/delegates/gpu/spi.h"

namespace tflite {
namespace gpu {
namespace cl {

// Creates an EglSync from OpenCL event. Source event does not need to outlive
// returned sync and could be safely destroyed.
//
// Depends on EGL 1.5.
absl::Status CreateEglSyncFromClEvent(cl_event event, EGLDisplay display,
                                      EglSync* sync);

// Returns true if 'CreateEglSyncFromClEvent' is supported.
bool IsEglSyncFromClEventSupported();

// Creates CL event from EGL sync.
// Created event could only be consumed by AcquiredGlObject::Acquire call as
// a 'wait_event'.
absl::Status CreateClEventFromEglSync(cl_context context,
                                      const EglSync& egl_sync, CLEvent* event);

// Returns true if 'CreateClEventFromEglSync' is supported.
bool IsClEventFromEglSyncSupported(const CLDevice& device);

// Creates new CL memory object from OpenGL buffer.
absl::Status CreateClMemoryFromGlBuffer(GLuint gl_ssbo_id,
                                        AccessType access_type,
                                        CLContext* context, CLMemory* memory);

// Creates new CL memory object from OpenGL texture.
absl::Status CreateClMemoryFromGlTexture(GLenum texture_target,
                                         GLuint texture_id,
                                         AccessType access_type,
                                         CLContext* context, CLMemory* memory);

// Returns true if GL objects could be shared with OpenCL context.
bool IsGlSharingSupported(const CLDevice& device);

// RAII-wrapper for GL objects acquired into CL context.
class AcquiredGlObjects {
 public:
  static bool IsSupported(const CLDevice& device);

  AcquiredGlObjects() : AcquiredGlObjects({}, nullptr) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSgl_interopDTh mht_0(mht_0_v, 248, "", "./tensorflow/lite/delegates/gpu/cl/gl_interop.h", "AcquiredGlObjects");
}

  // Quitely releases OpenGL objects. It is recommended to call Release()
  // explicitly to properly handle potential errors.
  ~AcquiredGlObjects();

  // Acquires memory from the OpenGL context. Memory must be created by either
  // CreateClMemoryFromGlBuffer or CreateClMemoryFromGlTexture calls.
  // If 'acquire_event' is not nullptr, it will be signared once acquisition is
  // complete.
  static absl::Status Acquire(const std::vector<cl_mem>& memory,
                              cl_command_queue queue,
                              const std::vector<cl_event>& wait_events,
                              CLEvent* acquire_event /* optional */,
                              AcquiredGlObjects* objects);

  // Releases OpenCL memory back to OpenGL context. If 'release_event' is not
  // nullptr, it will be signalled once release is complete.
  absl::Status Release(const std::vector<cl_event>& wait_events,
                       CLEvent* release_event /* optional */);

 private:
  AcquiredGlObjects(const std::vector<cl_mem>& memory, cl_command_queue queue)
      : memory_(memory), queue_(queue) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSgl_interopDTh mht_1(mht_1_v, 274, "", "./tensorflow/lite/delegates/gpu/cl/gl_interop.h", "AcquiredGlObjects");
}

  std::vector<cl_mem> memory_;
  cl_command_queue queue_;
};

// Incapsulates all complicated GL-CL synchronization. It manages life time of
// all appropriate events to ensure fast synchronization whenever possible.
class GlInteropFabric {
 public:
  GlInteropFabric(EGLDisplay egl_display, Environment* environment);

  // Ensures proper GL->CL synchronization is in place before
  // GL objects that are mapped to CL objects are used.
  absl::Status Start();

  // Puts appropriate CL->GL synchronization after all work is complete.
  absl::Status Finish();

  // Registers memory to be used from GL context. Such CL memory object must
  // be created with CreateClMemoryFromGlBuffer or CreateClMemoryFromGlTexture
  // call.
  void RegisterMemory(cl_mem memory);

  // Unregisters memory registered with RegisterMemory call.
  void UnregisterMemory(cl_mem memory);

 private:
  bool is_enabled() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSgl_interopDTh mht_2(mht_2_v, 305, "", "./tensorflow/lite/delegates/gpu/cl/gl_interop.h", "is_enabled");
 return egl_display_ && !memory_.empty(); }

  bool is_egl_sync_supported_;
  bool is_egl_to_cl_mapping_supported_;
  bool is_cl_to_egl_mapping_supported_;

  const EGLDisplay egl_display_;
  cl_context context_;
  cl_command_queue queue_;
  std::vector<cl_mem> memory_;
  AcquiredGlObjects gl_objects_;  // transient during Start/Finish calls.
};

// Copies data from(to) GL buffer to(from) CL buffer using CPU.
class GlClBufferCopier : public TensorObjectConverter {
 public:
  static bool IsSupported(const ObjectDef& input, const ObjectDef& output) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSgl_interopDTh mht_3(mht_3_v, 324, "", "./tensorflow/lite/delegates/gpu/cl/gl_interop.h", "IsSupported");

    return input.data_type == output.data_type &&
           input.data_layout == output.data_layout &&
           ((input.object_type == ObjectType::OPENGL_SSBO &&
             output.object_type == ObjectType::OPENCL_BUFFER) ||
            (input.object_type == ObjectType::OPENCL_BUFFER &&
             output.object_type == ObjectType::OPENGL_SSBO));
  }

  GlClBufferCopier(const TensorObjectDef& input_def,
                   const TensorObjectDef& output_def, Environment* environment);

  absl::Status Convert(const TensorObject& input_obj,
                       const TensorObject& output_obj) override;

 private:
  size_t size_in_bytes_;
  CLCommandQueue* queue_ = nullptr;
};

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_GL_INTEROP_H_
