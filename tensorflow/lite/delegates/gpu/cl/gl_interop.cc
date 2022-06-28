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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSgl_interopDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSgl_interopDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSgl_interopDTcc() {
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

#include "tensorflow/lite/delegates/gpu/cl/gl_interop.h"

#include "absl/strings/str_cat.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_errors.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_call.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_sync.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {

#ifndef EGL_VERSION_1_5
typedef void* EGLSync;
#define EGL_SYNC_CL_EVENT 0x30FE
#define EGL_CL_EVENT_HANDLE 0x309C
#define EGL_NO_SYNC 0
#endif /* EGL_VERSION_1_5 */

// TODO(b/131897059): replace with 64 version when EGL 1.5 is available.
// it should use KHR_cl_event2 extension. More details are in b/129974818.
using PFNEGLCREATESYNCPROC = EGLSync(EGLAPIENTRYP)(
    EGLDisplay dpy, EGLenum type, const EGLAttrib* attrib_list);

PFNEGLCREATESYNCPROC g_eglCreateSync = nullptr;

}  // namespace

absl::Status CreateEglSyncFromClEvent(cl_event event, EGLDisplay display,
                                      EglSync* sync) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSgl_interopDTcc mht_0(mht_0_v, 214, "", "./tensorflow/lite/delegates/gpu/cl/gl_interop.cc", "CreateEglSyncFromClEvent");

  if (!IsEglSyncFromClEventSupported()) {
    return absl::UnimplementedError(
        "CreateEglSyncFromClEvent is not supported");
  }
  EGLSync egl_sync;
  const EGLAttrib attributes[] = {EGL_CL_EVENT_HANDLE,
                                  reinterpret_cast<EGLAttrib>(event), EGL_NONE};
  RETURN_IF_ERROR(TFLITE_GPU_CALL_EGL(g_eglCreateSync, &egl_sync, display,
                                      EGL_SYNC_CL_EVENT, attributes));
  if (egl_sync == EGL_NO_SYNC) {
    return absl::InternalError("Returned empty EGL sync");
  }
  *sync = EglSync(display, egl_sync);
  return absl::OkStatus();
}

bool IsEglSyncFromClEventSupported() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSgl_interopDTcc mht_1(mht_1_v, 234, "", "./tensorflow/lite/delegates/gpu/cl/gl_interop.cc", "IsEglSyncFromClEventSupported");

  // In C++11, static initializers are guaranteed to be evaluated only once.
  static bool supported = []() -> bool {
    // This function requires EGL 1.5 to work
    g_eglCreateSync = reinterpret_cast<PFNEGLCREATESYNCPROC>(
        eglGetProcAddress("eglCreateSync"));
    // eglQueryString accepts EGL_NO_DISPLAY only starting EGL 1.5
    if (!eglQueryString(EGL_NO_DISPLAY, EGL_EXTENSIONS)) {
      g_eglCreateSync = nullptr;
    }
    return (g_eglCreateSync != nullptr);
  }();
  return supported;
}

absl::Status CreateClEventFromEglSync(cl_context context,
                                      const EglSync& egl_sync, CLEvent* event) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSgl_interopDTcc mht_2(mht_2_v, 253, "", "./tensorflow/lite/delegates/gpu/cl/gl_interop.cc", "CreateClEventFromEglSync");

  cl_int error_code;
  cl_event new_event = clCreateEventFromEGLSyncKHR(
      context, egl_sync.sync(), egl_sync.display(), &error_code);
  if (error_code != CL_SUCCESS) {
    return absl::InternalError(
        absl::StrCat("Unable to create CL sync from EGL sync. ",
                     CLErrorCodeToString(error_code)));
  }
  *event = CLEvent(new_event);
  return absl::OkStatus();
}

bool IsClEventFromEglSyncSupported(const CLDevice& device) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSgl_interopDTcc mht_3(mht_3_v, 269, "", "./tensorflow/lite/delegates/gpu/cl/gl_interop.cc", "IsClEventFromEglSyncSupported");

  return device.GetInfo().SupportsExtension("cl_khr_egl_event");
}

absl::Status CreateClMemoryFromGlBuffer(GLuint gl_ssbo_id,
                                        AccessType access_type,
                                        CLContext* context, CLMemory* memory) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSgl_interopDTcc mht_4(mht_4_v, 278, "", "./tensorflow/lite/delegates/gpu/cl/gl_interop.cc", "CreateClMemoryFromGlBuffer");

  cl_int error_code;
  auto mem = clCreateFromGLBuffer(context->context(), ToClMemFlags(access_type),
                                  gl_ssbo_id, &error_code);
  if (error_code != CL_SUCCESS) {
    return absl::InternalError(
        absl::StrCat("Unable to acquire CL buffer from GL buffer. ",
                     CLErrorCodeToString(error_code)));
  }
  *memory = CLMemory(mem, true);
  return absl::OkStatus();
}

absl::Status CreateClMemoryFromGlTexture(GLenum texture_target,
                                         GLuint texture_id,
                                         AccessType access_type,
                                         CLContext* context, CLMemory* memory) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSgl_interopDTcc mht_5(mht_5_v, 297, "", "./tensorflow/lite/delegates/gpu/cl/gl_interop.cc", "CreateClMemoryFromGlTexture");

  cl_int error_code;
  auto mem =
      clCreateFromGLTexture(context->context(), ToClMemFlags(access_type),
                            texture_target, 0, texture_id, &error_code);
  if (error_code != CL_SUCCESS) {
    return absl::InternalError(
        absl::StrCat("Unable to create CL buffer from GL texture. ",
                     CLErrorCodeToString(error_code)));
  }
  *memory = CLMemory(mem, true);
  return absl::OkStatus();
}

bool IsGlSharingSupported(const CLDevice& device) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSgl_interopDTcc mht_6(mht_6_v, 314, "", "./tensorflow/lite/delegates/gpu/cl/gl_interop.cc", "IsGlSharingSupported");

  return clCreateFromGLBuffer && clCreateFromGLTexture &&
         device.GetInfo().SupportsExtension("cl_khr_gl_sharing");
}

AcquiredGlObjects::~AcquiredGlObjects() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSgl_interopDTcc mht_7(mht_7_v, 322, "", "./tensorflow/lite/delegates/gpu/cl/gl_interop.cc", "AcquiredGlObjects::~AcquiredGlObjects");
 Release({}, nullptr).IgnoreError(); }

absl::Status AcquiredGlObjects::Acquire(
    const std::vector<cl_mem>& memory, cl_command_queue queue,
    const std::vector<cl_event>& wait_events, CLEvent* acquire_event,
    AcquiredGlObjects* objects) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSgl_interopDTcc mht_8(mht_8_v, 330, "", "./tensorflow/lite/delegates/gpu/cl/gl_interop.cc", "AcquiredGlObjects::Acquire");

  if (!memory.empty()) {
    cl_event new_event;
    cl_int error_code = clEnqueueAcquireGLObjects(
        queue, memory.size(), memory.data(), wait_events.size(),
        wait_events.data(), acquire_event ? &new_event : nullptr);
    if (error_code != CL_SUCCESS) {
      return absl::InternalError(absl::StrCat("Unable to acquire GL object. ",
                                              CLErrorCodeToString(error_code)));
    }
    if (acquire_event) {
      *acquire_event = CLEvent(new_event);
    }
    clFlush(queue);
  }
  *objects = AcquiredGlObjects(memory, queue);
  return absl::OkStatus();
}

absl::Status AcquiredGlObjects::Release(
    const std::vector<cl_event>& wait_events, CLEvent* release_event) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSgl_interopDTcc mht_9(mht_9_v, 353, "", "./tensorflow/lite/delegates/gpu/cl/gl_interop.cc", "AcquiredGlObjects::Release");

  if (queue_ && !memory_.empty()) {
    cl_event new_event;
    cl_int error_code = clEnqueueReleaseGLObjects(
        queue_, memory_.size(), memory_.data(), wait_events.size(),
        wait_events.data(), release_event ? &new_event : nullptr);
    if (error_code != CL_SUCCESS) {
      return absl::InternalError(absl::StrCat("Unable to release GL object. ",
                                              CLErrorCodeToString(error_code)));
    }
    if (release_event) {
      *release_event = CLEvent(new_event);
    }
    clFlush(queue_);
    queue_ = nullptr;
  }
  return absl::OkStatus();
}

GlInteropFabric::GlInteropFabric(EGLDisplay egl_display,
                                 Environment* environment)
    : is_egl_sync_supported_(true),
      is_egl_to_cl_mapping_supported_(
          IsClEventFromEglSyncSupported(environment->device())),
      is_cl_to_egl_mapping_supported_(IsEglSyncFromClEventSupported()),
      egl_display_(egl_display),
      context_(environment->context().context()),
      queue_(environment->queue()->queue()) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSgl_interopDTcc mht_10(mht_10_v, 383, "", "./tensorflow/lite/delegates/gpu/cl/gl_interop.cc", "GlInteropFabric::GlInteropFabric");
}

void GlInteropFabric::RegisterMemory(cl_mem memory) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSgl_interopDTcc mht_11(mht_11_v, 388, "", "./tensorflow/lite/delegates/gpu/cl/gl_interop.cc", "GlInteropFabric::RegisterMemory");

  memory_.push_back(memory);
}

void GlInteropFabric::UnregisterMemory(cl_mem memory) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSgl_interopDTcc mht_12(mht_12_v, 395, "", "./tensorflow/lite/delegates/gpu/cl/gl_interop.cc", "GlInteropFabric::UnregisterMemory");

  auto it = std::find(memory_.begin(), memory_.end(), memory);
  if (it != memory_.end()) {
    memory_.erase(it);
  }
}

absl::Status GlInteropFabric::Start() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSgl_interopDTcc mht_13(mht_13_v, 405, "", "./tensorflow/lite/delegates/gpu/cl/gl_interop.cc", "GlInteropFabric::Start");

  if (!is_enabled()) {
    return absl::OkStatus();
  }

  // In GL-CL interoperability, we need to make sure GL finished processing of
  // all commands that might affect GL objects. There are a few ways:
  //   a) glFinish
  //      slow, but portable
  //   b) EglSync + ClientWait
  //      faster alternative for glFinish, but still slow as it stalls GPU
  //      pipeline.
  //   c) EglSync->CLEvent or GlSync->CLEvent mapping
  //      Fast, as it allows to map sync to CL event and use it as a dependency
  //      later without stalling GPU pipeline.
  CLEvent inbound_event;
  std::vector<cl_event> inbound_events;
  if (is_egl_sync_supported_) {
    EglSync sync;
    RETURN_IF_ERROR(EglSync::NewFence(egl_display_, &sync));
    if (is_egl_to_cl_mapping_supported_) {
      // (c) EglSync->CLEvent or GlSync->CLEvent mapping
      glFlush();
      RETURN_IF_ERROR(CreateClEventFromEglSync(context_, sync, &inbound_event));
      inbound_events.push_back(inbound_event.event());
    } else {
      // (b) EglSync + ClientWait
      RETURN_IF_ERROR(sync.ClientWait());
    }
  } else {
    // (a) glFinish / GL fence sync
    RETURN_IF_ERROR(gl::GlActiveSyncWait());
  }

  // Acquire all GL objects needed while processing.
  return AcquiredGlObjects::Acquire(memory_, queue_, inbound_events, nullptr,
                                    &gl_objects_);
}

absl::Status GlInteropFabric::Finish() {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSgl_interopDTcc mht_14(mht_14_v, 447, "", "./tensorflow/lite/delegates/gpu/cl/gl_interop.cc", "GlInteropFabric::Finish");

  if (!is_enabled()) {
    return absl::OkStatus();
  }
  CLEvent outbound_event;
  RETURN_IF_ERROR(gl_objects_.Release({}, &outbound_event));

  // if (is_egl_sync_supported_ && is_cl_to_egl_mapping_supported_) {
  //   EglSync egl_outbound_sync;
  //   RETURN_IF_ERROR(CreateEglSyncFromClEvent(outbound_event.event(),
  //                                            egl_display_,
  //                                            &egl_outbound_sync));
  //   // Instruct GL pipeline to wait until corresponding CL event is signaled.
  //   RETURN_IF_ERROR(egl_outbound_sync.ServerWait());
  //   glFlush();
  // } else {
  //   // Slower option if proper sync is not supported. It is equivalent to
  //   // clFinish, but, hopefully, faster.
  //   outbound_event.Wait();
  // }

  // This slow sync is the only working solution right now. We have to debug why
  // above version is not working fast and reliable.
  outbound_event.Wait();
  return absl::OkStatus();
}

GlClBufferCopier::GlClBufferCopier(const TensorObjectDef& input_def,
                                   const TensorObjectDef& output_def,
                                   Environment* environment) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSgl_interopDTcc mht_15(mht_15_v, 479, "", "./tensorflow/lite/delegates/gpu/cl/gl_interop.cc", "GlClBufferCopier::GlClBufferCopier");

  queue_ = environment->queue();
  size_in_bytes_ =
      NumElements(input_def) * SizeOf(input_def.object_def.data_type);
}

absl::Status GlClBufferCopier::Convert(const TensorObject& input_obj,
                                       const TensorObject& output_obj) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSgl_interopDTcc mht_16(mht_16_v, 489, "", "./tensorflow/lite/delegates/gpu/cl/gl_interop.cc", "GlClBufferCopier::Convert");

  if (absl::holds_alternative<OpenGlBuffer>(input_obj)) {
    auto ssbo = absl::get_if<OpenGlBuffer>(&input_obj);
    auto cl_mem = absl::get_if<OpenClBuffer>(&output_obj);
    RETURN_IF_ERROR(
        TFLITE_GPU_CALL_GL(glBindBuffer, GL_SHADER_STORAGE_BUFFER, ssbo->id));
    void* ptr;
    RETURN_IF_ERROR(TFLITE_GPU_CALL_GL(glMapBufferRange, &ptr,
                                       GL_SHADER_STORAGE_BUFFER, 0,
                                       size_in_bytes_, GL_MAP_READ_BIT));
    RETURN_IF_ERROR(
        queue_->EnqueueWriteBuffer(cl_mem->memobj, size_in_bytes_, ptr));
    RETURN_IF_ERROR(
        TFLITE_GPU_CALL_GL(glUnmapBuffer, GL_SHADER_STORAGE_BUFFER));
  } else {
    auto cl_mem = absl::get_if<OpenClBuffer>(&input_obj);
    auto ssbo = absl::get_if<OpenGlBuffer>(&output_obj);
    RETURN_IF_ERROR(
        TFLITE_GPU_CALL_GL(glBindBuffer, GL_SHADER_STORAGE_BUFFER, ssbo->id));
    void* ptr;
    RETURN_IF_ERROR(TFLITE_GPU_CALL_GL(glMapBufferRange, &ptr,
                                       GL_SHADER_STORAGE_BUFFER, 0,
                                       size_in_bytes_, GL_MAP_WRITE_BIT));
    RETURN_IF_ERROR(
        queue_->EnqueueReadBuffer(cl_mem->memobj, size_in_bytes_, ptr));
    RETURN_IF_ERROR(
        TFLITE_GPU_CALL_GL(glUnmapBuffer, GL_SHADER_STORAGE_BUFFER));
  }
  return absl::OkStatus();
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
