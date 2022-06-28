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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSegl_environmentDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSegl_environmentDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSegl_environmentDTcc() {
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

#include "tensorflow/lite/delegates/gpu/gl/egl_environment.h"

#include "absl/memory/memory.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_call.h"
#include "tensorflow/lite/delegates/gpu/gl/request_gpu_info.h"

namespace tflite {
namespace gpu {
namespace gl {
namespace {

// TODO(akulik): detect power management event when all contexts are destroyed
// and OpenGL ES is reinitialized. See eglMakeCurrent

absl::Status InitDisplay(EGLDisplay* egl_display) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSegl_environmentDTcc mht_0(mht_0_v, 200, "", "./tensorflow/lite/delegates/gpu/gl/egl_environment.cc", "InitDisplay");

  RETURN_IF_ERROR(
      TFLITE_GPU_CALL_EGL(eglGetDisplay, egl_display, EGL_DEFAULT_DISPLAY));
  if (*egl_display == EGL_NO_DISPLAY) {
    return absl::UnavailableError("eglGetDisplay returned nullptr");
  }
  bool is_initialized;
  RETURN_IF_ERROR(TFLITE_GPU_CALL_EGL(eglInitialize, &is_initialized,
                                      *egl_display, nullptr, nullptr));
  if (!is_initialized) {
    return absl::InternalError("No EGL error, but eglInitialize failed");
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status EglEnvironment::NewEglEnvironment(
    std::unique_ptr<EglEnvironment>* egl_environment) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSegl_environmentDTcc mht_1(mht_1_v, 221, "", "./tensorflow/lite/delegates/gpu/gl/egl_environment.cc", "EglEnvironment::NewEglEnvironment");

  *egl_environment = absl::make_unique<EglEnvironment>();
  RETURN_IF_ERROR((*egl_environment)->Init());
  return absl::OkStatus();
}

EglEnvironment::~EglEnvironment() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSegl_environmentDTcc mht_2(mht_2_v, 230, "", "./tensorflow/lite/delegates/gpu/gl/egl_environment.cc", "EglEnvironment::~EglEnvironment");

  if (dummy_framebuffer_ != GL_INVALID_INDEX) {
    glDeleteFramebuffers(1, &dummy_framebuffer_);
  }
  if (dummy_texture_ != GL_INVALID_INDEX) {
    glDeleteTextures(1, &dummy_texture_);
  }
}

absl::Status EglEnvironment::Init() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSegl_environmentDTcc mht_3(mht_3_v, 242, "", "./tensorflow/lite/delegates/gpu/gl/egl_environment.cc", "EglEnvironment::Init");

  bool is_bound;
  RETURN_IF_ERROR(
      TFLITE_GPU_CALL_EGL(eglBindAPI, &is_bound, EGL_OPENGL_ES_API));
  if (!is_bound) {
    return absl::InternalError("No EGL error, but eglBindAPI failed");
  }

  // Re-use context and display if it was created on this thread.
  if (eglGetCurrentContext() != EGL_NO_CONTEXT) {
    display_ = eglGetCurrentDisplay();
    context_ =
        EglContext(eglGetCurrentContext(), display_, EGL_NO_CONFIG_KHR, false);
  } else {
    RETURN_IF_ERROR(InitDisplay(&display_));

    absl::Status status = InitConfiglessContext();
    if (!status.ok()) {
      status = InitSurfacelessContext();
    }
    if (!status.ok()) {
      status = InitPBufferContext();
    }
    if (!status.ok()) {
      return status;
    }
  }

  if (gpu_info_.vendor == GpuVendor::kUnknown) {
    RETURN_IF_ERROR(RequestGpuInfo(&gpu_info_));
  }
  // TODO(akulik): when do we need ForceSyncTurning?
  ForceSyncTurning();
  return absl::OkStatus();
}

absl::Status EglEnvironment::InitConfiglessContext() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSegl_environmentDTcc mht_4(mht_4_v, 281, "", "./tensorflow/lite/delegates/gpu/gl/egl_environment.cc", "EglEnvironment::InitConfiglessContext");

  RETURN_IF_ERROR(CreateConfiglessContext(display_, EGL_NO_CONTEXT, &context_));
  return context_.MakeCurrentSurfaceless();
}

absl::Status EglEnvironment::InitSurfacelessContext() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSegl_environmentDTcc mht_5(mht_5_v, 289, "", "./tensorflow/lite/delegates/gpu/gl/egl_environment.cc", "EglEnvironment::InitSurfacelessContext");

  RETURN_IF_ERROR(
      CreateSurfacelessContext(display_, EGL_NO_CONTEXT, &context_));
  RETURN_IF_ERROR(context_.MakeCurrentSurfaceless());

  // PowerVR support EGL_KHR_surfaceless_context, but glFenceSync crashes on
  // PowerVR when it is surface-less.
  RETURN_IF_ERROR(RequestGpuInfo(&gpu_info_));
  if (gpu_info_.IsPowerVR()) {
    return absl::UnavailableError(
        "Surface-less context is not properly supported on powervr.");
  }
  return absl::OkStatus();
}

absl::Status EglEnvironment::InitPBufferContext() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSegl_environmentDTcc mht_6(mht_6_v, 307, "", "./tensorflow/lite/delegates/gpu/gl/egl_environment.cc", "EglEnvironment::InitPBufferContext");

  RETURN_IF_ERROR(CreatePBufferContext(display_, EGL_NO_CONTEXT, &context_));
  RETURN_IF_ERROR(CreatePbufferRGBSurface(context_.config(), display_, 1, 1,
                                          &surface_read_));
  RETURN_IF_ERROR(CreatePbufferRGBSurface(context_.config(), display_, 1, 1,
                                          &surface_draw_));
  return context_.MakeCurrent(surface_read_.surface(), surface_draw_.surface());
}

void EglEnvironment::ForceSyncTurning() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSegl_environmentDTcc mht_7(mht_7_v, 319, "", "./tensorflow/lite/delegates/gpu/gl/egl_environment.cc", "EglEnvironment::ForceSyncTurning");

  glGenFramebuffers(1, &dummy_framebuffer_);
  glBindFramebuffer(GL_FRAMEBUFFER, dummy_framebuffer_);

  glGenTextures(1, &dummy_texture_);
  glBindTexture(GL_TEXTURE_2D, dummy_texture_);
  glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, 4, 4);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                         dummy_texture_, 0);

  GLenum draw_buffers[1] = {GL_COLOR_ATTACHMENT0};
  glDrawBuffers(1, draw_buffers);

  glViewport(0, 0, 4, 4);
  glClear(GL_COLOR_BUFFER_BIT);
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
