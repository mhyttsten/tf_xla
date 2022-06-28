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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSegl_contextDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSegl_contextDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSegl_contextDTcc() {
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

#include "tensorflow/lite/delegates/gpu/gl/egl_context.h"

#include <cstring>

#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_call.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_errors.h"

namespace tflite {
namespace gpu {
namespace gl {
namespace {

absl::Status GetConfig(EGLDisplay display, const EGLint* attributes,
                       EGLConfig* config) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSegl_contextDTcc mht_0(mht_0_v, 199, "", "./tensorflow/lite/delegates/gpu/gl/egl_context.cc", "GetConfig");

  EGLint config_count;
  bool chosen = eglChooseConfig(display, attributes, config, 1, &config_count);
  RETURN_IF_ERROR(GetOpenGlErrors());
  if (!chosen || config_count == 0) {
    return absl::InternalError("No EGL error, but eglChooseConfig failed.");
  }
  return absl::OkStatus();
}

absl::Status CreateContext(EGLDisplay display, EGLContext shared_context,
                           EGLConfig config, EglContext* egl_context) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSegl_contextDTcc mht_1(mht_1_v, 213, "", "./tensorflow/lite/delegates/gpu/gl/egl_context.cc", "CreateContext");

  static const EGLint attributes[] = {EGL_CONTEXT_CLIENT_VERSION, 3,
#ifdef _DEBUG  // Add debugging bit
                                      EGL_CONTEXT_FLAGS_KHR,
                                      EGL_CONTEXT_OPENGL_DEBUG_BIT_KHR,
#endif
                                      EGL_NONE};
  EGLContext context =
      eglCreateContext(display, config, shared_context, attributes);
  RETURN_IF_ERROR(GetOpenGlErrors());
  if (context == EGL_NO_CONTEXT) {
    return absl::InternalError("No EGL error, but eglCreateContext failed.");
  }
  *egl_context = EglContext(context, display, config, true);
  return absl::OkStatus();
}

bool HasExtension(EGLDisplay display, const char* name) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSegl_contextDTcc mht_2(mht_2_v, 234, "", "./tensorflow/lite/delegates/gpu/gl/egl_context.cc", "HasExtension");

  return std::strstr(eglQueryString(display, EGL_EXTENSIONS), name);
}

}  // namespace

void EglContext::Invalidate() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSegl_contextDTcc mht_3(mht_3_v, 243, "", "./tensorflow/lite/delegates/gpu/gl/egl_context.cc", "EglContext::Invalidate");

  if (context_ != EGL_NO_CONTEXT) {
    if (has_ownership_) {
      eglMakeCurrent(display_, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
      eglDestroyContext(display_, context_);
    }
    context_ = EGL_NO_CONTEXT;
  }
  has_ownership_ = false;
}

EglContext::EglContext(EglContext&& other)
    : context_(other.context_),
      display_(other.display_),
      config_(other.config_),
      has_ownership_(other.has_ownership_) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSegl_contextDTcc mht_4(mht_4_v, 261, "", "./tensorflow/lite/delegates/gpu/gl/egl_context.cc", "EglContext::EglContext");

  other.context_ = EGL_NO_CONTEXT;
  other.has_ownership_ = false;
}

EglContext& EglContext::operator=(EglContext&& other) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSegl_contextDTcc mht_5(mht_5_v, 269, "", "./tensorflow/lite/delegates/gpu/gl/egl_context.cc", "=");

  if (this != &other) {
    Invalidate();
    using std::swap;
    swap(context_, other.context_);
    display_ = other.display_;
    config_ = other.config_;
    swap(has_ownership_, other.has_ownership_);
  }
  return *this;
}

absl::Status EglContext::MakeCurrent(EGLSurface read, EGLSurface write) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSegl_contextDTcc mht_6(mht_6_v, 284, "", "./tensorflow/lite/delegates/gpu/gl/egl_context.cc", "EglContext::MakeCurrent");

  bool is_made_current = eglMakeCurrent(display_, write, read, context_);
  RETURN_IF_ERROR(GetOpenGlErrors());
  if (!is_made_current) {
    return absl::InternalError("No EGL error, but eglMakeCurrent failed.");
  }
  return absl::OkStatus();
}

bool EglContext::IsCurrent() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSegl_contextDTcc mht_7(mht_7_v, 296, "", "./tensorflow/lite/delegates/gpu/gl/egl_context.cc", "EglContext::IsCurrent");

  return context_ == eglGetCurrentContext();
}

absl::Status CreateConfiglessContext(EGLDisplay display,
                                     EGLContext shared_context,
                                     EglContext* egl_context) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSegl_contextDTcc mht_8(mht_8_v, 305, "", "./tensorflow/lite/delegates/gpu/gl/egl_context.cc", "CreateConfiglessContext");

  if (!HasExtension(display, "EGL_KHR_no_config_context")) {
    return absl::UnavailableError("EGL_KHR_no_config_context not supported");
  }
  return CreateContext(display, shared_context, EGL_NO_CONFIG_KHR, egl_context);
}

absl::Status CreateSurfacelessContext(EGLDisplay display,
                                      EGLContext shared_context,
                                      EglContext* egl_context) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSegl_contextDTcc mht_9(mht_9_v, 317, "", "./tensorflow/lite/delegates/gpu/gl/egl_context.cc", "CreateSurfacelessContext");

  if (!HasExtension(display, "EGL_KHR_create_context")) {
    return absl::UnavailableError("EGL_KHR_create_context not supported");
  }
  if (!HasExtension(display, "EGL_KHR_surfaceless_context")) {
    return absl::UnavailableError("EGL_KHR_surfaceless_context not supported");
  }
  const EGLint attributes[] = {EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT_KHR,
                               EGL_NONE};
  EGLConfig config;
  RETURN_IF_ERROR(GetConfig(display, attributes, &config));
  return CreateContext(display, shared_context, config, egl_context);
}

absl::Status CreatePBufferContext(EGLDisplay display, EGLContext shared_context,
                                  EglContext* egl_context) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSegl_contextDTcc mht_10(mht_10_v, 335, "", "./tensorflow/lite/delegates/gpu/gl/egl_context.cc", "CreatePBufferContext");

  const EGLint attributes[] = {
      EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,     EGL_BIND_TO_TEXTURE_RGB,
      EGL_TRUE,         EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT_KHR,
      EGL_NONE};
  EGLConfig config;
  RETURN_IF_ERROR(GetConfig(display, attributes, &config));
  return CreateContext(display, shared_context, config, egl_context);
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
