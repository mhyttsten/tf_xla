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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_errorsDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_errorsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_errorsDTcc() {
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

#include "tensorflow/lite/delegates/gpu/gl/gl_errors.h"

#include <string>
#include <vector>

#include "absl/strings/str_join.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/gl/portable_egl.h"
#include "tensorflow/lite/delegates/gpu/gl/portable_gl31.h"

namespace tflite {
namespace gpu {
namespace gl {
namespace {

const char* ErrorToString(GLenum error) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_errorsDTcc mht_0(mht_0_v, 200, "", "./tensorflow/lite/delegates/gpu/gl/gl_errors.cc", "ErrorToString");

  switch (error) {
    case GL_INVALID_ENUM:
      return "[GL_INVALID_ENUM]: An unacceptable value is specified for an "
             "enumerated argument.";
    case GL_INVALID_VALUE:
      return "[GL_INVALID_VALUE]: A numeric argument is out of range.";
    case GL_INVALID_OPERATION:
      return "[GL_INVALID_OPERATION]: The specified operation is not allowed "
             "in the current state.";
    case GL_INVALID_FRAMEBUFFER_OPERATION:
      return "[GL_INVALID_FRAMEBUFFER_OPERATION]: The framebuffer object is "
             "not complete.";
    case GL_OUT_OF_MEMORY:
      return "[GL_OUT_OF_MEMORY]: There is not enough memory left to execute "
             "the command.";
  }
  return "[UNKNOWN_GL_ERROR]";
}

struct ErrorFormatter {
  void operator()(std::string* out, GLenum error) const {
    absl::StrAppend(out, ErrorToString(error));
  }
};

}  // namespace

// TODO(akulik): create new error space for GL error.

absl::Status GetOpenGlErrors() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_errorsDTcc mht_1(mht_1_v, 233, "", "./tensorflow/lite/delegates/gpu/gl/gl_errors.cc", "GetOpenGlErrors");

#ifdef __EMSCRIPTEN__
  // This check is not recommended on WebGL, since it will force a wait on the
  // GPU process.
  return absl::OkStatus();
#else
  auto error = glGetError();
  if (error == GL_NO_ERROR) {
    return absl::OkStatus();
  }
  auto error2 = glGetError();
  if (error2 == GL_NO_ERROR) {
    return absl::InternalError(ErrorToString(error));
  }
  std::vector<GLenum> errors = {error, error2};
  for (error = glGetError(); error != GL_NO_ERROR; error = glGetError()) {
    errors.push_back(error);
  }
  return absl::InternalError(absl::StrJoin(errors, ",", ErrorFormatter()));
#endif  // __EMSCRIPTEN__
}

absl::Status GetEglError() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_errorsDTcc mht_2(mht_2_v, 258, "", "./tensorflow/lite/delegates/gpu/gl/gl_errors.cc", "GetEglError");

  EGLint error = eglGetError();
  switch (error) {
    case EGL_SUCCESS:
      return absl::OkStatus();
    case EGL_NOT_INITIALIZED:
      return absl::InternalError(
          "EGL is not initialized, or could not be initialized, for the "
          "specified EGL display connection.");
    case EGL_BAD_ACCESS:
      return absl::InternalError(
          "EGL cannot access a requested resource (for example a context is "
          "bound in another thread).");
    case EGL_BAD_ALLOC:
      return absl::InternalError(
          "EGL failed to allocate resources for the requested operation.");
    case EGL_BAD_ATTRIBUTE:
      return absl::InternalError(
          "An unrecognized attribute or attribute value was passed in the "
          "attribute list.");
    case EGL_BAD_CONTEXT:
      return absl::InternalError(
          "An EGLContext argument does not name a valid EGL rendering "
          "context.");
    case EGL_BAD_CONFIG:
      return absl::InternalError(
          "An EGLConfig argument does not name a valid EGL frame buffer "
          "configuration.");
    case EGL_BAD_CURRENT_SURFACE:
      return absl::InternalError(
          "The current surface of the calling thread is a window, pixel buffer "
          "or pixmap that is no longer valid.");
    case EGL_BAD_DISPLAY:
      return absl::InternalError(
          "An EGLDisplay argument does not name a valid EGL display "
          "connection.");
    case EGL_BAD_SURFACE:
      return absl::InternalError(
          "An EGLSurface argument does not name a valid surface (window, pixel "
          "buffer or pixmap) configured for GL rendering.");
    case EGL_BAD_MATCH:
      return absl::InternalError(
          "Arguments are inconsistent (for example, a valid context requires "
          "buffers not supplied by a valid surface).");
    case EGL_BAD_PARAMETER:
      return absl::InternalError("One or more argument values are invalid.");
    case EGL_BAD_NATIVE_PIXMAP:
      return absl::InternalError(
          "A NativePixmapType argument does not refer to a valid native "
          "pixmap.");
    case EGL_BAD_NATIVE_WINDOW:
      return absl::InternalError(
          "A NativeWindowType argument does not refer to a valid native "
          "window.");
    case EGL_CONTEXT_LOST:
      return absl::InternalError(
          "A power management event has occurred. The application must destroy "
          "all contexts and reinitialize OpenGL ES state and objects to "
          "continue rendering.");
  }
  return absl::UnknownError("EGL error: " + std::to_string(error));
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
