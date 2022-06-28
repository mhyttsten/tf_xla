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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSegl_syncDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSegl_syncDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSegl_syncDTcc() {
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

#include "tensorflow/lite/delegates/gpu/cl/egl_sync.h"

#include "tensorflow/lite/delegates/gpu/gl/gl_call.h"

namespace tflite {
namespace gpu {
namespace cl {

namespace {

bool HasExtension(EGLDisplay display, const char* extension) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("extension: \"" + (extension == nullptr ? std::string("nullptr") : std::string((char*)extension)) + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSegl_syncDTcc mht_0(mht_0_v, 196, "", "./tensorflow/lite/delegates/gpu/cl/egl_sync.cc", "HasExtension");

  const char* extensions = eglQueryString(display, EGL_EXTENSIONS);
  return extensions && std::strstr(extensions, extension);
}

absl::Status IsEglFenceSyncSupported(EGLDisplay display) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSegl_syncDTcc mht_1(mht_1_v, 204, "", "./tensorflow/lite/delegates/gpu/cl/egl_sync.cc", "IsEglFenceSyncSupported");

  static bool supported = HasExtension(display, "EGL_KHR_fence_sync");
  if (supported) {
    return absl::OkStatus();
  }
  return absl::InternalError("Not supported: EGL_KHR_fence_sync");
}

absl::Status IsEglWaitSyncSupported(EGLDisplay display) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSegl_syncDTcc mht_2(mht_2_v, 215, "", "./tensorflow/lite/delegates/gpu/cl/egl_sync.cc", "IsEglWaitSyncSupported");

  static bool supported = HasExtension(display, "EGL_KHR_wait_sync");
  if (supported) {
    return absl::OkStatus();
  }
  return absl::InternalError("Not supported: EGL_KHR_wait_sync");
}

}  // anonymous namespace

absl::Status EglSync::NewFence(EGLDisplay display, EglSync* sync) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSegl_syncDTcc mht_3(mht_3_v, 228, "", "./tensorflow/lite/delegates/gpu/cl/egl_sync.cc", "EglSync::NewFence");

  RETURN_IF_ERROR(IsEglFenceSyncSupported(display));
  static auto* egl_create_sync_khr =
      reinterpret_cast<decltype(&eglCreateSyncKHR)>(
          eglGetProcAddress("eglCreateSyncKHR"));
  if (egl_create_sync_khr == nullptr) {
    // Needs extension: EGL_KHR_fence_sync (EGL) / GL_OES_EGL_sync (OpenGL ES).
    return absl::InternalError(
        "Not supported / bad EGL implementation: eglCreateSyncKHR.");
  }
  EGLSyncKHR egl_sync;
  RETURN_IF_ERROR(TFLITE_GPU_CALL_EGL(*egl_create_sync_khr, &egl_sync, display,
                                      EGL_SYNC_FENCE_KHR, nullptr));
  if (egl_sync == EGL_NO_SYNC_KHR) {
    return absl::InternalError("Returned empty KHR EGL sync");
  }
  *sync = EglSync(display, egl_sync);
  return absl::OkStatus();
}

EglSync& EglSync::operator=(EglSync&& sync) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSegl_syncDTcc mht_4(mht_4_v, 251, "", "./tensorflow/lite/delegates/gpu/cl/egl_sync.cc", "=");

  if (this != &sync) {
    Invalidate();
    std::swap(sync_, sync.sync_);
    display_ = sync.display_;
  }
  return *this;
}

void EglSync::Invalidate() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSegl_syncDTcc mht_5(mht_5_v, 263, "", "./tensorflow/lite/delegates/gpu/cl/egl_sync.cc", "EglSync::Invalidate");

  if (sync_ != EGL_NO_SYNC_KHR) {
    static auto* egl_destroy_sync_khr =
        reinterpret_cast<decltype(&eglDestroySyncKHR)>(
            eglGetProcAddress("eglDestroySyncKHR"));
    // Needs extension: EGL_KHR_fence_sync (EGL) / GL_OES_EGL_sync (OpenGL ES).
    if (IsEglFenceSyncSupported(display_).ok() && egl_destroy_sync_khr) {
      // Note: we're doing nothing when the function pointer is nullptr, or the
      // call returns EGL_FALSE.
      (*egl_destroy_sync_khr)(display_, sync_);
    }
    sync_ = EGL_NO_SYNC_KHR;
  }
}

absl::Status EglSync::ServerWait() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSegl_syncDTcc mht_6(mht_6_v, 281, "", "./tensorflow/lite/delegates/gpu/cl/egl_sync.cc", "EglSync::ServerWait");

  RETURN_IF_ERROR(IsEglWaitSyncSupported(display_));
  static auto* egl_wait_sync_khr = reinterpret_cast<decltype(&eglWaitSyncKHR)>(
      eglGetProcAddress("eglWaitSyncKHR"));
  if (egl_wait_sync_khr == nullptr) {
    // Needs extension: EGL_KHR_wait_sync
    return absl::InternalError("Not supported: eglWaitSyncKHR.");
  }
  EGLint result;
  RETURN_IF_ERROR(
      TFLITE_GPU_CALL_EGL(*egl_wait_sync_khr, &result, display_, sync_, 0));
  return result == EGL_TRUE ? absl::OkStatus()
                            : absl::InternalError("eglWaitSync failed");
}

absl::Status EglSync::ClientWait() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSegl_syncDTcc mht_7(mht_7_v, 299, "", "./tensorflow/lite/delegates/gpu/cl/egl_sync.cc", "EglSync::ClientWait");

  RETURN_IF_ERROR(IsEglFenceSyncSupported(display_));
  static auto* egl_client_wait_sync_khr =
      reinterpret_cast<decltype(&eglClientWaitSyncKHR)>(
          eglGetProcAddress("eglClientWaitSyncKHR"));
  if (egl_client_wait_sync_khr == nullptr) {
    // Needs extension: EGL_KHR_fence_sync (EGL) / GL_OES_EGL_sync (OpenGL ES).
    return absl::InternalError("Not supported: eglClientWaitSyncKHR.");
  }
  EGLint result;
  // TODO(akulik): make it active wait for better performance
  RETURN_IF_ERROR(
      TFLITE_GPU_CALL_EGL(*egl_client_wait_sync_khr, &result, display_, sync_,
                          EGL_SYNC_FLUSH_COMMANDS_BIT_KHR, EGL_FOREVER_KHR));
  return result == EGL_CONDITION_SATISFIED_KHR
             ? absl::OkStatus()
             : absl::InternalError("eglClientWaitSync failed");
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
