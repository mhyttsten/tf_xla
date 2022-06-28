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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_syncDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_syncDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_syncDTcc() {
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

#include "tensorflow/lite/delegates/gpu/gl/gl_sync.h"

#ifdef __ARM_ACLE
#include <arm_acle.h>
#endif  // __ARM_ACLE

#include "tensorflow/lite/delegates/gpu/gl/gl_errors.h"

namespace tflite {
namespace gpu {
namespace gl {

absl::Status GlSyncWait() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_syncDTcc mht_0(mht_0_v, 197, "", "./tensorflow/lite/delegates/gpu/gl/gl_sync.cc", "GlSyncWait");

  GlSync sync;
  RETURN_IF_ERROR(GlSync::NewSync(&sync));
  // Flush sync and loop afterwards without it.
  GLenum status = glClientWaitSync(sync.sync(), GL_SYNC_FLUSH_COMMANDS_BIT,
                                   /* timeout ns = */ 0);
  while (true) {
    switch (status) {
      case GL_TIMEOUT_EXPIRED:
        break;
      case GL_CONDITION_SATISFIED:
      case GL_ALREADY_SIGNALED:
        return absl::OkStatus();
      case GL_WAIT_FAILED:
        return GetOpenGlErrors();
    }
    status = glClientWaitSync(sync.sync(), 0, /* timeout ns = */ 10000000);
  }
  return absl::OkStatus();
}

absl::Status GlActiveSyncWait() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_syncDTcc mht_1(mht_1_v, 221, "", "./tensorflow/lite/delegates/gpu/gl/gl_sync.cc", "GlActiveSyncWait");

  GlSync sync;
  RETURN_IF_ERROR(GlSync::NewSync(&sync));
  // Since creating a Sync object is itself a GL command it *must* be flushed.
  // Otherwise glGetSynciv may never succeed. Perform a flush with
  // glClientWaitSync call.
  GLenum status = glClientWaitSync(sync.sync(), GL_SYNC_FLUSH_COMMANDS_BIT,
                                   /* timeout ns = */ 0);
  switch (status) {
    case GL_TIMEOUT_EXPIRED:
      break;
    case GL_CONDITION_SATISFIED:
    case GL_ALREADY_SIGNALED:
      return absl::OkStatus();
    case GL_WAIT_FAILED:
      return GetOpenGlErrors();
  }

  // Start active loop.
  GLint result = GL_UNSIGNALED;
  while (true) {
    glGetSynciv(sync.sync(), GL_SYNC_STATUS, sizeof(GLint), nullptr, &result);
    if (result == GL_SIGNALED) {
      return absl::OkStatus();
    }
#ifdef __ARM_ACLE
    // Try to save CPU power by yielding CPU to another thread.
    __yield();
#endif
  }
}

absl::Status GlShaderSync::NewSync(GlShaderSync* gl_sync) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_syncDTcc mht_2(mht_2_v, 256, "", "./tensorflow/lite/delegates/gpu/gl/gl_sync.cc", "GlShaderSync::NewSync");

  GlShaderSync sync;
  RETURN_IF_ERROR(CreatePersistentBuffer(sizeof(int), &sync.flag_buffer_));
  static const std::string* kCode = new std::string(R"(#version 310 es
  layout(local_size_x = 1, local_size_y = 1) in;
  layout(std430) buffer;
  layout(binding = 0) buffer Output {
    int elements[];
  } output_data;
  void main() {
    output_data.elements[0] = 1;
  })");
  GlShader shader;
  RETURN_IF_ERROR(GlShader::CompileShader(GL_COMPUTE_SHADER, *kCode, &shader));
  RETURN_IF_ERROR(GlProgram::CreateWithShader(shader, &sync.flag_program_));
  *gl_sync = std::move(sync);
  return absl::OkStatus();
}

// How it works: GPU writes a buffer and CPU checks the buffer value to be
// changed. The buffer is accessible for writing by GPU and reading by CPU
// simultaneously - persistent buffer or buffer across shild context can be used
// for that.
absl::Status GlShaderSync::Wait() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_syncDTcc mht_3(mht_3_v, 282, "", "./tensorflow/lite/delegates/gpu/gl/gl_sync.cc", "GlShaderSync::Wait");

  if (!flag_buffer_.is_valid()) {
    return absl::UnavailableError("GlShaderSync is not initialized.");
  }
  RETURN_IF_ERROR(flag_buffer_.BindToIndex(0));
  volatile int* flag_ptr_ = reinterpret_cast<int*>(flag_buffer_.data());
  *flag_ptr_ = 0;
  RETURN_IF_ERROR(flag_program_.Dispatch({1, 1, 1}));
  // glFlush must be called to upload GPU task. Adreno won't start executing
  // the task without glFlush.
  glFlush();
  // Wait for the value is being updated by the shader.
  while (*flag_ptr_ != 1) {
  }
  return absl::OkStatus();
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
