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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_bufferDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_bufferDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_bufferDTcc() {
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

#include "tensorflow/lite/delegates/gpu/gl/gl_buffer.h"

#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {
namespace gl {

absl::Status CopyBuffer(const GlBuffer& read_buffer,
                        const GlBuffer& write_buffer) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_bufferDTcc mht_0(mht_0_v, 194, "", "./tensorflow/lite/delegates/gpu/gl/gl_buffer.cc", "CopyBuffer");

  if (read_buffer.bytes_size() != write_buffer.bytes_size()) {
    return absl::InvalidArgumentError(
        "Read buffer does not match write buffer size.");
  }
  gl_buffer_internal::BufferBinder read_buffer_binder(GL_COPY_READ_BUFFER,
                                                      read_buffer.id());
  gl_buffer_internal::BufferBinder write_buffer_binder(GL_COPY_WRITE_BUFFER,
                                                       write_buffer.id());
  return TFLITE_GPU_CALL_GL(glCopyBufferSubData, GL_COPY_READ_BUFFER,
                            GL_COPY_WRITE_BUFFER, read_buffer.offset(),
                            write_buffer.offset(), read_buffer.bytes_size());
}

absl::Status GetSSBOSize(GLuint id, int64_t* size_bytes) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_bufferDTcc mht_1(mht_1_v, 211, "", "./tensorflow/lite/delegates/gpu/gl/gl_buffer.cc", "GetSSBOSize");

  GLuint prev_id;
  RETURN_IF_ERROR(TFLITE_GPU_CALL_GL(glGetIntegerv,
                                     GL_SHADER_STORAGE_BUFFER_BINDING,
                                     reinterpret_cast<GLint*>(&prev_id)));
  gl_buffer_internal::BufferBinder binder(GL_SHADER_STORAGE_BUFFER, id,
                                          prev_id);
  return TFLITE_GPU_CALL_GL(glGetBufferParameteri64v, GL_SHADER_STORAGE_BUFFER,
                            GL_BUFFER_SIZE, size_bytes);
}

GlBuffer::GlBuffer(GlBuffer&& buffer)
    : GlBuffer(buffer.target_, buffer.id_, buffer.bytes_size_, buffer.offset_,
               buffer.has_ownership_) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_bufferDTcc mht_2(mht_2_v, 227, "", "./tensorflow/lite/delegates/gpu/gl/gl_buffer.cc", "GlBuffer::GlBuffer");

  buffer.has_ownership_ = false;
}

GlBuffer& GlBuffer::operator=(GlBuffer&& buffer) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_bufferDTcc mht_3(mht_3_v, 234, "", "./tensorflow/lite/delegates/gpu/gl/gl_buffer.cc", "=");

  if (this != &buffer) {
    Invalidate();

    target_ = buffer.target_;
    bytes_size_ = buffer.bytes_size_;
    offset_ = buffer.offset_;
    has_ownership_ = buffer.has_ownership_;
    id_ = buffer.id_;
    buffer.has_ownership_ = false;
  }
  return *this;
}

GlBuffer::~GlBuffer() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_bufferDTcc mht_4(mht_4_v, 251, "", "./tensorflow/lite/delegates/gpu/gl/gl_buffer.cc", "GlBuffer::~GlBuffer");
 Invalidate(); }

void GlBuffer::Invalidate() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_bufferDTcc mht_5(mht_5_v, 256, "", "./tensorflow/lite/delegates/gpu/gl/gl_buffer.cc", "GlBuffer::Invalidate");

  if (has_ownership_ && id_ != GL_INVALID_INDEX) {
    TFLITE_GPU_CALL_GL(glDeleteBuffers, 1, &id_).IgnoreError();
    id_ = GL_INVALID_INDEX;
  }
}

absl::Status GlBuffer::BindToIndex(uint32_t index) const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_bufferDTcc mht_6(mht_6_v, 266, "", "./tensorflow/lite/delegates/gpu/gl/gl_buffer.cc", "GlBuffer::BindToIndex");

  return TFLITE_GPU_CALL_GL(glBindBufferRange, target_, index, id_, offset_,
                            bytes_size_);
}

absl::Status GlBuffer::MakeView(size_t offset, size_t bytes_size,
                                GlBuffer* gl_buffer) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_bufferDTcc mht_7(mht_7_v, 275, "", "./tensorflow/lite/delegates/gpu/gl/gl_buffer.cc", "GlBuffer::MakeView");

  if (offset + bytes_size > bytes_size_) {
    return absl::OutOfRangeError("GlBuffer view is out of range.");
  }
  *gl_buffer = GlBuffer(target_, id_, bytes_size, offset_ + offset,
                        /*has_ownership=*/false);
  return absl::OkStatus();
}

GlBuffer GlBuffer::MakeRef() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_bufferDTcc mht_8(mht_8_v, 287, "", "./tensorflow/lite/delegates/gpu/gl/gl_buffer.cc", "GlBuffer::MakeRef");

  return GlBuffer(target_, id_, bytes_size_, offset_,
                  /* has_ownership = */ false);
}

GlPersistentBuffer::GlPersistentBuffer(GLenum target, GLuint id,
                                       size_t bytes_size, size_t offset,
                                       bool has_ownership, void* data)
    : GlBuffer(target, id, bytes_size, offset, has_ownership), data_(data) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_bufferDTcc mht_9(mht_9_v, 298, "", "./tensorflow/lite/delegates/gpu/gl/gl_buffer.cc", "GlPersistentBuffer::GlPersistentBuffer");
}

GlPersistentBuffer::GlPersistentBuffer()
    : GlPersistentBuffer(GL_INVALID_ENUM, GL_INVALID_INDEX, 0, 0, false,
                         nullptr) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_bufferDTcc mht_10(mht_10_v, 305, "", "./tensorflow/lite/delegates/gpu/gl/gl_buffer.cc", "GlPersistentBuffer::GlPersistentBuffer");
}

GlPersistentBuffer::GlPersistentBuffer(GlPersistentBuffer&& buffer)
    : GlBuffer(std::move(buffer)), data_(buffer.data_) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_bufferDTcc mht_11(mht_11_v, 311, "", "./tensorflow/lite/delegates/gpu/gl/gl_buffer.cc", "GlPersistentBuffer::GlPersistentBuffer");
}

GlPersistentBuffer& GlPersistentBuffer::operator=(GlPersistentBuffer&& buffer) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_bufferDTcc mht_12(mht_12_v, 316, "", "./tensorflow/lite/delegates/gpu/gl/gl_buffer.cc", "=");

  if (this != &buffer) {
    data_ = buffer.data_;
    GlBuffer::operator=(std::move(buffer));
  }
  return *this;
}

GlPersistentBuffer::~GlPersistentBuffer() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_bufferDTcc mht_13(mht_13_v, 327, "", "./tensorflow/lite/delegates/gpu/gl/gl_buffer.cc", "GlPersistentBuffer::~GlPersistentBuffer");

  if (!data_) return;
  gl_buffer_internal::BufferBinder binder(GL_SHADER_STORAGE_BUFFER, id());
  glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
}

absl::Status CreatePersistentBuffer(size_t size,
                                    GlPersistentBuffer* gl_buffer) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_bufferDTcc mht_14(mht_14_v, 337, "", "./tensorflow/lite/delegates/gpu/gl/gl_buffer.cc", "CreatePersistentBuffer");

  PFNGLBUFFERSTORAGEEXTPROC glBufferStorageEXT = nullptr;
  glBufferStorageEXT = reinterpret_cast<PFNGLBUFFERSTORAGEEXTPROC>(
      eglGetProcAddress("glBufferStorageEXT"));
  if (!glBufferStorageEXT) {
    return absl::UnavailableError("glBufferStorageEXT is not supported");
  }
  gl_buffer_internal::BufferId id;
  gl_buffer_internal::BufferBinder binder(GL_SHADER_STORAGE_BUFFER, id.id());
  RETURN_IF_ERROR(TFLITE_GPU_CALL_GL(
      glBufferStorageEXT, GL_SHADER_STORAGE_BUFFER, size, nullptr,
      GL_MAP_COHERENT_BIT_EXT | GL_MAP_READ_BIT | GL_MAP_WRITE_BIT |
          GL_MAP_PERSISTENT_BIT_EXT));
  void* data = nullptr;
  RETURN_IF_ERROR(TFLITE_GPU_CALL_GL(
      glMapBufferRange, &data, GL_SHADER_STORAGE_BUFFER, 0, size,
      GL_MAP_READ_BIT | GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT_EXT));
  *gl_buffer = GlPersistentBuffer{
      GL_SHADER_STORAGE_BUFFER, id.Release(), size, 0, true, data};
  return absl::OkStatus();
}

namespace gl_buffer_internal {

BufferMapper::BufferMapper(GLenum target, size_t offset, size_t bytes,
                           GLbitfield access)
    : target_(target),
      data_(glMapBufferRange(target_, offset, bytes, access)) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_bufferDTcc mht_15(mht_15_v, 367, "", "./tensorflow/lite/delegates/gpu/gl/gl_buffer.cc", "BufferMapper::BufferMapper");
}

BufferMapper::~BufferMapper() {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_bufferDTcc mht_16(mht_16_v, 372, "", "./tensorflow/lite/delegates/gpu/gl/gl_buffer.cc", "BufferMapper::~BufferMapper");

  TFLITE_GPU_CALL_GL(glUnmapBuffer, target_).IgnoreError();
}

};  // namespace gl_buffer_internal

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
