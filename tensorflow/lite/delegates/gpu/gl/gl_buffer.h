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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_GL_GL_BUFFER_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_GL_GL_BUFFER_H_
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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_bufferDTh {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_bufferDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_bufferDTh() {
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


#include <cstring>
#include <functional>
#include <vector>

#include "absl/types/span.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_call.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_errors.h"
#include "tensorflow/lite/delegates/gpu/gl/portable_gl31.h"

namespace tflite {
namespace gpu {
namespace gl {

// Buffer is an RAII wrapper for OpenGL buffer object.
// See https://www.khronos.org/opengl/wiki/Buffer_Object for more information.
//
// Buffer is moveable but not copyable.
class GlBuffer {
 public:
  // @param has_ownership indicates that GlBuffer is responsible for
  // corresponding GL buffer deletion.
  GlBuffer(GLenum target, GLuint id, size_t bytes_size, size_t offset,
           bool has_ownership)
      : target_(target),
        id_(id),
        bytes_size_(bytes_size),
        offset_(offset),
        has_ownership_(has_ownership) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_bufferDTh mht_0(mht_0_v, 216, "", "./tensorflow/lite/delegates/gpu/gl/gl_buffer.h", "GlBuffer");
}

  // Creates invalid buffer.
  GlBuffer() : GlBuffer(GL_INVALID_ENUM, GL_INVALID_INDEX, 0, 0, false) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_bufferDTh mht_1(mht_1_v, 222, "", "./tensorflow/lite/delegates/gpu/gl/gl_buffer.h", "GlBuffer");
}

  // Move-only
  GlBuffer(GlBuffer&& buffer);
  GlBuffer& operator=(GlBuffer&& buffer);
  GlBuffer(const GlBuffer&) = delete;
  GlBuffer& operator=(const GlBuffer&) = delete;

  ~GlBuffer();

  // Reads data from buffer into CPU memory. Data should point to a region that
  // has at least bytes_size available.
  template <typename T>
  absl::Status Read(absl::Span<T> data) const;

  // Writes data to a buffer.
  template <typename T>
  absl::Status Write(absl::Span<const T> data);

  // Maps GPU memory to CPU address space and calls reader that may read from
  // that memory.
  template <typename T>
  absl::Status MappedRead(
      const std::function<absl::Status(absl::Span<const T>)>& reader) const;

  // Maps GPU memory to CPU address space and calls writer that may write into
  // that memory.
  template <typename T>
  absl::Status MappedWrite(
      const std::function<absl::Status(absl::Span<T>)>& writer);

  absl::Status MakeView(size_t offset, size_t bytes_size, GlBuffer* gl_buffer);

  // Makes a copy without ownership of the buffer.
  GlBuffer MakeRef();

  // Binds a buffer to an index.
  absl::Status BindToIndex(uint32_t index) const;

  // Releases the ownership of the buffer object.
  void Release() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_bufferDTh mht_2(mht_2_v, 265, "", "./tensorflow/lite/delegates/gpu/gl/gl_buffer.h", "Release");
 has_ownership_ = false; }

  size_t bytes_size() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_bufferDTh mht_3(mht_3_v, 270, "", "./tensorflow/lite/delegates/gpu/gl/gl_buffer.h", "bytes_size");
 return bytes_size_; }

  const GLenum target() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_bufferDTh mht_4(mht_4_v, 275, "", "./tensorflow/lite/delegates/gpu/gl/gl_buffer.h", "target");
 return target_; }

  const GLuint id() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_bufferDTh mht_5(mht_5_v, 280, "", "./tensorflow/lite/delegates/gpu/gl/gl_buffer.h", "id");
 return id_; }

  bool is_valid() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_bufferDTh mht_6(mht_6_v, 285, "", "./tensorflow/lite/delegates/gpu/gl/gl_buffer.h", "is_valid");
 return id_ != GL_INVALID_INDEX; }

  size_t offset() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_bufferDTh mht_7(mht_7_v, 290, "", "./tensorflow/lite/delegates/gpu/gl/gl_buffer.h", "offset");
 return offset_; }

  // @return true if this object actually owns corresponding GL buffer
  //         and manages it's lifetime.
  bool has_ownership() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_bufferDTh mht_8(mht_8_v, 297, "", "./tensorflow/lite/delegates/gpu/gl/gl_buffer.h", "has_ownership");
 return has_ownership_; }

 private:
  void Invalidate();

  GLenum target_;
  GLuint id_;
  size_t bytes_size_;
  size_t offset_;
  bool has_ownership_;
};

absl::Status CopyBuffer(const GlBuffer& read_buffer,
                        const GlBuffer& write_buffer);

absl::Status GetSSBOSize(GLuint id, int64_t* size_bytes);

// Creates new shader storage buffer that will be modified and used many
// times.
//
// See https://www.khronos.org/opengl/wiki/Shader_Storage_Buffer_Object for
// details.
template <typename T>
absl::Status CreateReadWriteShaderStorageBuffer(uint32_t num_elements,
                                                GlBuffer* gl_buffer);

// Creates new shader storage buffer that will be filled with data once which
// will be used many times.
template <typename T>
absl::Status CreateReadOnlyShaderStorageBuffer(absl::Span<const T> data,
                                               GlBuffer* gl_buffer);

// Adapts raw Buffer::Read method to read data into a vector.
template <typename T>
absl::Status AppendFromBuffer(const GlBuffer& buffer, std::vector<T>* data) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_bufferDTh mht_9(mht_9_v, 334, "", "./tensorflow/lite/delegates/gpu/gl/gl_buffer.h", "AppendFromBuffer");

  if (buffer.bytes_size() % sizeof(T) != 0) {
    return absl::InvalidArgumentError("Buffer is not aligned");
  }
  size_t num_elements = buffer.bytes_size() / sizeof(T);
  data->resize(data->size() + num_elements);
  return buffer.Read<T>(
      absl::MakeSpan(data->data() + data->size() - num_elements, num_elements));
}

// Persistent buffer provides CPU pointer to the buffer that is valid all the
// time. A user should properly synchronize the access to the buffer on CPU and
// GPU sides.
class GlPersistentBuffer : public GlBuffer {
 public:
  GlPersistentBuffer(GLenum target, GLuint id, size_t bytes_size, size_t offset,
                     bool has_ownership, void* data);
  GlPersistentBuffer();

  // Move-only
  GlPersistentBuffer(GlPersistentBuffer&& buffer);
  GlPersistentBuffer& operator=(GlPersistentBuffer&& buffer);
  GlPersistentBuffer(const GlPersistentBuffer&) = delete;
  GlPersistentBuffer& operator=(const GlPersistentBuffer&) = delete;

  ~GlPersistentBuffer();

  void* data() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_bufferDTh mht_10(mht_10_v, 364, "", "./tensorflow/lite/delegates/gpu/gl/gl_buffer.h", "data");
 return data_; }

 private:
  void* data_;
};

// Creates read-write persistent buffer with valid CPU pointer
absl::Status CreatePersistentBuffer(size_t size, GlPersistentBuffer* gl_buffer);

////////////////////////////////////////////////////////////////////////////////
// Implementation details are below.

namespace gl_buffer_internal {

// RAII for creating and/or owning buffer id.
class BufferId {
 public:
  BufferId() : id_(GL_INVALID_INDEX) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_bufferDTh mht_11(mht_11_v, 384, "", "./tensorflow/lite/delegates/gpu/gl/gl_buffer.h", "BufferId");

    TFLITE_GPU_CALL_GL(glGenBuffers, 1 /* number of buffers */, &id_)
        .IgnoreError();
    // only possible error here is when a number of buffers is negative.
  }

  explicit BufferId(GLuint id) : id_(id) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_bufferDTh mht_12(mht_12_v, 393, "", "./tensorflow/lite/delegates/gpu/gl/gl_buffer.h", "BufferId");
}

  ~BufferId() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_bufferDTh mht_13(mht_13_v, 398, "", "./tensorflow/lite/delegates/gpu/gl/gl_buffer.h", "~BufferId");

    if (id_ != GL_INVALID_INDEX) {
      TFLITE_GPU_CALL_GL(glDeleteBuffers, 1, &id_).IgnoreError();
    }
  }

  GLuint id() const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_bufferDTh mht_14(mht_14_v, 407, "", "./tensorflow/lite/delegates/gpu/gl/gl_buffer.h", "id");
 return id_; }

  GLuint Release() {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_bufferDTh mht_15(mht_15_v, 412, "", "./tensorflow/lite/delegates/gpu/gl/gl_buffer.h", "Release");

    GLuint id = GL_INVALID_INDEX;
    std::swap(id, id_);
    return id;
  }

 private:
  GLuint id_;
};

// RAII for binding and unbinding a buffer.
class BufferBinder {
 public:
  BufferBinder(GLenum target, GLuint id) : target_(target), prev_id_(0) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_bufferDTh mht_16(mht_16_v, 428, "", "./tensorflow/lite/delegates/gpu/gl/gl_buffer.h", "BufferBinder");

    TFLITE_GPU_CALL_GL(glBindBuffer, target_, id).IgnoreError();
  }

  BufferBinder(GLenum target, GLuint id, GLuint prev_id)
      : target_(target), prev_id_(prev_id) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_bufferDTh mht_17(mht_17_v, 436, "", "./tensorflow/lite/delegates/gpu/gl/gl_buffer.h", "BufferBinder");

    TFLITE_GPU_CALL_GL(glBindBuffer, target_, id).IgnoreError();
  }

  ~BufferBinder() {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_bufferDTh mht_18(mht_18_v, 443, "", "./tensorflow/lite/delegates/gpu/gl/gl_buffer.h", "~BufferBinder");

    TFLITE_GPU_CALL_GL(glBindBuffer, target_, prev_id_).IgnoreError();
  }

 private:
  const GLenum target_;
  GLuint prev_id_;
};

// RAII for mapping and unmapping a buffer.
class BufferMapper {
 public:
  BufferMapper(GLenum target, size_t offset, size_t bytes, GLbitfield access);

  ~BufferMapper();

  void* data() {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_bufferDTh mht_19(mht_19_v, 462, "", "./tensorflow/lite/delegates/gpu/gl/gl_buffer.h", "data");
 return data_; }

 private:
  const GLenum target_;
  void* data_;
};

}  // namespace gl_buffer_internal

template <typename T>
absl::Status CreateReadWriteShaderStorageBuffer(uint32_t num_elements,
                                                GlBuffer* gl_buffer) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_bufferDTh mht_20(mht_20_v, 476, "", "./tensorflow/lite/delegates/gpu/gl/gl_buffer.h", "CreateReadWriteShaderStorageBuffer");

  gl_buffer_internal::BufferId id;
  gl_buffer_internal::BufferBinder binder(GL_SHADER_STORAGE_BUFFER, id.id());
  // TODO(akulik): benchmark DYNAMIC vs STREAM buffer
  RETURN_IF_ERROR(TFLITE_GPU_CALL_GL(glBufferData, GL_SHADER_STORAGE_BUFFER,
                                     num_elements * sizeof(T), nullptr,
                                     GL_STREAM_COPY));
  *gl_buffer = GlBuffer{GL_SHADER_STORAGE_BUFFER, id.Release(),
                        num_elements * sizeof(T), 0, true};
  return absl::OkStatus();
}

template <typename T>
absl::Status CreateReadOnlyShaderStorageBuffer(absl::Span<const T> data,
                                               GlBuffer* gl_buffer) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_bufferDTh mht_21(mht_21_v, 493, "", "./tensorflow/lite/delegates/gpu/gl/gl_buffer.h", "CreateReadOnlyShaderStorageBuffer");

  gl_buffer_internal::BufferId id;
  gl_buffer_internal::BufferBinder binder(GL_SHADER_STORAGE_BUFFER, id.id());
  RETURN_IF_ERROR(TFLITE_GPU_CALL_GL(glBufferData, GL_SHADER_STORAGE_BUFFER,
                                     data.size() * sizeof(T), data.data(),
                                     GL_STATIC_READ));
  *gl_buffer = GlBuffer{GL_SHADER_STORAGE_BUFFER, id.Release(),
                        data.size() * sizeof(T), 0, true};
  return absl::OkStatus();
}

template <typename T>
absl::Status GlBuffer::Read(absl::Span<T> data) const {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_bufferDTh mht_22(mht_22_v, 508, "", "./tensorflow/lite/delegates/gpu/gl/gl_buffer.h", "GlBuffer::Read");

  if (data.size() * sizeof(T) < bytes_size()) {
    return absl::InvalidArgumentError(
        "Read from buffer failed. Destination data is shorter than buffer.");
  }
  // TODO(akulik): glCopyBufferSubData is actually available in ES 3.1, try it.
  return MappedRead<T>([this, data](absl::Span<const T> src) {
    std::memcpy(data.data(), src.data(), bytes_size());
    return absl::OkStatus();
  });
}

template <typename T>
absl::Status GlBuffer::Write(absl::Span<const T> data) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_bufferDTh mht_23(mht_23_v, 524, "", "./tensorflow/lite/delegates/gpu/gl/gl_buffer.h", "GlBuffer::Write");

  if (data.size() * sizeof(T) > bytes_size_) {
    return absl::InvalidArgumentError(
        "Write to buffer failed. Source data is larger than buffer.");
  }
  gl_buffer_internal::BufferBinder binder(target_, id_);
  return TFLITE_GPU_CALL_GL(glBufferSubData, target_, offset_, bytes_size_,
                            data.data());
}

template <typename T>
absl::Status GlBuffer::MappedRead(
    const std::function<absl::Status(absl::Span<const T> d)>& reader) const {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_bufferDTh mht_24(mht_24_v, 539, "", "./tensorflow/lite/delegates/gpu/gl/gl_buffer.h", "GlBuffer::MappedRead");

  if (bytes_size_ % sizeof(T) != 0) {
    return absl::InvalidArgumentError("Buffer is not aligned");
  }
  gl_buffer_internal::BufferBinder binder(target_, id_);
  gl_buffer_internal::BufferMapper mapper(target_, offset_, bytes_size_,
                                          GL_MAP_READ_BIT);
  if (!mapper.data()) {
    return GetOpenGlErrors();
  }
  return reader(absl::MakeSpan(reinterpret_cast<const T*>(mapper.data()),
                               bytes_size_ / sizeof(T)));
}

template <typename T>
absl::Status GlBuffer::MappedWrite(
    const std::function<absl::Status(absl::Span<T> d)>& writer) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_bufferDTh mht_25(mht_25_v, 558, "", "./tensorflow/lite/delegates/gpu/gl/gl_buffer.h", "GlBuffer::MappedWrite");

  if (bytes_size_ % sizeof(T) != 0) {
    return absl::InvalidArgumentError("Buffer is not aligned");
  }
  gl_buffer_internal::BufferBinder binder(target_, id_);
  gl_buffer_internal::BufferMapper mapper(target_, offset_, bytes_size_,
                                          GL_MAP_WRITE_BIT);
  if (!mapper.data()) {
    return GetOpenGlErrors();
  }
  return writer(absl::MakeSpan(reinterpret_cast<T*>(mapper.data()),
                               bytes_size_ / sizeof(T)));
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_GL_GL_BUFFER_H_
