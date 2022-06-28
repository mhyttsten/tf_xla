/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_STREAM_EXECUTOR_TPU_NONCOPYABLE_BUFFER_H_
#define TENSORFLOW_STREAM_EXECUTOR_TPU_NONCOPYABLE_BUFFER_H_
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
class MHTracer_DTPStensorflowPSstream_executorPStpuPSnoncopyable_bufferDTh {
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
   MHTracer_DTPStensorflowPSstream_executorPStpuPSnoncopyable_bufferDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSstream_executorPStpuPSnoncopyable_bufferDTh() {
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


#include <memory>

#include "absl/base/casts.h"
#include "absl/functional/function_ref.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mem.h"

namespace tensorflow {
namespace tpu {

using BufferDeallocator = std::function<void(void*)>;
using OwnedDataPtr = std::unique_ptr<uint8_t[], BufferDeallocator>;
using BufferAllocator = absl::FunctionRef<OwnedDataPtr(size_t)>;

inline OwnedDataPtr DefaultAllocator(size_t size) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPSnoncopyable_bufferDTh mht_0(mht_0_v, 204, "", "./tensorflow/stream_executor/tpu/noncopyable_buffer.h", "DefaultAllocator");

  return {static_cast<uint8_t*>(malloc(size)), free};
}

// Uncopyable buffer type with optional ownership of the underlying data. If
// data is not owned then ensuring lifetime of the data exceeds the lifetime of
// the buffer is the responsibility of the user.
class NoncopyableBuffer {
 public:
  NoncopyableBuffer() = default;

  // Allocate an owning buffer without initializing the data. Useful when it
  // will be filled by a subsequent function and want to avoid initialization
  // cost. Size is specified in number of bytes.
  explicit NoncopyableBuffer(size_t size,
                             BufferAllocator allocator = DefaultAllocator)
      : data_(allocator(size)), buf_(data_.get()), size_(size) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPSnoncopyable_bufferDTh mht_1(mht_1_v, 223, "", "./tensorflow/stream_executor/tpu/noncopyable_buffer.h", "NoncopyableBuffer");
}

  // Allocates an owning buffer and initializes it with the specified data. Size
  // is specified in number of uint32's.
  NoncopyableBuffer(size_t size_in_u32s, absl::optional<uint32_t> value,
                    BufferAllocator allocator = DefaultAllocator)
      : NoncopyableBuffer(size_in_u32s * sizeof(uint32_t), allocator) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPSnoncopyable_bufferDTh mht_2(mht_2_v, 232, "", "./tensorflow/stream_executor/tpu/noncopyable_buffer.h", "NoncopyableBuffer");

#ifndef MEMORY_SANITIZER
    if (!value.has_value()) {
      return;
    }
#endif
    uint32_t* data_u32 = reinterpret_cast<uint32_t*>(data_.get());
    uint32_t v = value.value_or(0);
    for (uint32_t *p = data_u32, *e = data_u32 + size_in_u32s; p < e; ++p) {
      *p = v;
    }
  }

  // Directly use buf pointer without copying it to owning data_. This delays
  // the memcpy until mutable access is requested. "buf" is not owned by this
  // data structure, so it is the user's duty to ensure the live range of "buf"
  // is longer than this data structure.
  NoncopyableBuffer(const uint8_t* buf, size_t size)  // Size is in uint8's.
      : buf_(buf), size_(size) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPSnoncopyable_bufferDTh mht_3(mht_3_v, 253, "", "./tensorflow/stream_executor/tpu/noncopyable_buffer.h", "NoncopyableBuffer");
}
  NoncopyableBuffer(const uint32_t* buf,
                    size_t size_in_u32s)  // Size is in uint32_t's.
      : buf_(buf), size_(size_in_u32s * sizeof(uint32_t)) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPSnoncopyable_bufferDTh mht_4(mht_4_v, 259, "", "./tensorflow/stream_executor/tpu/noncopyable_buffer.h", "NoncopyableBuffer");
}

  NoncopyableBuffer(const NoncopyableBuffer&) = delete;
  NoncopyableBuffer(NoncopyableBuffer&&) = default;

  NoncopyableBuffer& operator=(const NoncopyableBuffer&) = delete;
  NoncopyableBuffer& operator=(NoncopyableBuffer&&) = default;

  // Ensure that the buffer owns the data and returns a mutable view into the
  // owned data for modification.
  template <typename T>
  absl::Span<T> mutable_data() {
    static_assert(std::is_arithmetic<T>::value, "Must be arithmetic type.");
    EnsureDataOwned();
    DCHECK_EQ(size_ % sizeof(T), 0);
    return absl::Span<T>(reinterpret_cast<T*>(data_.get()), size_ / sizeof(T));
  }

  template <typename T>
  absl::Span<const T> const_data() const {
    static_assert(std::is_arithmetic<T>::value, "Must be arithmetic type.");
    DCHECK_EQ(size_ % sizeof(T), 0);
    return absl::Span<const T>(static_cast<const T*>(buf_), size_ / sizeof(T));
  }
  // Clone the content to a given buffer.
  void CloneTo(void* buf) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPSnoncopyable_bufferDTh mht_5(mht_5_v, 287, "", "./tensorflow/stream_executor/tpu/noncopyable_buffer.h", "CloneTo");
 memcpy(buf, buf_, size_); }

  // Return true if data is owned by this buffer (have been copied to `data_`).
  bool owns_data() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPSnoncopyable_bufferDTh mht_6(mht_6_v, 293, "", "./tensorflow/stream_executor/tpu/noncopyable_buffer.h", "owns_data");
 return data_ != nullptr; }

  // Returns a copy of the object that owns its buffer.
  NoncopyableBuffer Clone(size_t alignment = 1) const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPSnoncopyable_bufferDTh mht_7(mht_7_v, 299, "", "./tensorflow/stream_executor/tpu/noncopyable_buffer.h", "Clone");

    auto clone = alignment <= 1
                     ? NoncopyableBuffer(size_)
                     : NoncopyableBuffer(AlignedAlloc(size_, alignment), size_);
    memcpy(clone.data_.get(), buf_, size_);
    return clone;
  }

  // Ensure that the buffer owns the data.
  void EnsureDataOwned() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPSnoncopyable_bufferDTh mht_8(mht_8_v, 311, "", "./tensorflow/stream_executor/tpu/noncopyable_buffer.h", "EnsureDataOwned");

    if (data_ == nullptr) {
      data_ = OwnedDataPtr(static_cast<uint8_t*>(malloc(size_)), free);
      memcpy(data_.get(), buf_, size_);
      buf_ = data_.get();
    }
  }

 private:
  NoncopyableBuffer(OwnedDataPtr data, size_t size)
      : data_(std::move(data)), buf_(data_.get()), size_(size) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPSnoncopyable_bufferDTh mht_9(mht_9_v, 324, "", "./tensorflow/stream_executor/tpu/noncopyable_buffer.h", "NoncopyableBuffer");
}

  static OwnedDataPtr AlignedAlloc(size_t size, size_t alignment) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPSnoncopyable_bufferDTh mht_10(mht_10_v, 329, "", "./tensorflow/stream_executor/tpu/noncopyable_buffer.h", "AlignedAlloc");

    return OwnedDataPtr(
        static_cast<uint8_t*>(port::AlignedMalloc(size, alignment)),
        port::AlignedFree);
  }
  // If data_ != nullptr then buf_ == data_.get()
  OwnedDataPtr data_{nullptr, free};  // Owning data pointer.
  const void* buf_;                   // Non-owning data pointer.
  size_t size_;                       // Size in number of bytes.
};

}  // namespace tpu
}  // namespace tensorflow

#endif  // TENSORFLOW_STREAM_EXECUTOR_TPU_NONCOPYABLE_BUFFER_H_
