/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_PJRT_TRACKED_TFRT_CPU_DEVICE_BUFFER_H_
#define TENSORFLOW_COMPILER_XLA_PJRT_TRACKED_TFRT_CPU_DEVICE_BUFFER_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStracked_tfrt_cpu_device_bufferDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStracked_tfrt_cpu_device_bufferDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStracked_tfrt_cpu_device_bufferDTh() {
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


#include <functional>
#include <memory>
#include <utility>

#include "absl/container/inlined_vector.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/cpu_function_runtime.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/mem.h"
#include "tfrt/host_context/async_value_ref.h"  // from @tf_runtime

namespace xla {

class MaybeOwningCpuMemory {
 public:
  MaybeOwningCpuMemory() = default;

  // Non-owning.
  explicit MaybeOwningCpuMemory(void* buf, size_t size)
      : buf_(buf), size_(size) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStracked_tfrt_cpu_device_bufferDTh mht_0(mht_0_v, 209, "", "./tensorflow/compiler/xla/pjrt/tracked_tfrt_cpu_device_buffer.h", "MaybeOwningCpuMemory");
}

  // Owning.
  using OwnedDataPtr =
      std::unique_ptr<uint8_t[], decltype(tensorflow::port::AlignedFree)*>;
  explicit MaybeOwningCpuMemory(OwnedDataPtr data, size_t size)
      : buf_(data.get()), data_(std::move(data)), size_(size) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStracked_tfrt_cpu_device_bufferDTh mht_1(mht_1_v, 218, "", "./tensorflow/compiler/xla/pjrt/tracked_tfrt_cpu_device_buffer.h", "MaybeOwningCpuMemory");
}

  // Move-only.
  MaybeOwningCpuMemory(MaybeOwningCpuMemory&&) = default;
  MaybeOwningCpuMemory& operator=(MaybeOwningCpuMemory&&) = default;
  MaybeOwningCpuMemory(const MaybeOwningCpuMemory&) = delete;
  MaybeOwningCpuMemory& operator=(const MaybeOwningCpuMemory&) = delete;

  // Owning.
  static StatusOr<std::shared_ptr<MaybeOwningCpuMemory>> AllocateShared(
      size_t size) {
    uint8_t* data = static_cast<uint8_t*>(tensorflow::port::AlignedMalloc(
        size, cpu_function_runtime::MinAlign()));
    if (!data) {
      return ResourceExhausted("Out of memory allocating %d bytes.", size);
    }
    return std::make_shared<MaybeOwningCpuMemory>(
        OwnedDataPtr{data, tensorflow::port::AlignedFree}, size);
  }

  void* data() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStracked_tfrt_cpu_device_bufferDTh mht_2(mht_2_v, 241, "", "./tensorflow/compiler/xla/pjrt/tracked_tfrt_cpu_device_buffer.h", "data");
 return buf_; }
  size_t size() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStracked_tfrt_cpu_device_bufferDTh mht_3(mht_3_v, 245, "", "./tensorflow/compiler/xla/pjrt/tracked_tfrt_cpu_device_buffer.h", "size");
 return size_; }
  bool owns_data() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStracked_tfrt_cpu_device_bufferDTh mht_4(mht_4_v, 249, "", "./tensorflow/compiler/xla/pjrt/tracked_tfrt_cpu_device_buffer.h", "owns_data");
 return data_ != nullptr; }

 private:
  void* buf_ = nullptr;                  // Non-owning data pointer.
  OwnedDataPtr data_ = {nullptr, free};  // Owning data pointer;
  size_t size_ = 0;                      // Size in number of bytes.
};

// tfrt::AsyncValueRef<CpuEvent> is used to indicate the completion of a CPU
// operation, e.g., data transfer or running a program.
struct CpuEvent {
  CpuEvent() = default;
};

// Class that represents CPU buffers. It optionally owns the buffers. It also
// tracks the definition and usage of the memory to allow for synchronized usage
// and deletion of CPU memory.
class TrackedTfrtCpuDeviceBuffer {
 public:
  // For non-tuple, takes a single buffer.
  // For tuple, takes the leaf buffers. Tuple index table created internally.
  // Nested tuple is not supported.
  TrackedTfrtCpuDeviceBuffer(
      bool is_tuple,
      absl::InlinedVector<std::shared_ptr<MaybeOwningCpuMemory>, 4> buffers,
      absl::InlinedVector<tfrt::AsyncValueRef<CpuEvent>, 4> definition_events,
      std::function<void()> on_delete_callback = nullptr);

  // Move-only.
  TrackedTfrtCpuDeviceBuffer(TrackedTfrtCpuDeviceBuffer&&) = default;
  TrackedTfrtCpuDeviceBuffer& operator=(TrackedTfrtCpuDeviceBuffer&&) = default;
  TrackedTfrtCpuDeviceBuffer(const TrackedTfrtCpuDeviceBuffer&) = delete;
  TrackedTfrtCpuDeviceBuffer& operator=(const TrackedTfrtCpuDeviceBuffer&) =
      delete;

  ~TrackedTfrtCpuDeviceBuffer();

  absl::Span<const std::shared_ptr<MaybeOwningCpuMemory>> Buffers() {
    return buffers_;
  }

  std::shared_ptr<MaybeOwningCpuMemory> Buffer(const ShapeIndex& shape_index);

  absl::Span<const tfrt::AsyncValueRef<CpuEvent>> DefinitionEvents() const {
    return definition_events_;
  }

  absl::Span<const tfrt::AsyncValueRef<CpuEvent>> UsageEvents() const {
    return usage_events_;
  }

  void AddUsageEvents(absl::Span<tfrt::AsyncValueRef<CpuEvent>> events);

  absl::InlinedVector<std::shared_ptr<MaybeOwningCpuMemory>, 4>
  ConsumeBuffers();

  // Return the usage events for the buffers. After
  // LockUseAndTransferUsageEvents is called, it is illegal to AddUsageEvent.
  absl::InlinedVector<tfrt::AsyncValueRef<CpuEvent>, 4>
  LockUseAndTransferUsageEvents();

  // Relinquishes ownership of the buffer's device memory, e.g., after the
  // buffer is passed to a computation that aliases its inputs to outputs.
  void ReleaseDeviceMemory();

 private:
  bool is_tuple_;
  absl::Mutex mu_;
  // If tuple, tuple index table is created and stored.
  std::shared_ptr<MaybeOwningCpuMemory> tuple_index_table_;
  // If non-tuple, `buffers_` contains 1 buffer; otherwise all leaf buffers.
  absl::InlinedVector<std::shared_ptr<MaybeOwningCpuMemory>, 4> buffers_;
  // Definition events are associated with CPU operations that write to the
  // buffers.
  absl::InlinedVector<tfrt::AsyncValueRef<CpuEvent>, 4> definition_events_;
  // Usage events are associated with CPU operations that read from the buffers.
  absl::InlinedVector<tfrt::AsyncValueRef<CpuEvent>, 4> usage_events_
      TF_GUARDED_BY(mu_);
  // A callback to call when the TrackedTfrtCpuDeviceBuffer is about to be
  // destroyed.
  std::function<void()> on_delete_callback_;
};
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PJRT_TRACKED_TFRT_CPU_DEVICE_BUFFER_H_
