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
class MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStracked_tfrt_cpu_device_bufferDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStracked_tfrt_cpu_device_bufferDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStracked_tfrt_cpu_device_bufferDTcc() {
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

#include "tensorflow/compiler/xla/pjrt/tracked_tfrt_cpu_device_buffer.h"

#include <atomic>
#include <functional>
#include <utility>

#include "absl/base/casts.h"
#include "absl/synchronization/mutex.h"
#include "tfrt/host_context/async_value_ref.h"  // from @tf_runtime

namespace xla {

TrackedTfrtCpuDeviceBuffer::TrackedTfrtCpuDeviceBuffer(
    bool is_tuple,
    absl::InlinedVector<std::shared_ptr<MaybeOwningCpuMemory>, 4> buffers,
    absl::InlinedVector<tfrt::AsyncValueRef<CpuEvent>, 4> definition_events,
    std::function<void()> on_delete_callback)
    : is_tuple_(is_tuple),
      buffers_(std::move(buffers)),
      definition_events_(std::move(definition_events)),
      on_delete_callback_(std::move(on_delete_callback)) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStracked_tfrt_cpu_device_bufferDTcc mht_0(mht_0_v, 205, "", "./tensorflow/compiler/xla/pjrt/tracked_tfrt_cpu_device_buffer.cc", "TrackedTfrtCpuDeviceBuffer::TrackedTfrtCpuDeviceBuffer");

  if (is_tuple) {
    size_t index_table_byte_size = buffers_.size() * sizeof(void*);
    // We assume tuple table allocations will not fail.
    tuple_index_table_ =
        MaybeOwningCpuMemory::AllocateShared(index_table_byte_size)
            .ValueOrDie();
    uintptr_t* index_table =
        reinterpret_cast<uintptr_t*>(tuple_index_table_->data());
    for (int i = 0; i < buffers_.size(); ++i) {
      index_table[i] = absl::bit_cast<uintptr_t>(buffers_[i]->data());
    }
  }
}

TrackedTfrtCpuDeviceBuffer::~TrackedTfrtCpuDeviceBuffer() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStracked_tfrt_cpu_device_bufferDTcc mht_1(mht_1_v, 223, "", "./tensorflow/compiler/xla/pjrt/tracked_tfrt_cpu_device_buffer.cc", "TrackedTfrtCpuDeviceBuffer::~TrackedTfrtCpuDeviceBuffer");

  ReleaseDeviceMemory();
  if (on_delete_callback_) {
    on_delete_callback_();
  }
}

std::shared_ptr<MaybeOwningCpuMemory> TrackedTfrtCpuDeviceBuffer::Buffer(
    const ShapeIndex& shape_index) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStracked_tfrt_cpu_device_bufferDTcc mht_2(mht_2_v, 234, "", "./tensorflow/compiler/xla/pjrt/tracked_tfrt_cpu_device_buffer.cc", "TrackedTfrtCpuDeviceBuffer::Buffer");

  if (shape_index.empty()) {
    // shape_index={}
    if (is_tuple_) return tuple_index_table_;
    return buffers_[0];
  }
  // shape_index={i}
  CHECK(is_tuple_);
  CHECK_EQ(shape_index.size(), 1) << "nested tuple not supported";
  return buffers_[shape_index[0]];
}

void TrackedTfrtCpuDeviceBuffer::AddUsageEvents(
    absl::Span<tfrt::AsyncValueRef<CpuEvent>> events) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStracked_tfrt_cpu_device_bufferDTcc mht_3(mht_3_v, 250, "", "./tensorflow/compiler/xla/pjrt/tracked_tfrt_cpu_device_buffer.cc", "TrackedTfrtCpuDeviceBuffer::AddUsageEvents");

  absl::MutexLock lock(&mu_);
  // Periodically remove available usage events to prevent memory blowup.
  if (usage_events_.size() >= 1024) {
    int i = 0;
    while (i < usage_events_.size()) {
      auto& event = usage_events_.at(i);
      if (event.IsAvailable()) {
        using std::swap;
        swap(event, usage_events_.back());
        usage_events_.pop_back();
        continue;
      }
      ++i;
    }
  }
  for (auto& ev : events) {
    usage_events_.push_back(std::move(ev));
  }
}

absl::InlinedVector<std::shared_ptr<MaybeOwningCpuMemory>, 4>
TrackedTfrtCpuDeviceBuffer::ConsumeBuffers() {
  return std::move(buffers_);
}

absl::InlinedVector<tfrt::AsyncValueRef<CpuEvent>, 4>
TrackedTfrtCpuDeviceBuffer::LockUseAndTransferUsageEvents() {
  absl::MutexLock lock(&mu_);
  return std::move(usage_events_);
}

void TrackedTfrtCpuDeviceBuffer::ReleaseDeviceMemory() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStracked_tfrt_cpu_device_bufferDTcc mht_4(mht_4_v, 285, "", "./tensorflow/compiler/xla/pjrt/tracked_tfrt_cpu_device_buffer.cc", "TrackedTfrtCpuDeviceBuffer::ReleaseDeviceMemory");

  tuple_index_table_.reset();
  buffers_.clear();
  definition_events_.clear();
  {
    absl::MutexLock lock(&mu_);
    usage_events_.clear();
  }
}

}  // namespace xla
