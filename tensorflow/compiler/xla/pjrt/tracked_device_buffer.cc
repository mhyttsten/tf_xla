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
class MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStracked_device_bufferDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStracked_device_bufferDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStracked_device_bufferDTcc() {
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

#include "tensorflow/compiler/xla/pjrt/tracked_device_buffer.h"

#include <atomic>
#include <iterator>
#include <memory>

#include "absl/synchronization/mutex.h"
#include "tensorflow/compiler/xla/pjrt/local_device_state.h"
#include "tensorflow/compiler/xla/service/shaped_buffer.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/device_memory_allocator.h"
#include "tensorflow/stream_executor/event.h"
#include "tensorflow/stream_executor/stream.h"

namespace xla {

void BufferSequencingEvent::SetSequencingEvent(EventPool::Handle event,
                                               se::Stream* stream) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStracked_device_bufferDTcc mht_0(mht_0_v, 203, "", "./tensorflow/compiler/xla/pjrt/tracked_device_buffer.cc", "BufferSequencingEvent::SetSequencingEvent");

  absl::MutexLock lock(&mu_);
  CHECK(!event_.event());
  event_ = std::move(event);
  CHECK(streams_defined_on_.empty());
  streams_defined_on_.push_back(stream);
  sequence_number_.store(event_.sequence_number(), std::memory_order_seq_cst);
}

bool BufferSequencingEvent::EventHasBeenRecorded() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStracked_device_bufferDTcc mht_1(mht_1_v, 215, "", "./tensorflow/compiler/xla/pjrt/tracked_device_buffer.cc", "BufferSequencingEvent::EventHasBeenRecorded");

  return event_.event() != nullptr;
}

uint64_t BufferSequencingEvent::sequence_number() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStracked_device_bufferDTcc mht_2(mht_2_v, 222, "", "./tensorflow/compiler/xla/pjrt/tracked_device_buffer.cc", "BufferSequencingEvent::sequence_number");

  uint64_t seq = sequence_number_.load(std::memory_order_seq_cst);
  CHECK_NE(seq, 0);
  return seq;
}

void BufferSequencingEvent::WaitForEventOnStream(se::Stream* stream) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStracked_device_bufferDTcc mht_3(mht_3_v, 231, "", "./tensorflow/compiler/xla/pjrt/tracked_device_buffer.cc", "BufferSequencingEvent::WaitForEventOnStream");

  absl::MutexLock lock(&mu_);

  // We cannot wait for an event until ThenRecordEvent has been called; on GPU
  // newly created events are deemed to have already happened past.
  mu_.Await(
      absl::Condition(this, &BufferSequencingEvent::EventHasBeenRecorded));

  // The set of defined streams is expected to be very small indeed (usually
  // 1-2), so a simple linear scan should be fast enough.
  if (std::find(streams_defined_on_.begin(), streams_defined_on_.end(),
                stream) != streams_defined_on_.end()) {
    // stream is in streams_defined_on_; it doesn't need to be waited on.
    return;
  }

  stream->ThenWaitFor(event_.event());
  streams_defined_on_.push_back(stream);
}

bool BufferSequencingEvent::DefinedOn(se::Stream* stream) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStracked_device_bufferDTcc mht_4(mht_4_v, 254, "", "./tensorflow/compiler/xla/pjrt/tracked_device_buffer.cc", "BufferSequencingEvent::DefinedOn");

  absl::MutexLock lock(&mu_);

  // We cannot wait for an event until ThenRecordEvent has been called; on GPU
  // newly created events are deemed to have already happened past.
  mu_.Await(
      absl::Condition(this, &BufferSequencingEvent::EventHasBeenRecorded));

  // The set of defined streams is expected to be very small indeed (usually
  // 1-2), so a simple linear scan should be fast enough.
  return std::find(streams_defined_on_.begin(), streams_defined_on_.end(),
                   stream) != streams_defined_on_.end();
}

bool BufferSequencingEvent::IsComplete() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStracked_device_bufferDTcc mht_5(mht_5_v, 271, "", "./tensorflow/compiler/xla/pjrt/tracked_device_buffer.cc", "BufferSequencingEvent::IsComplete");

  absl::MutexLock lock(&mu_);

  // We cannot wait for an event until ThenRecordEvent has been called; on
  // GPU newly created events are deemed to have already happened past.
  mu_.Await(
      absl::Condition(this, &BufferSequencingEvent::EventHasBeenRecorded));

  return event_.event()->PollForStatus() == se::Event::Status::kComplete;
}

/* static */ std::shared_ptr<TrackedDeviceBuffer>
TrackedDeviceBuffer::FromScopedShapedBuffer(
    ScopedShapedBuffer* shaped_buffer,
    absl::Span<const std::shared_ptr<BufferSequencingEvent>>
        definition_events) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStracked_device_bufferDTcc mht_6(mht_6_v, 289, "", "./tensorflow/compiler/xla/pjrt/tracked_device_buffer.cc", "TrackedDeviceBuffer::FromScopedShapedBuffer");

  ShapeTree<se::DeviceMemoryBase>::iterator iterator =
      shaped_buffer->buffers().begin();
  std::vector<se::DeviceMemoryBase> buffers;
  buffers.reserve(1);

  ShapeUtil::ForEachSubshape(
      shaped_buffer->on_device_shape(), [&](const Shape&, const ShapeIndex&) {
        CHECK(iterator != shaped_buffer->buffers().end());
        buffers.push_back(iterator->second);
        iterator->second = se::DeviceMemoryBase();
        ++iterator;
      });
  CHECK(iterator == shaped_buffer->buffers().end());
  return std::make_shared<TrackedDeviceBuffer>(
      shaped_buffer->memory_allocator(), shaped_buffer->device_ordinal(),
      absl::Span<se::DeviceMemoryBase>(buffers), definition_events,
      /*on_delete_callback=*/nullptr);
}

ShapedBuffer TrackedDeviceBuffer::AsShapedBuffer(
    const Shape& on_device_shape) const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStracked_device_bufferDTcc mht_7(mht_7_v, 313, "", "./tensorflow/compiler/xla/pjrt/tracked_device_buffer.cc", "TrackedDeviceBuffer::AsShapedBuffer");

  ShapedBuffer shaped_buffer(on_device_shape, device_ordinal_);
  ShapeTree<se::DeviceMemoryBase>::iterator iterator =
      shaped_buffer.buffers().begin();
  for (const se::DeviceMemoryBase& buf : device_memory_) {
    CHECK(iterator != shaped_buffer.buffers().end());
    iterator->second = buf;
    ++iterator;
  }
  CHECK(iterator == shaped_buffer.buffers().end());
  return shaped_buffer;
}

// See comment on ExecutionInput in xla/service/executable.h to understand
// the meaning of owned/unowned in that class.

void TrackedDeviceBuffer::AddToInputAsImmutable(
    ShapeTree<MaybeOwningDeviceMemory>::iterator* iterator,
    const ShapeTree<MaybeOwningDeviceMemory>::iterator& end) const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStracked_device_bufferDTcc mht_8(mht_8_v, 334, "", "./tensorflow/compiler/xla/pjrt/tracked_device_buffer.cc", "TrackedDeviceBuffer::AddToInputAsImmutable");

  for (const se::DeviceMemoryBase& buf : device_memory_) {
    CHECK(*iterator != end);
    // Set buffers to be case (1) in the comment on ExecutionInput.
    (*iterator)->second = MaybeOwningDeviceMemory(buf);
    ++(*iterator);
  }
}

void TrackedDeviceBuffer::AddToInputAsDonated(
    ShapeTree<MaybeOwningDeviceMemory>::iterator* iterator,
    const ShapeTree<MaybeOwningDeviceMemory>::iterator& end,
    ExecutionInput* execution_input,
    se::DeviceMemoryAllocator* allocator) const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStracked_device_bufferDTcc mht_9(mht_9_v, 350, "", "./tensorflow/compiler/xla/pjrt/tracked_device_buffer.cc", "TrackedDeviceBuffer::AddToInputAsDonated");

  for (const se::DeviceMemoryBase& buf : device_memory_) {
    CHECK(*iterator != end);
    // Set buffers to be case (2) in the comment on ExecutionInput.
    (*iterator)->second = MaybeOwningDeviceMemory(
        se::OwningDeviceMemory(buf, device_ordinal_, allocator));
    execution_input->SetUnownedIndex((*iterator)->first);
    ++(*iterator);
  }
}

TrackedDeviceBuffer::TrackedDeviceBuffer(
    se::DeviceMemoryAllocator* allocator, int device_ordinal,
    absl::Span<se::DeviceMemoryBase const> device_memory,
    absl::Span<const std::shared_ptr<BufferSequencingEvent>> definition_events,
    std::function<void()> on_delete_callback)
    : allocator_(allocator),
      device_ordinal_(device_ordinal),
      device_memory_(device_memory.begin(), device_memory.end()),
      definition_events_(std::make_move_iterator(definition_events.begin()),
                         std::make_move_iterator(definition_events.end())),
      in_use_(true),
      on_delete_callback_(std::move(on_delete_callback)) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStracked_device_bufferDTcc mht_10(mht_10_v, 375, "", "./tensorflow/compiler/xla/pjrt/tracked_device_buffer.cc", "TrackedDeviceBuffer::TrackedDeviceBuffer");
}

TrackedDeviceBuffer::~TrackedDeviceBuffer() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStracked_device_bufferDTcc mht_11(mht_11_v, 380, "", "./tensorflow/compiler/xla/pjrt/tracked_device_buffer.cc", "TrackedDeviceBuffer::~TrackedDeviceBuffer");

  if (allocator_) {
    for (const se::DeviceMemoryBase& buffer : device_memory_) {
      Status status = allocator_->Deallocate(device_ordinal_, buffer);
      if (!status.ok()) {
        LOG(ERROR) << "Buffer deallocation failed: " << status;
      }
    }
  }
  if (on_delete_callback_) {
    on_delete_callback_();
  }
}

void TrackedDeviceBuffer::AddUsageEvent(
    se::Stream* usage_stream, std::shared_ptr<BufferSequencingEvent> event,
    bool reference_held) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStracked_device_bufferDTcc mht_12(mht_12_v, 399, "", "./tensorflow/compiler/xla/pjrt/tracked_device_buffer.cc", "TrackedDeviceBuffer::AddUsageEvent");

  CHECK(in_use_);

  for (auto& existing : usage_events_) {
    if (existing.stream == usage_stream) {
      if (*existing.event < *event) {
        existing.event = event;
        existing.reference_held = reference_held;
      }
      return;
    }
  }
  usage_events_.push_back({usage_stream, event, reference_held});
}

TrackedDeviceBuffer::StreamAndEventContainer
TrackedDeviceBuffer::LockUseAndTransferUsageEvents() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStracked_device_bufferDTcc mht_13(mht_13_v, 418, "", "./tensorflow/compiler/xla/pjrt/tracked_device_buffer.cc", "TrackedDeviceBuffer::LockUseAndTransferUsageEvents");

  CHECK(in_use_);
  in_use_ = false;
  return std::move(usage_events_);
}

void GetDeviceBufferEvents(
    const TrackedDeviceBuffer& buffer, bool get_usage_events,
    absl::flat_hash_set<BufferSequencingEvent*>* events) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStracked_device_bufferDTcc mht_14(mht_14_v, 429, "", "./tensorflow/compiler/xla/pjrt/tracked_device_buffer.cc", "GetDeviceBufferEvents");

  if (get_usage_events) {
    for (const auto& e : buffer.usage_events()) {
      events->insert(e.event.get());
    }
  } else {
    for (const auto& e : buffer.definition_events()) {
      events->insert(e.get());
    }
  }
}

void WaitForBufferDefinitionEventsOnStream(const TrackedDeviceBuffer& buffer,
                                           se::Stream* stream) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStracked_device_bufferDTcc mht_15(mht_15_v, 445, "", "./tensorflow/compiler/xla/pjrt/tracked_device_buffer.cc", "WaitForBufferDefinitionEventsOnStream");

  absl::flat_hash_set<BufferSequencingEvent*> events;
  GetDeviceBufferEvents(buffer, /*get_usage_events=*/false, &events);
  for (BufferSequencingEvent* event : events) {
    event->WaitForEventOnStream(stream);
  }
}

}  // namespace xla
