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

#ifndef TENSORFLOW_COMPILER_XLA_PJRT_TRACKED_DEVICE_BUFFER_H_
#define TENSORFLOW_COMPILER_XLA_PJRT_TRACKED_DEVICE_BUFFER_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStracked_device_bufferDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStracked_device_bufferDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStracked_device_bufferDTh() {
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

#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/xla/pjrt/event_pool.h"
#include "tensorflow/compiler/xla/pjrt/local_device_state.h"
#include "tensorflow/compiler/xla/service/shaped_buffer.h"
#include "tensorflow/compiler/xla/service/transfer_manager.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/device_memory_allocator.h"
#include "tensorflow/stream_executor/stream.h"

namespace xla {

// A BufferSequencingEvent keeps track of dependencies of a buffer on each
// stream it has been used on.
//
// Each logical buffer in an XLA computation may be defined (i.e., written to)
// at most once. We call the operation that writes the buffer's value on some
// stream (e.g., a transfer or compute kernel) the buffer's definition event.
//
// After the operation that populates the value of a buffer has been enqueued on
// 'stream', RecordOnStream(stream) should also be called to trigger the
// definition event after the operation has completed.
//
// After the buffer is read on 'stream' another event should be added so that
// it is possible to sequence buffer donation after all reads have completed.
//
// Since different streams are not necessarily synchronized with one another,
// if we wish to consume the value of the buffer on a different stream, we
// should first call WaitForEventOnStream(stream), which add a cross-stream
// from 'stream' to the buffer's definition event, causing 'stream' to pause
// until the definition event has been triggered, if needed. Operations on
// 'stream' may then assume that the buffer is valid and its contents correspond
// to the desired buffer.
//
// The dependency logic caches the set of streams at the tail of which the
// definition event is known to have occurred; waiting for the same event on the
// same stream causes no additional waiting.
class BufferSequencingEvent {
 public:
  BufferSequencingEvent() = default;

  // Sets the sequencing event to 'event', which is recorded on 'stream'. Must
  // be called at most once. Unblocks any other host threads that are blocked in
  // WaitForEventOnStream.
  void SetSequencingEvent(EventPool::Handle event, se::Stream* stream);

  // Adds synchronization events to 'stream' that wait for this event to be
  // defined on 'stream'. Does nothing if the event is already known to have
  // occurred by the tail of 'stream'. If RecordOnStream has not yet been
  // called, blocks the calling thread until the event has been recorded.
  void WaitForEventOnStream(se::Stream* stream);

  // Returns true if the event is known to have occurred by the tail of
  // 'stream'. If RecordOnStream has not yet been called, blocks the calling
  // thread until the event has been recorded.
  bool DefinedOn(se::Stream* stream);

  // Returns true if the event is known by the host to have already occurred. If
  // RecordOnStream has not yet been called, blocks the calling thread until the
  // event has been recorded.
  bool IsComplete();

  // Compares the sequence numbers of two recorded events. It is illegal to call
  // the comparison operators unless both events have been recorded.
  inline bool operator<(const BufferSequencingEvent& rhs) const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStracked_device_bufferDTh mht_0(mht_0_v, 254, "", "./tensorflow/compiler/xla/pjrt/tracked_device_buffer.h", "operator<");

    return sequence_number() < rhs.sequence_number();
  }
  inline bool operator>(const BufferSequencingEvent& rhs) const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStracked_device_bufferDTh mht_1(mht_1_v, 260, "", "./tensorflow/compiler/xla/pjrt/tracked_device_buffer.h", "operator>");

    return rhs < *this;
  }
  inline bool operator<=(const BufferSequencingEvent& rhs) const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStracked_device_bufferDTh mht_2(mht_2_v, 266, "", "./tensorflow/compiler/xla/pjrt/tracked_device_buffer.h", "=");

    return !(*this > rhs);
  }
  inline bool operator>=(const BufferSequencingEvent& rhs) const {
    return !(*this < rhs);
  }

 private:
  bool EventHasBeenRecorded() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  uint64_t sequence_number() const;

  // An event that is triggered when the content of one or more buffers has been
  // read or written. If this event is used as a definition event and is
  // nullptr, it is assumed that the buffer's content is always defined for
  // example because it uses storage borrowed from elsewhere.
  EventPool::Handle event_;

  // Cache of event_->sequence_number that avoids synchronization overhead.
  // TODO(phawkins): In fact, event_->sequence_number is unused beyond the
  // initial population of sequence_number_, and we could remove it if we
  // refactored the EventPool API.
  std::atomic<uint64_t> sequence_number_{0};

  mutable absl::Mutex mu_;
  // A list of all streams for which the buffer's content is known to be defined
  // at the tail of the queue, i.e., for any newly enqueued command.
  absl::InlinedVector<se::Stream*, 2> streams_defined_on_ ABSL_GUARDED_BY(mu_);
};

// Class that represents a tuple of device buffers. Like a ScopedShapedBuffer it
// owns all of the device memory in the tuple. It also tracks the definition and
// usage of the memory on streams, to allow for synchronized usage and deletion
// of memory under all of the allocation model semantics.
class TrackedDeviceBuffer {
 public:
  // Helper object to keep track of usage of the buffer on streams.
  struct StreamAndEvent {
    // A stream the buffer has been used on.
    se::Stream* stream;
    // An event that is later than the most recent usage of the buffer on
    // stream.
    std::shared_ptr<BufferSequencingEvent> event;
    // True if and only if a reference to the buffer is kept live until after
    // the host knows that event is complete.
    bool reference_held;
  };

  // Converts a ScopedShapedBuffer into a TrackedDeviceBuffer. Takes ownership
  // of the buffers of the shaped_buffer.
  static std::shared_ptr<TrackedDeviceBuffer> FromScopedShapedBuffer(
      ScopedShapedBuffer* shaped_buffer,
      absl::Span<const std::shared_ptr<BufferSequencingEvent>>
          definition_events);

  // Builds a ShapedBuffer view onto the buffers of 'tree'.
  ShapedBuffer AsShapedBuffer(const Shape& on_device_shape) const;

  // Adds the owned device buffers in order to 'iterator'. Used to add the
  // buffers to an ExecutionInput. We require but do not verify that 'iterator'
  // when passed in is pointing to a sub-tuple of the ExecutionInput whose
  // on_device_shape matches that of the TrackedDeviceBuffer. 'end' is used to
  // check that 'iterator' doesn't run out of bounds.
  void AddToInputAsImmutable(
      ShapeTree<MaybeOwningDeviceMemory>::iterator* iterator,
      const ShapeTree<MaybeOwningDeviceMemory>::iterator& end) const;

  // Adds the owned device buffers in order to 'iterator', marking them as
  // available to be donated. If donation succeeds, i.e., execution_input is
  // subsequently successfully enqueued to a computation,
  // this->ReleaseDeviceMemory() must be called to avoid freeing the device
  // memory twice. We require but do not verify that 'iterator' when passed in
  // is pointing to a sub-tuple of execution_input whose on_device_shape matches
  // that of the TrackedDeviceBuffer. 'end' is used to check that 'iterator'
  // doesn't run out of bounds.
  void AddToInputAsDonated(
      ShapeTree<MaybeOwningDeviceMemory>::iterator* iterator,
      const ShapeTree<MaybeOwningDeviceMemory>::iterator& end,
      ExecutionInput* execution_input,
      se::DeviceMemoryAllocator* allocator) const;

  se::DeviceMemoryAllocator* allocator() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStracked_device_bufferDTh mht_3(mht_3_v, 349, "", "./tensorflow/compiler/xla/pjrt/tracked_device_buffer.h", "allocator");
 return allocator_; }
  int device_ordinal() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStracked_device_bufferDTh mht_4(mht_4_v, 353, "", "./tensorflow/compiler/xla/pjrt/tracked_device_buffer.h", "device_ordinal");
 return device_ordinal_; }
  absl::InlinedVector<se::DeviceMemoryBase, 1>& device_memory() {
    return device_memory_;
  }
  const absl::InlinedVector<se::DeviceMemoryBase, 1>& device_memory() const {
    return device_memory_;
  }
  absl::Span<const std::shared_ptr<BufferSequencingEvent>> definition_events()
      const {
    return definition_events_;
  }
  absl::Span<const StreamAndEvent> usage_events() const {
    return usage_events_;
  }

  // Relinquishes ownership of the buffer's device memory, e.g., after the
  // buffer is passed to a computation that aliases its inputs to outputs.
  void ReleaseDeviceMemory() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStracked_device_bufferDTh mht_5(mht_5_v, 373, "", "./tensorflow/compiler/xla/pjrt/tracked_device_buffer.h", "ReleaseDeviceMemory");
 device_memory_.clear(); }

  // Indicates that the buffer has been used on a stream.
  //
  //   usage_stream:   a stream that the buffer was used on.
  //   event:          an event that has been recorded on usage_stream after the
  //                   buffer was used.
  //   reference_held: true if and only if the caller has caused a memory
  //                   reference to *this to stay live until after the host
  //                   is sure that the usage (transfer or execution) has
  //                   completed.
  void AddUsageEvent(se::Stream* usage_stream,
                     std::shared_ptr<BufferSequencingEvent> event,
                     bool reference_held);

  using StreamAndEventContainer = absl::InlinedVector<StreamAndEvent, 3>;
  // Returns the set of streams that the buffer was used on, and for each stream
  // an event later than the last use of the buffer. After
  // LockUseAndTransferUsageEvents is called it is illegal to use the buffer on
  // any stream and, e.g. AddUsageHold will CHECK fail.
  StreamAndEventContainer LockUseAndTransferUsageEvents();

  TrackedDeviceBuffer() : in_use_(true) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStracked_device_bufferDTh mht_6(mht_6_v, 398, "", "./tensorflow/compiler/xla/pjrt/tracked_device_buffer.h", "TrackedDeviceBuffer");
}
  TrackedDeviceBuffer(se::DeviceMemoryAllocator* allocator, int device_ordinal,
                      absl::Span<se::DeviceMemoryBase const> device_memory,
                      absl::Span<const std::shared_ptr<BufferSequencingEvent>>
                          definition_events,
                      std::function<void()> on_delete_callback);
  ~TrackedDeviceBuffer();

 private:
  // Are the buffers in device_memory_ owned? If so, which allocator and device
  // ordinal? May be nullptr, indicating the buffers are not owned.
  se::DeviceMemoryAllocator* allocator_;
  int device_ordinal_;

  // Each host-side buffer may have several buffers on-device.
  absl::InlinedVector<se::DeviceMemoryBase, 1> device_memory_;

  // Events that are triggered when the content of one or more buffers is ready
  // during multistream execution. May be nullptr, which is used in the
  // single-stream execution case where events are not necessary for buffer
  // event sequencing. All events must be triggered before the buffers can be
  // used.
  absl::InlinedVector<std::shared_ptr<BufferSequencingEvent>, 2>
      definition_events_;

  // in_use_ starts out true, and is set to false when the buffer is released
  // from its owning PjRtBuffer. Once in_use_ is false, the buffer may no
  // longer be used on any stream.
  bool in_use_;
  // Set of streams that the buffer has ever been used on, see comment on
  // StreamAndEvent.
  StreamAndEventContainer usage_events_;

  // A callback to call when the TrackedDeviceBuffer is about to be destroyed.
  std::function<void()> on_delete_callback_;
};

// Populates 'events' with the set of buffer events for buffer. If
// get_usage_events=true populates with the latest usage events, otherwise
// populates with the definition events.
void GetDeviceBufferEvents(const TrackedDeviceBuffer& buffer,
                           bool get_usage_events,
                           absl::flat_hash_set<BufferSequencingEvent*>* events);

// Waits for all of the definition events in a buffer on 'stream'.
void WaitForBufferDefinitionEventsOnStream(const TrackedDeviceBuffer& buffer,
                                           se::Stream* stream);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PJRT_TRACKED_DEVICE_BUFFER_H_
