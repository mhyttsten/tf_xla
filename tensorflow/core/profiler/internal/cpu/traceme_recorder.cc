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
class MHTracer_DTPStensorflowPScorePSprofilerPSinternalPScpuPStraceme_recorderDTcc {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPScpuPStraceme_recorderDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSinternalPScpuPStraceme_recorderDTcc() {
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

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/profiler/internal/cpu/traceme_recorder.h"

#include <stddef.h>

#include <algorithm>
#include <atomic>
#include <memory>
#include <new>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace profiler {
namespace internal {

std::atomic<int> g_trace_level(TraceMeRecorder::kTracingDisabled);

// g_trace_level implementation must be lock-free for faster execution of the
// TraceMe API. This can be commented (if compilation is failing) but execution
// might be slow (even when tracing is disabled).
static_assert(ATOMIC_INT_LOCK_FREE == 2, "Assumed atomic<int> was lock free");

}  // namespace internal

namespace {

// Track events created by ActivityStart and merge their data into events
// created by ActivityEnd. TraceMe records events in its destructor, so this
// results in complete events sorted by their end_time in the thread they ended.
// Within the same thread, the record created by ActivityStart must appear
// before the record created by ActivityEnd. Cross-thread events must be
// processed in a separate pass. A single map can be used because the
// activity_id is globally unique.
class SplitEventTracker {
 public:
  void AddStart(TraceMeRecorder::Event&& event) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPScpuPStraceme_recorderDTcc mht_0(mht_0_v, 226, "", "./tensorflow/core/profiler/internal/cpu/traceme_recorder.cc", "AddStart");

    DCHECK(event.IsStart());
    start_events_.emplace(event.ActivityId(), std::move(event));
  }

  void AddEnd(TraceMeRecorder::Event* event) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPScpuPStraceme_recorderDTcc mht_1(mht_1_v, 234, "", "./tensorflow/core/profiler/internal/cpu/traceme_recorder.cc", "AddEnd");

    DCHECK(event->IsEnd());
    if (!FindStartAndMerge(event)) {
      end_events_.push_back(event);
    }
  }

  void HandleCrossThreadEvents() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPScpuPStraceme_recorderDTcc mht_2(mht_2_v, 244, "", "./tensorflow/core/profiler/internal/cpu/traceme_recorder.cc", "HandleCrossThreadEvents");

    for (auto* event : end_events_) {
      FindStartAndMerge(event);
    }
  }

 private:
  // Finds the start of the given event and merges data into it.
  bool FindStartAndMerge(TraceMeRecorder::Event* event) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPScpuPStraceme_recorderDTcc mht_3(mht_3_v, 255, "", "./tensorflow/core/profiler/internal/cpu/traceme_recorder.cc", "FindStartAndMerge");

    auto iter = start_events_.find(event->ActivityId());
    if (iter == start_events_.end()) return false;
    auto& start_event = iter->second;
    event->name = std::move(start_event.name);
    event->start_time = start_event.start_time;
    start_events_.erase(iter);
    return true;
  }

  // Start events are collected from each ThreadLocalRecorder::Consume() call.
  // Their data is merged into end_events.
  absl::flat_hash_map<int64_t, TraceMeRecorder::Event> start_events_;

  // End events are stored in the output of TraceMeRecorder::Consume().
  std::vector<TraceMeRecorder::Event*> end_events_;
};

// A single-producer single-consumer queue of Events.
//
// Implemented as a linked-list of blocks containing numbered slots, with start
// and end pointers:
//
//  [ events........ | next-]--> [ events......... | next ]
//  ^start_block_ ^start_         ^end_block_ ^end_
//
// start_ is the first occupied slot, end_ is the first unoccupied slot.
//
// Push writes at end_, and then advances it, allocating a block if needed.
// Consume takes ownership of events in the range [start_, end_).
// Clear removes events in the range [start_, end_).
// The end_ pointer is atomic so Push and Consume can be concurrent.
//
// Push and Consume are lock free and each might be called from at most one
// thread. Push is only called by the owner thread. Consume is only called by
// the tracing control thread.
//
// Thus, Consume might race with Push, so Consume only removes events that were
// in the queue when it was invoked. If Push is called while Consume is active,
// the new event remains in the queue. Thus, the tracing control thread should
// call Consume when tracing stops to remove events created during tracing, and
// Clear when tracing starts again to remove any remaining events.
class EventQueue {
 public:
  EventQueue()
      : start_block_(new Block{/*start=*/0, /*next=*/nullptr}),
        start_(start_block_->start),
        end_block_(start_block_),
        end_(start_) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPScpuPStraceme_recorderDTcc mht_4(mht_4_v, 306, "", "./tensorflow/core/profiler/internal/cpu/traceme_recorder.cc", "EventQueue");
}

  // Memory should be deallocated and trace events destroyed on destruction.
  // This doesn't require global lock as this discards all the stored trace
  // events and we assume of destruction of this instance only after the last
  // Push() has been called.
  ~EventQueue() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPScpuPStraceme_recorderDTcc mht_5(mht_5_v, 315, "", "./tensorflow/core/profiler/internal/cpu/traceme_recorder.cc", "~EventQueue");

    Clear();
    DCHECK(Empty());
    delete end_block_;
  }

  // Add a new event to the back of the queue. Fast and lock-free.
  void Push(TraceMeRecorder::Event&& event) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPScpuPStraceme_recorderDTcc mht_6(mht_6_v, 325, "", "./tensorflow/core/profiler/internal/cpu/traceme_recorder.cc", "Push");

    size_t end = end_.load(std::memory_order_relaxed);
    new (&end_block_->events[end++ - end_block_->start].event)
        TraceMeRecorder::Event(std::move(event));
    if (TF_PREDICT_FALSE(end - end_block_->start == Block::kNumSlots)) {
      auto* new_block = new Block{end, nullptr};
      end_block_->next = new_block;
      end_block_ = new_block;
    }
    end_.store(end, std::memory_order_release);  // Write index after contents.
  }

  // Removes all events from the queue.
  void Clear() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPScpuPStraceme_recorderDTcc mht_7(mht_7_v, 341, "", "./tensorflow/core/profiler/internal/cpu/traceme_recorder.cc", "Clear");

    size_t end = end_.load(std::memory_order_acquire);
    while (start_ != end) {
      Pop();
    }
  }

  // Retrieve and remove all events in the queue at the time of invocation.
  // If Push is called while Consume is active, the new event will not be
  // removed from the queue.
  // Consume is only called from ThreadLocalRecorder::Clear, which in turn is
  // only called while holding TraceMeRecorder::Mutex, so Consume has a single
  // caller at a time.
  TF_MUST_USE_RESULT std::deque<TraceMeRecorder::Event> Consume(
      SplitEventTracker* split_event_tracker) {
    // Read index before contents.
    size_t end = end_.load(std::memory_order_acquire);
    std::deque<TraceMeRecorder::Event> result;
    while (start_ != end) {
      TraceMeRecorder::Event event = Pop();
      // Copy data from start events to end events. TraceMe records events in
      // its destructor, so this results in complete events sorted by their
      // end_time in the thread they ended. Within the same thread, the start
      // event must appear before the corresponding end event.
      if (event.IsStart()) {
        split_event_tracker->AddStart(std::move(event));
        continue;
      }
      result.emplace_back(std::move(event));
      if (result.back().IsEnd()) {
        split_event_tracker->AddEnd(&result.back());
      }
    }
    return result;
  }

 private:
  // Returns true if the queue is empty at the time of invocation.
  bool Empty() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPScpuPStraceme_recorderDTcc mht_8(mht_8_v, 382, "", "./tensorflow/core/profiler/internal/cpu/traceme_recorder.cc", "Empty");

    return (start_ == end_.load(std::memory_order_acquire));
  }

  // Remove one event off the front of the queue and return it.
  // REQUIRES: The queue must not be empty.
  TraceMeRecorder::Event Pop() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPScpuPStraceme_recorderDTcc mht_9(mht_9_v, 391, "", "./tensorflow/core/profiler/internal/cpu/traceme_recorder.cc", "Pop");

    DCHECK(!Empty());
    // Move the next event into the output.
    auto& event = start_block_->events[start_++ - start_block_->start].event;
    TraceMeRecorder::Event out = std::move(event);
    event.~Event();  // Events must be individually destroyed.
    // If we reach the end of a block, we own it and should delete it.
    // The next block is present: end always points to something.
    if (TF_PREDICT_FALSE(start_ - start_block_->start == Block::kNumSlots)) {
      auto* next_block = start_block_->next;
      delete start_block_;
      start_block_ = next_block;
      DCHECK_EQ(start_, start_block_->start);
    }
    return out;
  }

  struct Block {
    // The number of slots in a block is chosen so the block fits in 64 KiB.
    static constexpr size_t kSize = 1 << 16;
    static constexpr size_t kNumSlots =
        (kSize - (sizeof(size_t) + sizeof(Block*))) /
        sizeof(TraceMeRecorder::Event);

    size_t start;  // The number of the first slot.
    Block* next;
    // Defer construction of Event until the data is available.
    // Must also destroy manually, as the block may not fill entirely.
    union MaybeEvent {
      MaybeEvent() {}
      ~MaybeEvent() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPScpuPStraceme_recorderDTcc mht_10(mht_10_v, 424, "", "./tensorflow/core/profiler/internal/cpu/traceme_recorder.cc", "~MaybeEvent");
}
      TraceMeRecorder::Event event;
    } events[kNumSlots];
  };

  static_assert(sizeof(Block) <= Block::kSize, "");

  // Head of list for reading. Only accessed by consumer thread.
  Block* start_block_;
  size_t start_;
  // Tail of list for writing. Accessed by producer thread.
  Block* end_block_;
  std::atomic<size_t> end_;  // Atomic: also read by consumer thread.
};

}  // namespace

// To avoid unnecessary synchronization between threads, each thread has a
// ThreadLocalRecorder that independently records its events.
class TraceMeRecorder::ThreadLocalRecorder {
 public:
  // The recorder is created the first time TraceMeRecorder::Record() is called
  // on a thread.
  ThreadLocalRecorder() {
    auto* env = Env::Default();
    info_.tid = env->GetCurrentThreadId();
    env->GetCurrentThreadName(&info_.name);
  }

  uint32 ThreadId() const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPScpuPStraceme_recorderDTcc mht_11(mht_11_v, 456, "", "./tensorflow/core/profiler/internal/cpu/traceme_recorder.cc", "ThreadId");
 return info_.tid; }

  bool IsActive() const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPScpuPStraceme_recorderDTcc mht_12(mht_12_v, 461, "", "./tensorflow/core/profiler/internal/cpu/traceme_recorder.cc", "IsActive");
 return active_; }
  void SetActive(bool active) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPScpuPStraceme_recorderDTcc mht_13(mht_13_v, 465, "", "./tensorflow/core/profiler/internal/cpu/traceme_recorder.cc", "SetActive");
 active_ = active; }

  // Record is only called from the owner thread.
  void Record(TraceMeRecorder::Event&& event) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPScpuPStraceme_recorderDTcc mht_14(mht_14_v, 471, "", "./tensorflow/core/profiler/internal/cpu/traceme_recorder.cc", "Record");
 queue_.Push(std::move(event)); }

  // Clear is called from the control thread when tracing starts to remove any
  // elements added due to Record racing with Consume.
  void Clear() {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPScpuPStraceme_recorderDTcc mht_15(mht_15_v, 478, "", "./tensorflow/core/profiler/internal/cpu/traceme_recorder.cc", "Clear");
 queue_.Clear(); }

  // Consume is called from the control thread when tracing stops.
  TF_MUST_USE_RESULT TraceMeRecorder::ThreadEvents Consume(
      SplitEventTracker* split_event_tracker) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPScpuPStraceme_recorderDTcc mht_16(mht_16_v, 485, "", "./tensorflow/core/profiler/internal/cpu/traceme_recorder.cc", "Consume");

    return {info_, queue_.Consume(split_event_tracker)};
  }

 private:
  TraceMeRecorder::ThreadInfo info_;
  EventQueue queue_;
  bool active_ = true;
};

// An instance of this wrapper is allocated in thread_local storage.
// It creates the ThreadLocalRecorder and notifies TraceMeRecorder when the
// the first TraceMe on the thread is executed while tracing is active, or when
// the thread is destroyed.
class TraceMeRecorder::ThreadLocalRecorderWrapper {
 public:
  ThreadLocalRecorderWrapper()
      : recorder_(std::make_shared<TraceMeRecorder::ThreadLocalRecorder>()) {
    TraceMeRecorder::Get()->RegisterThread(recorder_->ThreadId(), recorder_);
  }

  void Record(TraceMeRecorder::Event&& event) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPScpuPStraceme_recorderDTcc mht_17(mht_17_v, 509, "", "./tensorflow/core/profiler/internal/cpu/traceme_recorder.cc", "Record");

    recorder_->Record(std::move(event));
  }

  ~ThreadLocalRecorderWrapper() {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPScpuPStraceme_recorderDTcc mht_18(mht_18_v, 516, "", "./tensorflow/core/profiler/internal/cpu/traceme_recorder.cc", "~ThreadLocalRecorderWrapper");

    recorder_->SetActive(false);
    TraceMeRecorder::Get()->UnregisterThread(recorder_->ThreadId());
  }

 private:
  // Ownership of ThreadLocalRecorder is shared with TraceMeRecorder.
  // If a thread is destroyed during tracing, its ThreadLocalRecorder is kept
  // alive until the end of tracing.
  std::shared_ptr<TraceMeRecorder::ThreadLocalRecorder> recorder_;
};

/*static*/ TraceMeRecorder* TraceMeRecorder::Get() {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPScpuPStraceme_recorderDTcc mht_19(mht_19_v, 531, "", "./tensorflow/core/profiler/internal/cpu/traceme_recorder.cc", "TraceMeRecorder::Get");

  static TraceMeRecorder* singleton = new TraceMeRecorder;
  return singleton;
}

void TraceMeRecorder::RegisterThread(
    uint32 tid, std::shared_ptr<ThreadLocalRecorder> thread) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPScpuPStraceme_recorderDTcc mht_20(mht_20_v, 540, "", "./tensorflow/core/profiler/internal/cpu/traceme_recorder.cc", "TraceMeRecorder::RegisterThread");

  mutex_lock lock(mutex_);
  threads_.insert_or_assign(tid, std::move(thread));
}

void TraceMeRecorder::UnregisterThread(uint32 tid) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPScpuPStraceme_recorderDTcc mht_21(mht_21_v, 548, "", "./tensorflow/core/profiler/internal/cpu/traceme_recorder.cc", "TraceMeRecorder::UnregisterThread");

  // If tracing is active, keep the ThreadLocalRecorder alive.
  if (Active()) return;
  // If tracing is inactive, destroy the ThreadLocalRecorder.
  mutex_lock lock(mutex_);
  threads_.erase(tid);
}

// This method is performance critical and should be kept fast. It is called
// when tracing starts. The mutex is held, so no threads can be
// registered/unregistered. This ensures only the control thread calls
// ThreadLocalRecorder::Clear().
void TraceMeRecorder::Clear() {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPScpuPStraceme_recorderDTcc mht_22(mht_22_v, 563, "", "./tensorflow/core/profiler/internal/cpu/traceme_recorder.cc", "TraceMeRecorder::Clear");

  for (auto& id_and_recorder : threads_) {
    auto& recorder = id_and_recorder.second;
    recorder->Clear();
    // We should not have an inactive ThreadLocalRecorder here. If a thread is
    // destroyed while tracing is inactive, its ThreadLocalRecorder is removed
    // in UnregisterThread.
    DCHECK(recorder->IsActive());
  }
}

// This method is performance critical and should be kept fast. It is called
// when tracing stops. The mutex is held, so no threads can be
// registered/unregistered. This ensures only the control thread calls
// ThreadLocalRecorder::Consume().
TraceMeRecorder::Events TraceMeRecorder::Consume() {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPScpuPStraceme_recorderDTcc mht_23(mht_23_v, 581, "", "./tensorflow/core/profiler/internal/cpu/traceme_recorder.cc", "TraceMeRecorder::Consume");

  TraceMeRecorder::Events result;
  result.reserve(threads_.size());
  SplitEventTracker split_event_tracker;
  for (auto iter = threads_.begin(); iter != threads_.end();) {
    auto& recorder = iter->second;
    TraceMeRecorder::ThreadEvents events =
        recorder->Consume(&split_event_tracker);
    if (!events.events.empty()) {
      result.push_back(std::move(events));
    }
    // We can have an active thread here. If a thread is destroyed while tracing
    // is active, its ThreadLocalRecorder is kept alive in UnregisterThread.
    if (!recorder->IsActive()) {
      threads_.erase(iter++);
    } else {
      ++iter;
    }
  }
  split_event_tracker.HandleCrossThreadEvents();
  return result;
}

bool TraceMeRecorder::StartRecording(int level) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPScpuPStraceme_recorderDTcc mht_24(mht_24_v, 607, "", "./tensorflow/core/profiler/internal/cpu/traceme_recorder.cc", "TraceMeRecorder::StartRecording");

  level = std::max(0, level);
  mutex_lock lock(mutex_);
  // Change trace_level_ while holding mutex_.
  int expected = kTracingDisabled;
  bool started = internal::g_trace_level.compare_exchange_strong(
      expected, level, std::memory_order_acq_rel);
  if (started) {
    // We may have old events in buffers because Record() raced with Stop().
    Clear();
  }
  return started;
}

void TraceMeRecorder::Record(Event&& event) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPScpuPStraceme_recorderDTcc mht_25(mht_25_v, 624, "", "./tensorflow/core/profiler/internal/cpu/traceme_recorder.cc", "TraceMeRecorder::Record");

  static thread_local ThreadLocalRecorderWrapper thread_local_recorder;
  thread_local_recorder.Record(std::move(event));
}

TraceMeRecorder::Events TraceMeRecorder::StopRecording() {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPScpuPStraceme_recorderDTcc mht_26(mht_26_v, 632, "", "./tensorflow/core/profiler/internal/cpu/traceme_recorder.cc", "TraceMeRecorder::StopRecording");

  TraceMeRecorder::Events events;
  mutex_lock lock(mutex_);
  // Change trace_level_ while holding mutex_.
  if (internal::g_trace_level.exchange(
          kTracingDisabled, std::memory_order_acq_rel) != kTracingDisabled) {
    events = Consume();
  }
  return events;
}

/*static*/ int64_t TraceMeRecorder::NewActivityId() {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPScpuPStraceme_recorderDTcc mht_27(mht_27_v, 646, "", "./tensorflow/core/profiler/internal/cpu/traceme_recorder.cc", "TraceMeRecorder::NewActivityId");

  // Activity IDs: To avoid contention over a counter, the top 32 bits identify
  // the originating thread, the bottom 32 bits name the event within a thread.
  // IDs may be reused after 4 billion events on one thread, or 2 billion
  // threads.
  static std::atomic<int32> thread_counter(1);  // avoid kUntracedActivity
  const thread_local static int32_t thread_id =
      thread_counter.fetch_add(1, std::memory_order_relaxed);
  thread_local static uint32 per_thread_activity_id = 0;
  return static_cast<int64_t>(thread_id) << 32 | per_thread_activity_id++;
}

}  // namespace profiler
}  // namespace tensorflow
