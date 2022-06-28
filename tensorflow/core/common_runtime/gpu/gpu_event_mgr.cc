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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_event_mgrDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_event_mgrDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_event_mgrDTcc() {
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

/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/device/device_event_mgr.h"
#include "tensorflow/core/platform/stacktrace.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {

namespace {
// The EventMgr has 1 thread for the polling loop and one to execute
// event callback functions. Issues for reconsideration:
//  - Is this the right number of threads?
//  - Should EventMgrs be shared between devices on a machine with multiple
//  devices of the same type?
static const int kNumThreads = 2;
}  // namespace

namespace device_event_mgr {
class ThreadLabel {
 public:
  static const char* GetValue() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_event_mgrDTcc mht_0(mht_0_v, 204, "", "./tensorflow/core/common_runtime/gpu/gpu_event_mgr.cc", "GetValue");
 return value_; }

  // v must be a static const because value_ will capture and use its value
  // until reset or thread terminates.
  static void SetValue(const char* v) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("v: \"" + (v == nullptr ? std::string("nullptr") : std::string((char*)v)) + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_event_mgrDTcc mht_1(mht_1_v, 212, "", "./tensorflow/core/common_runtime/gpu/gpu_event_mgr.cc", "SetValue");
 value_ = v; }

 private:
  static thread_local const char* value_;
};
thread_local const char* ThreadLabel::value_ = "";

void WarnIfInCallback(std::function<void()> f) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_event_mgrDTcc mht_2(mht_2_v, 222, "", "./tensorflow/core/common_runtime/gpu/gpu_event_mgr.cc", "WarnIfInCallback");

  const char* label = ThreadLabel::GetValue();
  if (label && !strcmp(label, "device_event_mgr")) {
    if (f) {
      f();
    } else {
      LOG(WARNING) << "Executing inside EventMgr callback thread: "
                   << CurrentStackTrace();
    }
  }
}

void InitThreadpoolLabels(thread::ThreadPool* threadpool) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_event_mgrDTcc mht_3(mht_3_v, 237, "", "./tensorflow/core/common_runtime/gpu/gpu_event_mgr.cc", "InitThreadpoolLabels");

  static const char* label = "device_event_mgr";
  mutex mu;
  int init_count = 0;
  condition_variable all_initialized;
  int exit_count = 0;
  condition_variable ready_to_exit;
  const int num_threads = threadpool->NumThreads();
  for (int i = 0; i < num_threads; ++i) {
    threadpool->Schedule([num_threads, &mu, &init_count, &all_initialized,
                          &exit_count, &ready_to_exit]() {
      device_event_mgr::ThreadLabel::SetValue(label);
      mutex_lock l(mu);
      ++init_count;
      if (init_count == num_threads) {
        all_initialized.notify_all();
      }
      while (init_count < num_threads) {
        all_initialized.wait(l);
      }
      if (++exit_count == num_threads) {
        ready_to_exit.notify_all();
      }
    });
  }
  {
    mutex_lock l(mu);
    while (exit_count < num_threads) {
      ready_to_exit.wait(l);
    }
  }
}
}  // namespace device_event_mgr

EventMgr::EventMgr(se::StreamExecutor* se, const GPUOptions& gpu_options)
    : exec_(se),
      polling_active_delay_usecs_(gpu_options.polling_active_delay_usecs()
                                      ? gpu_options.polling_active_delay_usecs()
                                      : 10),
      threadpool_(Env::Default(), "Device_Event_Manager", kNumThreads) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_event_mgrDTcc mht_4(mht_4_v, 279, "", "./tensorflow/core/common_runtime/gpu/gpu_event_mgr.cc", "EventMgr::EventMgr");

  device_event_mgr::InitThreadpoolLabels(&threadpool_);
  StartPollingLoop();
}

EventMgr::~EventMgr() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_event_mgrDTcc mht_5(mht_5_v, 287, "", "./tensorflow/core/common_runtime/gpu/gpu_event_mgr.cc", "EventMgr::~EventMgr");

  StopPollingLoop();

  // Events are owned by this object.
  for (auto& e : free_events_) {
    delete e;
  }
  while (!used_events_.empty()) {
    InUse* ue = &used_events_[0];
    delete ue->event;
    if (ue->func != nullptr) threadpool_.Schedule(ue->func);
    used_events_.pop_front();
  }
}

void EventMgr::StartPollingLoop() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_event_mgrDTcc mht_6(mht_6_v, 305, "", "./tensorflow/core/common_runtime/gpu/gpu_event_mgr.cc", "EventMgr::StartPollingLoop");

  CHECK(polling_stopped_ == nullptr);
  {
    mutex_lock l(mu_);
    stop_polling_ = false;
  }
  polling_stopped_.reset(new Notification);
  threadpool_.Schedule([this]() { PollLoop(); });
}

void EventMgr::StopPollingLoop() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_event_mgrDTcc mht_7(mht_7_v, 318, "", "./tensorflow/core/common_runtime/gpu/gpu_event_mgr.cc", "EventMgr::StopPollingLoop");

  if (polling_stopped_) {
    {
      mutex_lock l(mu_);
      stop_polling_ = true;
      events_pending_.notify_all();
    }
    polling_stopped_->WaitForNotification();
    polling_stopped_.reset(nullptr);
  }
}

// A polling loop to detect completion of device events.
//
// While one or more events is outstanding, poll for completed events.  When no
// events are outstanding, we sleep until one is enqueued.
void EventMgr::PollLoop() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_event_mgrDTcc mht_8(mht_8_v, 337, "", "./tensorflow/core/common_runtime/gpu/gpu_event_mgr.cc", "EventMgr::PollLoop");

  ToFreeVector to_free;
  while (true) {
    bool events_still_pending;
    {
      mutex_lock l(mu_);
      if (stop_polling_) {
        break;
      }
      if (used_events_.empty()) {
        events_pending_.wait(l);
      }
      PollEvents(true, &to_free);
      events_still_pending = !used_events_.empty();
    }
    FreeMemory(to_free);
    to_free.clear();

    if (events_still_pending) {
      Env::Default()->SleepForMicroseconds(polling_active_delay_usecs_);
    }
  }
  polling_stopped_->Notify();
}

void EventMgr::QueueInUse(se::Stream* stream, InUse in_use) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_event_mgrDTcc mht_9(mht_9_v, 365, "", "./tensorflow/core/common_runtime/gpu/gpu_event_mgr.cc", "EventMgr::QueueInUse");

  VLOG(2) << "QueueInUse  free_events_ " << free_events_.size()
          << " used_events_ " << used_events_.size();
  // Events are created on demand, and repeatedly reused.  There is no
  // limit placed here on the number of allocated Events.
  if (free_events_.empty()) {
    free_events_.push_back(new se::Event(exec_));
    free_events_.back()->Init();
  }
  se::Event* e = free_events_.back();
  free_events_.pop_back();
  stream->ThenRecordEvent(e);
  in_use.event = e;
  bool was_empty = used_events_.empty();
  used_events_.push_back(in_use);
  // Maybe wake up the polling thread
  if (was_empty) events_pending_.notify_all();
}

// This function must be called periodically to check whether pending
// events have recorded, and then retire them.  Initial observations
// suggest that typical behavior in a TensorFlow program is to have
// 0-3 events pending most of the time, but there are occasionally
// spikes of up to several hundred outstanding.  (If GPUKernelTracker
// is used to cap pending kernels there should never be more than
// that many.)
//
// NOTE: If all events are on the same stream, no later event will
// complete before an earlier event, except possibly if the earlier
// event transitions to an error state, so there's no advantage in
// looking past the first kPending event.  However, if we're using
// multiple streams there may be some gain in looking deeper.
// As a compromise, PollEvent() calls that are triggered by the queueing
// of a single event never look past the first kPending event.  Consequently
// those calls do an expected constant amount of work, unaffected by the
// length of the pending queue.  Calls coming from the dedicated
// polling thread always sweep the full queue.
void EventMgr::PollEvents(bool is_dedicated_poller,
                          gtl::InlinedVector<InUse, 4>* to_free) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_event_mgrDTcc mht_10(mht_10_v, 406, "", "./tensorflow/core/common_runtime/gpu/gpu_event_mgr.cc", "EventMgr::PollEvents");

  VLOG(2) << "PollEvents  free_events_ " << free_events_.size()
          << " used_events_ " << used_events_.size();
  // Sweep the remaining events in order.  If this is the dedicated
  // polling thread, check the entire set.  Otherwise, just sweep up to
  // the first non-complete record that is still pending.
  for (auto& iu : used_events_) {
    if (iu.event == nullptr) continue;
    se::Event::Status s = iu.event->PollForStatus();
    switch (s) {
      case se::Event::Status::kUnknown:
      case se::Event::Status::kError:
        // We don't expect to see these.  Someday maybe propagate
        // a Status error, but for now fail hard.
        LOG(FATAL) << "Unexpected Event status: " << static_cast<int>(s);
        break;
      case se::Event::Status::kPending:
        if (!is_dedicated_poller) return;  // quit processing queue
        break;
      case se::Event::Status::kComplete:
        // Make a copy of the InUse record so we can free it after releasing
        // the lock
        to_free->push_back(iu);
        free_events_.push_back(iu.event);
        // Mark this InUse record as completed.
        iu.event = nullptr;
    }
  }
  // Then clear any completed InUse records from the front of the queue.
  while (!used_events_.empty()) {
    InUse& iu = used_events_.front();
    if (iu.event == nullptr) {
      used_events_.pop_front();
    } else {
      break;
    }
  }
}

EventMgrFactory* EventMgrFactory::Singleton() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_event_mgrDTcc mht_11(mht_11_v, 448, "", "./tensorflow/core/common_runtime/gpu/gpu_event_mgr.cc", "EventMgrFactory::Singleton");

  static EventMgrFactory* instance = new EventMgrFactory;
  return instance;
}

EventMgr* EventMgrFactory::GetEventMgr(se::StreamExecutor* se,
                                       const GPUOptions& gpu_options) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_event_mgrDTcc mht_12(mht_12_v, 457, "", "./tensorflow/core/common_runtime/gpu/gpu_event_mgr.cc", "EventMgrFactory::GetEventMgr");

  mutex_lock l(mu_);
  // TODO(laigd): consider making gpu_options part of the key. It's not
  // currently since EventMgr depends only rely on field deferred_deletion_bytes
  // and polling_active_delay_usecs from gpu_options which are not used or
  // rarely used.
  auto itr = event_mgr_map_.find(se);
  if (itr == event_mgr_map_.end()) {
    auto event_mgr = new EventMgr(se, gpu_options);
    event_mgr_map_[se] = event_mgr;
    return event_mgr;
  } else {
    return itr->second;
  }
}

}  // namespace tensorflow
