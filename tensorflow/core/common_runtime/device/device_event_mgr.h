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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_DEVICE_DEVICE_EVENT_MGR_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_DEVICE_DEVICE_EVENT_MGR_H_
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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_event_mgrDTh {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_event_mgrDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_event_mgrDTh() {
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


#include <deque>
#include <vector>

#include "tensorflow/core/framework/log_memory.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"

namespace stream_executor {
class Event;
class Stream;
class StreamExecutor;
}  // namespace stream_executor

namespace tensorflow {

// TODO(annarev): Check if we can use a more general option representation here
// that could work for other device types as well.
class GPUOptions;

// The callback provided to EventMgr::ThenExecute must not block or take a long
// time.  If it does, performance may be impacted and device memory may be
// exhausted.  This macro is for checking that an EventMgr thread is not
// accidentally entering blocking parts of the code, e.g. the RPC subsystem.
//
// Intended use is something like
//
//   void RespondToAnRPC(Params* params) {
//      WARN_IF_IN_EVENT_MGR_THREAD;
//      if (params->status.ok()) { ...
//
namespace device_event_mgr {
// Logs a stack trace if current execution thread belongs to this EventMgr
// object.  If f is not nullptr, executes instead of  logging the stack trace.
// trace.
void WarnIfInCallback(std::function<void()> f);
}  // namespace device_event_mgr
#define WARN_IF_IN_EVENT_MGR_THREAD device_event_mgr::WarnIfInCallback(nullptr)

// An object to keep track of pending Events in the StreamExecutor streams
// and associated Tensors that cannot safely be deleted until the associated
// Events are recorded.
class EventMgr {
 public:
  virtual ~EventMgr();

  // Execute func when all pending stream actions have completed.
  // func must be brief and non-blocking since it executes in the one
  // thread used for all such callbacks and also buffer deletions.
  inline void ThenExecute(se::Stream* stream, std::function<void()> func) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_event_mgrDTh mht_0(mht_0_v, 242, "", "./tensorflow/core/common_runtime/device/device_event_mgr.h", "ThenExecute");

    ToFreeVector to_free;
    {
      mutex_lock l(mu_);
      QueueFunc(stream, std::move(func));
      PollEvents(false, &to_free);
    }
    FreeMemory(to_free);
  }

 private:
  friend class TEST_EventMgr;
  friend class TEST_EventMgrHelper;
  friend class EventMgrFactory;
  se::StreamExecutor* const exec_;
  const int32 polling_active_delay_usecs_;
  mutex mu_;
  condition_variable events_pending_ TF_GUARDED_BY(mu_);

  struct InUse {
    se::Event* event;
    std::function<void()> func;
  };

  typedef gtl::InlinedVector<InUse, 4> ToFreeVector;

  EventMgr(se::StreamExecutor* se, const GPUOptions& gpu_options);

  void FreeMemory(const ToFreeVector& to_free) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_event_mgrDTh mht_1(mht_1_v, 273, "", "./tensorflow/core/common_runtime/device/device_event_mgr.h", "FreeMemory");

    for (const auto& iu : to_free) {
      // The function must be called in another thread.
      if (iu.func != nullptr) threadpool_.Schedule(iu.func);
    }
  }

  // Stream-enqueue an unused Event and save with it a collection of
  // Tensors and/or a BufRec to be deleted only after the Event
  // records.
  void QueueInUse(se::Stream* stream, InUse in_use)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  void QueueFunc(se::Stream* stream, std::function<void()> func)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_event_mgrDTh mht_2(mht_2_v, 290, "", "./tensorflow/core/common_runtime/device/device_event_mgr.h", "QueueFunc");

    QueueInUse(stream, {nullptr, std::move(func)});
  }

  // This function should be called at roughly the same tempo as
  // QueueTensors() to check whether pending events have recorded,
  // and then retire them.  It appends InUse elements that need cleanup
  // to "*to_free".  The caller should call FreeMemory(to_free)
  // when this returns.
  void PollEvents(bool is_dedicated_poller, ToFreeVector* to_free)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // An internal polling loop that runs at a low frequency to clear
  // straggler Events.
  void PollLoop();

  // Setup/Teardown functions for the polling loop.
  void StartPollingLoop();
  void StopPollingLoop();

  // A stack of unused events
  std::vector<se::Event*> free_events_ TF_GUARDED_BY(mu_);

  // A FIFO queue of InUse events and associated tensors.
  std::deque<InUse> used_events_ TF_GUARDED_BY(mu_);

  bool stop_polling_ TF_GUARDED_BY(mu_);
  std::unique_ptr<Notification> polling_stopped_;

  // The main PollLoop for the event manager runs in this threadpool.
  thread::ThreadPool threadpool_;
};

// Manages all the EventMgr instances.
class EventMgrFactory {
 public:
  static EventMgrFactory* Singleton();

  EventMgr* GetEventMgr(se::StreamExecutor* se, const GPUOptions& gpu_options);

 private:
  mutex mu_;

  // Maintain one EventMgr per physical device (StreamExecutor is
  // per-physical-device).
  std::map<se::StreamExecutor*, EventMgr*> event_mgr_map_ TF_GUARDED_BY(mu_);
};

}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_DEVICE_DEVICE_EVENT_MGR_H_
