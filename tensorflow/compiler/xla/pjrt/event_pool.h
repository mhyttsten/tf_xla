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

#ifndef TENSORFLOW_COMPILER_XLA_PJRT_EVENT_POOL_H_
#define TENSORFLOW_COMPILER_XLA_PJRT_EVENT_POOL_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSevent_poolDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSevent_poolDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSevent_poolDTh() {
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
#include <stack>

#include "absl/synchronization/mutex.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/platform/stream_executor.h"

namespace xla {

class EventPool {
 public:
  class Handle {
   public:
    Handle() = default;
    ~Handle();

    Handle(const Handle&) = delete;
    Handle(Handle&&) = default;
    Handle& operator=(const Handle&) = delete;
    Handle& operator=(Handle&&) = default;

    // There is a total order on events handed out by the event pool. The most
    // useful aspect of this total order is that two events returned by
    // ThenAllocateAndRecordEvent on the same stream can be compared to see
    // which was recorded earlier on that stream.
    // Valid sequence numbers are > 0.
    inline bool operator<(const Handle& rhs) const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSevent_poolDTh mht_0(mht_0_v, 215, "", "./tensorflow/compiler/xla/pjrt/event_pool.h", "operator<");

      return sequence_number_ < rhs.sequence_number_;
    }
    inline bool operator>(const Handle& rhs) const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSevent_poolDTh mht_1(mht_1_v, 221, "", "./tensorflow/compiler/xla/pjrt/event_pool.h", "operator>");
 return rhs < *this; }
    inline bool operator<=(const Handle& rhs) const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSevent_poolDTh mht_2(mht_2_v, 225, "", "./tensorflow/compiler/xla/pjrt/event_pool.h", "=");
 return !(*this > rhs); }
    inline bool operator>=(const Handle& rhs) const { return !(*this < rhs); }

    se::Event* event() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSevent_poolDTh mht_3(mht_3_v, 231, "", "./tensorflow/compiler/xla/pjrt/event_pool.h", "event");
 return event_.get(); }
    uint64_t sequence_number() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSevent_poolDTh mht_4(mht_4_v, 235, "", "./tensorflow/compiler/xla/pjrt/event_pool.h", "sequence_number");
 return sequence_number_; }

   private:
    friend class EventPool;

    EventPool* pool_ = nullptr;
    std::unique_ptr<se::Event> event_;
    uint64_t sequence_number_;
  };

  // Initializes a new EventPool. If `allow_reuse` is true, then events will be
  // returned to the pool when their handles are deleted and made available to
  // subsequent allocations. Reuse only works on the GPU platform.
  explicit EventPool(bool allow_reuse);

  // Allocates a new (or reused) event from the pool, and records the event on
  // `stream`.
  //
  // Reuse is only possible on GPU. Event allocation and recording are coupled
  // in a single operation because on GPU it is recording an event that makes it
  // a "new" event. According to the CUDA documentation it is safe to call
  // cudaEventRecord even if that event may still be in use on the device; APIs
  // such as cudaStreamWaitEvent capture the state of the event at the time of
  // the host-side call and are not affected by a later host-side
  // cudaEventRecord.
  StatusOr<Handle> ThenAllocateAndRecordEvent(se::Stream* stream);

  // Version of ThenAllocateAndRecordEvent split into two phases; this is
  // sometimes helpful if we want to avoid failures by preallocating events.
  StatusOr<Handle> AllocateEvent(se::StreamExecutor* executor);
  void ThenRecordEvent(se::Stream* stream, EventPool::Handle& handle);

 private:
  const bool allow_reuse_;

  absl::Mutex mu_;
  std::stack<std::unique_ptr<se::Event>> free_events_ ABSL_GUARDED_BY(mu_);
  uint64_t next_sequence_number_ ABSL_GUARDED_BY(mu_);
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PJRT_EVENT_POOL_H_
