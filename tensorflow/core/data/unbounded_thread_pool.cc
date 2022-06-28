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
class MHTracer_DTPStensorflowPScorePSdataPSunbounded_thread_poolDTcc {
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
   MHTracer_DTPStensorflowPScorePSdataPSunbounded_thread_poolDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdataPSunbounded_thread_poolDTcc() {
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

#include "tensorflow/core/data/unbounded_thread_pool.h"

#include "absl/memory/memory.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/resource.h"
#include "tensorflow/core/platform/unbounded_work_queue.h"

namespace tensorflow {
namespace data {

// A logical implementation of the `tensorflow::Thread` interface that uses
// physical threads in an `UnboundedThreadPool` to perform the work.
//
// NOTE: This object represents a logical thread of control that may be mapped
// onto the same physical thread as other work items that are submitted to the
// same `UnboundedThreadPool`.
class UnboundedThreadPool::LogicalThreadWrapper : public Thread {
 public:
  explicit LogicalThreadWrapper(std::shared_ptr<Notification> done)
      : done_(std::move(done)) {}

  ~LogicalThreadWrapper() override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdataPSunbounded_thread_poolDTcc mht_0(mht_0_v, 208, "", "./tensorflow/core/data/unbounded_thread_pool.cc", "~LogicalThreadWrapper");

    // NOTE: The `Thread` destructor is expected to "join" the created thread,
    // but the physical thread may continue to execute after the work for this
    // thread is complete. We simulate this by waiting on a notification that
    // the thread's work function will notify when it is complete.
    done_->WaitForNotification();
  }

 private:
  std::shared_ptr<Notification> done_;
};

// A lightweight wrapper for creating logical threads in a `UnboundedThreadPool`
// that can be shared (e.g.) in an `IteratorContext`.
class UnboundedThreadPool::LogicalThreadFactory : public ThreadFactory {
 public:
  explicit LogicalThreadFactory(UnboundedThreadPool* pool) : pool_(pool) {}

  std::unique_ptr<Thread> StartThread(const string& name,
                                      std::function<void()> fn) override {
    auto done = std::make_shared<Notification>();
    pool_->ScheduleOnWorkQueue(std::move(fn), done);
    return absl::make_unique<LogicalThreadWrapper>(std::move(done));
  }

 private:
  UnboundedThreadPool* const pool_;  // Not owned.
};

std::shared_ptr<ThreadFactory> UnboundedThreadPool::get_thread_factory() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdataPSunbounded_thread_poolDTcc mht_1(mht_1_v, 240, "", "./tensorflow/core/data/unbounded_thread_pool.cc", "UnboundedThreadPool::get_thread_factory");

  return std::make_shared<LogicalThreadFactory>(this);
}

void UnboundedThreadPool::Schedule(std::function<void()> fn) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdataPSunbounded_thread_poolDTcc mht_2(mht_2_v, 247, "", "./tensorflow/core/data/unbounded_thread_pool.cc", "UnboundedThreadPool::Schedule");

  auto tagged_fn = [fn = std::move(fn)]() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSdataPSunbounded_thread_poolDTcc mht_3(mht_3_v, 251, "", "./tensorflow/core/data/unbounded_thread_pool.cc", "lambda");

    tensorflow::ResourceTagger tag(kTFDataResourceTag, "ThreadPool");
    fn();
  };
  ScheduleOnWorkQueue(std::move(tagged_fn), /*done=*/nullptr);
}

int UnboundedThreadPool::NumThreads() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSdataPSunbounded_thread_poolDTcc mht_4(mht_4_v, 261, "", "./tensorflow/core/data/unbounded_thread_pool.cc", "UnboundedThreadPool::NumThreads");
 return -1; }

int UnboundedThreadPool::CurrentThreadId() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSdataPSunbounded_thread_poolDTcc mht_5(mht_5_v, 266, "", "./tensorflow/core/data/unbounded_thread_pool.cc", "UnboundedThreadPool::CurrentThreadId");
 return -1; }

namespace {
void WorkQueueFunc(const std::function<void()>& fn,
                   std::shared_ptr<Notification> done) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSdataPSunbounded_thread_poolDTcc mht_6(mht_6_v, 273, "", "./tensorflow/core/data/unbounded_thread_pool.cc", "WorkQueueFunc");

  fn();
  if (done) {
    done->Notify();
  }
}
}  // namespace

void UnboundedThreadPool::ScheduleOnWorkQueue(
    std::function<void()> fn, std::shared_ptr<Notification> done) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSdataPSunbounded_thread_poolDTcc mht_7(mht_7_v, 285, "", "./tensorflow/core/data/unbounded_thread_pool.cc", "UnboundedThreadPool::ScheduleOnWorkQueue");

  unbounded_work_queue_.Schedule(
      std::bind(&WorkQueueFunc, std::move(fn), std::move(done)));
}

}  // namespace data
}  // namespace tensorflow
