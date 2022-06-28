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
class MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSunbounded_work_queueDTcc {
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
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSunbounded_work_queueDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSunbounded_work_queueDTcc() {
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

#include "tensorflow/core/platform/default/unbounded_work_queue.h"

#include "absl/memory/memory.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/numa.h"

namespace tensorflow {

UnboundedWorkQueue::UnboundedWorkQueue(Env* env, const string& thread_name,
                                       const ThreadOptions& thread_options)
    : env_(env), thread_name_(thread_name), thread_options_(thread_options) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("thread_name: \"" + thread_name + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSunbounded_work_queueDTcc mht_0(mht_0_v, 197, "", "./tensorflow/core/platform/default/unbounded_work_queue.cc", "UnboundedWorkQueue::UnboundedWorkQueue");
}

UnboundedWorkQueue::~UnboundedWorkQueue() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSunbounded_work_queueDTcc mht_1(mht_1_v, 202, "", "./tensorflow/core/platform/default/unbounded_work_queue.cc", "UnboundedWorkQueue::~UnboundedWorkQueue");

  {
    mutex_lock l(work_queue_mu_);
    // Wake up all `PooledThreadFunc` threads and cause them to terminate before
    // joining them when `threads_` is cleared.
    cancelled_ = true;
    work_queue_cv_.notify_all();
    if (!work_queue_.empty()) {
      LOG(ERROR) << "UnboundedWorkQueue named \"" << thread_name_ << "\" was "
                 << "deleted with pending work in its queue. This may indicate "
                 << "a potential use-after-free bug.";
    }
  }

  {
    mutex_lock l(thread_pool_mu_);
    // Clear the list of pooled threads, which will eventually terminate due to
    // the previous notification.
    //
    // NOTE: It is safe to do this while holding `thread_pool_mu_`, because
    // no subsequent calls to `this->Schedule()` should be issued after the
    // destructor starts.
    thread_pool_.clear();
  }
}

void UnboundedWorkQueue::Schedule(WorkFunction fn) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSunbounded_work_queueDTcc mht_2(mht_2_v, 231, "", "./tensorflow/core/platform/default/unbounded_work_queue.cc", "UnboundedWorkQueue::Schedule");

  // Enqueue a work item for the new thread's function, and wake up a
  // cached thread to process it.
  mutex_lock l(work_queue_mu_);
  work_queue_.push_back(std::move(fn));
  work_queue_cv_.notify_one();
  // NOTE: The queue may be non-empty, so we must account for queued work when
  // considering how many threads are free.
  if (work_queue_.size() > num_idle_threads_) {
    // Spawn a new physical thread to process the given function.
    // NOTE: `PooledThreadFunc` will eventually increment `num_idle_threads_`
    // at the beginning of its work loop.
    Thread* new_thread =
        env_->StartThread({}, thread_name_, [this]() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSunbounded_work_queueDTcc mht_3(mht_3_v, 247, "", "./tensorflow/core/platform/default/unbounded_work_queue.cc", "lambda");
 PooledThreadFunc(); });

    mutex_lock l(thread_pool_mu_);
    thread_pool_.emplace_back(new_thread);
  }
}

void UnboundedWorkQueue::PooledThreadFunc() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSunbounded_work_queueDTcc mht_4(mht_4_v, 257, "", "./tensorflow/core/platform/default/unbounded_work_queue.cc", "UnboundedWorkQueue::PooledThreadFunc");

  // If specified, make sure the thread runs on the correct NUMA node.
  if (thread_options_.numa_node != port::kNUMANoAffinity) {
    port::NUMASetThreadNodeAffinity(thread_options_.numa_node);
  }

  while (true) {
    WorkFunction fn;
    {
      mutex_lock l(work_queue_mu_);
      ++num_idle_threads_;
      while (!cancelled_ && work_queue_.empty()) {
        // Wait for a new work function to be submitted, or the cache to be
        // destroyed.
        work_queue_cv_.wait(l);
      }
      if (cancelled_) {
        return;
      }
      fn = std::move(work_queue_.front());
      work_queue_.pop_front();
      --num_idle_threads_;
    }

    fn();
  }
}

}  // namespace tensorflow
