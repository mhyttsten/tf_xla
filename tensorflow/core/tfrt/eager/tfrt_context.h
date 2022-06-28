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
#ifndef TENSORFLOW_CORE_TFRT_EAGER_TFRT_CONTEXT_H_
#define TENSORFLOW_CORE_TFRT_EAGER_TFRT_CONTEXT_H_
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
class MHTracer_DTPStensorflowPScorePStfrtPSeagerPStfrt_contextDTh {
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
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPStfrt_contextDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePStfrtPSeagerPStfrt_contextDTh() {
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
#include <utility>

#include "tensorflow/c/eager/immediate_execution_context.h"
#include "tensorflow/core/platform/threadpool_interface.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/tfrt/runtime/tf_threadpool_concurrent_work_queue.h"
#include "tfrt/host_context/resource_context.h"  // from @tf_runtime

namespace tensorflow {
class EagerContext;
class DynamicDeviceMgr;
}
namespace tfrt {
class HostContext;
class CoreRuntime;
class OpHandler;

namespace tf {

// Wraps an `Eigen::ThreadPoolInterface` as a
// `tensorflow::thread::ThreadPoolInterface`.
//
// Copied from internal directory: http://shortn/_jsmzLpQu7q
class ThreadPoolInterfaceWrapper
    : public tensorflow::thread::ThreadPoolInterface {
 public:
  explicit ThreadPoolInterfaceWrapper(Eigen::ThreadPoolInterface* thread_pool)
      : thread_pool_{thread_pool} {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPStfrt_contextDTh mht_0(mht_0_v, 215, "", "./tensorflow/core/tfrt/eager/tfrt_context.h", "ThreadPoolInterfaceWrapper");

    DCHECK(thread_pool);
  }

  void Schedule(std::function<void()> fn) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPStfrt_contextDTh mht_1(mht_1_v, 222, "", "./tensorflow/core/tfrt/eager/tfrt_context.h", "Schedule");

    return thread_pool().Schedule(std::move(fn));
  }

  void ScheduleWithHint(std::function<void()> fn, int start, int end) override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPStfrt_contextDTh mht_2(mht_2_v, 229, "", "./tensorflow/core/tfrt/eager/tfrt_context.h", "ScheduleWithHint");

    return thread_pool().ScheduleWithHint(std::move(fn), start, end);
  }

  void Cancel() override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPStfrt_contextDTh mht_3(mht_3_v, 236, "", "./tensorflow/core/tfrt/eager/tfrt_context.h", "Cancel");
 thread_pool().Cancel(); }

  int NumThreads() const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPStfrt_contextDTh mht_4(mht_4_v, 241, "", "./tensorflow/core/tfrt/eager/tfrt_context.h", "NumThreads");
 return thread_pool().NumThreads(); }

  int CurrentThreadId() const override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPStfrt_contextDTh mht_5(mht_5_v, 246, "", "./tensorflow/core/tfrt/eager/tfrt_context.h", "CurrentThreadId");

    return thread_pool().CurrentThreadId();
  }

 private:
  Eigen::ThreadPoolInterface& thread_pool() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPStfrt_contextDTh mht_6(mht_6_v, 254, "", "./tensorflow/core/tfrt/eager/tfrt_context.h", "thread_pool");

    DCHECK(thread_pool_);
    return *thread_pool_;
  }

  // Not owning pointer to the thread pool.
  Eigen::ThreadPoolInterface* thread_pool_ = nullptr;
};

// This class defines a list of objects needed to support execution with TFRT.
class TfrtContext {
 public:
  TfrtContext(
      const tensorflow::SessionOptions& opts,
      tensorflow::ContextDevicePlacementPolicy default_device_placement_policy,
      bool is_async);
  ~TfrtContext();

  HostContext* GetHostContext() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPStfrt_contextDTh mht_7(mht_7_v, 275, "", "./tensorflow/core/tfrt/eager/tfrt_context.h", "GetHostContext");
 return host_context_; }
  CoreRuntime* GetCoreRuntime() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPStfrt_contextDTh mht_8(mht_8_v, 279, "", "./tensorflow/core/tfrt/eager/tfrt_context.h", "GetCoreRuntime");
 return corert_.get(); }
  tensorflow::EagerContext* GetEagerContext() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPStfrt_contextDTh mht_9(mht_9_v, 283, "", "./tensorflow/core/tfrt/eager/tfrt_context.h", "GetEagerContext");
 return eager_context_; }
  const tensorflow::EagerContext* GetEagerContext() const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPStfrt_contextDTh mht_10(mht_10_v, 287, "", "./tensorflow/core/tfrt/eager/tfrt_context.h", "GetEagerContext");

    return eager_context_;
  }
  OpHandler* GetFallbackOpHandler() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPStfrt_contextDTh mht_11(mht_11_v, 293, "", "./tensorflow/core/tfrt/eager/tfrt_context.h", "GetFallbackOpHandler");
 return fallback_op_handler_; }

  ResourceContext* GetResourceContext() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPStfrt_contextDTh mht_12(mht_12_v, 298, "", "./tensorflow/core/tfrt/eager/tfrt_context.h", "GetResourceContext");
 return &resource_context_; }

  tensorflow::tfrt_stub::TfThreadPoolWorkQueue* GetTfThreadPoolWorkQueue() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPStfrt_contextDTh mht_13(mht_13_v, 303, "", "./tensorflow/core/tfrt/eager/tfrt_context.h", "GetTfThreadPoolWorkQueue");

    return tf_thread_pool_work_queue_.get();
  }

  const tensorflow::DeviceNameUtils::ParsedName& HostCPUParsedName() const;

  bool IsAsync() const;

 private:
  std::unique_ptr<CoreRuntime> corert_;
  ::tfrt::HostContext* host_context_;
  OpHandler* fallback_op_handler_;
  ResourceContext resource_context_;
  tensorflow::EagerContext* eager_context_;
  std::unique_ptr<ThreadPoolInterfaceWrapper> eager_ctx_thread_pool_;

  // Manage the local thread pool's lifetime because the wrapper does not own
  // the thread pool.
  std::unique_ptr<tensorflow::thread::ThreadPool> local_thread_pool_;
  std::unique_ptr<ThreadPoolInterfaceWrapper> local_thread_pool_wrapper_;
  std::unique_ptr<tensorflow::tfrt_stub::TfThreadPoolWorkQueue>
      tf_thread_pool_work_queue_;
};

}  // namespace tf
}  // namespace tfrt

#endif  // TENSORFLOW_CORE_TFRT_EAGER_TFRT_CONTEXT_H_
