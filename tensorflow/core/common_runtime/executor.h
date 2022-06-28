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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_EXECUTOR_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_EXECUTOR_H_
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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSexecutorDTh {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSexecutorDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSexecutorDTh() {
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


#include "absl/time/time.h"
#include "absl/types/optional.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/local_executor_params.h"
#include "tensorflow/core/framework/rendezvous.h"
#include "tensorflow/core/framework/session_state.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/threadpool_interface.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/util/managed_stack_trace.h"

namespace tensorflow {

class StepStatsCollector;

// Executor runs a graph computation.
// Example:
//   Graph* graph = ...;
//      ... construct graph ...
//   Executor* executor;
//   TF_CHECK_OK(NewSimpleExecutor(my_device, graph, &executor));
//   Rendezvous* rendezvous = NewNaiveRendezvous();
//   TF_CHECK_OK(rendezvous->Send("input", some_input_tensor));
//   TF_CHECK_OK(executor->Run({ExecutorOpts, rendezvous, nullptr}));
//   TF_CHECK_OK(rendezvous->Recv("output", &output_tensor));
//   ... ...
//
// Multiple threads can call Executor::Run concurrently.
class Executor {
 public:
  virtual ~Executor() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSexecutorDTh mht_0(mht_0_v, 223, "", "./tensorflow/core/common_runtime/executor.h", "~Executor");
}

  // RunAsync() executes the graph computation. "done" is run when the
  // graph computation completes. If any error happens during the
  // computation, "done" is run and the error is passed to "done".
  //
  // RunAsync() is given a few arguments in Args. The caller must
  // ensure objects passed in Args (rendezvous, stats_collector, etc.)
  // are alive at least until done is invoked. All pointers to the
  // argument objects can be nullptr.
  //
  // "step_id" is a process-wide unique identifier for the step being
  // run. Executors on different devices may receive the same step_id
  // in the case that a step runs Ops on more than one device. The
  // step_id is used for tracking resource usage of a given step.
  //
  // RunAsync() uses the given "rendezvous", if not null, as the
  // mechanism to communicate inputs and outputs of the underlying
  // graph computation.
  //
  // RunAsync() calls "stats_collector", if not null, to keep track of
  // stats. This allows us to collect statistics and traces on demand.
  //
  // RunAsync() is provided a "call_frame", if the executor is used
  // for executing a function, is used to pass arguments and return
  // values between the caller and the callee.
  //
  // RunAsync() uses "cancellation_manager", if not nullptr, to
  // register callbacks that should be called if the graph computation
  // is canceled. Note that the callbacks merely unblock any
  // long-running computation, and a canceled step will terminate by
  // returning/calling the DoneCallback as usual.
  //
  // RunAsync() dispatches closures to "runner". Typically, "runner"
  // is backed up by a bounded threadpool.
  //
  // "start_time_usecs" is a timestamp for the start of RunAsync()
  // execution. Used for system-wide latency metrics.
  struct Args {
    int64_t step_id = 0;
    RendezvousInterface* rendezvous = nullptr;
    StepStatsCollectorInterface* stats_collector = nullptr;
    CallFrameInterface* call_frame = nullptr;
    CancellationManager* cancellation_manager = nullptr;
    SessionState* session_state = nullptr;
    // Unique session identifier. Can be empty.
    string session_handle;
    TensorStore* tensor_store = nullptr;
    ScopedStepContainer* step_container = nullptr;
    CollectiveExecutor* collective_executor = nullptr;
    thread::ThreadPoolInterface* user_intra_op_threadpool = nullptr;
    CoordinationServiceAgent* coordination_service_agent = nullptr;
    int64_t start_time_usecs = 0;
    // The deadline for the kernel to complete by. Empty if unspecified.
    absl::optional<absl::Time> deadline;
    absl::optional<ManagedStackTrace> stack_trace = absl::nullopt;

    // If true, calls Sync() on the device.
    bool sync_on_finish = false;

    typedef std::function<void()> Closure;
    typedef std::function<void(Closure)> Runner;
    Runner runner = nullptr;

    // If true, all kernels will be treated as "inexpensive", and hence executed
    // on the scheduling thread.
    bool run_all_kernels_inline = false;
  };
  typedef std::function<void(const Status&)> DoneCallback;
  virtual void RunAsync(const Args& args, DoneCallback done) = 0;

  // Synchronous wrapper for RunAsync().
  virtual Status Run(const Args& args) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSexecutorDTh mht_1(mht_1_v, 298, "", "./tensorflow/core/common_runtime/executor.h", "Run");

    Status ret;
    Notification n;
    RunAsync(args, [&ret, &n](const Status& s) {
      ret = s;
      n.Notify();
    });
    n.WaitForNotification();
    return ret;
  }
};

// Creates an Executor that computes the given "graph".
//
// If successful, returns the constructed executor in "*executor". Otherwise,
// returns an error status.
//
// "params" provides a set of context for the executor. We expect that
// different context would provide different implementations.
::tensorflow::Status NewLocalExecutor(const LocalExecutorParams& params,
                                      const Graph& graph, Executor** executor);

// A class to help run multiple executors in parallel and wait until
// all of them are complete.
//
// ExecutorBarrier deletes itself after the function returned by Get()
// is called.
class ExecutorBarrier {
 public:
  typedef std::function<void(const Status&)> StatusCallback;

  // Create an ExecutorBarrier for 'num' different executors.
  //
  // 'r' is the shared Rendezvous object that is used to communicate
  // state.  If any of the executors experiences an error, the
  // rendezvous object will be aborted exactly once.
  //
  // 'done' is called after the last executor completes, and
  // ExecutorBarrier is deleted.
  ExecutorBarrier(size_t num, Rendezvous* r, StatusCallback done)
      : rendez_(r), done_cb_(done), pending_(num) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSexecutorDTh mht_2(mht_2_v, 341, "", "./tensorflow/core/common_runtime/executor.h", "ExecutorBarrier");
}

  ~ExecutorBarrier() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSexecutorDTh mht_3(mht_3_v, 346, "", "./tensorflow/core/common_runtime/executor.h", "~ExecutorBarrier");
}

  // Returns a closure that Executors must call when they are done
  // computing, passing the status of their execution as an argument.
  StatusCallback Get() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSexecutorDTh mht_4(mht_4_v, 353, "", "./tensorflow/core/common_runtime/executor.h", "Get");

    return std::bind(&ExecutorBarrier::WhenDone, this, std::placeholders::_1);
  }

 private:
  Rendezvous* rendez_ = nullptr;
  StatusCallback done_cb_ = nullptr;

  mutable mutex mu_;
  int pending_ TF_GUARDED_BY(mu_) = 0;
  StatusGroup status_group_ TF_GUARDED_BY(mu_);

  void WhenDone(const Status& s) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSexecutorDTh mht_5(mht_5_v, 368, "", "./tensorflow/core/common_runtime/executor.h", "WhenDone");

    Rendezvous* error_rendez = nullptr;
    StatusCallback done = nullptr;
    Status status;

    {
      mutex_lock l(mu_);

      // If we are the first error encountered, trigger an abort of the
      // Rendezvous object by this thread only.
      if (status_group_.ok() && !s.ok()) {
        error_rendez = rendez_;
        error_rendez->Ref();
      }

      if (!s.ok() && !StatusGroup::IsDerived(s) &&
          !status_group_.HasLogMessages()) {
        status_group_.AttachLogMessages();
      }

      status_group_.Update(s);

      // If this is the last call to WhenDone, call the final callback
      // below.
      if (--pending_ == 0) {
        CHECK(done_cb_ != nullptr);
        std::swap(done, done_cb_);
        status = status_group_.as_summary_status();
      }
    }

    if (error_rendez != nullptr) {
      error_rendez->StartAbort(
          errors::Aborted("Stopping remaining executors."));
      error_rendez->Unref();
    }

    if (done != nullptr) {
      delete this;
      if (!status.ok()) {
        VLOG(1) << "ExecutorBarrier finished with bad status: " << status;
      }
      done(status);
    }
  }

  TF_DISALLOW_COPY_AND_ASSIGN(ExecutorBarrier);
};

// A few helpers to facilitate create/delete kernels.

// Creates a kernel based on "props" on device "device". The kernel can
// access the functions in the "flib". The caller takes ownership of
// returned "*kernel".
Status CreateNonCachedKernel(Device* device, FunctionLibraryRuntime* flib,
                             const std::shared_ptr<const NodeProperties>& props,
                             int graph_def_version, OpKernel** kernel);

// Deletes "kernel" returned by CreateKernel.
void DeleteNonCachedKernel(OpKernel* kernel);

}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_EXECUTOR_H_
