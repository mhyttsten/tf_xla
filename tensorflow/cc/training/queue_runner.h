/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CC_TRAINING_QUEUE_RUNNER_H_
#define TENSORFLOW_CC_TRAINING_QUEUE_RUNNER_H_
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
class MHTracer_DTPStensorflowPSccPStrainingPSqueue_runnerDTh {
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
   MHTracer_DTPStensorflowPSccPStrainingPSqueue_runnerDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSccPStrainingPSqueue_runnerDTh() {
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
#include <string>
#include <unordered_set>
#include <vector>

#include "tensorflow/cc/training/coordinator.h"
#include "tensorflow/core/lib/core/blocking_counter.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tensorflow/core/protobuf/queue_runner.pb.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {

/// QueueRunner class imitates the behavior of the python version of QueueRunner
/// which creates a thread for each enqueue op, runs close op on completion.
class QueueRunner : public RunnerInterface {
 public:
  /// Creates a new QueueRunner from proto.
  // TODO(yuefengz): we may want to initialize from queues and ops in the
  // future.
  static Status New(const QueueRunnerDef& queue_runner_def,
                    std::unique_ptr<QueueRunner>* result);

  /// Creates a new QueueRunner with a coordinator, see coordinator.h for usage.
  static Status New(const QueueRunnerDef& queue_runner_def, Coordinator* coord,
                    std::unique_ptr<QueueRunner>* result);

  /// Adds a callback that the queue runner will call when it detects an error.
  void AddErrorCallback(const std::function<void(Status)>& cb);

  /// Delete the previously registered callbacks.
  void ClearErrorCallbacks();

  /// The destructor would join all the threads.
  ~QueueRunner();

  /// Starts the queue runner with the given session.
  Status Start(Session* sess);

  /// Starts the queue runner with the given session and sets the run arguments
  /// for sess->Run. It also collects and stores the cost model.
  Status StartAndCollectCostGraph(Session* sess,
                                  const RunOptions& run_options = RunOptions());

  /// Starts the queue runner with the given session, and wait for up to the
  /// specified time (in milliseconds) for the queues to start to fill up.
  Status Start(Session* sess, int wait_for_ms);
  Status StartAndCollectCostGraph(Session* session, int wait_for_ms,
                                  const RunOptions& run_options = RunOptions());

  /// Requests to stop and runs the cancel op. It would be called in a separate
  /// thread when coordinator is set. If there is no coordinator it should be
  /// called before calling Join.
  void Stop(Session* sess);

  /// Joins all the threads. Returns okay if all threads run successfully;
  /// otherwise returns the first captured failure status.
  Status Join() final;

  /// Returns the latest status.
  Status GetStatus();

  // Returns the stored cost model.
  Status ExportCostGraph(CostGraphDef* cost_graph) const override;

 private:
  QueueRunner() : coord_(nullptr), stopped_(false), cg_mu_(nullptr) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSccPStrainingPSqueue_runnerDTh mht_0(mht_0_v, 258, "", "./tensorflow/cc/training/queue_runner.h", "QueueRunner");
}

  // Initializes the instance with the QueueRunnerDef proto.
  Status Init(const QueueRunnerDef& queue_runner_def);

  // The Run function for each thread.
  void Run(Session* sess, const string& enqueue_op);

  // Updates the internal status; it only keeps OK or the first unexpected error
  // status.
  void UpdateStatus(const Status& status);

  bool IsQueueClosed(Status status) const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSccPStrainingPSqueue_runnerDTh mht_1(mht_1_v, 273, "", "./tensorflow/cc/training/queue_runner.h", "IsQueueClosed");

    return queue_closed_exception_types_.count(
               static_cast<int>(status.code())) > 0;
  }

  bool IsRunning() const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSccPStrainingPSqueue_runnerDTh mht_2(mht_2_v, 281, "", "./tensorflow/cc/training/queue_runner.h", "IsRunning");
 return !stopped_; }

  void SetRunArgumentsAndCostGraph(const RunOptions& run_options);

  Status RealRun(Session* sess, const string& op, bool update_costs);

  string queue_name_;
  std::vector<string> enqueue_op_names_;
  string close_op_name_;
  string cancel_op_name_;
  // code::Code casted to int to avoid a hash function.
  std::unordered_set<int> queue_closed_exception_types_;

  std::unique_ptr<thread::ThreadPool> thread_pool_;
  mutex mu_;
  int runs_ = 0;
  Status status_ TF_GUARDED_BY(mu_);
  Status enqueue_status_ TF_GUARDED_BY(mu_);
  std::unique_ptr<BlockingCounter> counter_;

  Coordinator* coord_;

  std::atomic<bool> stopped_;

  mutex cb_mu_;
  std::vector<std::function<void(Status)>> callbacks_;

  mutable std::unique_ptr<mutex> cg_mu_;
  std::unique_ptr<CostGraphDef> cost_graph_ TF_GUARDED_BY(cg_mu_);
  RunOptions run_options_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CC_TRAINING_QUEUE_RUNNER_H_
