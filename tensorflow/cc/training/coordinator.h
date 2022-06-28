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

#ifndef TENSORFLOW_CC_TRAINING_COORDINATOR_H_
#define TENSORFLOW_CC_TRAINING_COORDINATOR_H_
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
class MHTracer_DTPStensorflowPSccPStrainingPScoordinatorDTh {
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
   MHTracer_DTPStensorflowPSccPStrainingPScoordinatorDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSccPStrainingPScoordinatorDTh() {
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


#include <atomic>
#include <memory>
#include <unordered_set>
#include <vector>

#include "tensorflow/core/framework/cost_graph.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"

namespace tensorflow {

/// The abstract interface for runners which must implement the Join and the
/// IsRunning function.
class RunnerInterface {
 public:
  virtual ~RunnerInterface() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSccPStrainingPScoordinatorDTh mht_0(mht_0_v, 206, "", "./tensorflow/cc/training/coordinator.h", "~RunnerInterface");
}
  virtual Status Join() = 0;
  virtual Status ExportCostGraph(CostGraphDef* cost_graph) const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSccPStrainingPScoordinatorDTh mht_1(mht_1_v, 211, "", "./tensorflow/cc/training/coordinator.h", "ExportCostGraph");

    return Status(error::INVALID_ARGUMENT, "No cost model to export.");
  }
  /// Returns true iff the runner is running, i.e. if it is trying to populate
  /// its queue.
  virtual bool IsRunning() const = 0;
};

/// Coordinator class manages the termination of a collection of QueueRunners.
/// Without a coordinator, QueueRunners have to be joined in a specific order;
/// otherwise the QueueRunner::Join() could sometimes hang. The
/// Coordinator::RequestStop() plays the key role which notifies all running
/// threads under a coordinator to stop. This function could be called by any
/// thread or any client.
/// Usage, in the client:
///   Coordinator coord;
///   std::unique_ptr<QueueRunner> qr(&coord, ...);
///   qr.Start(session);
///   coord.RegisterRunner(std::move(qr));
///   /// do some work
///   TF_CHECK_OK(coord.Join());
/// In each thread of QueueRunner, the coordinator needs to be used as:
///   void Run() {
///     while (!coord->ShouldStop()) {
///       /// do some work
///       if (error) {
///         coord->RequestStop();
///         coord->ReportStatus(error_status);
///       }
///     }
///   }
class Coordinator {
 public:
  Coordinator();

  /// Constructor with a list of error codes which would not be taken as errors
  /// in status reporting.
  Coordinator(const std::vector<error::Code>& clean_stop_errors);

  /// In the destructor, RequestStop() and Join() would be called.
  ~Coordinator();

  /// Registers a runner, i.e. a unit of running threads which is usually a
  /// QueueRunner. It takes the ownership of runner to avoid lifecycle-related
  /// problems. Note, the coordinator would not start these threads; they are
  /// supposed to be in running state when they are registered here.
  Status RegisterRunner(std::unique_ptr<RunnerInterface> runner);

  /// Returns true iff all the registered runners have been stopped.
  bool AllRunnersStopped();

  /// Requests all running threads to stop.
  Status RequestStop();

  /// Returns true if its RequestStop() has been called.
  bool ShouldStop();

  /// Joins all threads, returns OK or the first reported and unexpected status.
  Status Join();

  /// Reports status to the coordinator. This is usually called by threads.
  void ReportStatus(const Status& status);

  /// Returns the latest status.
  Status GetStatus();

  /// Returns immediately if the coordinator is stopped or blocks until
  /// RequestStop() is called.
  void WaitForStop();

  // Returns the cost graph from stored run metadata in registered runners.
  Status ExportCostGraph(CostGraphDef* cost_graph) const;

 private:
  std::unordered_set<int> clean_stop_errors_;
  condition_variable wait_for_stop_;

  mutex mu_;
  bool should_stop_ TF_GUARDED_BY(mu_);

  mutex status_lock_;
  Status status_ TF_GUARDED_BY(status_lock_);

  mutable mutex runners_lock_;
  std::vector<std::unique_ptr<RunnerInterface>> runners_
      TF_GUARDED_BY(runners_lock_);

  TF_DISALLOW_COPY_AND_ASSIGN(Coordinator);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CC_TRAINING_COORDINATOR_H_
