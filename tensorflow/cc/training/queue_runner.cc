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
class MHTracer_DTPStensorflowPSccPStrainingPSqueue_runnerDTcc {
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
   MHTracer_DTPStensorflowPSccPStrainingPSqueue_runnerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSccPStrainingPSqueue_runnerDTcc() {
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

#include "tensorflow/cc/training/queue_runner.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {

Status QueueRunner::New(const QueueRunnerDef& queue_runner_def,
                        std::unique_ptr<QueueRunner>* result) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSccPStrainingPSqueue_runnerDTcc mht_0(mht_0_v, 192, "", "./tensorflow/cc/training/queue_runner.cc", "QueueRunner::New");

  result->reset(new QueueRunner());
  return (*result)->Init(queue_runner_def);
}

Status QueueRunner::New(const QueueRunnerDef& queue_runner_def,
                        Coordinator* coord,
                        std::unique_ptr<QueueRunner>* result) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSccPStrainingPSqueue_runnerDTcc mht_1(mht_1_v, 202, "", "./tensorflow/cc/training/queue_runner.cc", "QueueRunner::New");

  result->reset(new QueueRunner());
  (*result)->coord_ = coord;
  return (*result)->Init(queue_runner_def);
}

void QueueRunner::AddErrorCallback(const std::function<void(Status)>& cb) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSccPStrainingPSqueue_runnerDTcc mht_2(mht_2_v, 211, "", "./tensorflow/cc/training/queue_runner.cc", "QueueRunner::AddErrorCallback");

  mutex_lock l(cb_mu_);
  callbacks_.push_back(cb);
}

void QueueRunner::ClearErrorCallbacks() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSccPStrainingPSqueue_runnerDTcc mht_3(mht_3_v, 219, "", "./tensorflow/cc/training/queue_runner.cc", "QueueRunner::ClearErrorCallbacks");

  mutex_lock l(cb_mu_);
  callbacks_.clear();
}

Status QueueRunner::Init(const QueueRunnerDef& queue_runner_def) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSccPStrainingPSqueue_runnerDTcc mht_4(mht_4_v, 227, "", "./tensorflow/cc/training/queue_runner.cc", "QueueRunner::Init");

  queue_name_ = queue_runner_def.queue_name();
  enqueue_op_names_.clear();
  enqueue_op_names_.insert(enqueue_op_names_.end(),
                           queue_runner_def.enqueue_op_name().begin(),
                           queue_runner_def.enqueue_op_name().end());
  size_t op_names_size = enqueue_op_names_.size();
  if (op_names_size > kint32max) {
    return Status(error::INVALID_ARGUMENT,
                  "Enqueue ops to run cannot exceed kint32max");
  }
  runs_ = static_cast<int>(op_names_size);
  if (runs_ == 0) {
    return Status(error::INVALID_ARGUMENT, "Empty enqueue ops to run.");
  }
  close_op_name_ = queue_runner_def.close_op_name();
  cancel_op_name_ = queue_runner_def.cancel_op_name();
  if (queue_runner_def.queue_closed_exception_types_size() == 0) {
    queue_closed_exception_types_.insert(error::OUT_OF_RANGE);
  } else {
    for (const auto& code : queue_runner_def.queue_closed_exception_types()) {
      queue_closed_exception_types_.insert(static_cast<int>(code));
    }
  }

  int nthreads = runs_;
  if (coord_) {
    // One more thread to call Stop()
    nthreads++;
  }
  thread_pool_.reset(new thread::ThreadPool(
      Env::Default(), SanitizeThreadSuffix(queue_name_), nthreads));

  return Status::OK();
}

QueueRunner::~QueueRunner() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSccPStrainingPSqueue_runnerDTcc mht_5(mht_5_v, 266, "", "./tensorflow/cc/training/queue_runner.cc", "QueueRunner::~QueueRunner");

  // Cannot run Stop() here because the session might already be closed or
  // destroyed.
  Join().IgnoreError();
}

Status QueueRunner::Start(Session* sess) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSccPStrainingPSqueue_runnerDTcc mht_6(mht_6_v, 275, "", "./tensorflow/cc/training/queue_runner.cc", "QueueRunner::Start");
 return Start(sess, 0); }

Status QueueRunner::StartAndCollectCostGraph(Session* sess,
                                             const RunOptions& run_options) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSccPStrainingPSqueue_runnerDTcc mht_7(mht_7_v, 281, "", "./tensorflow/cc/training/queue_runner.cc", "QueueRunner::StartAndCollectCostGraph");

  SetRunArgumentsAndCostGraph(run_options);
  return Start(sess, 0);
}

Status QueueRunner::Start(Session* sess, int wait_for) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSccPStrainingPSqueue_runnerDTcc mht_8(mht_8_v, 289, "", "./tensorflow/cc/training/queue_runner.cc", "QueueRunner::Start");

  counter_.reset(new BlockingCounter(runs_));
  for (const string& enqueue_op : enqueue_op_names_) {
    thread_pool_->Schedule(
        std::bind(&QueueRunner::Run, this, sess, enqueue_op));
  }
  if (coord_) {
    thread_pool_->Schedule(std::bind(&QueueRunner::Stop, this, sess));
  }
  // Wait for up to 'wait_for' milliseconds.
  if (wait_for > 0) {
    if (!counter_->WaitFor(std::chrono::milliseconds(wait_for))) {
      return Status(error::DEADLINE_EXCEEDED,
                    "Queues not fed before the timeout");
    }
    // Check the status of the queue runner as well as the result of the enqueue
    // operations.
    mutex_lock l(mu_);
    if (!enqueue_status_.ok()) {
      return enqueue_status_;
    } else {
      return status_;
    }
  }
  return Status::OK();
}

Status QueueRunner::StartAndCollectCostGraph(Session* session, int wait_for_ms,
                                             const RunOptions& run_options) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSccPStrainingPSqueue_runnerDTcc mht_9(mht_9_v, 320, "", "./tensorflow/cc/training/queue_runner.cc", "QueueRunner::StartAndCollectCostGraph");

  SetRunArgumentsAndCostGraph(run_options);
  return Start(session, wait_for_ms);
}

void QueueRunner::Stop(Session* sess) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSccPStrainingPSqueue_runnerDTcc mht_10(mht_10_v, 328, "", "./tensorflow/cc/training/queue_runner.cc", "QueueRunner::Stop");

  if (coord_ != nullptr) {
    coord_->WaitForStop();
  }
  if (!cancel_op_name_.empty()) {
    UpdateStatus(RealRun(sess, cancel_op_name_, false));
  }
  stopped_ = true;
}

Status QueueRunner::Join() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSccPStrainingPSqueue_runnerDTcc mht_11(mht_11_v, 341, "", "./tensorflow/cc/training/queue_runner.cc", "QueueRunner::Join");

  thread_pool_.reset();
  mutex_lock l(mu_);
  return status_;
}

void QueueRunner::UpdateStatus(const Status& status) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSccPStrainingPSqueue_runnerDTcc mht_12(mht_12_v, 350, "", "./tensorflow/cc/training/queue_runner.cc", "QueueRunner::UpdateStatus");

  {
    mutex_lock l(mu_);
    if (!status_.ok() || status.ok() || IsQueueClosed(status)) {
      return;
    }
    status_ = status;
  }
  if (coord_) {
    coord_->ReportStatus(status);
  }
  mutex_lock l(cb_mu_);
  for (auto& cb : callbacks_) {
    cb(status);
  }
}

void QueueRunner::Run(Session* sess, const string& enqueue_op) {
   std::vector<std::string> mht_13_v;
   mht_13_v.push_back("enqueue_op: \"" + enqueue_op + "\"");
   MHTracer_DTPStensorflowPSccPStrainingPSqueue_runnerDTcc mht_13(mht_13_v, 371, "", "./tensorflow/cc/training/queue_runner.cc", "QueueRunner::Run");

  bool first_iteration = true;
  Status status;
  while (status.ok()) {
    if (coord_ && coord_->ShouldStop()) {
      break;
    }
    status = RealRun(sess, enqueue_op, true);
    if (first_iteration) {
      if (!status.ok()) {
        mutex_lock l(mu_);
        enqueue_status_ = status;
      }
      counter_->DecrementCount();
      first_iteration = false;
    }
  }
  bool last_run = false;
  {
    mutex_lock l(mu_);
    runs_--;
    last_run = (runs_ == 0);
  }

  // Close the queue unless the coordinator is shutting down since the cancel op
  // will be run anyway in this case.
  if (IsQueueClosed(status) && (!coord_ || !coord_->ShouldStop())) {
    if (last_run && !close_op_name_.empty()) {
      UpdateStatus(RealRun(sess, close_op_name_, false));
    }
  } else if (!status.ok()) {
    LOG(ERROR) << "Queue runner thread got a failure status: "
               << status.ToString();
    UpdateStatus(status);
    if (coord_) {
      coord_->RequestStop().IgnoreError();
    }
  }
}

Status QueueRunner::GetStatus() {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSccPStrainingPSqueue_runnerDTcc mht_14(mht_14_v, 414, "", "./tensorflow/cc/training/queue_runner.cc", "QueueRunner::GetStatus");

  mutex_lock l(mu_);
  return status_;
}

Status QueueRunner::ExportCostGraph(CostGraphDef* cost_graph) const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSccPStrainingPSqueue_runnerDTcc mht_15(mht_15_v, 422, "", "./tensorflow/cc/training/queue_runner.cc", "QueueRunner::ExportCostGraph");

  if (!cg_mu_) {
    return Status(error::FAILED_PRECONDITION,
                  "This QueueRunner doesn't collect a cost graph.");
  }
  mutex_lock l(*cg_mu_);
  cost_graph->MergeFrom(*cost_graph_);
  return Status::OK();
}

void QueueRunner::SetRunArgumentsAndCostGraph(const RunOptions& run_options) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSccPStrainingPSqueue_runnerDTcc mht_16(mht_16_v, 435, "", "./tensorflow/cc/training/queue_runner.cc", "QueueRunner::SetRunArgumentsAndCostGraph");

  cg_mu_.reset(new mutex());
  {
    mutex_lock l(*cg_mu_);
    cost_graph_.reset(new CostGraphDef());
  }
  run_options_ = run_options;
}

Status QueueRunner::RealRun(Session* sess, const string& op,
                            bool update_costs) {
   std::vector<std::string> mht_17_v;
   mht_17_v.push_back("op: \"" + op + "\"");
   MHTracer_DTPStensorflowPSccPStrainingPSqueue_runnerDTcc mht_17(mht_17_v, 449, "", "./tensorflow/cc/training/queue_runner.cc", "QueueRunner::RealRun");

  Status s;
  if (update_costs && cg_mu_) {
    RunMetadata metadata;
    s = sess->Run(run_options_, {}, {}, {op}, nullptr, &metadata);
    mutex_lock l(*cg_mu_);
    cost_graph_->Swap(metadata.mutable_cost_graph());
  } else {
    s = sess->Run({}, {}, {op}, nullptr);
  }
  return s;
}

}  // namespace tensorflow
