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
class MHTracer_DTPStensorflowPSccPStrainingPScoordinatorDTcc {
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
   MHTracer_DTPStensorflowPSccPStrainingPScoordinatorDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSccPStrainingPScoordinatorDTcc() {
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

#include "tensorflow/cc/training/coordinator.h"

namespace tensorflow {

Coordinator::Coordinator() : Coordinator(std::vector<error::Code>()) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSccPStrainingPScoordinatorDTcc mht_0(mht_0_v, 189, "", "./tensorflow/cc/training/coordinator.cc", "Coordinator::Coordinator");
}

Coordinator::Coordinator(const std::vector<error::Code>& clean_stop_errors)
    : should_stop_(false) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSccPStrainingPScoordinatorDTcc mht_1(mht_1_v, 195, "", "./tensorflow/cc/training/coordinator.cc", "Coordinator::Coordinator");

  if (clean_stop_errors.empty()) {
    clean_stop_errors_.insert(error::OUT_OF_RANGE);
  } else {
    for (const auto& code : clean_stop_errors) {
      clean_stop_errors_.insert(static_cast<int>(code));
    }
  }
}

Coordinator::~Coordinator() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSccPStrainingPScoordinatorDTcc mht_2(mht_2_v, 208, "", "./tensorflow/cc/training/coordinator.cc", "Coordinator::~Coordinator");

  RequestStop().IgnoreError();
  Join().IgnoreError();
}

Status Coordinator::RegisterRunner(std::unique_ptr<RunnerInterface> runner) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSccPStrainingPScoordinatorDTcc mht_3(mht_3_v, 216, "", "./tensorflow/cc/training/coordinator.cc", "Coordinator::RegisterRunner");

  {
    mutex_lock l(mu_);
    if (should_stop_) {
      return Status(error::FAILED_PRECONDITION,
                    "The coordinator has been stopped.");
    }
  }
  mutex_lock l(runners_lock_);
  runners_.push_back(std::move(runner));
  return Status::OK();
}

bool Coordinator::AllRunnersStopped() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSccPStrainingPScoordinatorDTcc mht_4(mht_4_v, 232, "", "./tensorflow/cc/training/coordinator.cc", "Coordinator::AllRunnersStopped");

  mutex_lock l(runners_lock_);
  for (const auto& runner : runners_) {
    if (runner->IsRunning()) {
      return false;
    }
  }
  return true;
}

Status Coordinator::RequestStop() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSccPStrainingPScoordinatorDTcc mht_5(mht_5_v, 245, "", "./tensorflow/cc/training/coordinator.cc", "Coordinator::RequestStop");

  mutex_lock l(mu_);
  if (should_stop_) {
    return Status(error::FAILED_PRECONDITION,
                  "The Coordinator is not running.");
  }
  should_stop_ = true;
  wait_for_stop_.notify_all();
  return Status::OK();
}

bool Coordinator::ShouldStop() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSccPStrainingPScoordinatorDTcc mht_6(mht_6_v, 259, "", "./tensorflow/cc/training/coordinator.cc", "Coordinator::ShouldStop");

  mutex_lock l(mu_);
  return should_stop_;
}

Status Coordinator::Join() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSccPStrainingPScoordinatorDTcc mht_7(mht_7_v, 267, "", "./tensorflow/cc/training/coordinator.cc", "Coordinator::Join");

  // TODO(yuefengz): deal with stragglers.
  {
    mutex_lock l(mu_);
    if (!should_stop_) {
      return Status(error::FAILED_PRECONDITION,
                    "Joining coordinator without requesting to stop.");
    }
  }

  {
    mutex_lock l(runners_lock_);
    for (const auto& t : runners_) {
      ReportStatus(t->Join());
    }
    runners_.clear();
  }
  return GetStatus();
}

void Coordinator::ReportStatus(const Status& status) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSccPStrainingPScoordinatorDTcc mht_8(mht_8_v, 290, "", "./tensorflow/cc/training/coordinator.cc", "Coordinator::ReportStatus");

  mutex_lock l(status_lock_);
  if (status.ok() || !status_.ok() ||
      clean_stop_errors_.count(static_cast<int>(status.code())) > 0) {
    return;
  }
  status_ = status;
}

Status Coordinator::GetStatus() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSccPStrainingPScoordinatorDTcc mht_9(mht_9_v, 302, "", "./tensorflow/cc/training/coordinator.cc", "Coordinator::GetStatus");

  mutex_lock l(status_lock_);
  return status_;
}

void Coordinator::WaitForStop() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSccPStrainingPScoordinatorDTcc mht_10(mht_10_v, 310, "", "./tensorflow/cc/training/coordinator.cc", "Coordinator::WaitForStop");

  mutex_lock l(mu_);
  while (!should_stop_) {
    wait_for_stop_.wait(l);
  }
}

Status Coordinator::ExportCostGraph(CostGraphDef* cost_graph) const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSccPStrainingPScoordinatorDTcc mht_11(mht_11_v, 320, "", "./tensorflow/cc/training/coordinator.cc", "Coordinator::ExportCostGraph");

  mutex_lock l(runners_lock_);
  for (auto& t : runners_) {
    Status s = t->ExportCostGraph(cost_graph);
    if (!s.ok()) {
      return s;
    }
  }
  return Status::OK();
}

}  // namespace tensorflow
