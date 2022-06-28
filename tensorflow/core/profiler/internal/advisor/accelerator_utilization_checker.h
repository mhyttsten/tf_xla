/* Copyright 2016 The TensorFlow Authors All Rights Reserved.

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
// This checker checks the accelerator's utilization.
#ifndef TENSORFLOW_CORE_PROFILER_INTERNAL_ADVISOR_ACCELERATOR_UTILIZATION_CHECKER_H_
#define TENSORFLOW_CORE_PROFILER_INTERNAL_ADVISOR_ACCELERATOR_UTILIZATION_CHECKER_H_
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
class MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSadvisorPSaccelerator_utilization_checkerDTh {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSadvisorPSaccelerator_utilization_checkerDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSadvisorPSaccelerator_utilization_checkerDTh() {
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


#include "absl/strings/str_format.h"
#include "tensorflow/core/profiler/internal/advisor/checker.h"

namespace tensorflow {
namespace tfprof {

struct ExecStats {
 public:
  // Earliest start time of a step.
  int64_t start_micros;
  // Latest finish time of a step.
  int64_t end_micros;
  // The duration spent on running a kernel during a step.
  int64_t exec_micros;
};

class AcceleratorUtilizationChecker : public Checker {
 public:
  string name() const override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSadvisorPSaccelerator_utilization_checkerDTh mht_0(mht_0_v, 206, "", "./tensorflow/core/profiler/internal/advisor/accelerator_utilization_checker.h", "name");
 return kCheckers[0]; }

 private:
  AdviceProto::Checker Check(const AdvisorOptionsProto::CheckerOption& options,
                             const TFStats* stats) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSadvisorPSaccelerator_utilization_checkerDTh mht_1(mht_1_v, 213, "", "./tensorflow/core/profiler/internal/advisor/accelerator_utilization_checker.h", "Check");

    if (!stats) {
      absl::FPrintF(
          stderr, "Missing profiles (e.g. graph, run_meta). Skip %s\n", name());
      return reports_;
    }
    for (const auto& n : stats->nodes()) {
      BuildExecStats(n.second.get());
    }
    return CheckInternal();
  }

  AdviceProto::Checker CheckInternal() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSadvisorPSaccelerator_utilization_checkerDTh mht_2(mht_2_v, 228, "", "./tensorflow/core/profiler/internal/advisor/accelerator_utilization_checker.h", "CheckInternal");

    for (const auto& s : accelerator_exec_stats_) {
      const ExecStats& stat = s.second;
      int64_t total_micros = stat.end_micros - stat.start_micros;
      if (total_micros <= 0) continue;
      double utilization = 1.0 * stat.exec_micros / total_micros;
      if (utilization >= 0.5) {
        reports_.add_reports(absl::StrFormat("device: %s utilization: %.2f",
                                             s.first, utilization));
      } else if (utilization < 0.5 && utilization > 0.2) {
        reports_.add_reports(absl::StrFormat("device: %s low utilization: %.2f",
                                             s.first, utilization));
      } else if (utilization <= 0.2) {
        reports_.add_reports(absl::StrFormat("device: %s low utilization: %.2f",
                                             s.first, utilization));
      }
    }
    return reports_;
  }

  void BuildExecStats(const TFGraphNode* node) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSadvisorPSaccelerator_utilization_checkerDTh mht_3(mht_3_v, 251, "", "./tensorflow/core/profiler/internal/advisor/accelerator_utilization_checker.h", "BuildExecStats");

    const auto& execs = node->all_op_execs();
    if (execs.empty()) {
      return;
    }
    if (!IsPlacedOnAccelerator(node->canonical_device())) {
      return;
    }

    if (accelerator_exec_stats_.find(node->canonical_device()) ==
        accelerator_exec_stats_.end()) {
      accelerator_exec_stats_.insert(
          std::pair<string, ExecStats>(node->canonical_device(), ExecStats()));
    }
    ExecStats& stats = accelerator_exec_stats_.at(node->canonical_device());

    // TODO(xpan): Use multiple steps?
    const ExecStep& exec = execs.rbegin()->second;

    if (stats.start_micros == 0) {
      stats.start_micros = exec.all_start_micros();
    } else if (exec.all_start_micros() != 0) {
      stats.start_micros =
          std::min(stats.start_micros, exec.all_start_micros());
    }
    stats.end_micros = std::max(stats.end_micros, exec.latest_end_micros());
    stats.exec_micros += exec.accelerator_exec_micros();
  }

  std::map<string, ExecStats> accelerator_exec_stats_;
  std::map<string, int64_t> ps_placement_;
  AdviceProto::Checker reports_;
};

}  // namespace tfprof
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_INTERNAL_ADVISOR_ACCELERATOR_UTILIZATION_CHECKER_H_
