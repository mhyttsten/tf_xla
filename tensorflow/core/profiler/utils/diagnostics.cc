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
class MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSdiagnosticsDTcc {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSdiagnosticsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSdiagnosticsDTcc() {
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

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/profiler/utils/diagnostics.h"

#include "absl/algorithm/container.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/profiler/protobuf/steps_db.pb.h"

namespace tensorflow {
namespace profiler {

const absl::string_view kErrorIncompleteStep =
    "Incomplete step observed and hence the step time is unknown."
    "Instead, we use the trace duration as the step time. This may happen"
    " if your profiling duration is shorter than the step time. In this"
    " case, you may try to profile longer.";

const absl::string_view kErrorEmptyIntersect =
    "Although there are steps observed on some host(s), the intersection of "
    "the steps over all hosts is empty (because the differences among "
    "individual host's step sequences are too big). Consequently, the overall "
    "step time is "
    "unknown.";

const absl::string_view kErrorNoStepMarker =
    "No step marker observed and hence the step time is unknown."
    " This may happen if (1) training steps are not instrumented (e.g., if"
    " you are not using Keras) or (2) the profiling duration is shorter"
    " than the step time. For (1), you need to add step instrumentation;"
    " for (2), you may try to profile longer.";

const absl::string_view kNoDeviceTraceCollected =
    "No device trace was collected. This might happen if your job hadn't been "
    "run on the device when sampling was turned on. You could try the sampling"
    " again later.";

const absl::string_view kStepsDropped =
    " steps dropped. This might happen when you profile many hosts and/or many "
    "steps. You could try to profile shorter or reduce the number of hosts "
    "you profile.";

void PopulateStepDiagnostics(const OpStats& op_stats, Diagnostics* diag) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSdiagnosticsDTcc mht_0(mht_0_v, 225, "", "./tensorflow/core/profiler/utils/diagnostics.cc", "PopulateStepDiagnostics");

  if (op_stats.step_db().use_incomplete_step()) {
    *diag->add_warnings() = std::string(kErrorIncompleteStep);
  } else if (op_stats.step_db().step_sequence().empty()) {
    *diag->add_warnings() = op_stats.step_db().empty_intersect()
                                ? std::string(kErrorEmptyIntersect)
                                : std::string(kErrorNoStepMarker);
  }
  if (op_stats.step_db().num_steps_dropped()) {
    *diag->add_warnings() =
        absl::StrCat(op_stats.step_db().num_steps_dropped(), kStepsDropped);
  }
}

void PopulateOverviewDiagnostics(const OpStats& op_stats, Diagnostics* diag) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSdiagnosticsDTcc mht_1(mht_1_v, 242, "", "./tensorflow/core/profiler/utils/diagnostics.cc", "PopulateOverviewDiagnostics");

  *diag->mutable_errors() = op_stats.diagnostics().errors();
  absl::c_sort(*diag->mutable_errors());
  if (diag->errors().empty()) {
    // Shows run-environment error only if there is no other existing error.
    if (op_stats.run_environment().device_type() != "CPU" &&
        op_stats.run_environment().device_core_count() <= 0) {
      *diag->add_errors() = std::string(kNoDeviceTraceCollected);
    }
  }
  *diag->mutable_warnings() = op_stats.diagnostics().warnings();
  PopulateStepDiagnostics(op_stats, diag);
}

}  // namespace profiler
}  // namespace tensorflow
