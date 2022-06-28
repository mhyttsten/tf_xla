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
// This checker checks the most expensive operations.
#ifndef TENSORFLOW_CORE_PROFILER_INTERNAL_ADVISOR_EXPENSIVE_OPERATION_CHECKER_H_
#define TENSORFLOW_CORE_PROFILER_INTERNAL_ADVISOR_EXPENSIVE_OPERATION_CHECKER_H_
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
class MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSadvisorPSexpensive_operation_checkerDTh {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSadvisorPSexpensive_operation_checkerDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSadvisorPSexpensive_operation_checkerDTh() {
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
#include "absl/strings/str_join.h"
#include "tensorflow/core/profiler/internal/advisor/checker.h"

namespace tensorflow {
namespace tfprof {

class ExpensiveOperationChecker : public Checker {
 public:
  string name() const override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSadvisorPSexpensive_operation_checkerDTh mht_0(mht_0_v, 197, "", "./tensorflow/core/profiler/internal/advisor/expensive_operation_checker.h", "name");
 return kCheckers[2]; }

 private:
  AdviceProto::Checker Check(const AdvisorOptionsProto::CheckerOption& options,
                             const TFStats* stats) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSadvisorPSexpensive_operation_checkerDTh mht_1(mht_1_v, 204, "", "./tensorflow/core/profiler/internal/advisor/expensive_operation_checker.h", "Check");

    if (!stats) {
      absl::FPrintF(
          stderr, "Missing profiles (e.g. graph, run_meta). Skip %s\n", name());
      return reports_;
    }
    if (stats->steps().empty()) {
      absl::FPrintF(stderr, "Missing RunMetadata info. Skip %s\n", name());
    }
    CheckOpView(stats);
    CheckScopeView(stats);
    CheckCodeView(stats);
    return reports_;
  }

  void CheckOpView(const TFStats* stats) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSadvisorPSexpensive_operation_checkerDTh mht_2(mht_2_v, 222, "", "./tensorflow/core/profiler/internal/advisor/expensive_operation_checker.h", "CheckOpView");

    if (stats->steps().empty()) {
      absl::FPrintF(stderr, "Missing run_meta for %s\n", name());
      return;
    }
    Options opts(3, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, -1, "micros", {".*"}, {".*"},
                 {}, {".*"}, {}, false, {"micros", "occurrence"}, "none", {});
    const MultiGraphNodeProto root = stats->ShowMultiGraphNode("op", opts);
    if (root.children_size() == 0) {
      return;
    }
    const MultiGraphNodeProto* node = &root;
    std::vector<string> outputs;
    for (int i = 0; i < 3 && node->children_size() > 0; ++i) {
      node = &node->children(0);
      outputs.push_back(absl::StrFormat(
          "top %d operation type: %s, "
          "cpu: %s, accelerator: %s, total: %s (%.2f%%)",
          i + 1, node->name(), FormatTime(node->cpu_exec_micros()),
          FormatTime(node->accelerator_exec_micros()),
          FormatTime(node->exec_micros()),
          100.0 * node->exec_micros() / (root.total_exec_micros() + 1e-10)));
    }
    reports_.add_reports(absl::StrJoin(outputs, "\n"));
  }

  void CheckCodeView(const TFStats* stats) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSadvisorPSexpensive_operation_checkerDTh mht_3(mht_3_v, 251, "", "./tensorflow/core/profiler/internal/advisor/expensive_operation_checker.h", "CheckCodeView");

    if (!stats->has_code_traces()) {
      absl::FPrintF(stderr, "Missing op_log (code traces) for %s\n", name());
      return;
    }
    Options opts(100, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, -1, "micros", {".*"},
                 {".*"}, {}, {".*"}, {}, false, {"micros"}, "none", {});
    const MultiGraphNodeProto root = stats->ShowMultiGraphNode("code", opts);
    const MultiGraphNodeProto* node = &root;
    // A trick here is: Usually, codes in library file are usually referenced
    // only once, while user's own code are referenced multiple times.
    while (node->children_size() == 1) {
      node = &node->children(0);
    }
    if (node->children_size() == 0) {
      return;
    }

    std::vector<string> outputs;
    CodeViewHelper(node, 0, &outputs);
    reports_.add_reports(absl::StrJoin(outputs, "\n"));
  }

  void CheckScopeView(const TFStats* stats) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSadvisorPSexpensive_operation_checkerDTh mht_4(mht_4_v, 277, "", "./tensorflow/core/profiler/internal/advisor/expensive_operation_checker.h", "CheckScopeView");

    Options opts(100, 0, 0, 0, 0, 100, 0, 0, 0, 0, 0, -1, "micros", {".*"},
                 {".*"}, {}, {".*"}, {}, false, {"micros"}, "none", {});
    const GraphNodeProto root = stats->ShowGraphNode("scope", opts);
    if (root.children_size() == 0) {
      return;
    }
    std::vector<string> outputs;
    for (int i = 0; i < 3 && i < root.children_size(); ++i) {
      const GraphNodeProto& node = root.children(i);
      outputs.push_back(absl::StrFormat(
          "top %d graph node: %s, cpu: %s, accelerator: %s, total: %s", i + 1,
          node.name(), FormatTime(node.cpu_exec_micros()),
          FormatTime(node.accelerator_exec_micros()),
          FormatTime(node.exec_micros())));
    }
    reports_.add_reports(absl::StrJoin(outputs, "\n"));
  }

  void CodeViewHelper(const MultiGraphNodeProto* node, int depth,
                      std::vector<string>* outputs) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSadvisorPSexpensive_operation_checkerDTh mht_5(mht_5_v, 300, "", "./tensorflow/core/profiler/internal/advisor/expensive_operation_checker.h", "CodeViewHelper");

    if (node->children_size() <= 1 || depth > 3) {
      return;
    }
    for (int j = 0; j < 3 && j < node->children_size(); ++j) {
      const MultiGraphNodeProto* c = &node->children(j);
      if (c->total_exec_micros() < 1000) {
        continue;
      }
      outputs->push_back(
          absl::StrFormat("%s%s, cpu: %s, accelerator: %s, total: %s",
                          std::string(depth * 2, ' '), c->name(),
                          FormatTime(c->total_cpu_exec_micros()),
                          FormatTime(c->total_accelerator_exec_micros()),
                          FormatTime(c->total_exec_micros())));
      CodeViewHelper(c, depth + 1, outputs);
    }
  }

  AdviceProto::Checker reports_;
};

}  // namespace tfprof
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_INTERNAL_ADVISOR_EXPENSIVE_OPERATION_CHECKER_H_
