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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSoptimization_registryDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSoptimization_registryDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSoptimization_registryDTcc() {
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

#include "tensorflow/core/common_runtime/optimization_registry.h"

#include "tensorflow/core/framework/metrics.h"
#include "tensorflow/core/util/dump_graph.h"

namespace tensorflow {

// static
OptimizationPassRegistry* OptimizationPassRegistry::Global() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSoptimization_registryDTcc mht_0(mht_0_v, 193, "", "./tensorflow/core/common_runtime/optimization_registry.cc", "OptimizationPassRegistry::Global");

  static OptimizationPassRegistry* global_optimization_registry =
      new OptimizationPassRegistry;
  return global_optimization_registry;
}

void OptimizationPassRegistry::Register(
    Grouping grouping, int phase, std::unique_ptr<GraphOptimizationPass> pass) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSoptimization_registryDTcc mht_1(mht_1_v, 203, "", "./tensorflow/core/common_runtime/optimization_registry.cc", "OptimizationPassRegistry::Register");

  groups_[grouping][phase].push_back(std::move(pass));
}

Status OptimizationPassRegistry::RunGrouping(
    Grouping grouping, const GraphOptimizationPassOptions& options) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSoptimization_registryDTcc mht_2(mht_2_v, 211, "", "./tensorflow/core/common_runtime/optimization_registry.cc", "OptimizationPassRegistry::RunGrouping");

  auto dump_graph = [&](std::string& prefix) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSoptimization_registryDTcc mht_3(mht_3_v, 215, "", "./tensorflow/core/common_runtime/optimization_registry.cc", "lambda");

    if (options.graph) {
      DumpGraphToFile(
          strings::StrCat(prefix, "_",
                          reinterpret_cast<uintptr_t>((*options.graph).get())),
          **options.graph, options.flib_def);
    }
    if (options.partition_graphs) {
      for (auto& part : *options.partition_graphs) {
        DumpGraphToFile(
            strings::StrCat(prefix, "_partition_", part.first, "_",
                            reinterpret_cast<uintptr_t>(part.second.get())),
            *part.second, options.flib_def);
      }
    }
  };

  VLOG(1) << "Starting optimization of a group " << grouping;
  if (VLOG_IS_ON(2)) {
    std::string prefix = strings::StrCat("before_grouping_", grouping);
    dump_graph(prefix);
  }
  auto group = groups_.find(grouping);
  if (group != groups_.end()) {
    static const char* kGraphOptimizationCategory = "GraphOptimizationPass";
    tensorflow::metrics::ScopedCounter<2> group_timings(
        tensorflow::metrics::GetGraphOptimizationCounter(),
        {kGraphOptimizationCategory, "*"});
    for (auto& phase : group->second) {
      VLOG(1) << "Running optimization phase " << phase.first;
      for (auto& pass : phase.second) {
        VLOG(1) << "Running optimization pass: " << pass->name();

        tensorflow::metrics::ScopedCounter<2> pass_timings(
            tensorflow::metrics::GetGraphOptimizationCounter(),
            {kGraphOptimizationCategory, pass->name()});
        Status s = pass->Run(options);

        if (!s.ok()) return s;
        pass_timings.ReportAndStop();
        if (VLOG_IS_ON(5)) {
          std::string prefix =
              strings::StrCat("after_group_", grouping, "_phase_", phase.first,
                              "_", pass->name());
          dump_graph(prefix);
        }
      }
    }
    group_timings.ReportAndStop();
  }
  VLOG(1) << "Finished optimization of a group " << grouping;
  if (VLOG_IS_ON(2)) {
    std::string prefix = strings::StrCat("after_grouping_", grouping);
    dump_graph(prefix);
  }
  return Status::OK();
}

void OptimizationPassRegistry::LogGrouping(Grouping grouping, int vlog_level) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSoptimization_registryDTcc mht_4(mht_4_v, 276, "", "./tensorflow/core/common_runtime/optimization_registry.cc", "OptimizationPassRegistry::LogGrouping");

  auto group = groups_.find(grouping);
  if (group != groups_.end()) {
    for (auto& phase : group->second) {
      for (auto& pass : phase.second) {
        VLOG(vlog_level) << "Registered optimization pass grouping " << grouping
                         << " phase " << phase.first << ": " << pass->name();
      }
    }
  }
}

void OptimizationPassRegistry::LogAllGroupings(int vlog_level) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSoptimization_registryDTcc mht_5(mht_5_v, 291, "", "./tensorflow/core/common_runtime/optimization_registry.cc", "OptimizationPassRegistry::LogAllGroupings");

  for (auto group = groups_.begin(); group != groups_.end(); ++group) {
    LogGrouping(group->first, vlog_level);
  }
}

}  // namespace tensorflow
