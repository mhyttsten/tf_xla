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

#ifndef TENSORFLOW_CORE_UTIL_STAT_SUMMARIZER_H_
#define TENSORFLOW_CORE_UTIL_STAT_SUMMARIZER_H_
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
class MHTracer_DTPStensorflowPScorePSutilPSstat_summarizerDTh {
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
   MHTracer_DTPStensorflowPScorePSutilPSstat_summarizerDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSutilPSstat_summarizerDTh() {
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


#include <stdlib.h>

#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <string>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/stat_summarizer_options.h"
#include "tensorflow/core/util/stats_calculator.h"

namespace tensorflow {

class GraphDef;
class StepStats;
class NodeExecStats;

// A StatSummarizer assists in performance analysis of Graph executions.
//
// It summarizes time spent executing (on GPU/CPU), memory used etc. across
// multiple executions of a single Graph from the StepStats collected during
// graph execution.
//
// See tensorflow/tools/benchmark/benchmark_model.cc for an example usage.
class StatSummarizer {
 public:
  explicit StatSummarizer(const StatSummarizerOptions& options);

  // Deprecated: Use StatSummarizer(const StatSummarizerOptions&) instead. The
  // GraphDef is not needed by the StatSummarizer.
  explicit StatSummarizer(const tensorflow::GraphDef& tensorflow_graph);

  ~StatSummarizer();

  // Adds another run's StepStats output to the aggregate counts.
  void ProcessStepStats(const StepStats& step_stats);

  // Returns a string detailing the accumulated runtime stats in a tab-separated
  // format which can be pasted into a spreadsheet for further analysis.
  std::string GetOutputString() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSutilPSstat_summarizerDTh mht_0(mht_0_v, 231, "", "./tensorflow/core/util/stat_summarizer.h", "GetOutputString");

    return stats_calculator_->GetOutputString();
  }

  std::string ShortSummary() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSutilPSstat_summarizerDTh mht_1(mht_1_v, 238, "", "./tensorflow/core/util/stat_summarizer.h", "ShortSummary");

    return stats_calculator_->GetShortSummary();
  }

  // Prints the string returned by GetOutputString().
  void PrintStepStats() const;

  // Prints the output tensor sizes and types for each node.
  void PrintOutputs() const;

  void ComputeStatsByType(
      std::map<std::string, int64_t>* node_type_map_count,
      std::map<std::string, int64_t>* node_type_map_time,
      std::map<std::string, int64_t>* node_type_map_memory,
      std::map<std::string, int64_t>* node_type_map_times_called,
      int64_t* accumulated_us) const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSutilPSstat_summarizerDTh mht_2(mht_2_v, 256, "", "./tensorflow/core/util/stat_summarizer.h", "ComputeStatsByType");

    stats_calculator_->ComputeStatsByType(
        node_type_map_count, node_type_map_time, node_type_map_memory,
        node_type_map_times_called, accumulated_us);
  }

  std::string GetStatsByNodeType() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSutilPSstat_summarizerDTh mht_3(mht_3_v, 265, "", "./tensorflow/core/util/stat_summarizer.h", "GetStatsByNodeType");

    return stats_calculator_->GetStatsByNodeType();
  }

  std::string GetStatsByMetric(const string& title,
                               StatsCalculator::SortingMetric sorting_metric,
                               int num_stats) const {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("title: \"" + title + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSstat_summarizerDTh mht_4(mht_4_v, 275, "", "./tensorflow/core/util/stat_summarizer.h", "GetStatsByMetric");

    return stats_calculator_->GetStatsByMetric(title, sorting_metric,
                                               num_stats);
  }

  int num_runs() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSutilPSstat_summarizerDTh mht_5(mht_5_v, 283, "", "./tensorflow/core/util/stat_summarizer.h", "num_runs");
 return stats_calculator_->num_runs(); }

  // Returns stats of total microseconds spent by all nodes in each run.
  const Stat<int64_t>& run_total_us() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSutilPSstat_summarizerDTh mht_6(mht_6_v, 289, "", "./tensorflow/core/util/stat_summarizer.h", "run_total_us");

    return stats_calculator_->run_total_us();
  }

 private:
  void Validate(const std::vector<TensorDescription>* outputs,
                const NodeExecStats& ns) const;

  std::map<std::string, std::vector<TensorDescription> > outputs_;

  std::unique_ptr<StatsCalculator> stats_calculator_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_STAT_SUMMARIZER_H_
