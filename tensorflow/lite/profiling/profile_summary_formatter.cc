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
class MHTracer_DTPStensorflowPSlitePSprofilingPSprofile_summary_formatterDTcc {
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
   MHTracer_DTPStensorflowPSlitePSprofilingPSprofile_summary_formatterDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSprofilingPSprofile_summary_formatterDTcc() {
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

#include "tensorflow/lite/profiling/profile_summary_formatter.h"

#include <memory>
#include <sstream>

namespace tflite {
namespace profiling {

std::string ProfileSummaryDefaultFormatter::GetOutputString(
    const std::map<uint32_t, std::unique_ptr<tensorflow::StatsCalculator>>&
        stats_calculator_map,
    const tensorflow::StatsCalculator& delegate_stats_calculator) const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSprofilingPSprofile_summary_formatterDTcc mht_0(mht_0_v, 196, "", "./tensorflow/lite/profiling/profile_summary_formatter.cc", "ProfileSummaryDefaultFormatter::GetOutputString");

  return GenerateReport("profile", /*include_output_string*/ true,
                        stats_calculator_map, delegate_stats_calculator);
}

std::string ProfileSummaryDefaultFormatter::GetShortSummary(
    const std::map<uint32_t, std::unique_ptr<tensorflow::StatsCalculator>>&
        stats_calculator_map,
    const tensorflow::StatsCalculator& delegate_stats_calculator) const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSprofilingPSprofile_summary_formatterDTcc mht_1(mht_1_v, 207, "", "./tensorflow/lite/profiling/profile_summary_formatter.cc", "ProfileSummaryDefaultFormatter::GetShortSummary");

  return GenerateReport("summary", /*include_output_string*/ false,
                        stats_calculator_map, delegate_stats_calculator);
}

std::string ProfileSummaryDefaultFormatter::GenerateReport(
    const std::string& tag, bool include_output_string,
    const std::map<uint32_t, std::unique_ptr<tensorflow::StatsCalculator>>&
        stats_calculator_map,
    const tensorflow::StatsCalculator& delegate_stats_calculator) const {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("tag: \"" + tag + "\"");
   MHTracer_DTPStensorflowPSlitePSprofilingPSprofile_summary_formatterDTcc mht_2(mht_2_v, 220, "", "./tensorflow/lite/profiling/profile_summary_formatter.cc", "ProfileSummaryDefaultFormatter::GenerateReport");

  std::stringstream stream;
  bool has_non_primary_graph =
      (stats_calculator_map.size() - stats_calculator_map.count(0)) > 0;
  for (const auto& stats_calc : stats_calculator_map) {
    auto subgraph_index = stats_calc.first;
    auto subgraph_stats = stats_calc.second.get();
    if (has_non_primary_graph) {
      if (subgraph_index == 0) {
        stream << "Primary graph " << tag << ":" << std::endl;
      } else {
        stream << "Subgraph (index: " << subgraph_index << ") " << tag << ":"
               << std::endl;
      }
    }
    if (include_output_string) {
      stream << subgraph_stats->GetOutputString();
    }
    if (subgraph_index != 0) {
      stream << "Subgraph (index: " << subgraph_index << ") ";
    }
    stream << subgraph_stats->GetShortSummary() << std::endl;
  }

  if (delegate_stats_calculator.num_runs() > 0) {
    stream << "Delegate internal: " << std::endl;
    if (include_output_string) {
      stream << delegate_stats_calculator.GetOutputString();
    }
    stream << delegate_stats_calculator.GetShortSummary() << std::endl;
  }

  return stream.str();
}

tensorflow::StatSummarizerOptions
ProfileSummaryDefaultFormatter::GetStatSummarizerOptions() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSprofilingPSprofile_summary_formatterDTcc mht_3(mht_3_v, 259, "", "./tensorflow/lite/profiling/profile_summary_formatter.cc", "ProfileSummaryDefaultFormatter::GetStatSummarizerOptions");

  auto options = tensorflow::StatSummarizerOptions();
  // Summary will be manually handled per subgraphs in order to keep the
  // compatibility.
  options.show_summary = false;
  options.show_memory = false;
  return options;
}

tensorflow::StatSummarizerOptions
ProfileSummaryCSVFormatter::GetStatSummarizerOptions() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSprofilingPSprofile_summary_formatterDTcc mht_4(mht_4_v, 272, "", "./tensorflow/lite/profiling/profile_summary_formatter.cc", "ProfileSummaryCSVFormatter::GetStatSummarizerOptions");

  auto options = ProfileSummaryDefaultFormatter::GetStatSummarizerOptions();
  options.format_as_csv = true;
  return options;
}

}  // namespace profiling
}  // namespace tflite
