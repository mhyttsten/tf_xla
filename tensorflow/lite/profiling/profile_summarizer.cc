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
class MHTracer_DTPStensorflowPSlitePSprofilingPSprofile_summarizerDTcc {
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
   MHTracer_DTPStensorflowPSlitePSprofilingPSprofile_summarizerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSprofilingPSprofile_summarizerDTcc() {
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

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/profiling/profile_summarizer.h"

#include <memory>
#include <sstream>

#include "tensorflow/lite/profiling/memory_info.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace profiling {
namespace {

struct OperatorDetails {
  uint32_t subgraph_index;
  uint32_t node_index;
  std::string op_description;
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
};

std::string GetTensorName(const tflite::Interpreter& interpreter,
                          int tensor_index) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSprofilingPSprofile_summarizerDTcc mht_0(mht_0_v, 206, "", "./tensorflow/lite/profiling/profile_summarizer.cc", "GetTensorName");

  const auto tensor = interpreter.tensor(tensor_index);
  if (tensor == nullptr || tensor->name == nullptr) {
    return "Unknown";
  }
  return tensor->name;
}
std::vector<std::string> GetTensorNames(const tflite::Interpreter& interpreter,
                                        const TfLiteIntArray* tensor_indices) {
  std::vector<std::string> tensors;
  tensors.reserve(tensor_indices->size);
  for (int i = 0; i < tensor_indices->size; i++) {
    tensors.push_back(GetTensorName(interpreter, tensor_indices->data[i]));
  }
  return tensors;
}

std::string ToString(const std::vector<std::string>& str_vector) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSprofilingPSprofile_summarizerDTcc mht_1(mht_1_v, 226, "", "./tensorflow/lite/profiling/profile_summarizer.cc", "ToString");

  std::stringstream stream;
  stream << "[";
  bool first = true;
  for (const auto& s : str_vector) {
    if (!first) {
      stream << ", ";
    } else {
      first = false;
    }
    stream << s;
  }
  stream << "]";
  return stream.str();
}

OperatorDetails GetOperatorDetails(const tflite::Interpreter& interpreter,
                                   uint32_t subgraph_index,
                                   uint32_t node_index) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSprofilingPSprofile_summarizerDTcc mht_2(mht_2_v, 247, "", "./tensorflow/lite/profiling/profile_summarizer.cc", "GetOperatorDetails");

  auto subgraph =
      const_cast<tflite::Interpreter&>(interpreter).subgraph(subgraph_index);
  auto node_reg = subgraph->node_and_registration(node_index);
  auto inputs = node_reg->first.inputs;
  auto outputs = node_reg->first.outputs;
  const char* profiling_string =
      interpreter.OpProfilingString(node_reg->second, &node_reg->first);
  OperatorDetails details;
  if (profiling_string) {
    details.op_description = std::string(profiling_string);
  }
  details.inputs = GetTensorNames(interpreter, inputs);
  details.outputs = GetTensorNames(interpreter, outputs);
  return details;
}

}  // namespace

ProfileSummarizer::ProfileSummarizer(
    std::shared_ptr<ProfileSummaryFormatter> summary_formatter)
    : summary_formatter_(summary_formatter) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSprofilingPSprofile_summarizerDTcc mht_3(mht_3_v, 271, "", "./tensorflow/lite/profiling/profile_summarizer.cc", "ProfileSummarizer::ProfileSummarizer");

  // Create stats calculator for the primary graph.
  stats_calculator_map_[0] = std::unique_ptr<tensorflow::StatsCalculator>(
      new tensorflow::StatsCalculator(
          summary_formatter_->GetStatSummarizerOptions()));

  // Create stats calculator for the delegation op.
  delegate_stats_calculator_ = std::unique_ptr<tensorflow::StatsCalculator>(
      new tensorflow::StatsCalculator(
          summary_formatter_->GetStatSummarizerOptions()));
}
void ProfileSummarizer::ProcessProfiles(
    const std::vector<const ProfileEvent*>& profile_stats,
    const tflite::Interpreter& interpreter) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSprofilingPSprofile_summarizerDTcc mht_4(mht_4_v, 287, "", "./tensorflow/lite/profiling/profile_summarizer.cc", "ProfileSummarizer::ProcessProfiles");

  if (profile_stats.empty()) return;

  std::vector<const ProfileEvent*> events;
  std::copy_if(profile_stats.begin(), profile_stats.end(),
               std::back_inserter(events), [](const ProfileEvent* e) {
                 return e->end_timestamp_us >= e->begin_timestamp_us;
               });
  // Sort with begin_time.
  std::sort(events.begin(), events.end(),
            [](const ProfileEvent* const& a, const ProfileEvent* const& b) {
              return a->begin_timestamp_us < b->begin_timestamp_us;
            });
  if (events.empty()) {
    return;
  }

  int64_t base_start_us = events[0]->begin_timestamp_us;
  int node_num = 0;

  // Total time will be accumulated per subgraph.
  std::map<uint32_t, int64_t> total_us_per_subgraph_map;
  int64_t delegate_internal_total_us = 0;

  for (auto event : events) {
    const auto subgraph_index = event->extra_event_metadata;
    auto stats_calculator = GetStatsCalculator(subgraph_index);
    int64_t start_us = event->begin_timestamp_us - base_start_us;
    int64_t node_exec_time =
        event->end_timestamp_us - event->begin_timestamp_us;
    if (event->event_type == Profiler::EventType::OPERATOR_INVOKE_EVENT) {
      // When recording an OPERATOR_INVOKE_EVENT, we have recorded the node
      // index as event_metadata. See the macro
      // TFLITE_SCOPED_TAGGED_OPERATOR_PROFILE defined in
      // tensorflow/lite/core/api/profiler.h for details.
      const auto node_index = event->event_metadata;

      const auto op_details =
          GetOperatorDetails(interpreter, subgraph_index, node_index);
      std::string type_in_stats(event->tag);
      if (!op_details.op_description.empty()) {
        type_in_stats += "/" + op_details.op_description;
      }

      const auto node_name = ToString(op_details.outputs);
      // Append node index to node name because 'stats_calculator' can not
      // distinguish two nodes w/ the same 'node_name'.
      const auto node_name_in_stats =
          node_name + ":" + std::to_string(node_index);

      stats_calculator->AddNodeStats(node_name_in_stats, type_in_stats,
                                     node_num, start_us, node_exec_time,
                                     0 /*memory */);
    } else if (event->event_type ==
               Profiler::EventType::DELEGATE_OPERATOR_INVOKE_EVENT) {
      const std::string node_name(event->tag);
      // Append event_metadata to node name because 'stats_calculator' can not
      // distinguish two nodes w/ the same 'node_name'.
      const auto node_name_in_stats =
          "Delegate/" + node_name + ":" + std::to_string(event->event_metadata);

      delegate_stats_calculator_->AddNodeStats(
          node_name_in_stats, "DelegateOpInvoke", node_num, start_us,
          node_exec_time, 0 /*memory */);
    } else {
      // Note: a different stats_calculator could be used to record
      // non-op-invoke events so that these could be separated from
      // op-invoke-events in the final profiling stats report.
      const memory::MemoryUsage node_mem_usage =
          event->end_mem_usage - event->begin_mem_usage;
      std::string node_name(event->tag);
      if (node_name == "Invoke") {
        // Don't count the overall Invoke for profiling.
        continue;
      }
      node_name += "/" + std::to_string(event->extra_event_metadata);
      stats_calculator->AddNodeStats(node_name, event->tag, node_num, start_us,
                                     node_exec_time,
                                     node_mem_usage.max_rss_kb * 1000.0);
    }

    // Add total time except actual delegate ops since the elapsed time of the
    // delegate ops inside are already combined at a fused DELEGATE op.
    if (event->event_type !=
        Profiler::EventType::DELEGATE_OPERATOR_INVOKE_EVENT) {
      total_us_per_subgraph_map[subgraph_index] += node_exec_time;
    } else {
      delegate_internal_total_us += node_exec_time;
    }
    ++node_num;
  }

  for (auto& total_us_per_subgraph_pair : total_us_per_subgraph_map) {
    auto stats_calculator =
        GetStatsCalculator(total_us_per_subgraph_pair.first);
    stats_calculator->UpdateRunTotalUs(total_us_per_subgraph_pair.second);
  }
  if (delegate_internal_total_us > 0) {
    delegate_stats_calculator_->UpdateRunTotalUs(delegate_internal_total_us);
  }
}

tensorflow::StatsCalculator* ProfileSummarizer::GetStatsCalculator(
    uint32_t subgraph_index) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSprofilingPSprofile_summarizerDTcc mht_5(mht_5_v, 393, "", "./tensorflow/lite/profiling/profile_summarizer.cc", "ProfileSummarizer::GetStatsCalculator");

  if (stats_calculator_map_.count(subgraph_index) == 0) {
    stats_calculator_map_[subgraph_index] =
        std::unique_ptr<tensorflow::StatsCalculator>(
            new tensorflow::StatsCalculator(
                summary_formatter_->GetStatSummarizerOptions()));
  }
  return stats_calculator_map_[subgraph_index].get();
}

}  // namespace profiling
}  // namespace tflite
