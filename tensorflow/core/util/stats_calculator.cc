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
class MHTracer_DTPStensorflowPScorePSutilPSstats_calculatorDTcc {
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
   MHTracer_DTPStensorflowPScorePSutilPSstats_calculatorDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSutilPSstats_calculatorDTcc() {
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

#include "tensorflow/core/util/stats_calculator.h"

#include <iomanip>
#include <map>
#include <queue>
#include <sstream>
#include <string>

namespace tensorflow {

StatsCalculator::StatsCalculator(const StatSummarizerOptions& options)
    : options_(options) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSutilPSstats_calculatorDTcc mht_0(mht_0_v, 196, "", "./tensorflow/core/util/stats_calculator.cc", "StatsCalculator::StatsCalculator");
}

std::string StatsCalculator::GetShortSummary() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSutilPSstats_calculatorDTcc mht_1(mht_1_v, 201, "", "./tensorflow/core/util/stats_calculator.cc", "StatsCalculator::GetShortSummary");

  std::stringstream stream;
  stream << "Timings (microseconds): ";
  run_total_us_.OutputToStream(&stream);
  stream << std::endl;

  stream << "Memory (bytes): ";
  memory_.OutputToStream(&stream);
  stream << std::endl;

  stream << details_.size() << " nodes observed" << std::endl;
  return stream.str();
}

std::ostream& InitField(std::ostream& stream, int width) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSutilPSstats_calculatorDTcc mht_2(mht_2_v, 218, "", "./tensorflow/core/util/stats_calculator.cc", "InitField");

  stream << "\t" << std::right << std::setw(width) << std::fixed
         << std::setprecision(3);
  return stream;
}

std::string StatsCalculator::HeaderString(const std::string& title) const {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("title: \"" + title + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSstats_calculatorDTcc mht_3(mht_3_v, 228, "", "./tensorflow/core/util/stats_calculator.cc", "StatsCalculator::HeaderString");

  std::stringstream stream;

  stream << "============================== " << title
         << " ==============================" << std::endl;
  if (options_.format_as_csv) {
    stream << "node type, start, first, avg_ms, %, cdf%, mem KB, times called, "
              "name";
  } else {
    InitField(stream, 24) << "[node type]";
    InitField(stream, 17) << "[start]";
    InitField(stream, 9) << "[first]";
    InitField(stream, 9) << "[avg ms]";
    InitField(stream, 8) << "[%]";
    InitField(stream, 8) << "[cdf%]";
    InitField(stream, 10) << "[mem KB]";
    InitField(stream, 9) << "[times called]";
    stream << "\t"
           << "[Name]";
  }
  return stream.str();
}

std::string StatsCalculator::ColumnString(const Detail& detail,
                                          const int64_t cumulative_stat_on_node,
                                          const Stat<int64_t>& stat) const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSutilPSstats_calculatorDTcc mht_4(mht_4_v, 256, "", "./tensorflow/core/util/stats_calculator.cc", "StatsCalculator::ColumnString");

  const double start_ms = detail.start_us.avg() / 1000.0;
  const double first_time_ms = detail.rel_end_us.first() / 1000.0;
  const double avg_time_ms = detail.rel_end_us.avg() / 1000.0;
  const double percentage = detail.rel_end_us.sum() * 100.0 / stat.sum();
  const double cdf_percentage = (cumulative_stat_on_node * 100.0f) / stat.sum();
  const int64_t times_called = detail.times_called / num_runs();

  std::stringstream stream;
  if (options_.format_as_csv) {
    std::string name(detail.name);
    std::replace(name.begin(), name.end(), ',', '\t');
    stream << detail.type << ", " << start_ms << ", " << first_time_ms << ", "
           << avg_time_ms << ", " << percentage << "%, " << cdf_percentage
           << "%, " << detail.mem_used.newest() / 1000.0 << ", " << times_called
           << ", " << name;
  } else {
    InitField(stream, 24) << detail.type;
    InitField(stream, 17) << start_ms;
    InitField(stream, 9) << first_time_ms;
    InitField(stream, 9) << avg_time_ms;
    InitField(stream, 7) << percentage << "%";
    InitField(stream, 7) << cdf_percentage << "%";
    InitField(stream, 10) << detail.mem_used.newest() / 1000.0;
    InitField(stream, 9) << times_called;
    stream << "\t" << detail.name;
  }

  return stream.str();
}

void StatsCalculator::OrderNodesByMetric(
    SortingMetric metric, std::vector<const Detail*>* details) const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSutilPSstats_calculatorDTcc mht_5(mht_5_v, 291, "", "./tensorflow/core/util/stats_calculator.cc", "StatsCalculator::OrderNodesByMetric");

  std::priority_queue<std::pair<std::string, const Detail*>> sorted_list;
  const int num_nodes = details_.size();

  for (const auto& det : details_) {
    const Detail* detail = &(det.second);
    std::stringstream stream;
    stream << std::setw(20) << std::right << std::setprecision(10)
           << std::fixed;

    switch (metric) {
      case BY_NAME:
        stream << detail->name;
        break;
      case BY_RUN_ORDER:
        stream << num_nodes - detail->run_order;
        break;
      case BY_TIME:
        stream << detail->rel_end_us.avg();
        break;
      case BY_MEMORY:
        stream << detail->mem_used.avg();
        break;
      case BY_TYPE:
        stream << detail->type;
        break;
      default:
        stream << "";
        break;
    }

    sorted_list.emplace(stream.str(), detail);
  }

  while (!sorted_list.empty()) {
    auto entry = sorted_list.top();
    sorted_list.pop();
    details->push_back(entry.second);
  }
}

void StatsCalculator::ComputeStatsByType(
    std::map<std::string, int64_t>* node_type_map_count,
    std::map<std::string, int64_t>* node_type_map_time,
    std::map<std::string, int64_t>* node_type_map_memory,
    std::map<std::string, int64_t>* node_type_map_times_called,
    int64_t* accumulated_us) const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSutilPSstats_calculatorDTcc mht_6(mht_6_v, 340, "", "./tensorflow/core/util/stats_calculator.cc", "StatsCalculator::ComputeStatsByType");

  int64_t run_count = run_total_us_.count();

  for (const auto& det : details_) {
    const std::string node_name = det.first;
    const Detail& detail = det.second;

    int64_t curr_time_val =
        static_cast<int64_t>(detail.rel_end_us.sum() / run_count);
    *accumulated_us += curr_time_val;

    int64_t curr_memory_val = detail.mem_used.newest();

    const std::string& node_type = detail.type;

    (*node_type_map_count)[node_type] += 1;
    (*node_type_map_time)[node_type] += curr_time_val;
    (*node_type_map_memory)[node_type] += curr_memory_val;
    (*node_type_map_times_called)[node_type] += detail.times_called / run_count;
  }
}

std::string StatsCalculator::GetStatsByNodeType() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSutilPSstats_calculatorDTcc mht_7(mht_7_v, 365, "", "./tensorflow/core/util/stats_calculator.cc", "StatsCalculator::GetStatsByNodeType");

  std::stringstream stream;

  stream << "Number of nodes executed: " << details_.size() << std::endl;

  stream << "============================== Summary by node type "
            "=============================="
         << std::endl;

  std::map<std::string, int64_t> node_type_map_count;
  std::map<std::string, int64_t> node_type_map_time;
  std::map<std::string, int64_t> node_type_map_memory;
  std::map<std::string, int64_t> node_type_map_times_called;
  int64_t accumulated_us = 0;

  ComputeStatsByType(&node_type_map_count, &node_type_map_time,
                     &node_type_map_memory, &node_type_map_times_called,
                     &accumulated_us);

  // Sort them.
  std::priority_queue<std::pair<int64_t, std::pair<std::string, int64_t>>>
      timings;
  for (const auto& node_type : node_type_map_time) {
    const int64_t mem_used = node_type_map_memory[node_type.first];
    timings.emplace(node_type.second,
                    std::pair<std::string, int64_t>(node_type.first, mem_used));
  }

  if (options_.format_as_csv) {
    stream << "node type, count, avg_ms, avg %, cdf %, mem KB, times called\n";
  } else {
    InitField(stream, 24) << "[Node type]";
    InitField(stream, 9) << "[count]";
    InitField(stream, 10) << "[avg ms]";
    InitField(stream, 11) << "[avg %]";
    InitField(stream, 11) << "[cdf %]";
    InitField(stream, 10) << "[mem KB]";
    InitField(stream, 10) << "[times called]";
    stream << std::endl;
  }

  float cdf = 0.0f;
  while (!timings.empty()) {
    auto entry = timings.top();
    timings.pop();

    const std::string node_type = entry.second.first;
    const float memory = entry.second.second / 1000.0f;

    const int64_t node_type_total_us = entry.first;
    const float time_per_run_ms = node_type_total_us / 1000.0f;

    const float percentage =
        ((entry.first / static_cast<float>(accumulated_us)) * 100.0f);
    cdf += percentage;

    if (options_.format_as_csv) {
      stream << node_type << ", " << node_type_map_count[node_type] << ", "
             << time_per_run_ms << ", " << percentage << "%, " << cdf << "%, "
             << memory << ", " << node_type_map_times_called[node_type]
             << std::endl;
    } else {
      InitField(stream, 24) << node_type;
      InitField(stream, 9) << node_type_map_count[node_type];
      InitField(stream, 10) << time_per_run_ms;
      InitField(stream, 10) << percentage << "%";
      InitField(stream, 10) << cdf << "%";
      InitField(stream, 10) << memory;
      InitField(stream, 9) << node_type_map_times_called[node_type];
      stream << std::endl;
    }
  }
  stream << std::endl;
  return stream.str();
}

std::string StatsCalculator::GetStatsByMetric(const std::string& title,
                                              SortingMetric sorting_metric,
                                              int num_stats) const {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("title: \"" + title + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSstats_calculatorDTcc mht_8(mht_8_v, 447, "", "./tensorflow/core/util/stats_calculator.cc", "StatsCalculator::GetStatsByMetric");

  std::vector<const Detail*> details;
  OrderNodesByMetric(sorting_metric, &details);

  double cumulative_stat_on_node = 0;

  std::stringstream stream;
  stream << HeaderString(title) << std::endl;
  int stat_num = 0;
  for (auto detail : details) {
    ++stat_num;
    if (num_stats > 0 && stat_num > num_stats) {
      break;
    }

    // TODO(andrewharp): Make this keep track of the particular metric for cdf.
    cumulative_stat_on_node += detail->rel_end_us.sum();
    stream << ColumnString(*detail, cumulative_stat_on_node, run_total_us_)
           << std::endl;
  }
  stream << std::endl;
  return stream.str();
}

std::string StatsCalculator::GetOutputString() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSutilPSstats_calculatorDTcc mht_9(mht_9_v, 474, "", "./tensorflow/core/util/stats_calculator.cc", "StatsCalculator::GetOutputString");

  std::stringstream stream;
  if (options_.show_run_order) {
    stream << GetStatsByMetric("Run Order", BY_RUN_ORDER,
                               options_.run_order_limit);
  }
  if (options_.show_time) {
    stream << GetStatsByMetric("Top by Computation Time", BY_TIME,
                               options_.time_limit);
  }
  if (options_.show_memory) {
    stream << GetStatsByMetric("Top by Memory Use", BY_MEMORY,
                               options_.memory_limit);
  }
  if (options_.show_type) {
    stream << GetStatsByNodeType();
  }
  if (options_.show_summary) {
    stream << GetShortSummary() << std::endl;
  }
  return stream.str();
}

void StatsCalculator::AddNodeStats(const std::string& name,
                                   const std::string& type, int64_t run_order,
                                   int64_t start_us, int64_t rel_end_us,
                                   int64_t mem_used) {
   std::vector<std::string> mht_10_v;
   mht_10_v.push_back("name: \"" + name + "\"");
   mht_10_v.push_back("type: \"" + type + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSstats_calculatorDTcc mht_10(mht_10_v, 505, "", "./tensorflow/core/util/stats_calculator.cc", "StatsCalculator::AddNodeStats");

  Detail* detail = nullptr;
  if (details_.find(name) == details_.end()) {
    details_.insert({name, {}});
    detail = &details_.at(name);
    detail->type = type;
    detail->name = name;
    detail->run_order = run_order;
  } else {
    detail = &details_.at(name);
  }
  detail->start_us.UpdateStat(start_us);
  detail->rel_end_us.UpdateStat(rel_end_us);
  detail->mem_used.UpdateStat(mem_used);
  detail->times_called++;
}

}  // namespace tensorflow
