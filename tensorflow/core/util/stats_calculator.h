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

#ifndef TENSORFLOW_CORE_UTIL_STATS_CALCULATOR_H_
#define TENSORFLOW_CORE_UTIL_STATS_CALCULATOR_H_
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
class MHTracer_DTPStensorflowPScorePSutilPSstats_calculatorDTh {
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
   MHTracer_DTPStensorflowPScorePSutilPSstats_calculatorDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSutilPSstats_calculatorDTh() {
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

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "tensorflow/core/util/stat_summarizer_options.h"

namespace tensorflow {

template <typename ValueType, typename HighPrecisionValueType = double>
class Stat {
 public:
  void UpdateStat(ValueType v) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSutilPSstats_calculatorDTh mht_0(mht_0_v, 205, "", "./tensorflow/core/util/stats_calculator.h", "UpdateStat");

    if (count_ == 0) {
      first_ = v;
    }

    newest_ = v;
    max_ = std::max(v, max_);
    min_ = std::min(v, min_);
    ++count_;
    sum_ += v;
    squared_sum_ += static_cast<HighPrecisionValueType>(v) * v;
  }

  void Reset() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSutilPSstats_calculatorDTh mht_1(mht_1_v, 221, "", "./tensorflow/core/util/stats_calculator.h", "Reset");
 new (this) Stat<ValueType, HighPrecisionValueType>(); }

  bool empty() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSutilPSstats_calculatorDTh mht_2(mht_2_v, 226, "", "./tensorflow/core/util/stats_calculator.h", "empty");
 return count_ == 0; }

  ValueType first() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSutilPSstats_calculatorDTh mht_3(mht_3_v, 231, "", "./tensorflow/core/util/stats_calculator.h", "first");
 return first_; }

  ValueType newest() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSutilPSstats_calculatorDTh mht_4(mht_4_v, 236, "", "./tensorflow/core/util/stats_calculator.h", "newest");
 return newest_; }

  ValueType max() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSutilPSstats_calculatorDTh mht_5(mht_5_v, 241, "", "./tensorflow/core/util/stats_calculator.h", "max");
 return max_; }

  ValueType min() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSutilPSstats_calculatorDTh mht_6(mht_6_v, 246, "", "./tensorflow/core/util/stats_calculator.h", "min");
 return min_; }

  int64_t count() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSutilPSstats_calculatorDTh mht_7(mht_7_v, 251, "", "./tensorflow/core/util/stats_calculator.h", "count");
 return count_; }

  ValueType sum() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSutilPSstats_calculatorDTh mht_8(mht_8_v, 256, "", "./tensorflow/core/util/stats_calculator.h", "sum");
 return sum_; }

  HighPrecisionValueType squared_sum() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSutilPSstats_calculatorDTh mht_9(mht_9_v, 261, "", "./tensorflow/core/util/stats_calculator.h", "squared_sum");
 return squared_sum_; }

  bool all_same() const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSutilPSstats_calculatorDTh mht_10(mht_10_v, 266, "", "./tensorflow/core/util/stats_calculator.h", "all_same");
 return (count_ == 0 || min_ == max_); }

  HighPrecisionValueType avg() const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSutilPSstats_calculatorDTh mht_11(mht_11_v, 271, "", "./tensorflow/core/util/stats_calculator.h", "avg");

    return empty() ? std::numeric_limits<ValueType>::quiet_NaN()
                   : static_cast<HighPrecisionValueType>(sum_) / count_;
  }

  // Returns sample variance.
  ValueType sample_variance() const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSutilPSstats_calculatorDTh mht_12(mht_12_v, 280, "", "./tensorflow/core/util/stats_calculator.h", "sample_variance");

    return all_same()
               ? 0
               : (squared_sum_ - std::pow(sum_, 2.0) / count_) / (count_ - 1);
  }

  // Returns population variance.
  ValueType variance() const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSutilPSstats_calculatorDTh mht_13(mht_13_v, 290, "", "./tensorflow/core/util/stats_calculator.h", "variance");

    return all_same() ? 0 : (squared_sum_ / count_) - (avg() * avg());
  }

  // Returns population stddev.
  ValueType std_deviation() const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSutilPSstats_calculatorDTh mht_14(mht_14_v, 298, "", "./tensorflow/core/util/stats_calculator.h", "std_deviation");

    return all_same() ? 0 : std::sqrt(variance());
  }

  void OutputToStream(std::ostream* stream) const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSutilPSstats_calculatorDTh mht_15(mht_15_v, 305, "", "./tensorflow/core/util/stats_calculator.h", "OutputToStream");

    if (empty()) {
      *stream << "count=0";
    } else if (all_same()) {
      *stream << "count=" << count_ << " curr=" << newest_;
      if (count_ > 1) *stream << "(all same)";
    } else {
      *stream << "count=" << count_ << " first=" << first_
              << " curr=" << newest_ << " min=" << min_ << " max=" << max_
              << " avg=" << avg() << " std=" << std_deviation();
    }
  }

  friend std::ostream& operator<<(std::ostream& stream,
                                  const Stat<ValueType>& stat) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSutilPSstats_calculatorDTh mht_16(mht_16_v, 322, "", "./tensorflow/core/util/stats_calculator.h", "operator<<");

    stat.OutputToStream(&stream);
    return stream;
  }

 private:
  ValueType first_ = 0;
  ValueType newest_ = 0;
  ValueType max_ = std::numeric_limits<ValueType>::min();
  ValueType min_ = std::numeric_limits<ValueType>::max();
  int64_t count_ = 0;
  ValueType sum_ = 0;
  HighPrecisionValueType squared_sum_ = 0;
};

// A StatsCalculator assists in performance analysis of Graph executions.
//
// It summarizes time spent executing (on GPU/CPU), memory used etc for
// graph execution.
//
// For example usage see StatsSummarizer.
class StatsCalculator {
 public:
  enum SortingMetric {
    BY_NAME,
    BY_RUN_ORDER,
    BY_TIME,
    BY_MEMORY,
    BY_TYPE,
  };

  explicit StatsCalculator(const StatSummarizerOptions& options);

  // Returns a string detailing the accumulated runtime stats in a tab-separated
  // format which can be pasted into a spreadsheet for further analysis.
  std::string GetOutputString() const;

  std::string GetShortSummary() const;

  void ComputeStatsByType(
      std::map<std::string, int64_t>* node_type_map_count,
      std::map<std::string, int64_t>* node_type_map_time,
      std::map<std::string, int64_t>* node_type_map_memory,
      std::map<std::string, int64_t>* node_type_map_times_called,
      int64_t* accumulated_us) const;

  std::string GetStatsByNodeType() const;

  std::string GetStatsByMetric(const std::string& title,
                               SortingMetric sorting_metric,
                               int num_stats) const;

  // Returns number of runs.
  int num_runs() const {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSutilPSstats_calculatorDTh mht_17(mht_17_v, 378, "", "./tensorflow/core/util/stats_calculator.h", "num_runs");
 return static_cast<int>(run_total_us_.count()); }

  // Returns stats of total microseconds spent by all nodes in each run.
  const Stat<int64_t>& run_total_us() const {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSutilPSstats_calculatorDTh mht_18(mht_18_v, 384, "", "./tensorflow/core/util/stats_calculator.h", "run_total_us");
 return run_total_us_; }

  void UpdateRunTotalUs(int64_t run_total_us) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSutilPSstats_calculatorDTh mht_19(mht_19_v, 389, "", "./tensorflow/core/util/stats_calculator.h", "UpdateRunTotalUs");

    run_total_us_.UpdateStat(run_total_us);
  }

  void UpdateMemoryUsed(int64_t memory) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSutilPSstats_calculatorDTh mht_20(mht_20_v, 396, "", "./tensorflow/core/util/stats_calculator.h", "UpdateMemoryUsed");
 memory_.UpdateStat(memory); }

  struct Detail {
    std::string name;
    std::string type;
    int64_t run_order;
    Stat<int64_t> start_us;
    Stat<int64_t> rel_end_us;
    Stat<int64_t> mem_used;
    int64_t times_called;
  };

  const std::map<std::string, Detail>& GetDetails() const { return details_; }

  void AddNodeStats(const std::string& name, const std::string& type,
                    int64_t run_order, int64_t start_us, int64_t rel_end_us,
                    int64_t mem_used);

 private:
  void OrderNodesByMetric(SortingMetric sorting_metric,
                          std::vector<const Detail*>* details) const;

  std::string HeaderString(const std::string& title) const;
  std::string ColumnString(const Detail& detail,
                           const int64_t cumulative_stat_on_node,
                           const Stat<int64_t>& stat) const;

  Stat<int64_t> run_total_us_;
  Stat<int64_t> memory_;

  std::map<std::string, Detail> details_;
  StatSummarizerOptions options_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_STATS_CALCULATOR_H_
