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
class MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_tf_functionsDTcc {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_tf_functionsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_tf_functionsDTcc() {
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
You may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/profiler/convert/xplane_to_tf_functions.h"

#include <algorithm>
#include <stack>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/utils/math_utils.h"
#include "tensorflow/core/profiler/utils/tf_xplane_visitor.h"
#include "tensorflow/core/profiler/utils/timespan.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"
#include "tensorflow/core/profiler/utils/xplane_visitor.h"

namespace tensorflow {
namespace profiler {

namespace {

std::pair<TfFunctionExecutionMode, TfFunctionCompiler> Decode(
    absl::string_view function_name, absl::string_view mode) {
  // mode is one of ["eager", "concrete", "traced-xla", "traced-nonXla",
  // "notTraced-xla", "notTraced-nonXla"]
  if (mode == "eager") return {EAGER_MODE, INVALID_COMPILER};
  if (mode == "concrete") return {CONCRETE_MODE, INVALID_COMPILER};
  if (mode == "traced-xla") return {TRACED_MODE, XLA_COMPILER};
  if (mode == "traced-nonXla") return {TRACED_MODE, OTHER_COMPILER};
  if (mode == "notTraced-xla") return {NOT_TRACED_MODE, XLA_COMPILER};
  if (mode == "notTraced-nonXla") return {NOT_TRACED_MODE, OTHER_COMPILER};
  // Shouldn't reach here.
  LOG(ERROR) << absl::StrCat("tf-function '", function_name,
                             "' has an unexpected execution mode '", mode, "'")
             << std::endl;
  return {INVALID_MODE, INVALID_COMPILER};
  DCHECK(false);
}

double ComputeExpensiveCallPercent(const TfFunction& tf_function) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_tf_functionsDTcc mht_0(mht_0_v, 231, "", "./tensorflow/core/profiler/convert/xplane_to_tf_functions.cc", "ComputeExpensiveCallPercent");

  // Computes the expensiveness in terms of time (rather than count).
  uint64 total_call_time_ps = 0;
  uint64 expensive_call_time_ps = 0;
  for (const auto& mode_metrics : tf_function.metrics()) {
    const auto mode = mode_metrics.first;
    const auto& metrics = mode_metrics.second;
    total_call_time_ps += metrics.self_time_ps();
    if (mode == TRACED_MODE || mode == EAGER_MODE) {
      expensive_call_time_ps += metrics.self_time_ps();
    }
  }
  return SafeDivide(100.0 * expensive_call_time_ps, total_call_time_ps);
}

// Each invocation of a tf-function creates an ActivationRecord.
struct ActivationRecord {
  std::string function_name;               // name of the tf-function.
  Timespan timespan;                       // timespan of this invocation.
  TfFunctionExecutionMode execution_mode;  // execution mode.
  TfFunctionCompiler compiler;             // compiler used.
  int64_t tracing_count;  // the total tracing count of this function when this
                          // invocation happened.
  uint64 children_duration_ps;  // Sum of the duration of all (immediate)
                                // children tf-functions of this function.
  ActivationRecord()
      : function_name(""),
        execution_mode(INVALID_MODE),
        compiler(INVALID_COMPILER),
        tracing_count(0),
        children_duration_ps(0) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_tf_functionsDTcc mht_1(mht_1_v, 264, "", "./tensorflow/core/profiler/convert/xplane_to_tf_functions.cc", "ActivationRecord");
}
  ActivationRecord(absl::string_view name, const Timespan& timespan,
                   TfFunctionExecutionMode exe_mode,
                   TfFunctionCompiler compiler, int64_t tracing_cnt)
      : function_name(std::string(name)),
        timespan(timespan),
        execution_mode(exe_mode),
        compiler(compiler),
        tracing_count(tracing_cnt),
        children_duration_ps(0) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_tf_functionsDTcc mht_2(mht_2_v, 277, "", "./tensorflow/core/profiler/convert/xplane_to_tf_functions.cc", "ActivationRecord");
}
  std::string DebugString() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_tf_functionsDTcc mht_3(mht_3_v, 281, "", "./tensorflow/core/profiler/convert/xplane_to_tf_functions.cc", "DebugString");

    return absl::StrCat("{", function_name, ", ",
                        TfFunctionExecutionMode_Name(execution_mode), ", ",
                        TfFunctionCompiler_Name(compiler),
                        ", tracing_count:", tracing_count,
                        ", children_duration:", children_duration_ps,
                        " ps, timespan:", timespan.DebugString(), "}");
  }
};

// Entry or exit point of a tf-function.
struct EntryOrExit {
  bool is_entry;        // true for entry, false for exit.
  int64_t index;        // index to the ActivationRecord.
  uint64 timestamp_ps;  // the time when this entry/exit happens.
  EntryOrExit() : is_entry(false), index(-1), timestamp_ps(0) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_tf_functionsDTcc mht_4(mht_4_v, 299, "", "./tensorflow/core/profiler/convert/xplane_to_tf_functions.cc", "EntryOrExit");
}
  EntryOrExit(bool is_entry, int64_t index, uint64 timestamp_ps)
      : is_entry(is_entry), index(index), timestamp_ps(timestamp_ps) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_tf_functionsDTcc mht_5(mht_5_v, 304, "", "./tensorflow/core/profiler/convert/xplane_to_tf_functions.cc", "EntryOrExit");
}
  std::string DebugString() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_tf_functionsDTcc mht_6(mht_6_v, 308, "", "./tensorflow/core/profiler/convert/xplane_to_tf_functions.cc", "DebugString");

    std::string entry_or_exit = is_entry ? "entry, " : "exit,  ";
    return absl::StrCat("{", entry_or_exit, "idx:", index,
                        ", timestamp:", timestamp_ps, "}");
  }
};

TfFunctionCompiler CombineCompilers(TfFunctionCompiler a,
                                    TfFunctionCompiler b) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_tf_functionsDTcc mht_7(mht_7_v, 319, "", "./tensorflow/core/profiler/convert/xplane_to_tf_functions.cc", "CombineCompilers");

  if (a == INVALID_COMPILER) return b;
  if (b == INVALID_COMPILER) return a;
  if (a == b) return a;
  return MIXED_COMPILER;
}

void CombineTfFunctionMetrics(const TfFunctionMetrics& src,
                              TfFunctionMetrics* dst) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_tf_functionsDTcc mht_8(mht_8_v, 330, "", "./tensorflow/core/profiler/convert/xplane_to_tf_functions.cc", "CombineTfFunctionMetrics");

  dst->set_count(src.count() + dst->count());
  dst->set_self_time_ps(src.self_time_ps() + dst->self_time_ps());
}

void CombineTfFunction(const TfFunction& src, TfFunction* dst) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_tf_functionsDTcc mht_9(mht_9_v, 338, "", "./tensorflow/core/profiler/convert/xplane_to_tf_functions.cc", "CombineTfFunction");

  dst->set_total_tracing_count(
      std::max(src.total_tracing_count(), dst->total_tracing_count()));
  dst->set_compiler(CombineCompilers(src.compiler(), dst->compiler()));
  for (const auto& mode_metrics : src.metrics()) {
    int32_t execution_mode = mode_metrics.first;
    const TfFunctionMetrics& src_metrics = mode_metrics.second;
    TfFunctionMetrics* dst_metrics =
        gtl::FindOrNull(*dst->mutable_metrics(), execution_mode);
    if (dst_metrics == nullptr) {
      (*dst->mutable_metrics())[execution_mode] = src_metrics;
    } else {
      CombineTfFunctionMetrics(src_metrics, dst_metrics);
    }
  }
  dst->set_expensive_call_percent(ComputeExpensiveCallPercent(*dst));
}

// Execution history of all tf-functions invoked.
class TfFunctionExecutions {
 public:
  explicit TfFunctionExecutions(const XLineVisitor& line) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_tf_functionsDTcc mht_10(mht_10_v, 362, "", "./tensorflow/core/profiler/convert/xplane_to_tf_functions.cc", "TfFunctionExecutions");

    // Creates points_ and activations_ from line.
    line.ForEachEvent([&](const XEventVisitor& event) {
      absl::string_view mode;
      int64_t tracing_count = 0;
      event.ForEachStat([&mode, &tracing_count](const XStatVisitor& stat) {
        if (!stat.Type().has_value()) return;
        switch (stat.Type().value()) {
          case StatType::kTfFunctionCall:
            mode = stat.StrOrRefValue();
            break;
          case StatType::kTfFunctionTracingCount:
            tracing_count = stat.IntValue();
            break;
        }
      });
      if (mode.empty()) return;

      // event is a tf-function.
      int64_t index = activations_.size();
      auto timespan = event.GetTimespan();
      auto mode_compiler = Decode(event.Name(), mode);
      ActivationRecord activation_record =
          ActivationRecord(event.Name(), timespan, mode_compiler.first,
                           mode_compiler.second, tracing_count);
      activations_.push_back(activation_record);
      EntryOrExit entry_point =
          EntryOrExit(/*is_entry=*/true, index, timespan.begin_ps());
      EntryOrExit exit_point =
          EntryOrExit(/*is_entry=*/false, index, timespan.end_ps());
      points_.push_back(entry_point);
      points_.push_back(exit_point);
    });

    // Sorts points_ in ascending order of timestamps.
    auto ascending_in_timestamp = [](const EntryOrExit& a,
                                     const EntryOrExit& b) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_tf_functionsDTcc mht_11(mht_11_v, 401, "", "./tensorflow/core/profiler/convert/xplane_to_tf_functions.cc", "lambda");

      return a.timestamp_ps < b.timestamp_ps;
    };
    absl::c_sort(points_, ascending_in_timestamp);

    // Calculates the children duration for each activation record.
    CalculateChildrenDurations();
  }

  std::string DebugString() const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_tf_functionsDTcc mht_12(mht_12_v, 413, "", "./tensorflow/core/profiler/convert/xplane_to_tf_functions.cc", "DebugString");

    std::string result = "\nActivations:\n";
    for (int i = 0, end = activations_.size(); i < end; i++) {
      absl::StrAppend(&result, "[", i, "] ", activations_[i].DebugString(),
                      "\n");
    }
    absl::StrAppend(&result, "tf-function Entry/Exit Points:\n");
    for (const auto& pt : points_) {
      absl::StrAppend(&result, pt.DebugString(), "\n");
    }
    return result;
  }

  // Converts this execution history to a TfFunctionDb.
  TfFunctionDb ConvertToTfFunctionDb() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_tf_functionsDTcc mht_13(mht_13_v, 430, "", "./tensorflow/core/profiler/convert/xplane_to_tf_functions.cc", "ConvertToTfFunctionDb");

    TfFunctionDb result;
    for (const auto& record : activations_) {
      TfFunction* fun = &(*result.mutable_tf_functions())[record.function_name];
      fun->set_total_tracing_count(
          std::max(static_cast<int64_t>(fun->total_tracing_count()),
                   record.tracing_count));
      fun->set_compiler(CombineCompilers(fun->compiler(), record.compiler));
      // The self-time of this function is the difference between the duration
      // of this function and the duration of its children.
      uint64 self_time_ps =
          record.timespan.duration_ps() - record.children_duration_ps;
      // Updates the metrics for this execution mode with this invocation.
      TfFunctionMetrics* metrics =
          &(*fun->mutable_metrics())[record.execution_mode];
      metrics->set_count(metrics->count() + 1);
      metrics->set_self_time_ps(metrics->self_time_ps() + self_time_ps);
    }
    for (auto& name_fun : *result.mutable_tf_functions()) {
      TfFunction& fun = name_fun.second;
      fun.set_expensive_call_percent(ComputeExpensiveCallPercent(fun));
    }
    return result;
  }

  // Calculates the children duration of every tf-function.
  void CalculateChildrenDurations() {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_tf_functionsDTcc mht_14(mht_14_v, 459, "", "./tensorflow/core/profiler/convert/xplane_to_tf_functions.cc", "CalculateChildrenDurations");

    std::stack<int64_t> call_stack;
    for (const auto& pt : points_) {
      if (pt.is_entry) {
        // Function entry.
        call_stack.push(pt.index);
      } else {
        // Function exit.
        DCHECK(call_stack.top() == pt.index);  // must be well nested.
        uint64 call_duration = activations_[pt.index].timespan.duration_ps();
        call_stack.pop();
        if (!call_stack.empty()) {
          // call_stack.top() is the parent tf-function; adds call_duration to
          // its children_duration.
          activations_[call_stack.top()].children_duration_ps += call_duration;
        }
      }
    }
  }

 private:
  // ActivationRecords for all tf-function invocations.
  std::vector<ActivationRecord> activations_;
  // Entry and exit points of all invocations.
  std::vector<EntryOrExit> points_;
};

}  // namespace

std::string DebugString(const TfFunctionDb& tf_function_db) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_tf_functionsDTcc mht_15(mht_15_v, 491, "", "./tensorflow/core/profiler/convert/xplane_to_tf_functions.cc", "DebugString");

  std::string str;
  protobuf::TextFormat::PrintToString(tf_function_db, &str);
  return str;
}

void CombineTfFunctionDb(const TfFunctionDb& src, TfFunctionDb* dst) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_tf_functionsDTcc mht_16(mht_16_v, 500, "", "./tensorflow/core/profiler/convert/xplane_to_tf_functions.cc", "CombineTfFunctionDb");

  for (const auto& name_function : src.tf_functions()) {
    const auto& name = name_function.first;
    const auto& src_fun = name_function.second;
    TfFunction* dst_fun = gtl::FindOrNull(*dst->mutable_tf_functions(), name);
    if (dst_fun == nullptr) {
      (*dst->mutable_tf_functions())[name] = src_fun;
    } else {
      CombineTfFunction(src_fun, dst_fun);
    }
  }
}

TfFunctionDb ConvertHostThreadsXLineToTfFunctionDb(const XLineVisitor& line) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_tf_functionsDTcc mht_17(mht_17_v, 516, "", "./tensorflow/core/profiler/convert/xplane_to_tf_functions.cc", "ConvertHostThreadsXLineToTfFunctionDb");

  TfFunctionExecutions tf_function_executions = TfFunctionExecutions(line);
  return tf_function_executions.ConvertToTfFunctionDb();
}

}  // namespace profiler
}  // namespace tensorflow
