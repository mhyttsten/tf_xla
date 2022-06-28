/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_TOOLS_BENCHMARK_BENCHMARK_PERFORMANCE_OPTIONS_H_
#define TENSORFLOW_LITE_TOOLS_BENCHMARK_BENCHMARK_PERFORMANCE_OPTIONS_H_
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
class MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSbenchmark_performance_optionsDTh {
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
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSbenchmark_performance_optionsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSbenchmark_performance_optionsDTh() {
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


#include <memory>
#include <string>
#include <vector>

#include "absl/memory/memory.h"
#include "tensorflow/lite/tools/benchmark/benchmark_model.h"
#include "tensorflow/lite/tools/benchmark/benchmark_params.h"

namespace tflite {
namespace benchmark {

class MultiRunStatsRecorder : public BenchmarkListener {
 public:
  // BenchmarkListener::OnBenchmarkStart is invoked after each run's
  // BenchmarkModel::Init. However, some run could fail during Init, e.g.
  // delegate fails to be created etc. To still record such run, we will call
  // the following function right before a run starts.
  void MarkBenchmarkStart(const BenchmarkParams& params) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSbenchmark_performance_optionsDTh mht_0(mht_0_v, 205, "", "./tensorflow/lite/tools/benchmark/benchmark_performance_options.h", "MarkBenchmarkStart");

    results_.emplace_back(EachRunResult());
    auto& current = results_.back();
    current.completed = false;
    current.params = absl::make_unique<BenchmarkParams>();
    current.params->Merge(params, true /* overwrite*/);
  }

  void OnBenchmarkEnd(const BenchmarkResults& results) final {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSbenchmark_performance_optionsDTh mht_1(mht_1_v, 216, "", "./tensorflow/lite/tools/benchmark/benchmark_performance_options.h", "OnBenchmarkEnd");

    auto& current = results_.back();
    current.completed = true;
    current.metrics = results;
  }

  virtual void OutputStats();

 protected:
  struct EachRunResult {
    bool completed = false;
    std::unique_ptr<BenchmarkParams> params;
    BenchmarkResults metrics;
  };
  std::vector<EachRunResult> results_;

  // Use this to order the runs by the average inference time in increasing
  // order (i.e. the fastest run ranks first.). If the run didn't complete,
  // we consider it to be slowest.
  struct EachRunStatsEntryComparator {
    bool operator()(const EachRunResult& i, const EachRunResult& j) {
      if (!i.completed) return false;
      if (!j.completed) return true;
      return i.metrics.inference_time_us().avg() <
             j.metrics.inference_time_us().avg();
    }
  };

  virtual std::string PerfOptionName(const BenchmarkParams& params) const;
};

// Benchmarks all performance options on a model by repeatedly invoking the
// single-performance-option run on a passed-in 'BenchmarkModel' object.
class BenchmarkPerformanceOptions {
 public:
  // Doesn't own the memory of 'single_option_run'.
  explicit BenchmarkPerformanceOptions(
      BenchmarkModel* single_option_run,
      std::unique_ptr<MultiRunStatsRecorder> all_run_stats =
          absl::make_unique<MultiRunStatsRecorder>());

  virtual ~BenchmarkPerformanceOptions() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSbenchmark_performance_optionsDTh mht_2(mht_2_v, 260, "", "./tensorflow/lite/tools/benchmark/benchmark_performance_options.h", "~BenchmarkPerformanceOptions");
}

  // Just run the benchmark just w/ default parameter values.
  void Run();
  void Run(int argc, char** argv);

 protected:
  static BenchmarkParams DefaultParams();

  BenchmarkPerformanceOptions(
      BenchmarkParams params, BenchmarkModel* single_option_run,
      std::unique_ptr<MultiRunStatsRecorder> all_run_stats);

  // Unparsable flags will remain in 'argv' in the original order and 'argc'
  // will be updated accordingly.
  bool ParseFlags(int* argc, char** argv);
  virtual std::vector<Flag> GetFlags();

  bool ParsePerfOptions();
  virtual std::vector<std::string> GetValidPerfOptions() const;
  bool HasOption(const std::string& option) const;

  virtual void ResetPerformanceOptions();
  virtual void CreatePerformanceOptions();

  BenchmarkParams params_;
  std::vector<std::string> perf_options_;

  // The object that drives a single-performance-option run.
  BenchmarkModel* const single_option_run_;          // Doesn't own the memory.
  BenchmarkParams* const single_option_run_params_;  // Doesn't own the memory.

  // Each element is a set of performance-affecting benchmark parameters to be
  // all set for a particular benchmark run.
  std::vector<BenchmarkParams> all_run_params_;

  std::unique_ptr<MultiRunStatsRecorder> all_run_stats_;
};

}  // namespace benchmark
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_BENCHMARK_BENCHMARK_PERFORMANCE_OPTIONS_H_
