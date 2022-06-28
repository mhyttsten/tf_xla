/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_TOOLS_BENCHMARK_BENCHMARK_MODEL_H_
#define TENSORFLOW_LITE_TOOLS_BENCHMARK_BENCHMARK_MODEL_H_
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
class MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSbenchmark_modelDTh {
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
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSbenchmark_modelDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSbenchmark_modelDTh() {
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


#include <cmath>
#include <cstdint>
#include <limits>
#include <ostream>
#include <string>
#include <unordered_set>
#include <vector>

#include "tensorflow/core/util/stats_calculator.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/profiling/memory_info.h"
#include "tensorflow/lite/profiling/memory_usage_monitor.h"
#include "tensorflow/lite/tools/benchmark/benchmark_params.h"
#include "tensorflow/lite/tools/command_line_flags.h"

namespace tflite {
namespace benchmark {

enum RunType {
  WARMUP,
  REGULAR,
};

class BenchmarkResults {
 public:
  BenchmarkResults() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSbenchmark_modelDTh mht_0(mht_0_v, 213, "", "./tensorflow/lite/tools/benchmark/benchmark_model.h", "BenchmarkResults");
}
  BenchmarkResults(double model_size_mb, int64_t startup_latency_us,
                   uint64_t input_bytes,
                   tensorflow::Stat<int64_t> warmup_time_us,
                   tensorflow::Stat<int64_t> inference_time_us,
                   const profiling::memory::MemoryUsage& init_mem_usage,
                   const profiling::memory::MemoryUsage& overall_mem_usage,
                   float peak_mem_mb)
      : model_size_mb_(model_size_mb),
        startup_latency_us_(startup_latency_us),
        input_bytes_(input_bytes),
        warmup_time_us_(warmup_time_us),
        inference_time_us_(inference_time_us),
        init_mem_usage_(init_mem_usage),
        overall_mem_usage_(overall_mem_usage),
        peak_mem_mb_(peak_mem_mb) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSbenchmark_modelDTh mht_1(mht_1_v, 231, "", "./tensorflow/lite/tools/benchmark/benchmark_model.h", "BenchmarkResults");
}

  const double model_size_mb() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSbenchmark_modelDTh mht_2(mht_2_v, 236, "", "./tensorflow/lite/tools/benchmark/benchmark_model.h", "model_size_mb");
 return model_size_mb_; }
  tensorflow::Stat<int64_t> inference_time_us() const {
    return inference_time_us_;
  }
  tensorflow::Stat<int64_t> warmup_time_us() const { return warmup_time_us_; }
  int64_t startup_latency_us() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSbenchmark_modelDTh mht_3(mht_3_v, 244, "", "./tensorflow/lite/tools/benchmark/benchmark_model.h", "startup_latency_us");
 return startup_latency_us_; }
  uint64_t input_bytes() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSbenchmark_modelDTh mht_4(mht_4_v, 248, "", "./tensorflow/lite/tools/benchmark/benchmark_model.h", "input_bytes");
 return input_bytes_; }
  double throughput_MB_per_second() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSbenchmark_modelDTh mht_5(mht_5_v, 252, "", "./tensorflow/lite/tools/benchmark/benchmark_model.h", "throughput_MB_per_second");

    double bytes_per_sec = (input_bytes_ * inference_time_us_.count() * 1e6) /
                           inference_time_us_.sum();
    return bytes_per_sec / (1024.0 * 1024.0);
  }

  const profiling::memory::MemoryUsage& init_mem_usage() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSbenchmark_modelDTh mht_6(mht_6_v, 261, "", "./tensorflow/lite/tools/benchmark/benchmark_model.h", "init_mem_usage");

    return init_mem_usage_;
  }
  const profiling::memory::MemoryUsage& overall_mem_usage() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSbenchmark_modelDTh mht_7(mht_7_v, 267, "", "./tensorflow/lite/tools/benchmark/benchmark_model.h", "overall_mem_usage");

    return overall_mem_usage_;
  }
  float peak_mem_mb() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSbenchmark_modelDTh mht_8(mht_8_v, 273, "", "./tensorflow/lite/tools/benchmark/benchmark_model.h", "peak_mem_mb");
 return peak_mem_mb_; }

 private:
  double model_size_mb_ = 0.0;
  int64_t startup_latency_us_ = 0;
  uint64_t input_bytes_ = 0;
  tensorflow::Stat<int64_t> warmup_time_us_;
  tensorflow::Stat<int64_t> inference_time_us_;
  profiling::memory::MemoryUsage init_mem_usage_;
  profiling::memory::MemoryUsage overall_mem_usage_;
  // An invalid value could happen when we don't monitor memory footprint for
  // the inference, or the memory usage info isn't available on the benchmarking
  // platform.
  float peak_mem_mb_ =
      profiling::memory::MemoryUsageMonitor::kInvalidMemUsageMB;
};

class BenchmarkListener {
 public:
  // Called before the (outer) inference loop begins.
  // Note that this is called *after* the interpreter has been initialized, but
  // *before* any warmup runs have been executed.
  virtual void OnBenchmarkStart(const BenchmarkParams& params) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSbenchmark_modelDTh mht_9(mht_9_v, 298, "", "./tensorflow/lite/tools/benchmark/benchmark_model.h", "OnBenchmarkStart");
}
  // Called before a single (inner) inference call starts.
  virtual void OnSingleRunStart(RunType runType) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSbenchmark_modelDTh mht_10(mht_10_v, 303, "", "./tensorflow/lite/tools/benchmark/benchmark_model.h", "OnSingleRunStart");
}
  // Called before a single (inner) inference call ends.
  virtual void OnSingleRunEnd() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSbenchmark_modelDTh mht_11(mht_11_v, 308, "", "./tensorflow/lite/tools/benchmark/benchmark_model.h", "OnSingleRunEnd");
}
  // Called after the (outer) inference loop begins.
  virtual void OnBenchmarkEnd(const BenchmarkResults& results) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSbenchmark_modelDTh mht_12(mht_12_v, 313, "", "./tensorflow/lite/tools/benchmark/benchmark_model.h", "OnBenchmarkEnd");
}
  virtual ~BenchmarkListener() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSbenchmark_modelDTh mht_13(mht_13_v, 317, "", "./tensorflow/lite/tools/benchmark/benchmark_model.h", "~BenchmarkListener");
}
};

// A listener that forwards its method calls to a collection of listeners.
class BenchmarkListeners : public BenchmarkListener {
 public:
  // Added a listener to the listener collection.
  // |listener| is not owned by the instance of |BenchmarkListeners|.
  // |listener| should not be null and should outlast the instance of
  // |BenchmarkListeners|.
  void AddListener(BenchmarkListener* listener) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSbenchmark_modelDTh mht_14(mht_14_v, 330, "", "./tensorflow/lite/tools/benchmark/benchmark_model.h", "AddListener");

    listeners_.push_back(listener);
  }

  // Remove all listeners after [index] including the one at 'index'.
  void RemoveListeners(int index) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSbenchmark_modelDTh mht_15(mht_15_v, 338, "", "./tensorflow/lite/tools/benchmark/benchmark_model.h", "RemoveListeners");

    if (index >= NumListeners()) return;
    listeners_.resize(index);
  }

  int NumListeners() const {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSbenchmark_modelDTh mht_16(mht_16_v, 346, "", "./tensorflow/lite/tools/benchmark/benchmark_model.h", "NumListeners");
 return listeners_.size(); }

  void OnBenchmarkStart(const BenchmarkParams& params) override {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSbenchmark_modelDTh mht_17(mht_17_v, 351, "", "./tensorflow/lite/tools/benchmark/benchmark_model.h", "OnBenchmarkStart");

    for (auto listener : listeners_) {
      listener->OnBenchmarkStart(params);
    }
  }

  void OnSingleRunStart(RunType runType) override {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSbenchmark_modelDTh mht_18(mht_18_v, 360, "", "./tensorflow/lite/tools/benchmark/benchmark_model.h", "OnSingleRunStart");

    for (auto listener : listeners_) {
      listener->OnSingleRunStart(runType);
    }
  }

  void OnSingleRunEnd() override {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSbenchmark_modelDTh mht_19(mht_19_v, 369, "", "./tensorflow/lite/tools/benchmark/benchmark_model.h", "OnSingleRunEnd");

    for (auto listener : listeners_) {
      listener->OnSingleRunEnd();
    }
  }

  void OnBenchmarkEnd(const BenchmarkResults& results) override {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSbenchmark_modelDTh mht_20(mht_20_v, 378, "", "./tensorflow/lite/tools/benchmark/benchmark_model.h", "OnBenchmarkEnd");

    for (auto listener : listeners_) {
      listener->OnBenchmarkEnd(results);
    }
  }

  ~BenchmarkListeners() override {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSbenchmark_modelDTh mht_21(mht_21_v, 387, "", "./tensorflow/lite/tools/benchmark/benchmark_model.h", "~BenchmarkListeners");
}

 private:
  // Use vector so listeners are invoked in the order they are added.
  std::vector<BenchmarkListener*> listeners_;
};

// Benchmark listener that just logs the results of benchmark run.
class BenchmarkLoggingListener : public BenchmarkListener {
 public:
  void OnBenchmarkEnd(const BenchmarkResults& results) override;
};

template <typename T>
Flag CreateFlag(const char* name, BenchmarkParams* params,
                const std::string& usage) {
   std::vector<std::string> mht_22_v;
   mht_22_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   mht_22_v.push_back("usage: \"" + usage + "\"");
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSbenchmark_modelDTh mht_22(mht_22_v, 407, "", "./tensorflow/lite/tools/benchmark/benchmark_model.h", "CreateFlag");

  return Flag(
      name,
      [params, name](const T& val, int argv_position) {
        params->Set<T>(name, val, argv_position);
      },
      params->Get<T>(name), usage, Flag::kOptional);
}

// Benchmarks a model.
//
// Subclasses need to implement initialization and running of the model.
// The results can be collected by adding BenchmarkListener(s).
class BenchmarkModel {
 public:
  static BenchmarkParams DefaultParams();
  BenchmarkModel();
  explicit BenchmarkModel(BenchmarkParams params)
      : params_(std::move(params)) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSbenchmark_modelDTh mht_23(mht_23_v, 428, "", "./tensorflow/lite/tools/benchmark/benchmark_model.h", "BenchmarkModel");
}
  virtual ~BenchmarkModel() {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSbenchmark_modelDTh mht_24(mht_24_v, 432, "", "./tensorflow/lite/tools/benchmark/benchmark_model.h", "~BenchmarkModel");
}
  virtual TfLiteStatus Init() = 0;
  virtual TfLiteStatus Run(int argc, char** argv);
  virtual TfLiteStatus Run();
  void AddListener(BenchmarkListener* listener) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSbenchmark_modelDTh mht_25(mht_25_v, 439, "", "./tensorflow/lite/tools/benchmark/benchmark_model.h", "AddListener");

    listeners_.AddListener(listener);
  }
  // Remove all listeners after [index] including the one at 'index'.
  void RemoveListeners(int index) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSbenchmark_modelDTh mht_26(mht_26_v, 446, "", "./tensorflow/lite/tools/benchmark/benchmark_model.h", "RemoveListeners");
 listeners_.RemoveListeners(index); }
  int NumListeners() const {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSbenchmark_modelDTh mht_27(mht_27_v, 450, "", "./tensorflow/lite/tools/benchmark/benchmark_model.h", "NumListeners");
 return listeners_.NumListeners(); }

  BenchmarkParams* mutable_params() {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSbenchmark_modelDTh mht_28(mht_28_v, 455, "", "./tensorflow/lite/tools/benchmark/benchmark_model.h", "mutable_params");
 return &params_; }

  // Unparsable flags will remain in 'argv' in the original order and 'argc'
  // will be updated accordingly.
  TfLiteStatus ParseFlags(int* argc, char** argv);

 protected:
  virtual void LogParams();
  virtual TfLiteStatus ValidateParams();

  TfLiteStatus ParseFlags(int argc, char** argv) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSbenchmark_modelDTh mht_29(mht_29_v, 468, "", "./tensorflow/lite/tools/benchmark/benchmark_model.h", "ParseFlags");

    return ParseFlags(&argc, argv);
  }
  virtual std::vector<Flag> GetFlags();

  // Get the model file size if it's available.
  virtual int64_t MayGetModelFileSize() {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSbenchmark_modelDTh mht_30(mht_30_v, 477, "", "./tensorflow/lite/tools/benchmark/benchmark_model.h", "MayGetModelFileSize");
 return -1; }
  virtual uint64_t ComputeInputBytes() = 0;
  virtual tensorflow::Stat<int64_t> Run(int min_num_times, float min_secs,
                                        float max_secs, RunType run_type,
                                        TfLiteStatus* invoke_status);
  // Prepares input data for benchmark. This can be used to initialize input
  // data that has non-trivial cost.
  virtual TfLiteStatus PrepareInputData();

  virtual TfLiteStatus ResetInputsAndOutputs();
  virtual TfLiteStatus RunImpl() = 0;

  // Create a MemoryUsageMonitor to report peak memory footprint if specified.
  virtual std::unique_ptr<profiling::memory::MemoryUsageMonitor>
  MayCreateMemoryUsageMonitor() const;

  BenchmarkParams params_;
  BenchmarkListeners listeners_;
};

}  // namespace benchmark
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_BENCHMARK_BENCHMARK_MODEL_H_
