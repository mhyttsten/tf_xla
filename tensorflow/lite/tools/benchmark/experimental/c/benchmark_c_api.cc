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
class MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSexperimentalPScPSbenchmark_c_apiDTcc {
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
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSexperimentalPScPSbenchmark_c_apiDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSexperimentalPScPSbenchmark_c_apiDTcc() {
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

#include "tensorflow/lite/tools/benchmark/experimental/c/benchmark_c_api.h"

#include "tensorflow/core/util/stats_calculator.h"
#include "tensorflow/lite/tools/benchmark/benchmark_tflite_model.h"

extern "C" {

// -----------------------------------------------------------------------------
// C APIs corresponding to tflite::benchmark::BenchmarkResults type.
// -----------------------------------------------------------------------------
struct TfLiteBenchmarkResults {
  const tflite::benchmark::BenchmarkResults* results;
};

// Converts the given int64_t stat into a TfLiteBenchmarkInt64Stat struct.
TfLiteBenchmarkInt64Stat ConvertStat(const tensorflow::Stat<int64_t>& stat) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSexperimentalPScPSbenchmark_c_apiDTcc mht_0(mht_0_v, 200, "", "./tensorflow/lite/tools/benchmark/experimental/c/benchmark_c_api.cc", "ConvertStat");

  return {
      stat.empty(),    stat.first(), stat.newest(),        stat.max(),
      stat.min(),      stat.count(), stat.sum(),           stat.squared_sum(),
      stat.all_same(), stat.avg(),   stat.std_deviation(),
  };
}

TfLiteBenchmarkInt64Stat TfLiteBenchmarkResultsGetInferenceTimeMicroseconds(
    const TfLiteBenchmarkResults* results) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSexperimentalPScPSbenchmark_c_apiDTcc mht_1(mht_1_v, 212, "", "./tensorflow/lite/tools/benchmark/experimental/c/benchmark_c_api.cc", "TfLiteBenchmarkResultsGetInferenceTimeMicroseconds");

  return ConvertStat(results->results->inference_time_us());
}

TfLiteBenchmarkInt64Stat TfLiteBenchmarkResultsGetWarmupTimeMicroseconds(
    const TfLiteBenchmarkResults* results) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSexperimentalPScPSbenchmark_c_apiDTcc mht_2(mht_2_v, 220, "", "./tensorflow/lite/tools/benchmark/experimental/c/benchmark_c_api.cc", "TfLiteBenchmarkResultsGetWarmupTimeMicroseconds");

  return ConvertStat(results->results->warmup_time_us());
}

int64_t TfLiteBenchmarkResultsGetStartupLatencyMicroseconds(
    const TfLiteBenchmarkResults* results) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSexperimentalPScPSbenchmark_c_apiDTcc mht_3(mht_3_v, 228, "", "./tensorflow/lite/tools/benchmark/experimental/c/benchmark_c_api.cc", "TfLiteBenchmarkResultsGetStartupLatencyMicroseconds");

  return results->results->startup_latency_us();
}

uint64_t TfLiteBenchmarkResultsGetInputBytes(
    const TfLiteBenchmarkResults* results) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSexperimentalPScPSbenchmark_c_apiDTcc mht_4(mht_4_v, 236, "", "./tensorflow/lite/tools/benchmark/experimental/c/benchmark_c_api.cc", "TfLiteBenchmarkResultsGetInputBytes");

  return results->results->input_bytes();
}

double TfLiteBenchmarkResultsGetThroughputMbPerSecond(
    const TfLiteBenchmarkResults* results) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSexperimentalPScPSbenchmark_c_apiDTcc mht_5(mht_5_v, 244, "", "./tensorflow/lite/tools/benchmark/experimental/c/benchmark_c_api.cc", "TfLiteBenchmarkResultsGetThroughputMbPerSecond");

  return results->results->throughput_MB_per_second();
}

// -----------------------------------------------------------------------------
// C APIs corresponding to tflite::benchmark::BenchmarkListener type.
// -----------------------------------------------------------------------------
class BenchmarkListenerAdapter : public tflite::benchmark::BenchmarkListener {
 public:
  void OnBenchmarkStart(
      const tflite::benchmark::BenchmarkParams& params) override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSexperimentalPScPSbenchmark_c_apiDTcc mht_6(mht_6_v, 257, "", "./tensorflow/lite/tools/benchmark/experimental/c/benchmark_c_api.cc", "OnBenchmarkStart");

    if (on_benchmark_start_fn_ != nullptr) {
      on_benchmark_start_fn_(user_data_);
    }
  }

  void OnSingleRunStart(tflite::benchmark::RunType runType) override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSexperimentalPScPSbenchmark_c_apiDTcc mht_7(mht_7_v, 266, "", "./tensorflow/lite/tools/benchmark/experimental/c/benchmark_c_api.cc", "OnSingleRunStart");

    if (on_single_run_start_fn_ != nullptr) {
      on_single_run_start_fn_(user_data_, runType == tflite::benchmark::WARMUP
                                              ? TfLiteBenchmarkWarmup
                                              : TfLiteBenchmarkRegular);
    }
  }

  void OnSingleRunEnd() override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSexperimentalPScPSbenchmark_c_apiDTcc mht_8(mht_8_v, 277, "", "./tensorflow/lite/tools/benchmark/experimental/c/benchmark_c_api.cc", "OnSingleRunEnd");

    if (on_single_run_end_fn_ != nullptr) {
      on_single_run_end_fn_(user_data_);
    }
  }

  void OnBenchmarkEnd(
      const tflite::benchmark::BenchmarkResults& results) override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSexperimentalPScPSbenchmark_c_apiDTcc mht_9(mht_9_v, 287, "", "./tensorflow/lite/tools/benchmark/experimental/c/benchmark_c_api.cc", "OnBenchmarkEnd");

    if (on_benchmark_end_fn_ != nullptr) {
      TfLiteBenchmarkResults* wrapper = new TfLiteBenchmarkResults{&results};
      on_benchmark_end_fn_(user_data_, wrapper);
      delete wrapper;
    }
  }

  // Keep the user_data pointer provided when setting the callbacks.
  void* user_data_;

  // Function pointers set by the TfLiteBenchmarkListenerSetCallbacks call.
  // Only non-null callbacks will be actually called.
  void (*on_benchmark_start_fn_)(void* user_data);
  void (*on_single_run_start_fn_)(void* user_data,
                                  TfLiteBenchmarkRunType runType);
  void (*on_single_run_end_fn_)(void* user_data);
  void (*on_benchmark_end_fn_)(void* user_data,
                               TfLiteBenchmarkResults* results);
};

struct TfLiteBenchmarkListener {
  std::unique_ptr<BenchmarkListenerAdapter> adapter;
};

TfLiteBenchmarkListener* TfLiteBenchmarkListenerCreate() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSexperimentalPScPSbenchmark_c_apiDTcc mht_10(mht_10_v, 315, "", "./tensorflow/lite/tools/benchmark/experimental/c/benchmark_c_api.cc", "TfLiteBenchmarkListenerCreate");

  std::unique_ptr<BenchmarkListenerAdapter> adapter(
      new BenchmarkListenerAdapter());
  return new TfLiteBenchmarkListener{std::move(adapter)};
}

void TfLiteBenchmarkListenerDelete(TfLiteBenchmarkListener* listener) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSexperimentalPScPSbenchmark_c_apiDTcc mht_11(mht_11_v, 324, "", "./tensorflow/lite/tools/benchmark/experimental/c/benchmark_c_api.cc", "TfLiteBenchmarkListenerDelete");

  delete listener;
}

void TfLiteBenchmarkListenerSetCallbacks(
    TfLiteBenchmarkListener* listener, void* user_data,
    void (*on_benchmark_start_fn)(void* user_data),
    void (*on_single_run_start_fn)(void* user_data,
                                   TfLiteBenchmarkRunType runType),
    void (*on_single_run_end_fn)(void* user_data),
    void (*on_benchmark_end_fn)(void* user_data,
                                TfLiteBenchmarkResults* results)) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSexperimentalPScPSbenchmark_c_apiDTcc mht_12(mht_12_v, 338, "", "./tensorflow/lite/tools/benchmark/experimental/c/benchmark_c_api.cc", "TfLiteBenchmarkListenerSetCallbacks");

  listener->adapter->user_data_ = user_data;
  listener->adapter->on_benchmark_start_fn_ = on_benchmark_start_fn;
  listener->adapter->on_single_run_start_fn_ = on_single_run_start_fn;
  listener->adapter->on_single_run_end_fn_ = on_single_run_end_fn;
  listener->adapter->on_benchmark_end_fn_ = on_benchmark_end_fn;
}

// -----------------------------------------------------------------------------
// C APIs corresponding to tflite::benchmark::BenchmarkTfLiteModel type.
// -----------------------------------------------------------------------------
struct TfLiteBenchmarkTfLiteModel {
  std::unique_ptr<tflite::benchmark::BenchmarkTfLiteModel> benchmark_model;
};

TfLiteBenchmarkTfLiteModel* TfLiteBenchmarkTfLiteModelCreate() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSexperimentalPScPSbenchmark_c_apiDTcc mht_13(mht_13_v, 356, "", "./tensorflow/lite/tools/benchmark/experimental/c/benchmark_c_api.cc", "TfLiteBenchmarkTfLiteModelCreate");

  std::unique_ptr<tflite::benchmark::BenchmarkTfLiteModel> benchmark_model(
      new tflite::benchmark::BenchmarkTfLiteModel());
  return new TfLiteBenchmarkTfLiteModel{std::move(benchmark_model)};
}

void TfLiteBenchmarkTfLiteModelDelete(
    TfLiteBenchmarkTfLiteModel* benchmark_model) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSexperimentalPScPSbenchmark_c_apiDTcc mht_14(mht_14_v, 366, "", "./tensorflow/lite/tools/benchmark/experimental/c/benchmark_c_api.cc", "TfLiteBenchmarkTfLiteModelDelete");

  delete benchmark_model;
}

TfLiteStatus TfLiteBenchmarkTfLiteModelInit(
    TfLiteBenchmarkTfLiteModel* benchmark_model) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSexperimentalPScPSbenchmark_c_apiDTcc mht_15(mht_15_v, 374, "", "./tensorflow/lite/tools/benchmark/experimental/c/benchmark_c_api.cc", "TfLiteBenchmarkTfLiteModelInit");

  return benchmark_model->benchmark_model->Init();
}

TfLiteStatus TfLiteBenchmarkTfLiteModelRun(
    TfLiteBenchmarkTfLiteModel* benchmark_model) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSexperimentalPScPSbenchmark_c_apiDTcc mht_16(mht_16_v, 382, "", "./tensorflow/lite/tools/benchmark/experimental/c/benchmark_c_api.cc", "TfLiteBenchmarkTfLiteModelRun");

  return benchmark_model->benchmark_model->Run();
}

TfLiteStatus TfLiteBenchmarkTfLiteModelRunWithArgs(
    TfLiteBenchmarkTfLiteModel* benchmark_model, int argc, char** argv) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSexperimentalPScPSbenchmark_c_apiDTcc mht_17(mht_17_v, 390, "", "./tensorflow/lite/tools/benchmark/experimental/c/benchmark_c_api.cc", "TfLiteBenchmarkTfLiteModelRunWithArgs");

  return benchmark_model->benchmark_model->Run(argc, argv);
}

void TfLiteBenchmarkTfLiteModelAddListener(
    TfLiteBenchmarkTfLiteModel* benchmark_model,
    const TfLiteBenchmarkListener* listener) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSexperimentalPScPSbenchmark_c_apiDTcc mht_18(mht_18_v, 399, "", "./tensorflow/lite/tools/benchmark/experimental/c/benchmark_c_api.cc", "TfLiteBenchmarkTfLiteModelAddListener");

  return benchmark_model->benchmark_model->AddListener(listener->adapter.get());
}

}  // extern "C"
