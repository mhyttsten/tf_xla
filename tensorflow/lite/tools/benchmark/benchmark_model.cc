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
class MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSbenchmark_modelDTcc {
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
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSbenchmark_modelDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSbenchmark_modelDTcc() {
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

#include "tensorflow/lite/tools/benchmark/benchmark_model.h"

#include <iostream>
#include <sstream>

#include "tensorflow/lite/profiling/memory_info.h"
#include "tensorflow/lite/profiling/time.h"
#include "tensorflow/lite/tools/benchmark/benchmark_utils.h"
#include "tensorflow/lite/tools/logging.h"

namespace tflite {
namespace benchmark {
using tensorflow::Stat;

constexpr int kMemoryCheckIntervalMs = 50;

BenchmarkParams BenchmarkModel::DefaultParams() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSbenchmark_modelDTcc mht_0(mht_0_v, 201, "", "./tensorflow/lite/tools/benchmark/benchmark_model.cc", "BenchmarkModel::DefaultParams");

  BenchmarkParams params;
  params.AddParam("num_runs", BenchmarkParam::Create<int32_t>(50));
  params.AddParam("min_secs", BenchmarkParam::Create<float>(1.0f));
  params.AddParam("max_secs", BenchmarkParam::Create<float>(150.0f));
  params.AddParam("run_delay", BenchmarkParam::Create<float>(-1.0f));
  params.AddParam("run_frequency", BenchmarkParam::Create<float>(-1.0f));
  params.AddParam("num_threads", BenchmarkParam::Create<int32_t>(-1));
  params.AddParam("use_caching", BenchmarkParam::Create<bool>(false));
  params.AddParam("benchmark_name", BenchmarkParam::Create<std::string>(""));
  params.AddParam("output_prefix", BenchmarkParam::Create<std::string>(""));
  params.AddParam("warmup_runs", BenchmarkParam::Create<int32_t>(1));
  params.AddParam("warmup_min_secs", BenchmarkParam::Create<float>(0.5f));
  params.AddParam("verbose", BenchmarkParam::Create<bool>(false));
  params.AddParam("dry_run", BenchmarkParam::Create<bool>(false));
  params.AddParam("report_peak_memory_footprint",
                  BenchmarkParam::Create<bool>(false));
  params.AddParam("memory_footprint_check_interval_ms",
                  BenchmarkParam::Create<int32_t>(kMemoryCheckIntervalMs));
  return params;
}

BenchmarkModel::BenchmarkModel() : params_(DefaultParams()) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSbenchmark_modelDTcc mht_1(mht_1_v, 226, "", "./tensorflow/lite/tools/benchmark/benchmark_model.cc", "BenchmarkModel::BenchmarkModel");
}

void BenchmarkLoggingListener::OnBenchmarkEnd(const BenchmarkResults& results) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSbenchmark_modelDTcc mht_2(mht_2_v, 231, "", "./tensorflow/lite/tools/benchmark/benchmark_model.cc", "BenchmarkLoggingListener::OnBenchmarkEnd");

  auto inference_us = results.inference_time_us();
  auto init_us = results.startup_latency_us();
  auto warmup_us = results.warmup_time_us();
  auto init_mem_usage = results.init_mem_usage();
  auto overall_mem_usage = results.overall_mem_usage();
  TFLITE_LOG(INFO) << "Inference timings in us: "
                   << "Init: " << init_us << ", "
                   << "First inference: " << warmup_us.first() << ", "
                   << "Warmup (avg): " << warmup_us.avg() << ", "
                   << "Inference (avg): " << inference_us.avg();

  if (!init_mem_usage.IsSupported()) return;
  TFLITE_LOG(INFO)
      << "Note: as the benchmark tool itself affects memory footprint, the "
         "following is only APPROXIMATE to the actual memory footprint of the "
         "model at runtime. Take the information at your discretion.";
  TFLITE_LOG(INFO) << "Memory footprint delta from the start of the tool (MB): "
                   << "init=" << init_mem_usage.max_rss_kb / 1024.0
                   << " overall=" << overall_mem_usage.max_rss_kb / 1024.0;

  auto peak_mem_mb = results.peak_mem_mb();
  if (peak_mem_mb > 0) {
    TFLITE_LOG(INFO)
        << "Overall peak memory footprint (MB) via periodic monitoring: "
        << peak_mem_mb;
  }
}

std::vector<Flag> BenchmarkModel::GetFlags() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSbenchmark_modelDTcc mht_3(mht_3_v, 263, "", "./tensorflow/lite/tools/benchmark/benchmark_model.cc", "BenchmarkModel::GetFlags");

  return {
      CreateFlag<int32_t>(
          "num_runs", &params_,
          "expected number of runs, see also min_secs, max_secs"),
      CreateFlag<float>(
          "min_secs", &params_,
          "minimum number of seconds to rerun for, potentially making the "
          "actual number of runs to be greater than num_runs"),
      CreateFlag<float>(
          "max_secs", &params_,
          "maximum number of seconds to rerun for, potentially making the "
          "actual number of runs to be less than num_runs. Note if --max-secs "
          "is exceeded in the middle of a run, the benchmark will continue to "
          "the end of the run but will not start the next run."),
      CreateFlag<float>("run_delay", &params_, "delay between runs in seconds"),
      CreateFlag<float>(
          "run_frequency", &params_,
          "Execute at a fixed frequency, instead of a fixed delay."
          "Note if the targeted rate per second cannot be reached, the "
          "benchmark would start the next run immediately, trying its best to "
          "catch up. If set, this will override run_delay."),
      CreateFlag<int32_t>("num_threads", &params_, "number of threads"),
      CreateFlag<bool>(
          "use_caching", &params_,
          "Enable caching of prepacked weights matrices in matrix "
          "multiplication routines. Currently implies the use of the Ruy "
          "library."),
      CreateFlag<std::string>("benchmark_name", &params_, "benchmark name"),
      CreateFlag<std::string>("output_prefix", &params_,
                              "benchmark output prefix"),
      CreateFlag<int32_t>(
          "warmup_runs", &params_,
          "minimum number of runs performed on initialization, to "
          "allow performance characteristics to settle, see also "
          "warmup_min_secs"),
      CreateFlag<float>(
          "warmup_min_secs", &params_,
          "minimum number of seconds to rerun for, potentially making the "
          "actual number of warm-up runs to be greater than warmup_runs"),
      CreateFlag<bool>("verbose", &params_,
                       "Whether to log parameters whose values are not set. "
                       "By default, only log those parameters that are set by "
                       "parsing their values from the commandline flags."),
      CreateFlag<bool>("dry_run", &params_,
                       "Whether to run the tool just with simply loading the "
                       "model, allocating tensors etc. but without actually "
                       "invoking any op kernels."),
      CreateFlag<bool>(
          "report_peak_memory_footprint", &params_,
          "Report the peak memory footprint by periodically checking the "
          "memory footprint. Internally, a separate thread will be spawned for "
          "this periodic check. Therefore, the performance benchmark result "
          "could be affected."),
      CreateFlag<int32_t>("memory_footprint_check_interval_ms", &params_,
                          "The interval in millisecond between two consecutive "
                          "memory footprint checks. This is only used when "
                          "--report_peak_memory_footprint is set to true.")};
}

void BenchmarkModel::LogParams() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSbenchmark_modelDTcc mht_4(mht_4_v, 326, "", "./tensorflow/lite/tools/benchmark/benchmark_model.cc", "BenchmarkModel::LogParams");

  const bool verbose = params_.Get<bool>("verbose");
  TFLITE_LOG(INFO) << "Log parameter values verbosely: [" << verbose << "]";

  LOG_BENCHMARK_PARAM(int32_t, "num_runs", "Min num runs", verbose);
  LOG_BENCHMARK_PARAM(float, "min_secs", "Min runs duration (seconds)",
                      verbose);
  LOG_BENCHMARK_PARAM(float, "max_secs", "Max runs duration (seconds)",
                      verbose);
  LOG_BENCHMARK_PARAM(float, "run_delay", "Inter-run delay (seconds)", verbose);
  LOG_BENCHMARK_PARAM(float, "run_frequency",
                      "Number of prorated runs per second", verbose);
  LOG_BENCHMARK_PARAM(int32_t, "num_threads", "Num threads", verbose);
  LOG_BENCHMARK_PARAM(bool, "use_caching", "Use caching", verbose);
  LOG_BENCHMARK_PARAM(std::string, "benchmark_name", "Benchmark name", verbose);
  LOG_BENCHMARK_PARAM(std::string, "output_prefix", "Output prefix", verbose);
  LOG_BENCHMARK_PARAM(int32_t, "warmup_runs", "Min warmup runs", verbose);
  LOG_BENCHMARK_PARAM(float, "warmup_min_secs",
                      "Min warmup runs duration (seconds)", verbose);
  LOG_BENCHMARK_PARAM(bool, "dry_run", "Run w/o invoking kernels", verbose);
  LOG_BENCHMARK_PARAM(bool, "report_peak_memory_footprint",
                      "Report the peak memory footprint", verbose);
  LOG_BENCHMARK_PARAM(int32_t, "memory_footprint_check_interval_ms",
                      "Memory footprint check interval (ms)", verbose);
}

TfLiteStatus BenchmarkModel::PrepareInputData() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSbenchmark_modelDTcc mht_5(mht_5_v, 355, "", "./tensorflow/lite/tools/benchmark/benchmark_model.cc", "BenchmarkModel::PrepareInputData");
 return kTfLiteOk; }

TfLiteStatus BenchmarkModel::ResetInputsAndOutputs() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSbenchmark_modelDTcc mht_6(mht_6_v, 360, "", "./tensorflow/lite/tools/benchmark/benchmark_model.cc", "BenchmarkModel::ResetInputsAndOutputs");
 return kTfLiteOk; }

Stat<int64_t> BenchmarkModel::Run(int min_num_times, float min_secs,
                                  float max_secs, RunType run_type,
                                  TfLiteStatus* invoke_status) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSbenchmark_modelDTcc mht_7(mht_7_v, 367, "", "./tensorflow/lite/tools/benchmark/benchmark_model.cc", "BenchmarkModel::Run");

  Stat<int64_t> run_stats;
  TFLITE_LOG(INFO) << "Running benchmark for at least " << min_num_times
                   << " iterations and at least " << min_secs << " seconds but"
                   << " terminate if exceeding " << max_secs << " seconds.";
  int64_t now_us = profiling::time::NowMicros();
  int64_t min_finish_us = now_us + static_cast<int64_t>(min_secs * 1.e6f);
  int64_t max_finish_us = now_us + static_cast<int64_t>(max_secs * 1.e6f);

  *invoke_status = kTfLiteOk;
  float inter_run_sleep_time = params_.Get<float>("run_delay");
  auto run_frequency = params_.Get<float>("run_frequency");
  double manual_inter_run_gap = 1.0 / run_frequency;
  // float doesn't have sufficient precision for storing this number
  double next_run_finish_time = now_us * 1e-6 + manual_inter_run_gap;
  for (int run = 0; (run < min_num_times || now_us < min_finish_us) &&
                    now_us <= max_finish_us;
       run++) {
    ResetInputsAndOutputs();
    listeners_.OnSingleRunStart(run_type);
    int64_t start_us = profiling::time::NowMicros();
    TfLiteStatus status = RunImpl();
    int64_t end_us = profiling::time::NowMicros();
    listeners_.OnSingleRunEnd();

    run_stats.UpdateStat(end_us - start_us);
    if (run_frequency > 0) {
      inter_run_sleep_time =
          next_run_finish_time - profiling::time::NowMicros() * 1e-6;
      next_run_finish_time += manual_inter_run_gap;
    }
    // Note when "inter_run_sleep_time" is negative or 0.0,
    // the function will return immediately.
    util::SleepForSeconds(inter_run_sleep_time);
    now_us = profiling::time::NowMicros();

    if (status != kTfLiteOk) {
      *invoke_status = status;
    }
  }

  std::stringstream stream;
  run_stats.OutputToStream(&stream);
  TFLITE_LOG(INFO) << stream.str() << std::endl;

  return run_stats;
}

TfLiteStatus BenchmarkModel::ValidateParams() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSbenchmark_modelDTcc mht_8(mht_8_v, 418, "", "./tensorflow/lite/tools/benchmark/benchmark_model.cc", "BenchmarkModel::ValidateParams");

  if (params_.Get<bool>("report_peak_memory_footprint")) {
    const int32_t interval =
        params_.Get<int32_t>("memory_footprint_check_interval_ms");
    if (interval <= 0) {
      TFLITE_LOG(WARN) << "--memory_footprint_check_interval_ms is set to "
                       << interval
                       << " (ms), This value is invalid, and it will be set to "
                          "the default value "
                       << kMemoryCheckIntervalMs << " (ms).";
      params_.Set<int32_t>("memory_footprint_check_interval_ms",
                           kMemoryCheckIntervalMs);
    }
  }
  return kTfLiteOk;
}

TfLiteStatus BenchmarkModel::Run(int argc, char** argv) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSbenchmark_modelDTcc mht_9(mht_9_v, 438, "", "./tensorflow/lite/tools/benchmark/benchmark_model.cc", "BenchmarkModel::Run");

  TF_LITE_ENSURE_STATUS(ParseFlags(argc, argv));
  return Run();
}

TfLiteStatus BenchmarkModel::Run() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSbenchmark_modelDTcc mht_10(mht_10_v, 446, "", "./tensorflow/lite/tools/benchmark/benchmark_model.cc", "BenchmarkModel::Run");

  TF_LITE_ENSURE_STATUS(ValidateParams());

  LogParams();

  auto peak_memory_reporter = MayCreateMemoryUsageMonitor();
  if (peak_memory_reporter != nullptr) peak_memory_reporter->Start();
  const double model_size_mb = MayGetModelFileSize() / 1e6;
  const auto start_mem_usage = profiling::memory::GetMemoryUsage();
  int64_t initialization_start_us = profiling::time::NowMicros();
  TF_LITE_ENSURE_STATUS(Init());
  const auto init_end_mem_usage = profiling::memory::GetMemoryUsage();
  int64_t initialization_end_us = profiling::time::NowMicros();
  int64_t startup_latency_us = initialization_end_us - initialization_start_us;
  const auto init_mem_usage = init_end_mem_usage - start_mem_usage;

  if (model_size_mb > 0) {
    TFLITE_LOG(INFO) << "The input model file size (MB): " << model_size_mb;
  }
  TFLITE_LOG(INFO) << "Initialized session in " << startup_latency_us / 1e3
                   << "ms.";

  TF_LITE_ENSURE_STATUS(PrepareInputData());

  TfLiteStatus status = kTfLiteOk;
  uint64_t input_bytes = ComputeInputBytes();

  // Overwrite certain parameters when --dry_run=true is set.
  if (params_.Get<bool>("dry_run")) {
    params_.Set("warmup_runs", 0);
    params_.Set("warmup_min_secs", -1.0f);
    params_.Set("num_runs", 0);
    params_.Set("min_secs", -1.0f);
  }

  listeners_.OnBenchmarkStart(params_);
  Stat<int64_t> warmup_time_us =
      Run(params_.Get<int32_t>("warmup_runs"),
          params_.Get<float>("warmup_min_secs"), params_.Get<float>("max_secs"),
          WARMUP, &status);
  if (status != kTfLiteOk) {
    return status;
  }

  Stat<int64_t> inference_time_us =
      Run(params_.Get<int32_t>("num_runs"), params_.Get<float>("min_secs"),
          params_.Get<float>("max_secs"), REGULAR, &status);
  const auto overall_mem_usage =
      profiling::memory::GetMemoryUsage() - start_mem_usage;

  float peak_mem_mb = profiling::memory::MemoryUsageMonitor::kInvalidMemUsageMB;
  if (peak_memory_reporter != nullptr) {
    peak_memory_reporter->Stop();
    peak_mem_mb = peak_memory_reporter->GetPeakMemUsageInMB();
  }

  listeners_.OnBenchmarkEnd({model_size_mb, startup_latency_us, input_bytes,
                             warmup_time_us, inference_time_us, init_mem_usage,
                             overall_mem_usage, peak_mem_mb});
  return status;
}

TfLiteStatus BenchmarkModel::ParseFlags(int* argc, char** argv) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSbenchmark_modelDTcc mht_11(mht_11_v, 511, "", "./tensorflow/lite/tools/benchmark/benchmark_model.cc", "BenchmarkModel::ParseFlags");

  auto flag_list = GetFlags();
  const bool parse_result =
      Flags::Parse(argc, const_cast<const char**>(argv), flag_list);
  // "--help" flag is added in tools/delegates/default_execution_provider.cc. As
  // this is an optional dependency, we need to check whether "--help" exists or
  // not first.
  if (!parse_result ||
      (params_.HasParam("help") && params_.Get<bool>("help"))) {
    std::string usage = Flags::Usage(argv[0], flag_list);
    TFLITE_LOG(ERROR) << usage;
    // Returning kTfLiteError intentionally when "--help=true" is specified so
    // that the caller could check the return value to decide stopping the
    // execution.
    return kTfLiteError;
  }

  std::string unconsumed_args =
      Flags::ArgsToString(*argc, const_cast<const char**>(argv));
  if (!unconsumed_args.empty()) {
    TFLITE_LOG(WARN) << "Unconsumed cmdline flags: " << unconsumed_args;
  }

  return kTfLiteOk;
}

std::unique_ptr<profiling::memory::MemoryUsageMonitor>
BenchmarkModel::MayCreateMemoryUsageMonitor() const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSbenchmarkPSbenchmark_modelDTcc mht_12(mht_12_v, 541, "", "./tensorflow/lite/tools/benchmark/benchmark_model.cc", "BenchmarkModel::MayCreateMemoryUsageMonitor");

  if (!params_.Get<bool>("report_peak_memory_footprint")) return nullptr;

  return std::unique_ptr<profiling::memory::MemoryUsageMonitor>(
      new profiling::memory::MemoryUsageMonitor(
          params_.Get<int32_t>("memory_footprint_check_interval_ms")));
}

}  // namespace benchmark
}  // namespace tflite
