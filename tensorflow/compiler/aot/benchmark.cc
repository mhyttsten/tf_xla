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
class MHTracer_DTPStensorflowPScompilerPSaotPSbenchmarkDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSaotPSbenchmarkDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSaotPSbenchmarkDTcc() {
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

// The purpose of the benchmark library is to support building an aot binary
// with minimal dependencies, to demonstrate small binary sizes.
//
// KEEP THE DEPENDENCIES MINIMAL.

#include "tensorflow/compiler/aot/benchmark.h"

#include <sys/time.h>

#include <algorithm>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace tfcompile {
namespace benchmark {

// Returns current wall time in micros.
//
// TODO(b/33546473): Refactor tensorflow::Env::NowMicros() so that we can re-use
// the implementation without pulling in all of the Env dependencies.
static double NowMicros() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSaotPSbenchmarkDTcc mht_0(mht_0_v, 210, "", "./tensorflow/compiler/aot/benchmark.cc", "NowMicros");

  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return static_cast<uint64>(tv.tv_sec) * 1000000 + tv.tv_usec;
}

void DumpStatsToStdout(const Stats& stats) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSaotPSbenchmarkDTcc mht_1(mht_1_v, 219, "", "./tensorflow/compiler/aot/benchmark.cc", "DumpStatsToStdout");

  // Compute stats.
  std::vector<int64_t> sorted_us(stats.per_iter_us);
  std::sort(sorted_us.begin(), sorted_us.end());
  const size_t count_us = sorted_us.size();
  double sum_us = 0;
  size_t count_us_trimmed = 0;
  double sum_us_trimmed = 0;
  size_t count_us_best = 0;
  double sum_us_best = 0;
  static constexpr float trim_ratio = 0.25;
  static constexpr float best_ratio = 0.1;
  const size_t count_trimmed = count_us * trim_ratio;
  const size_t count_best = count_us * best_ratio;
  for (size_t i = 0; i < sorted_us.size(); ++i) {
    const int64_t us = sorted_us[i];
    sum_us += us;
    if (i >= count_trimmed && i < count_us - count_trimmed) {
      sum_us_trimmed += us;
      ++count_us_trimmed;
    }
    if (i < count_best) {
      sum_us_best += us;
      ++count_us_best;
    }
  }
  // Prepare nicely-formatted data.
  const int kBufSize = 1000;
  char buf[kBufSize];
  snprintf(buf, kBufSize, "Mean with %2.0f%% trimmed:", trim_ratio * 100);
  std::string label_trimmed(buf);
  snprintf(buf, kBufSize, "Mean of %2.0f%% best:", best_ratio * 100);
  std::string label_best(buf);
  std::vector<std::pair<std::string, double>> groups = {
      {"Best:", sorted_us.front()},
      {"Worst:", sorted_us.back()},
      {"Median:", sorted_us[count_us / 2]},
      {"Mean:", sum_us / count_us},
      {std::move(label_trimmed), sum_us_trimmed / count_us_trimmed},
      {std::move(label_best), sum_us_best / count_us_best},
  };
  int max_label_size = 0;
  double max_us = 0;
  for (const auto& g : groups) {
    if (g.first.size() > max_label_size) {
      max_label_size = g.first.size();
    }
    if (g.second > max_us) {
      max_us = g.second;
    }
  }
  int max_digits = 1;
  while (max_us >= 10.0) {
    max_us /= 10.0;
    ++max_digits;
  }
  // Dump stats out.
  printf("Benchmark ran %zu iterations over %lld us\n", count_us,
         static_cast<long long>(stats.total_us));  // NOLINT
  for (const auto& g : groups) {
    printf("  %-*s %*.3f us\n", max_label_size, g.first.c_str(), max_digits + 4,
           g.second);
  }
}

void Benchmark(const Options& options, const BenchmarkFn& fn, Stats* stats) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSaotPSbenchmarkDTcc mht_2(mht_2_v, 287, "", "./tensorflow/compiler/aot/benchmark.cc", "Benchmark");

  // If neither max_seconds or max_iters is set, stop at kDefaultMicros.
  const int64_t max_us = (options.max_micros <= 0 && options.max_iters <= 0)
                             ? Options::kDefaultMicros
                             : options.max_micros;
  // NOLINTNEXTLINE
  printf("Running benchmark for %lld us\n", static_cast<long long>(max_us));
  const int64_t start_us = NowMicros();
  int64_t iters = 0;
  while (true) {
    const int64_t iter_start_us = NowMicros();
    fn();
    const int64_t end_us = NowMicros();
    // Collect stats and decide whether to stop.
    stats->per_iter_us.push_back(end_us - iter_start_us);
    const int64_t total_us = end_us - start_us;
    ++iters;
    if ((max_us > 0 && total_us >= max_us) ||
        (options.max_iters > 0 && iters >= options.max_iters)) {
      stats->total_us = total_us;
      break;
    }
  }
}

}  // namespace benchmark
}  // namespace tfcompile
}  // namespace tensorflow
