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
class MHTracer_DTPStensorflowPScorePSframeworkPSrun_handler_utilDTcc {
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
   MHTracer_DTPStensorflowPScorePSframeworkPSrun_handler_utilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPSrun_handler_utilDTcc() {
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

/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/run_handler_util.h"

#include <cmath>

#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/str_util.h"

namespace tensorflow {

double ParamFromEnvWithDefault(const char* var_name, double default_value) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("var_name: \"" + (var_name == nullptr ? std::string("nullptr") : std::string((char*)var_name)) + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSrun_handler_utilDTcc mht_0(mht_0_v, 196, "", "./tensorflow/core/framework/run_handler_util.cc", "ParamFromEnvWithDefault");

  const char* val = std::getenv(var_name);
  double num;
  return (val && strings::safe_strtod(val, &num)) ? num : default_value;
}

std::vector<double> ParamFromEnvWithDefault(const char* var_name,
                                            std::vector<double> default_value) {
  const char* val = std::getenv(var_name);
  if (!val) {
    return default_value;
  }
  std::vector<string> splits = str_util::Split(val, ",");
  std::vector<double> result;
  result.reserve(splits.size());
  for (auto& split : splits) {
    double num;
    if (strings::safe_strtod(split, &num)) {
      result.push_back(num);
    } else {
      LOG(ERROR) << "Wrong format for " << var_name << ". Use default value.";
      return default_value;
    }
  }
  return result;
}

std::vector<int> ParamFromEnvWithDefault(const char* var_name,
                                         std::vector<int> default_value) {
  const char* val = std::getenv(var_name);
  if (!val) {
    return default_value;
  }
  std::vector<string> splits = str_util::Split(val, ",");
  std::vector<int> result;
  result.reserve(splits.size());
  for (auto& split : splits) {
    int num;
    if (strings::safe_strto32(split, &num)) {
      result.push_back(num);
    } else {
      LOG(ERROR) << "Wrong format for " << var_name << ". Use default value.";
      return default_value;
    }
  }
  return result;
}

bool ParamFromEnvBoolWithDefault(const char* var_name, bool default_value) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("var_name: \"" + (var_name == nullptr ? std::string("nullptr") : std::string((char*)var_name)) + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSrun_handler_utilDTcc mht_1(mht_1_v, 248, "", "./tensorflow/core/framework/run_handler_util.cc", "ParamFromEnvBoolWithDefault");

  const char* val = std::getenv(var_name);
  return (val) ? str_util::Lowercase(val) == "true" : default_value;
}

void ComputeInterOpSchedulingRanges(int num_active_requests, int num_threads,
                                    int min_threads_per_request,
                                    std::vector<std::uint_fast32_t>* start_vec,
                                    std::vector<std::uint_fast32_t>* end_vec) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSrun_handler_utilDTcc mht_2(mht_2_v, 259, "", "./tensorflow/core/framework/run_handler_util.cc", "ComputeInterOpSchedulingRanges");

  // Each request is expected to have weight W[i] = num_active_requests - i.
  // Therefore, total_weight = sum of all request weights.
  float total_weight = 0.5f * num_active_requests * (num_active_requests + 1);
  float demand_factor = static_cast<float>(num_threads) / total_weight;
  float last_cumulative_weight = 0.0;
  min_threads_per_request = std::max(1, min_threads_per_request);
  for (int i = 0; i != num_active_requests; i++) {
    float cumulative_weight =
        static_cast<float>(i + 1) *
        (num_active_requests - static_cast<float>(i) * 0.5f);
    float weight = cumulative_weight - last_cumulative_weight;
    // Quantize thread_demand by rounding up, and also satisfying
    // `min_threads_per_request` constraint.
    // Note: We subtract a small epsilon (0.00001) to prevent ceil(..) from
    // rounding weights like 4.0 to 5.
    int demand = std::max(
        min_threads_per_request,
        static_cast<int>(std::ceil(weight * demand_factor - 0.00001f)));
    // For the quantized range [start, end); compute the floor of real start,
    // and expand downwards from there with length `demand` and adjust for
    // boundary conditions.
    int start = last_cumulative_weight * demand_factor;
    int end = std::min(num_threads, start + demand);
    start = std::max(0, std::min(start, end - demand));
    start_vec->at(i) = start;
    end_vec->at(i) = end;
    last_cumulative_weight = cumulative_weight;
  }
}

void ComputeInterOpStealingRanges(int num_threads, int min_threads_per_domain,
                                  std::vector<std::uint_fast32_t>* start_vec,
                                  std::vector<std::uint_fast32_t>* end_vec) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSrun_handler_utilDTcc mht_3(mht_3_v, 295, "", "./tensorflow/core/framework/run_handler_util.cc", "ComputeInterOpStealingRanges");

  int steal_domain_size = std::min(min_threads_per_domain, num_threads);
  unsigned steal_start = 0, steal_end = steal_domain_size;
  for (int i = 0; i < num_threads; ++i) {
    if (i >= steal_end) {
      if (steal_end + steal_domain_size < num_threads) {
        steal_start = steal_end;
        steal_end += steal_domain_size;
      } else {
        steal_end = num_threads;
        steal_start = steal_end - steal_domain_size;
      }
    }
    start_vec->at(i) = steal_start;
    end_vec->at(i) = steal_end;
  }
}

std::vector<int> ChooseRequestsWithExponentialDistribution(
    int num_active_requests, int num_threads) {
  // Fraction of the total threads that will be evenly distributed across
  // requests. The rest of threads will be exponentially distributed across
  // requests.
  static const double kCapacityFractionForEvenDistribution =
      ParamFromEnvWithDefault("TF_RUN_HANDLER_EXP_DIST_EVEN_FRACTION", 0.5);

  // For the threads that will be exponentially distributed across requests,
  // a request will get allocated (kPowerBase - 1) times as much threads as
  // threads allocated to all requests that arrive after it. For example, the
  // oldest request will be allocated num_threads*(kPowerBase-1)/kPowerBase
  // number of threads.
  static const double kPowerBase =
      ParamFromEnvWithDefault("TF_RUN_HANDLER_EXP_DIST_POWER_BASE", 2.0);

  static const int kMinEvenThreadsFromEnv = static_cast<int>(
      ParamFromEnvWithDefault("TF_RUN_HANDLER_EXP_DIST_MIN_EVEN_THREADS", 1));
  static const int kMaxEvenThreadsFromEnv = static_cast<int>(
      ParamFromEnvWithDefault("TF_RUN_HANDLER_EXP_DIST_MAX_EVEN_THREADS", 3));

  std::vector<int> request_idx_list;
  request_idx_list.resize(num_threads);
  // Each request gets at least this number of threads that steal from it first.
  int min_threads_per_request =
      num_threads * kCapacityFractionForEvenDistribution / num_active_requests;
  min_threads_per_request =
      std::max(kMinEvenThreadsFromEnv, min_threads_per_request);
  min_threads_per_request =
      std::min(kMaxEvenThreadsFromEnv, min_threads_per_request);

  int num_remaining_threads =
      std::max(0, num_threads - num_active_requests * min_threads_per_request);
  int request_idx = -1;
  int num_threads_next_request = 0;

  for (int tid = 0; tid < num_threads; ++tid) {
    if (num_threads_next_request <= 0) {
      request_idx = std::min(num_active_requests - 1, request_idx + 1);
      int num_extra_threads_next_request =
          std::ceil(num_remaining_threads * (kPowerBase - 1.0) / kPowerBase);
      num_remaining_threads -= num_extra_threads_next_request;
      num_threads_next_request =
          num_extra_threads_next_request + min_threads_per_request;
    }
    num_threads_next_request--;
    request_idx_list[tid] = request_idx;
  }
  return request_idx_list;
}

}  // namespace tensorflow
