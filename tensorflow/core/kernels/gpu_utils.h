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

#ifndef TENSORFLOW_CORE_KERNELS_GPU_UTILS_H_
#define TENSORFLOW_CORE_KERNELS_GPU_UTILS_H_
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
class MHTracer_DTPStensorflowPScorePSkernelsPSgpu_utilsDTh {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSgpu_utilsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSgpu_utilsDTh() {
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


#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include <unordered_map>

#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/stream_executor/dnn.h"
#include "tensorflow/stream_executor/lazy_op_runner.h"

namespace stream_executor {
class RedzoneAllocator;
}  // namespace stream_executor

namespace tensorflow {

class NodeDef;
class AutotuneResult;

// Return whether the redzone check is disabled.
//
// Controlled by the TF_DISABLE_RZ_CHECK environment variable.
bool RedzoneCheckDisabled();

// Return an allocated buffer with redzones the size of `buffer`. Does
// *not* copy the contents of the `buffer` into the newly allocated buffer:
// assumes that buffer is a pure out-parameter.
//
// Returns `buffer` if RedzoneCheckDisabled() is true.
//
// On error, return `buffer`, and log an error message (once).
se::DeviceMemoryBase WrapRedzoneBestEffort(se::RedzoneAllocator* rz_allocator,
                                           se::DeviceMemoryBase buffer);

// Check the passed allocator for redzone violations.
// If violations have occurred, mark the corresponding autotune result
// as a failure.
void CheckRedzones(const se::RedzoneAllocator& rz_allocator,
                   AutotuneResult* autotune_result);

template <typename T>
inline se::DeviceMemory<T> AsDeviceMemory(const T* cuda_memory, uint64 size) {
  se::DeviceMemoryBase wrapped(const_cast<T*>(cuda_memory), size * sizeof(T));
  se::DeviceMemory<T> typed(wrapped);
  return typed;
}

// A helper class that looks up the best autotuned config from parameters.
// Due to the noisy nature of autotune, especially with multiple devices, it
// only accepts a config if its margin exceeds a threshold.
// For the same shape configs, if a new best config matches the previous best,
// they get promoted; otherwise, the winner gets demoted. This process stops
// when the winner's score exceeds the threshold.
// In a bad case when two configs are very close to each other and flips
// back and forth randomly, the expected number of experiments before autotune
// settles is O(threshold ^ 2). So we recommend that number of warmup runs
// for any benchmarks.
template <typename Parameters, typename Config>
class AutotuneMap {
 private:
  // Retrieves the hash code of Parameters class.
  struct Hasher {
    std::size_t operator()(const Parameters& parameter) const {
      return parameter.hash();
    }
  };

 public:
  bool Find(const Parameters& params, Config* config) const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSgpu_utilsDTh mht_0(mht_0_v, 262, "", "./tensorflow/core/kernels/gpu_utils.h", "Find");

    mutex_lock lock(mu_);
    auto iter = params_config_map_.find(params);
    if (iter == params_config_map_.end() ||
        (iter->second.score < min_score_threshold_ &&
         iter->second.count <= max_autotune_count_)) {
      return false;
    }
    *config = iter->second.config;
    return true;
  }
  void Insert(const Parameters& params, const Config& config) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSgpu_utilsDTh mht_1(mht_1_v, 276, "", "./tensorflow/core/kernels/gpu_utils.h", "Insert");

    mutex_lock lock(mu_);
    auto iter = params_config_map_.find(params);
    int new_score = 0;
    if (iter == params_config_map_.end()) {
      // Create a new entry if params is new.
      VLOG(1) << GetActionSummary("creates", params, config);
      params_config_map_.insert(
          std::make_pair(params, ValueType{config, 1, 1}));
      new_score = 1;
    } else if (iter->second.score < min_score_threshold_ &&
               iter->second.count <= max_autotune_count_) {
      DCHECK_GT(iter->second.score, 0);
      if (iter->second.config != config) {
        // If it is different from the current winner, demotes the winner.
        VLOG(1) << GetActionSummary("demotes", params, config);
        new_score = --iter->second.score;
        ++iter->second.count;
        if (new_score <= 0) {
          VLOG(1) << GetActionSummary("erases", params, config);
          params_config_map_.erase(iter);
        }
      } else {
        // If it is the same as the current winner, promotes the winner.
        VLOG(1) << GetActionSummary("promotes", params, config);
        new_score = ++iter->second.score;
        ++iter->second.count;
      }
    }
    if (new_score >= min_score_threshold_) {
      VLOG(1) << GetActionSummary("accepts", params, config);
    } else if (autotune_global_count_ >= max_autotune_global_count_) {
      // The autotuning exceeds the max iteration threshold and we accept the
      // the winner if it exists in the map, otherwise we accept the current
      // winner.
      auto winner = params_config_map_.find(params);
      if (winner == params_config_map_.end()) {
        VLOG(1) << GetActionSummary("creates", params, config);
        for (int i = 0; i < min_score_threshold_; ++i) {
          VLOG(1) << GetActionSummary("promotes", params, config);
        }
        params_config_map_.insert(
            std::make_pair(params, ValueType{config, min_score_threshold_, 1}));
      } else {
        int promotes_times = min_score_threshold_ - winner->second.score;
        for (int i = 0; i < promotes_times; ++i) {
          VLOG(1) << GetActionSummary("promotes", params, config);
        }
        winner->second.score = min_score_threshold_;
      }
      VLOG(1) << GetActionSummary("accepts", params, config);
    }
    autotune_global_count_++;
  }

  std::unordered_map<Parameters, Config, Hasher> GetMap() const {
    mutex_lock lock(mu_);
    std::unordered_map<Parameters, Config, Hasher> map;
    for (const auto& entry : params_config_map_) {
      map.insert(std::make_pair(entry.first, entry.second.config));
    }
    return map;
  }

  // Only for testing
  void ClearMap() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSgpu_utilsDTh mht_2(mht_2_v, 344, "", "./tensorflow/core/kernels/gpu_utils.h", "ClearMap");

    mutex_lock lock(mu_);
    params_config_map_.clear();
  }

 private:
  // Underlying data structure of values in the map.
  struct ValueType {
    Config config;
    int32 score;
    int32 count;
  };
  AutotuneMap(const std::string& name) : name_(name) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSgpu_utilsDTh mht_3(mht_3_v, 360, "", "./tensorflow/core/kernels/gpu_utils.h", "AutotuneMap");

    min_score_threshold_ = 1;
    int min_warmup_iterations = 10;
    const char* threshold_str = getenv("TF_AUTOTUNE_THRESHOLD");
    if (threshold_str != nullptr) {
      VLOG(1) << "TF_AUTOTUNE_THRESHOLD = " << threshold_str;
      strings::safe_strto32(threshold_str, &min_score_threshold_);
    }
    const char* min_warmup_iteration_str =
        getenv("TF_AUTOTUNE_MIN_WARMUP_ITERATIONS");
    if (min_warmup_iteration_str != nullptr) {
      strings::safe_strto32(min_warmup_iteration_str, &min_warmup_iterations);
    }
    min_score_threshold_ = std::max(min_score_threshold_, 1);
    max_autotune_count_ = std::max(
        5 * min_score_threshold_ * min_score_threshold_, min_warmup_iterations);
    max_autotune_global_count_ = 2 * max_autotune_count_;
    autotune_global_count_ = 0;
  }

  template <class Group, class Params, class Cfg>
  friend class AutotuneSingleton;

  std::string GetActionSummary(StringPiece action, const Parameters& params,
                               const Config& config) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSgpu_utilsDTh mht_4(mht_4_v, 387, "", "./tensorflow/core/kernels/gpu_utils.h", "GetActionSummary");

    return strings::Printf("autotune_map %s %s: %s -> (%s)", name_.c_str(),
                           string(action).c_str(), params.ToString().c_str(),
                           config.ToString().c_str());
  }

  mutable mutex mu_;

  std::unordered_map<Parameters, ValueType, Hasher> params_config_map_
      TF_GUARDED_BY(mu_);
  std::string name_;
  int32 min_score_threshold_;
  int32 max_autotune_count_;
  int32 max_autotune_global_count_;
  int32 autotune_global_count_;

  TF_DISALLOW_COPY_AND_ASSIGN(AutotuneMap);
};

// A Singleton helper that manages the global autotune results by groups.
// The caller specified arbitrary Group type that can distinguish between
// different autotune results, even if their Parameters and Configs are the
// same.
template <class Group, typename Parameters, typename Config>
class AutotuneSingleton {
 public:
  typedef AutotuneMap<Parameters, Config> AutotuneType;
  static AutotuneType* GetInstance() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSgpu_utilsDTh mht_5(mht_5_v, 417, "", "./tensorflow/core/kernels/gpu_utils.h", "GetInstance");

    static AutotuneType* instance = new AutotuneType(Group::name());
    return instance;
  }
};

// Logs convolution results to customized back-storage.
void LogConvAutotuneResults(se::dnn::ConvolutionKind kind,
                            se::dnn::DataType element_type,
                            se::DeviceMemoryBase input_buffer,
                            se::DeviceMemoryBase filter_buffer,
                            se::DeviceMemoryBase output_buffer,
                            const se::dnn::BatchDescriptor& input_desc,
                            const se::dnn::FilterDescriptor& filter_desc,
                            const se::dnn::BatchDescriptor& output_desc,
                            const se::dnn::ConvolutionDescriptor& conv_desc,
                            se::StreamExecutor* stream_exec,
                            absl::Span<const AutotuneResult> results);

// Logs fused convolution results to customized back-storage.
void LogFusedConvForwardAutotuneResults(
    se::dnn::DataType element_type, se::DeviceMemoryBase input_buffer,
    se::DeviceMemoryBase filter_buffer, se::DeviceMemoryBase output_buffer,
    se::DeviceMemoryBase bias_buffer, se::DeviceMemoryBase side_input_buffer,
    const se::dnn::BatchDescriptor& input_desc,
    const se::dnn::FilterDescriptor& filter_desc,
    const se::dnn::BatchDescriptor& output_desc,
    const se::dnn::ConvolutionDescriptor& conv_desc, double conv_scale,
    double side_value_scale, se::dnn::ActivationMode activation_mode,
    se::StreamExecutor* stream_exec, absl::Span<const AutotuneResult> results);

// Autotuning map entry for cuDNN-frontend-capable APIs.
//
// The longer-term intent is to remove the AlgorithmConfig variant and make this
// contain only the two LazyOpRunners, but for the time being ROCm is stuck on
// the legacy API and requires an AlgorithmConfig.
template <typename Op>
class AutotuneEntry {
 public:
  AutotuneEntry() : is_algorithm_config_(true) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSgpu_utilsDTh mht_6(mht_6_v, 459, "", "./tensorflow/core/kernels/gpu_utils.h", "AutotuneEntry");
}

  // Initialize with legacy-API AlgorithmConfig; used for the ROCm backend only.
  explicit AutotuneEntry(se::dnn::AlgorithmConfig config)
      : is_algorithm_config_(true), algorithm_config_(std::move(config)) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSgpu_utilsDTh mht_7(mht_7_v, 466, "", "./tensorflow/core/kernels/gpu_utils.h", "AutotuneEntry");
}

  AutotuneEntry(std::shared_ptr<se::dnn::LazyOpRunner<Op>> primary,
                std::shared_ptr<se::dnn::LazyOpRunner<Op>> no_scratch_fallback)
      : is_algorithm_config_(false),
        op_runners_{std::move(primary), std::move(no_scratch_fallback)} {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSgpu_utilsDTh mht_8(mht_8_v, 474, "", "./tensorflow/core/kernels/gpu_utils.h", "AutotuneEntry");
}

  // Initialize from config data, without pre-cached runners, such as when
  // loading AoT autotuning maps.
  AutotuneEntry(se::dnn::AlgorithmDesc primary,
                absl::optional<se::dnn::AlgorithmDesc> no_scratch_fallback)
      : AutotuneEntry(std::make_shared<se::dnn::LazyOpRunner<Op>>(primary),
                      no_scratch_fallback
                          ? std::make_shared<se::dnn::LazyOpRunner<Op>>(
                                *no_scratch_fallback)
                          : nullptr) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSgpu_utilsDTh mht_9(mht_9_v, 487, "", "./tensorflow/core/kernels/gpu_utils.h", "AutotuneEntry");
}

  // Initialize with pre-cached OpRunners, such as during autotuning.
  static StatusOr<AutotuneEntry> FromOpRunners(
      std::unique_ptr<const se::dnn::OpRunner<typename Op::Signature>> primary,
      std::unique_ptr<const se::dnn::OpRunner<typename Op::Signature>>
          no_cache_fallback) {
    TF_ASSIGN_OR_RETURN(
        auto primary_cache,
        se::dnn::LazyOpRunner<Op>::FromOpRunner(std::move(primary)));

    if (no_cache_fallback) {
      TF_ASSIGN_OR_RETURN(auto fallback_cache,
                          se::dnn::LazyOpRunner<Op>::FromOpRunner(
                              std::move(no_cache_fallback)));
      return AutotuneEntry(std::move(primary_cache), std::move(fallback_cache));

    } else {
      return AutotuneEntry(std::move(primary_cache), nullptr);
    }
  }

  struct OpRunners {
    OpRunners() = default;

    OpRunners(std::shared_ptr<se::dnn::LazyOpRunner<Op>> primary_,
              std::shared_ptr<se::dnn::LazyOpRunner<Op>> no_scratch_fallback_)
        : primary(std::move(primary_)),
          no_scratch_fallback(std::move(no_scratch_fallback_)) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSgpu_utilsDTh mht_10(mht_10_v, 518, "", "./tensorflow/core/kernels/gpu_utils.h", "OpRunners");
}

    // Null iff this 'OpRunners' is default-constructed as part of the
    // fake-variant in AutotuneEntry; users outside gpu_utils.h itself should
    // never see primary = nullptr.
    std::shared_ptr<se::dnn::LazyOpRunner<Op>> primary;
    std::shared_ptr<se::dnn::LazyOpRunner<Op>> no_scratch_fallback;  // Nullable

    bool operator==(const OpRunners& other) const {
      return *primary == *other.primary &&
             ((!no_scratch_fallback && !other.no_scratch_fallback) ||
              (no_scratch_fallback && other.no_scratch_fallback &&
               *no_scratch_fallback == *other.no_scratch_fallback));
    }
  };

  bool is_algorithm_config() const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSgpu_utilsDTh mht_11(mht_11_v, 537, "", "./tensorflow/core/kernels/gpu_utils.h", "is_algorithm_config");
 return is_algorithm_config_; }

  const se::dnn::AlgorithmConfig& GetAlgorithmConfig() const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSgpu_utilsDTh mht_12(mht_12_v, 542, "", "./tensorflow/core/kernels/gpu_utils.h", "GetAlgorithmConfig");

    DCHECK(is_algorithm_config_);
    return algorithm_config_;
  }

  const OpRunners& GetOpRunners() const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSgpu_utilsDTh mht_13(mht_13_v, 550, "", "./tensorflow/core/kernels/gpu_utils.h", "GetOpRunners");

    DCHECK(!is_algorithm_config_);
    return op_runners_;
  }

  // AutotuneMap needs to test equality to keep track of the number of times an
  // algorithm has won autotuning; for this purpose, we can use ToString to
  // determine whether runners are equal.
  bool operator==(const AutotuneEntry<Op>& other) const {
    if (is_algorithm_config_) {
      return other.is_algorithm_config_ &&
             algorithm_config_ == other.algorithm_config_;
    }

    return !other.is_algorithm_config_ && op_runners_ == other.op_runners_;
  }

  bool operator!=(const AutotuneEntry<Op>& other) const {
    return !(*this == other);
  }

  std::string ToString() const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSgpu_utilsDTh mht_14(mht_14_v, 574, "", "./tensorflow/core/kernels/gpu_utils.h", "ToString");

    if (is_algorithm_config_) {
      return algorithm_config_.ToString();
    }
    return absl::StrCat("{", op_runners_.primary->ToString(), ", ",
                        (op_runners_.no_scratch_fallback
                             ? op_runners_.no_scratch_fallback->ToString()
                             : "(op_runners have no fallback)"),
                        "}");
  }

 private:
  // NVCC is broken, so we can't use absl::variant here.  Just fake it with a
  // bool and both fields.
  bool is_algorithm_config_;
  se::dnn::AlgorithmConfig algorithm_config_;
  OpRunners op_runners_;
};

namespace internal {
StatusOr<std::tuple<int, int>> BestCudnnConvAlgorithmIndices(
    absl::Span<const AutotuneResult> results);
}  // namespace internal

// Returns the best algorithms for the config, one is the fastest, the other is
// other is fastest with 0 scratch space. Unsuccessful autotuning results are
// allowed and ignored.
StatusOr<se::dnn::AlgorithmConfig> BestCudnnConvAlgorithm(
    absl::Span<const AutotuneResult> results);

// Explicitly-instantiated with ConvOp and FusedConvOp.
//
// The definition can't be in the header because including .pb.h files in
// headers is forbidden.
template <typename Op>
StatusOr<AutotuneEntry<Op>> BestCudnnConvAlgorithm(
    absl::Span<const AutotuneResult> results,
    std::vector<
        std::unique_ptr<const se::dnn::OpRunner<typename Op::Signature>>>
        runners);

}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#endif  // TENSORFLOW_CORE_KERNELS_GPU_UTILS_H_
