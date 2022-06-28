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
class MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSvalidatorDTcc {
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
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSvalidatorDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSvalidatorDTcc() {
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

/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/validator.h"

#include <errno.h>
#include <time.h>

#include <iostream>

#include "tensorflow/lite/core/api/profiler.h"
#include "tensorflow/lite/experimental/acceleration/configuration/configuration_generated.h"
#include "tensorflow/lite/experimental/acceleration/configuration/delegate_registry.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/call_register.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/decode_jpeg_register.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/status_codes.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/minimal_logging.h"

#ifndef TEMP_FAILURE_RETRY
#ifdef __ANDROID__
#error "TEMP_FAILURE_RETRY not set although on Android"
#else  // ! defined(__ANDROID__)
#define TEMP_FAILURE_RETRY(exp) exp
#endif  // defined(__ANDROID__)
#endif  // defined(TEMP_FAILURE_RETRY)

namespace tflite {
namespace acceleration {

Validator::Validator(const std::string& model_path,
                     const ComputeSettings* compute_settings)
    : model_path_(model_path),
      compute_settings_(compute_settings),
      delegate_(nullptr, [](TfLiteDelegate*) {}) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("model_path: \"" + model_path + "\"");
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSvalidatorDTcc mht_0(mht_0_v, 216, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/validator.cc", "Validator::Validator");
}

Validator::Validator(int model_fd, size_t model_offset, size_t model_size,
                     const ComputeSettings* compute_settings)
    :
#ifndef _WIN32
      model_fd_(dup(model_fd)),
#else   // _WIN32
      model_fd_(-1),
#endif  // !_WIN32
      model_offset_(model_offset),
      model_size_(model_size),
      compute_settings_(compute_settings),
      delegate_(nullptr, [](TfLiteDelegate*) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSvalidatorDTcc mht_1(mht_1_v, 232, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/validator.cc", "Validator::Validator");
}) {
}

Validator::~Validator() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSvalidatorDTcc mht_2(mht_2_v, 238, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/validator.cc", "Validator::~Validator");

#ifndef _WIN32
  if (model_fd_ >= 0) {
    close(model_fd_);
  }
#endif  // !_WIN32
}

namespace {
std::unique_ptr<tflite::delegates::DelegatePluginInterface> LoadDelegatePlugin(
    const std::string& name, const tflite::TFLiteSettings& tflite_settings) {
  return tflite::delegates::DelegatePluginRegistry::CreateByName(
      name + "Plugin", tflite_settings);
}

constexpr int64_t kMicrosInSecond = 1000 * 1000;
constexpr int64_t kNanosInMicro = 1000;

// CLOCK_BOOTTIME is what Android uses for elapsed time. Wallclock on mobile
// devices can jump due to user actions or network time sync.
int64_t ElapsedTimeMicros() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSvalidatorDTcc mht_3(mht_3_v, 261, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/validator.cc", "ElapsedTimeMicros");

  struct timespec ts;
#if defined(__ANDROID__)
  int err = clock_gettime(CLOCK_BOOTTIME, &ts);
#elif defined(_WIN32)
  int err = 1;
#else
  int err = clock_gettime(CLOCK_MONOTONIC, &ts);
#endif
  if (err) {
    return -1;
  }
  return ts.tv_sec * kMicrosInSecond + ts.tv_nsec / kNanosInMicro;
}

class ValidatorProfiler : public ::tflite::Profiler {
 public:
  struct EventData {
    std::string tag;
    int64_t start_time_us = -1;
    int64_t end_time_us = -1;
  };
  const std::vector<EventData>& events() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSvalidatorDTcc mht_4(mht_4_v, 286, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/validator.cc", "events");
 return events_; }
  uint32_t BeginEvent(const char* tag, EventType event_type,
                      int64_t event_metadata1,
                      int64_t event_metadata2) override {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("tag: \"" + (tag == nullptr ? std::string("nullptr") : std::string((char*)tag)) + "\"");
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSvalidatorDTcc mht_5(mht_5_v, 293, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/validator.cc", "BeginEvent");

    if (event_type != EventType::DEFAULT) {
      return 0;
    }
    events_.push_back({tag, ElapsedTimeMicros(), -1});
    return events_.size();
  }
  void EndEvent(uint32_t event_handle) override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSvalidatorDTcc mht_6(mht_6_v, 303, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/validator.cc", "EndEvent");

    if (event_handle == 0) {
      return;
    }
    events_[event_handle - 1].end_time_us = ElapsedTimeMicros();
  }
  uint32_t handle_ = 0;

 private:
  std::vector<EventData> events_;
};

}  // namespace

MinibenchmarkStatus Validator::CheckModel(bool load_only) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSvalidatorDTcc mht_7(mht_7_v, 320, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/validator.cc", "Validator::CheckModel");

  if (validation_entrypoint_) {
    // Already done.
    return kMinibenchmarkSuccess;
  }
  if (model_path_.empty() && model_fd_ <= 0) {
    return kMinibenchmarkPreconditionNotMet;
  }
  if (!model_path_.empty()) {
    model_ = FlatBufferModel::VerifyAndBuildFromFile(model_path_.c_str());
  } else if (MMAPAllocation::IsSupported()) {
    auto allocation = std::make_unique<MMAPAllocation>(
        model_fd_, model_offset_, model_size_, tflite::DefaultErrorReporter());
    if (!allocation->valid()) {
      return kMinibenchmarkModelReadFailed;
    }
    model_ =
        FlatBufferModel::VerifyAndBuildFromAllocation(std::move(allocation));
  } else {
    return kMinibenchmarkUnsupportedPlatform;
  }
  if (!model_) {
    return kMinibenchmarkModelBuildFailed;
  }
  if (load_only) {
    return kMinibenchmarkSuccess;
  }

  resolver_.AddCustom("validation/call",
                      ::tflite::acceleration::ops::Register_CALL(), 1);
  resolver_.AddCustom(
      "validation/decode_jpeg",
      ::tflite::acceleration::decode_jpeg_kernel::Register_DECODE_JPEG(), 1);

  tflite::InterpreterBuilder(*model_, resolver_)(&interpreter_);
  if (!interpreter_) {
    return kMinibenchmarkInterpreterBuilderFailed;
  }
  main_model_ = interpreter_->subgraph(0);
  for (int i = 0; i < interpreter_->subgraphs_size(); i++) {
    Subgraph* subgraph = interpreter_->subgraph(i);
    if (subgraph->GetName() == "VALIDATION:main") {
      validation_entrypoint_ = subgraph;
      break;
    }
  }
  if (!validation_entrypoint_) {
    return kMinibenchmarkValidationSubgraphNotFound;
  }
  if (validation_entrypoint_->inputs().size() <= 1) {
    return kMinibenchmarkValidationSubgraphHasTooFewInputs;
  }
  if (validation_entrypoint_->inputs().size() >
      validation_entrypoint_->outputs().size()) {
    return kMinibenchmarkValidationSubgraphHasTooFewOutputs;
  }

  // Check if we have validation data embedded or need to run CPU for it. If
  // the data is embedded, there is already an allocation for it from the model.
  TfLiteTensor* first_input_tensor =
      validation_entrypoint_->tensor(validation_entrypoint_->inputs()[0]);
  if (!first_input_tensor->allocation) {
    // Run on CPU.
    if (validation_entrypoint_->AllocateTensors() != kTfLiteOk) {
      return kMinibenchmarkAllocateTensorsFailed;
    }
    // Set initial golden outputs to 0 to avoid accessing uninitialized memory.
    // Last input is jpeg, skip.
    for (int i = 0; i < validation_entrypoint_->inputs().size() - 1; i++) {
      TfLiteTensor* input_tensor =
          validation_entrypoint_->tensor(validation_entrypoint_->inputs()[i]);
      memset(input_tensor->data.raw, 0, input_tensor->bytes);
    }
    TfLiteStatus status = validation_entrypoint_->Invoke();
    if (status != kTfLiteOk) {
      return kMinibenchmarkInvokeFailed;
    }
    // Copy CPU outputs as golden. Last input is jpeg image data, skip.
    for (int i = 0; i < validation_entrypoint_->inputs().size() - 1; i++) {
      TfLiteTensor* input_tensor =
          validation_entrypoint_->tensor(validation_entrypoint_->inputs()[i]);
      TfLiteTensor* golden_tensor =
          validation_entrypoint_->tensor(validation_entrypoint_->outputs()[i]);
      if (input_tensor->bytes != golden_tensor->bytes) {
        return kMinibenchmarkValidationSubgraphInputsDontMatchOutputs;
      }
      memcpy(input_tensor->data.raw, golden_tensor->data.raw,
             input_tensor->bytes);
    }
  }

  return kMinibenchmarkSuccess;
}

MinibenchmarkStatus Validator::LoadDelegate() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSvalidatorDTcc mht_8(mht_8_v, 417, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/validator.cc", "Validator::LoadDelegate");

  if (!interpreter_ || !compute_settings_) {
    return kMinibenchmarkPreconditionNotMet;
  }

  Delegate which_delegate = Delegate_NONE;
  if (compute_settings_->tflite_settings()) {
    which_delegate = compute_settings_->tflite_settings()->delegate();
  }
  if (which_delegate == Delegate_NNAPI) {
    delegate_plugin_ =
        LoadDelegatePlugin("Nnapi", *compute_settings_->tflite_settings());
  } else if (which_delegate == Delegate_GPU) {
    delegate_plugin_ =
        LoadDelegatePlugin("Gpu", *compute_settings_->tflite_settings());
  } else if (which_delegate == Delegate_XNNPACK) {
    delegate_plugin_ =
        LoadDelegatePlugin("XNNPack", *compute_settings_->tflite_settings());
  } else if (which_delegate == Delegate_NONE) {
    return kMinibenchmarkSuccess;
  } else {
    return kMinibenchmarkDelegateNotSupported;
  }
  if (!delegate_plugin_) {
    return kMinibenchmarkDelegatePluginNotFound;
  }
  delegate_ = delegate_plugin_->Create();

  return kMinibenchmarkSuccess;
}

MinibenchmarkStatus Validator::ApplyComputeSettings(int* delegate_error_out) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSvalidatorDTcc mht_9(mht_9_v, 451, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/validator.cc", "Validator::ApplyComputeSettings");

  if (!delegate_error_out) {
    return kMinibenchmarkPreconditionNotMet;
  }
  *delegate_error_out = 0;
  Delegate which_delegate = Delegate_NONE;
  if (compute_settings_->tflite_settings()) {
    which_delegate = compute_settings_->tflite_settings()->delegate();
  }
  std::string delegate;
  if (which_delegate == Delegate_NONE) {
    delegate = "CPU";
  } else if (which_delegate == Delegate_GPU) {
    delegate = "GPU";
  } else if (which_delegate == Delegate_NNAPI) {
    delegate = "NNAPI";
  } else if (which_delegate == Delegate_XNNPACK) {
    delegate = "XNNPACK";
  }

  TFLITE_LOG_PROD(TFLITE_LOG_INFO, "Running mini-benchmark on %s",
                  delegate.c_str());
  if (which_delegate == Delegate_NONE) {
    return kMinibenchmarkSuccess;
  } else if (!delegate_) {
    return kMinibenchmarkPreconditionNotMet;
  }
  ValidatorProfiler profiler;
  main_model_->SetProfiler(&profiler, 0);
  TfLiteStatus status = interpreter_->ModifyGraphWithDelegate(delegate_.get());
  main_model_->SetProfiler(nullptr, 0);
  for (const auto& e : profiler.events()) {
    if (e.tag == "ModifyGraphWithDelegate" && e.start_time_us != -1 &&
        e.end_time_us != -1) {
      compilation_time_us_ = e.end_time_us - e.start_time_us;
      TFLITE_LOG_PROD(TFLITE_LOG_INFO, "  Compilation took %d us",
                      static_cast<int>(compilation_time_us_));
      break;
    }
  }
  if (status == kTfLiteOk) {
    return kMinibenchmarkSuccess;
  } else {
    *delegate_error_out = delegate_plugin_->GetDelegateErrno(delegate_.get());
    return kMinibenchmarkModifyGraphWithDelegateFailed;
  }
}

MinibenchmarkStatus Validator::RunValidation(Results* results_out) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSvalidatorDTcc mht_10(mht_10_v, 502, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/validator.cc", "Validator::RunValidation");

  if (!results_out) {
    return kMinibenchmarkPreconditionNotMet;
  }
  MinibenchmarkStatus mb_status = CheckModel();
  if (mb_status != kMinibenchmarkSuccess) {
    return mb_status;
  }
  mb_status = LoadDelegate();
  if (mb_status != kMinibenchmarkSuccess) {
    return mb_status;
  }
  mb_status = ApplyComputeSettings(&(results_out->delegate_error));
  if (mb_status != kMinibenchmarkSuccess) {
    return mb_status;
  }
  if (validation_entrypoint_->AllocateTensors() != kTfLiteOk) {
    return kMinibenchmarkAllocateTensorsFailed;
  }
  ValidatorProfiler profiler;
  main_model_->SetProfiler(&profiler, 0);
  TfLiteStatus status = validation_entrypoint_->Invoke();
  main_model_->SetProfiler(nullptr, 0);
  if (status != kTfLiteOk) {
    return kMinibenchmarkInvokeFailed;
  }
  const std::string kMetricPrefix = "metrics/";
  const std::string kOk("ok");
  for (int i : validation_entrypoint_->outputs()) {
    TfLiteTensor* tensor = validation_entrypoint_->tensor(i);
    std::string name = tensor->name;
    if (name.find(kMetricPrefix) != 0) {  // NOLINT
      continue;
    }
    name = name.substr(kMetricPrefix.size());
    if (kOk == name) {
      results_out->ok = *(tensor->data.b);
    } else {
      std::vector<float> values;
      int count = 1;
      for (int j = 0; j < tensor->dims->size; j++) {
        count *= tensor->dims->data[j];
      }
      values.reserve(count);
      for (int j = 0; j < count; j++) {
        values.push_back(tensor->data.f[j]);
        TFLITE_LOG_PROD(TFLITE_LOG_INFO, "  %s %.4f", name.c_str(),
                        tensor->data.f[j]);
      }
      results_out->metrics[name] = values;
    }
  }
  TFLITE_LOG_PROD(TFLITE_LOG_INFO, "  accuracy: %s",
                  results_out->ok ? "ok" : "not ok");
  results_out->compilation_time_us = compilation_time_us_;
  for (const auto& e : profiler.events()) {
    if (e.tag == "Invoke" && e.start_time_us != -1 && e.end_time_us != -1) {
      results_out->execution_time_us.push_back(e.end_time_us - e.start_time_us);
      TFLITE_LOG_PROD(TFLITE_LOG_INFO, "  Inference took %d us",
                      static_cast<int>(e.end_time_us - e.start_time_us));
    }
  }
  return kMinibenchmarkSuccess;
}

int64_t Validator::BootTimeMicros() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSvalidatorDTcc mht_11(mht_11_v, 570, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/validator.cc", "Validator::BootTimeMicros");
 return ElapsedTimeMicros(); }
int64_t Validator::WallTimeMicros() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSvalidatorDTcc mht_12(mht_12_v, 574, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/validator.cc", "Validator::WallTimeMicros");

  struct timespec ts;
#ifndef _WIN32
  int err = clock_gettime(CLOCK_REALTIME, &ts);
#else   // _WIN32
  int err = 1;
#endif  // !_WIN32
  if (err) {
    return -1;
  }
  return ts.tv_sec * kMicrosInSecond + ts.tv_nsec / kNanosInMicro;
}

}  // namespace acceleration
}  // namespace tflite
