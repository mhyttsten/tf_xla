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
class MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSmini_benchmark_implementationDTcc {
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
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSmini_benchmark_implementationDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSmini_benchmark_implementationDTcc() {
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
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/experimental/acceleration/configuration/configuration_generated.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/fb_storage.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/mini_benchmark.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/status_codes.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/validator_runner.h"
#include "tensorflow/lite/minimal_logging.h"
#include "tensorflow/lite/nnapi/sl/include/SupportLibrary.h"

namespace tflite {
namespace acceleration {

using ::tflite::nnapi::NnApiSupportLibrary;

// This class is used to store the results of a GetBestAcceleration and
// information on the events used to take the decision.
// The class is thread-compatible as MiniBenchmark.
class MemoizedBestAccelerationSelector {
 public:
  // Note 'settings' has to outlast this instance.
  MemoizedBestAccelerationSelector(const MinibenchmarkSettings& settings,
                                   const std::string model_namespace,
                                   const std::string& model_id,
                                   const std::string& storage_path)
      : settings_(settings),
        model_namespace_(model_namespace),
        model_id_(model_id),
        number_of_events_in_memoized_call_(0),
        memoised_result_(nullptr),
        storage_(storage_path, tflite::DefaultErrorReporter()) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("model_namespace: \"" + model_namespace + "\"");
   mht_0_v.push_back("model_id: \"" + model_id + "\"");
   mht_0_v.push_back("storage_path: \"" + storage_path + "\"");
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSmini_benchmark_implementationDTcc mht_0(mht_0_v, 224, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/mini_benchmark_implementation.cc", "MemoizedBestAccelerationSelector");

    storage_.Read();

    TFLITE_LOG_PROD_ONCE(TFLITE_LOG_INFO,
                         "Initializing BestAccelerationSelector for model (%s, "
                         "%s) and storage path %s. Storage has %zu events.\n",
                         model_namespace_.c_str(), model_id_.c_str(),
                         storage_path.c_str(), storage_.Count());

    // Read saved status from storage_.
    for (int i = storage_.Count() - 1; i >= 0; i--) {
      const MiniBenchmarkEvent* event = storage_.Get(i);
      if (event == nullptr || event->best_acceleration_decision() == nullptr) {
        continue;
      }

      const auto* best_decision = event->best_acceleration_decision();
      const TFLiteSettings* acceleration_settings =
          CreateAccelerationFromBenchmark(
              best_decision->min_latency_event(),
              best_decision->min_inference_time_us());
      Memoize(acceleration_settings, best_decision->number_of_source_events());
      TFLITE_LOG_PROD_ONCE(
          TFLITE_LOG_INFO,
          "Rebuilding memoised best acceleration from storage. It has been "
          "generated based on %d events.\n",
          number_of_events_in_memoized_call_);
      break;
    }
  }

  ComputeSettingsT GetBestAcceleration(
      const std::vector<const BenchmarkEvent*>& events) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSmini_benchmark_implementationDTcc mht_1(mht_1_v, 259, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/mini_benchmark_implementation.cc", "GetBestAcceleration");

    ComputeSettingsT result;
    if (events.empty()) {
      TFLITE_LOG_PROD_ONCE(
          TFLITE_LOG_INFO,
          "No completed events are available to calculate best "
          "acceleration result for model (%s, %s).\n",
          model_namespace_.c_str(), model_id_.c_str());
      return result;
    }
    if (memoised_result_ != nullptr &&
        (events.size() == number_of_events_in_memoized_call_)) {
      TFLITE_LOG_PROD_ONCE(TFLITE_LOG_INFO,
                           "Returning memoized best acceleration result for "
                           "model (%s, %s) based on %d events.\n",
                           model_namespace_.c_str(), model_id_.c_str(),
                           number_of_events_in_memoized_call_);
      memoised_result_->UnPackTo(&result);
      return result;
    }

    TFLITE_LOG_PROD_ONCE(TFLITE_LOG_INFO,
                         "Calculating best acceleration result for model (%s, "
                         "%s) based on %zu events.\n",
                         model_namespace_.c_str(), model_id_.c_str(),
                         events.size());

    int64_t min_latency = -1;
    auto min_latency_event = FindMinLatencyEvent(events, min_latency);
    if (min_latency_event == nullptr) {
      // We won't memoise a decision on no events.
      return result;
    }

    const TFLiteSettings* acceleration_settings =
        CreateAccelerationFromBenchmark(min_latency_event, min_latency);
    Memoize(acceleration_settings, static_cast<int>(events.size()));
    StoreBestAcceleration(min_latency_event, min_latency);
    memoised_result_->UnPackTo(&result);
    return result;
  }

  // Note that events used here are all benchmark-run-ok events, i.e. those
  // indicating the acceleration configuration has run successfully and produced
  // correct results.
  int NumEventsUsedInBestAcceleration() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSmini_benchmark_implementationDTcc mht_2(mht_2_v, 307, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/mini_benchmark_implementation.cc", "NumEventsUsedInBestAcceleration");

    return number_of_events_in_memoized_call_;
  }

 private:
  void Memoize(const TFLiteSettings* acceleration_settings, int num_events) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSmini_benchmark_implementationDTcc mht_3(mht_3_v, 315, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/mini_benchmark_implementation.cc", "Memoize");

    number_of_events_in_memoized_call_ = num_events;
    memoised_result_buffer_.Clear();
    flatbuffers::Offset<tflite::TFLiteSettings> tflite_setting_offset = 0;
    if (acceleration_settings != nullptr) {
      TFLiteSettingsT copy;
      acceleration_settings->UnPackTo(&copy);
      tflite_setting_offset =
          TFLiteSettings::Pack(memoised_result_buffer_, &copy);
    }
    auto compute_settings = CreateComputeSettings(
        memoised_result_buffer_, tflite::ExecutionPreference_ANY,
        tflite_setting_offset,
        memoised_result_buffer_.CreateString(model_namespace_),
        memoised_result_buffer_.CreateString(model_id_));
    memoised_result_buffer_.Finish(compute_settings);
    memoised_result_ = flatbuffers::GetRoot<ComputeSettings>(
        memoised_result_buffer_.GetBufferPointer());
  }

  // Stores the BestAcceleration to persistent storage to handle restart.
  void StoreBestAcceleration(const BenchmarkEvent* min_latency_event,
                             int64_t min_latency) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSmini_benchmark_implementationDTcc mht_4(mht_4_v, 340, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/mini_benchmark_implementation.cc", "StoreBestAcceleration");

    flatbuffers::FlatBufferBuilder fbb;
    tflite::BenchmarkEventT min_latency_ev_copy;
    min_latency_event->UnPackTo(&min_latency_ev_copy);
    auto best_acceleration_decision = tflite::CreateBestAccelerationDecision(
        fbb, number_of_events_in_memoized_call_,
        CreateBenchmarkEvent(fbb, &min_latency_ev_copy), min_latency);
    storage_.Append(
        &fbb, CreateMiniBenchmarkEvent(fbb, /*is_log_flushing_event=*/false,
                                       best_acceleration_decision));
  }

  const BenchmarkEvent* FindMinLatencyEvent(
      const std::vector<const BenchmarkEvent*>& events, int64_t& min_latency) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSmini_benchmark_implementationDTcc mht_5(mht_5_v, 356, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/mini_benchmark_implementation.cc", "FindMinLatencyEvent");

    const BenchmarkEvent* min_latency_event = nullptr;
    min_latency = -1;
    for (const BenchmarkEvent* ev : events) {
      for (int i = 0; i < ev->result()->inference_time_us()->size(); i++) {
        int64_t latency = ev->result()->inference_time_us()->Get(i);
        if (latency < 0) {
          continue;
        }
        if ((min_latency < 0) || (latency < min_latency)) {
          min_latency = latency;
          min_latency_event = ev;
        }
      }
    }

    return min_latency_event;
  }

  std::string GetDelegateFromBenchmarkEvent(const BenchmarkEvent* event) const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSmini_benchmark_implementationDTcc mht_6(mht_6_v, 378, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/mini_benchmark_implementation.cc", "GetDelegateFromBenchmarkEvent");

    if (event->tflite_settings()->delegate() == Delegate_NNAPI) {
      return "NNAPI";
    }
    if (event->tflite_settings()->delegate() == Delegate_GPU) {
      return "GPU";
    }
    if (event->tflite_settings()->delegate() == Delegate_XNNPACK) {
      return "XNNPACK";
    }

    return "CPU";
  }

  // Returns the acceleration settings in the list of settings_to_test
  // corresponding to the tflite settings of the provided min_latency_event.
  // Returns nullptr if no matching settings is found.
  const TFLiteSettings* FindAccelerationToTestFromMiniBenchmarkEvent(
      const BenchmarkEvent* min_latency_event) const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSmini_benchmark_implementationDTcc mht_7(mht_7_v, 399, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/mini_benchmark_implementation.cc", "FindAccelerationToTestFromMiniBenchmarkEvent");

    TFLiteSettingsT event_tflite_settings;
    min_latency_event->tflite_settings()->UnPackTo(&event_tflite_settings);
    for (int i = 0; i < settings_.settings_to_test()->size(); i++) {
      auto one_setting = settings_.settings_to_test()->Get(i);
      TFLiteSettingsT to_test_tflite_settings;
      one_setting->UnPackTo(&to_test_tflite_settings);
      if (to_test_tflite_settings == event_tflite_settings) {
        return one_setting;
      }
    }
    TFLITE_LOG_PROD_ONCE(
        TFLITE_LOG_WARNING,
        "Couldn't find  setting to test matching the best latency event for "
        "model %s, returning no acceleration.\n",
        model_id_.c_str());
    return nullptr;
  }

  const TFLiteSettings* CreateAccelerationFromBenchmark(
      const BenchmarkEvent* min_latency_event, int64_t min_latency) const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSmini_benchmark_implementationDTcc mht_8(mht_8_v, 422, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/mini_benchmark_implementation.cc", "CreateAccelerationFromBenchmark");

    std::string delegate = GetDelegateFromBenchmarkEvent(min_latency_event);
    TFLITE_LOG_PROD_ONCE(
        TFLITE_LOG_INFO,
        "Found best latency for %s with delegate %s ( %ld us).\n",
        model_id_.c_str(), delegate.c_str(), min_latency);

    if (min_latency_event->tflite_settings()->delegate() == Delegate_NONE) {
      TFLITE_LOG_PROD_ONCE(
          TFLITE_LOG_INFO,
          "Best latency for %s is without a delegate, not overring defaults.\n",
          model_id_.c_str());
      return nullptr;
    }

    return FindAccelerationToTestFromMiniBenchmarkEvent(min_latency_event);
  }

  const MinibenchmarkSettings& settings_;
  std::string model_namespace_;
  std::string model_id_;
  // The number of events we are basing our best acceleration decision on.
  // We are assuming that the events passed to this object will only increase
  // and never change.
  int number_of_events_in_memoized_call_ = 0;
  flatbuffers::FlatBufferBuilder memoised_result_buffer_;
  // Pointer to the 'memoised_result_buffer_' for convenience.
  const ComputeSettings* memoised_result_;

  FlatbufferStorage<MiniBenchmarkEvent> storage_;
};

class MiniBenchmarkImpl : public MiniBenchmark {
 public:
  MiniBenchmarkImpl(const MinibenchmarkSettings& settings,
                    const std::string& model_namespace,
                    const std::string& model_id)
      : model_namespace_(model_namespace), model_id_(model_id) {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("model_namespace: \"" + model_namespace + "\"");
   mht_9_v.push_back("model_id: \"" + model_id + "\"");
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSmini_benchmark_implementationDTcc mht_9(mht_9_v, 464, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/mini_benchmark_implementation.cc", "MiniBenchmarkImpl");

    // Keep a copy of the passed in 'settings' to simplify the memory
    // management. Otherwise, the 'settings' has to outlast this instance.
    MinibenchmarkSettingsT copy;
    settings.UnPackTo(&copy);
    settings_buffer_.Finish(
        MinibenchmarkSettings::Pack(settings_buffer_, &copy));
    settings_ = flatbuffers::GetRoot<MinibenchmarkSettings>(
        settings_buffer_.GetBufferPointer());

    is_enabled_ = BenchmarkIsEnabled();
    if (!is_enabled_) return;

    is_cpu_validation_specified_ = false;
    total_validation_tests_ = settings_->settings_to_test()->size();
    for (int i = 0; i < settings_->settings_to_test()->size(); i++) {
      auto one_setting = settings_->settings_to_test()->Get(i);
      if (one_setting->delegate() == Delegate_NONE) {
        is_cpu_validation_specified_ = true;
      }
    }
    // By default, will always add a cpu testing even if it's not requested.
    if (total_validation_tests_ != 0 && !is_cpu_validation_specified_) {
      total_validation_tests_ += 1;
    }

    const std::string local_event_fp = LocalEventStorageFileName(settings);
    best_acceleration_selector_.reset(new MemoizedBestAccelerationSelector(
        *settings_, model_namespace, model_id, local_event_fp));

    storage_.reset(new FlatbufferStorage<MiniBenchmarkEvent>(
        local_event_fp, tflite::DefaultErrorReporter()));
    storage_->Read();
    for (int i = storage_->Count() - 1; i >= 0; i--) {
      auto* event = storage_->Get(i);
      if (event != nullptr && event->initialization_failure()) {
        initialization_failure_logged_ = true;
        break;
      }
    }
  }

  ComputeSettingsT GetBestAcceleration() override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSmini_benchmark_implementationDTcc mht_10(mht_10_v, 509, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/mini_benchmark_implementation.cc", "GetBestAcceleration");

    if (!is_enabled_) return ComputeSettingsT();
    CreateValidatorIfNececessary();
    if (!validator_initialized_) return ComputeSettingsT();

    std::vector<const BenchmarkEvent*> events =
        validator_->GetSuccessfulResults();
    TFLITE_LOG_PROD_ONCE(TFLITE_LOG_INFO,
                         "Got %zu successful minibenchmark events for %s.\n",
                         events.size(), model_id_.c_str());

    return best_acceleration_selector_->GetBestAcceleration(events);
  }

  void TriggerMiniBenchmark() override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSmini_benchmark_implementationDTcc mht_11(mht_11_v, 526, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/mini_benchmark_implementation.cc", "TriggerMiniBenchmark");

    if (!is_enabled_) return;
    CreateValidatorIfNececessary();
    if (!validator_initialized_) return;

    std::vector<const TFLiteSettings*> settings;
    for (int i = 0; i < settings_->settings_to_test()->size(); i++) {
      auto one_setting = settings_->settings_to_test()->Get(i);
      settings.push_back(one_setting);
    }
    // By default, always add a cpu testing even if it's not requested.
    flatbuffers::FlatBufferBuilder cpu_fbb;
    if (!settings.empty() && !is_cpu_validation_specified_) {
      cpu_fbb.Finish(CreateTFLiteSettings(cpu_fbb));
      settings.push_back(
          flatbuffers::GetRoot<TFLiteSettings>(cpu_fbb.GetBufferPointer()));
    }
    int triggered = validator_->TriggerMissingValidation(settings);
    if (triggered > 0) {
      TFLITE_LOG_PROD(TFLITE_LOG_INFO,
                      "Triggered mini benchmark for %s with %d possibilities "
                      "(including CPU).\n",
                      model_id_.c_str(), triggered);
    }
  }

  std::vector<MiniBenchmarkEventT> MarkAndGetEventsToLog() override {
    if (!is_enabled_) return {};

    std::vector<tflite::MiniBenchmarkEventT> result;

    // Internal MiniBenchmarkEvents.
    storage_->Read();
    int storage_size_pre_flush = storage_->Count();
    if (storage_size_pre_flush != 0) {
      // Marking current events as read. Adding a MiniBenchmarkEvent
      // with the is_log_flushing_event flag set to true.
      flatbuffers::FlatBufferBuilder fbb;
      storage_->Append(&fbb, ::tflite::CreateMiniBenchmarkEvent(
                                 fbb, /*is_log_flushing_event=*/true));

      storage_->Read();
      for (int i = storage_size_pre_flush - 1; i >= 0; i--) {
        const MiniBenchmarkEvent* event = storage_->Get(i);
        if (event == nullptr || event->is_log_flushing_event()) {
          break;
        }
        tflite::MiniBenchmarkEventT mini_benchmark_event;
        event->UnPackTo(&mini_benchmark_event);
        result.push_back(std::move(mini_benchmark_event));
      }
    }
    std::reverse(result.begin(), result.end());

    // BenchmarkEvents from the validaton runner.
    std::vector<const BenchmarkEvent*> event_ptrs =
        validator_->GetAndFlushEventsToLog(event_timeout_us_);
    for (const auto* event_ptr : event_ptrs) {
      tflite::MiniBenchmarkEventT mini_benchmark_event;
      mini_benchmark_event.benchmark_event =
          std::unique_ptr<BenchmarkEventT>(event_ptr->UnPack());
      result.push_back(std::move(mini_benchmark_event));
    }

    return result;
  }

  void SetEventTimeoutForTesting(int64_t timeout_us) override {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSmini_benchmark_implementationDTcc mht_12(mht_12_v, 596, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/mini_benchmark_implementation.cc", "SetEventTimeoutForTesting");

    event_timeout_us_ = (timeout_us == -1)
                            ? ValidatorRunner::kDefaultEventTimeoutUs
                            : timeout_us;
  }

  int NumRemainingAccelerationTests() override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSmini_benchmark_implementationDTcc mht_13(mht_13_v, 605, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/mini_benchmark_implementation.cc", "NumRemainingAccelerationTests");

    // We return -1 when the overall mini-benchmark-related setup isn't properly
    // initialized.
    if (!is_enabled_ || !validator_initialized_) return -1;

    // We first check if there has been a previous execution of the runner
    // already using some of the validation results to decide the best
    // acceleration.
    const int to_complete_tests =
        total_validation_tests_ -
        best_acceleration_selector_->NumEventsUsedInBestAcceleration();
    // No remaining tests, skip reading the validation events log and just
    // return.
    if (to_complete_tests == 0) return 0;

    // Read the whole validation events log to find the number of completed
    // runs.
    return total_validation_tests_ - validator_->GetNumCompletedResults();
  }

 private:
  static std::string LocalEventStorageFileName(
      const MinibenchmarkSettings& settings) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSmini_benchmark_implementationDTcc mht_14(mht_14_v, 630, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/mini_benchmark_implementation.cc", "LocalEventStorageFileName");

    if (settings.storage_paths() == nullptr ||
        settings.storage_paths()->storage_file_path() == nullptr) {
      return "mini_benchmark.default.extra.fb";
    }
    return settings.storage_paths()->storage_file_path()->str() + ".extra.fb";
  }

  bool BenchmarkIsEnabled() const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSmini_benchmark_implementationDTcc mht_15(mht_15_v, 641, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/mini_benchmark_implementation.cc", "BenchmarkIsEnabled");

    if (settings_->settings_to_test() == nullptr) return false;
    if (settings_->settings_to_test()->size() <= 0) return false;

    const auto* storage_paths = settings_->storage_paths();
    if (storage_paths == nullptr) return false;
    if (storage_paths->storage_file_path() == nullptr ||
        storage_paths->storage_file_path()->str().empty()) {
      TFLITE_LOG_PROD_ONCE(
          TFLITE_LOG_ERROR,
          "Minibenchmark requested for %s but storage_file_path not set.\n",
          model_id_.c_str());
      return false;
    } else if (storage_paths->data_directory_path() == nullptr ||
               storage_paths->data_directory_path()->str().empty()) {
      TFLITE_LOG_PROD_ONCE(
          TFLITE_LOG_ERROR,
          "Minibenchmark requested for %s but data_directory_path not set.\n",
          model_id_.c_str());
      return false;
    }

    const auto* model_file = settings_->model_file();
    if (model_file == nullptr) return false;
    if (model_file->fd() <= 0 && (model_file->filename() == nullptr ||
                                  model_file->filename()->str().empty())) {
      TFLITE_LOG_PROD_ONCE(
          TFLITE_LOG_ERROR,
          "Minibenchmark requested for %s but model_file not set.\n",
          model_id_.c_str());
      return false;
    }

    return true;
  }

  MinibenchmarkStatus GetNnApiSlPointerIfPresent(
      const NnApiSLDriverImplFL5** nnapi_sl) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSmini_benchmark_implementationDTcc mht_16(mht_16_v, 681, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/mini_benchmark_implementation.cc", "GetNnApiSlPointerIfPresent");

    *nnapi_sl = nullptr;
    const auto& settings_to_test = *settings_->settings_to_test();
    for (const auto* setting_to_test : settings_to_test) {
      if (setting_to_test->nnapi_settings()) {
        const NnApiSLDriverImplFL5* curr_nnapi_sl_handle =
            reinterpret_cast<const NnApiSLDriverImplFL5*>(
                setting_to_test->nnapi_settings()->support_library_handle());

        if (*nnapi_sl != nullptr && *nnapi_sl != curr_nnapi_sl_handle) {
          return kMiniBenchmarkInvalidSupportLibraryConfiguration;
        }

        *nnapi_sl = curr_nnapi_sl_handle;
      }
    }
    return kMinibenchmarkSuccess;
  }

  void LogInitializationFailure(MinibenchmarkStatus status) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSmini_benchmark_implementationDTcc mht_17(mht_17_v, 703, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/mini_benchmark_implementation.cc", "LogInitializationFailure");

    if (!initialization_failure_logged_) {
      flatbuffers::FlatBufferBuilder fbb;
      storage_->Append(&fbb, CreateMiniBenchmarkEvent(
                                 fbb, /*is_log_flushing_event=*/false,
                                 /*best_acceleration_decision=*/0,
                                 ::tflite::CreateBenchmarkInitializationFailure(
                                     fbb, status)));
      initialization_failure_logged_ = true;
    }
  }

  void CreateValidatorIfNececessary() {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSmini_benchmark_implementationDTcc mht_18(mht_18_v, 718, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/mini_benchmark_implementation.cc", "CreateValidatorIfNececessary");

    if (validator_) return;

    const NnApiSLDriverImplFL5* nnapi_sl;
    MinibenchmarkStatus get_nnapi_sl_status =
        GetNnApiSlPointerIfPresent(&nnapi_sl);
    if (get_nnapi_sl_status != kMinibenchmarkSuccess) {
      LogInitializationFailure(get_nnapi_sl_status);
      return;
    }

    if (settings_->model_file()->fd() <= 0) {
      validator_ = std::make_unique<ValidatorRunner>(
          settings_->model_file()->filename()->str(),
          settings_->storage_paths()->storage_file_path()->str(),
          settings_->storage_paths()->data_directory_path()->str(), nnapi_sl);
    } else {
      validator_ = std::make_unique<ValidatorRunner>(
          settings_->model_file()->fd(), settings_->model_file()->offset(),
          settings_->model_file()->length(),
          settings_->storage_paths()->storage_file_path()->str(),
          settings_->storage_paths()->data_directory_path()->str(), nnapi_sl);
    }
    MinibenchmarkStatus status = validator_->Init();
    if (status == kMinibenchmarkValidationEntrypointSymbolNotFound) {
      TFLITE_LOG_PROD_ONCE(TFLITE_LOG_ERROR,
                           "Model %s does not contain a validation subgraph.",
                           model_id_.c_str());
    } else if (status != kMinibenchmarkSuccess) {
      TFLITE_LOG_PROD_ONCE(TFLITE_LOG_ERROR,
                           "ValidatorRunner::Init() failed for model %s.",
                           model_id_.c_str());
    } else {
      validator_initialized_ = true;
    }
    if (status != kMinibenchmarkSuccess) {
      LogInitializationFailure(status);
    }
  }

  flatbuffers::FlatBufferBuilder settings_buffer_;
  // Just a pointer to the 'settings_buffer_' for convenience.
  const MinibenchmarkSettings* settings_ = nullptr;
  bool is_enabled_ = false;
  int total_validation_tests_ = 0;
  bool is_cpu_validation_specified_ = false;

  std::unique_ptr<ValidatorRunner> validator_ = nullptr;
  bool validator_initialized_ = false;
  std::string model_namespace_;
  std::string model_id_;
  int64_t event_timeout_us_ = ValidatorRunner::kDefaultEventTimeoutUs;
  std::unique_ptr<MemoizedBestAccelerationSelector>
      best_acceleration_selector_ = nullptr;

  std::unique_ptr<FlatbufferStorage<MiniBenchmarkEvent>> storage_ = nullptr;
  bool initialization_failure_logged_ = false;
};

std::unique_ptr<MiniBenchmark> CreateMiniBenchmarkImpl(
    const MinibenchmarkSettings& settings, const std::string& model_namespace,
    const std::string& model_id) {
  return std::unique_ptr<MiniBenchmark>(
      new MiniBenchmarkImpl(settings, model_namespace, model_id));
}

TFLITE_REGISTER_MINI_BENCMARK_FACTORY_FUNCTION(Impl, CreateMiniBenchmarkImpl);
}  // namespace acceleration
}  // namespace tflite
