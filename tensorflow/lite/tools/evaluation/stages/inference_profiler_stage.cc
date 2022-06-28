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
class MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSinference_profiler_stageDTcc {
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
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSinference_profiler_stageDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSinference_profiler_stageDTcc() {
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
#include "tensorflow/lite/tools/evaluation/stages/inference_profiler_stage.h"

#include <cmath>
#include <limits>
#include <random>

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_config.pb.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_stages.pb.h"

namespace tflite {
namespace evaluation {
namespace {

// Parameters for a simple Gaussian distribution to generate values roughly in
// [0, 1).
constexpr float kGaussianFloatMean = 0.5;
constexpr float kGaussianStdDev = 1.0 / 3;

// TODO(b/131420973): Reconcile these with the functionality in
// testing/kernel_test.
template <typename T>
void GenerateRandomGaussianData(int64_t num_elements, float min, float max,
                                std::vector<T>* data) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSinference_profiler_stageDTcc mht_0(mht_0_v, 207, "", "./tensorflow/lite/tools/evaluation/stages/inference_profiler_stage.cc", "GenerateRandomGaussianData");

  data->clear();
  data->reserve(num_elements);

  static std::normal_distribution<double> distribution(kGaussianFloatMean,
                                                       kGaussianStdDev);
  static std::default_random_engine generator;
  for (int i = 0; i < num_elements; ++i) {
    auto rand_n = distribution(generator);
    while (rand_n < 0 || rand_n >= 1) {
      rand_n = distribution(generator);
    }
    auto rand_float = min + (max - min) * static_cast<float>(rand_n);
    data->push_back(static_cast<T>(rand_float));
  }
}

template <typename T>
float CalculateAverageError(T* reference, T* test, int64_t num_elements) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSinference_profiler_stageDTcc mht_1(mht_1_v, 228, "", "./tensorflow/lite/tools/evaluation/stages/inference_profiler_stage.cc", "CalculateAverageError");

  float error = 0;

  for (int i = 0; i < num_elements; i++) {
    float test_value = static_cast<float>(test[i]);
    float reference_value = static_cast<float>(reference[i]);
    error += std::abs(test_value - reference_value);
  }
  error /= num_elements;

  return error;
}

}  // namespace

TfLiteStatus InferenceProfilerStage::Init(
    const DelegateProviders* delegate_providers) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSinference_profiler_stageDTcc mht_2(mht_2_v, 247, "", "./tensorflow/lite/tools/evaluation/stages/inference_profiler_stage.cc", "InferenceProfilerStage::Init");

  // Initialize TfliteInferenceStage with the user-provided
  // TfliteInferenceParams.
  test_stage_.reset(new TfliteInferenceStage(config_));
  if (test_stage_->Init(delegate_providers) != kTfLiteOk) return kTfLiteError;
  LOG(INFO) << "Test interpreter has been initialized.";

  // Initialize a reference TfliteInferenceStage that uses the given model &
  // num_runs, but maintains the rest of TfliteInferenceParams to default.
  EvaluationStageConfig reference_config;
  reference_config.set_name("reference_inference");
  auto* params = reference_config.mutable_specification()
                     ->mutable_tflite_inference_params();
  params->set_model_file_path(
      config_.specification().tflite_inference_params().model_file_path());
  params->set_invocations_per_run(
      config_.specification().tflite_inference_params().invocations_per_run());
  reference_stage_.reset(new TfliteInferenceStage(reference_config));
  if (reference_stage_->Init() != kTfLiteOk) return kTfLiteError;
  LOG(INFO) << "Reference interpreter (1 thread on CPU) has been initialized.";

  model_info_ = reference_stage_->GetModelInfo();

  // Preprocess model input metadata for generating random data later.
  for (int i = 0; i < model_info_->inputs.size(); ++i) {
    const TfLiteType model_input_type = model_info_->inputs[i]->type;
    if (model_input_type == kTfLiteUInt8 || model_input_type == kTfLiteInt8 ||
        model_input_type == kTfLiteFloat32) {
    } else {
      LOG(ERROR) << "InferenceProfilerStage only supports float/int8/uint8 "
                    "input types";
      return kTfLiteError;
    }
    auto* input_shape = model_info_->inputs[i]->dims;
    int64_t total_num_elements = 1;
    for (int i = 0; i < input_shape->size; i++) {
      total_num_elements *= input_shape->data[i];
    }
    input_num_elements_.push_back(total_num_elements);
    float_tensors_.emplace_back();
    uint8_tensors_.emplace_back();
    int8_tensors_.emplace_back();
  }
  // Preprocess output metadata for calculating diffs later.
  for (int i = 0; i < model_info_->outputs.size(); ++i) {
    const TfLiteType model_output_type = model_info_->outputs[i]->type;
    if (model_output_type == kTfLiteUInt8 || model_output_type == kTfLiteInt8 ||
        model_output_type == kTfLiteFloat32) {
    } else {
      LOG(ERROR) << "InferenceProfilerStage only supports float/int8/uint8 "
                    "output types";
      return kTfLiteError;
    }
    auto* output_shape = model_info_->outputs[i]->dims;
    int64_t total_num_elements = 1;
    for (int i = 0; i < output_shape->size; i++) {
      total_num_elements *= output_shape->data[i];
    }
    output_num_elements_.push_back(total_num_elements);

    error_stats_.emplace_back();
  }

  return kTfLiteOk;
}

TfLiteStatus InferenceProfilerStage::Run() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSinference_profiler_stageDTcc mht_3(mht_3_v, 316, "", "./tensorflow/lite/tools/evaluation/stages/inference_profiler_stage.cc", "InferenceProfilerStage::Run");

  // Generate random inputs.
  std::vector<void*> input_ptrs;
  for (int i = 0; i < model_info_->inputs.size(); ++i) {
    const TfLiteType model_input_type = model_info_->inputs[i]->type;
    if (model_input_type == kTfLiteUInt8) {
      GenerateRandomGaussianData(
          input_num_elements_[i], std::numeric_limits<uint8_t>::min(),
          std::numeric_limits<uint8_t>::max(), &uint8_tensors_[i]);
      input_ptrs.push_back(uint8_tensors_[i].data());
    } else if (model_input_type == kTfLiteInt8) {
      GenerateRandomGaussianData(
          input_num_elements_[i], std::numeric_limits<int8_t>::min(),
          std::numeric_limits<int8_t>::max(), &int8_tensors_[i]);
      input_ptrs.push_back(int8_tensors_[i].data());
    } else if (model_input_type == kTfLiteFloat32) {
      GenerateRandomGaussianData(input_num_elements_[i], -1, 1,
                                 &(float_tensors_[i]));
      input_ptrs.push_back(float_tensors_[i].data());
    }
  }

  // Run both inference stages.
  test_stage_->SetInputs(input_ptrs);
  reference_stage_->SetInputs(input_ptrs);
  if (test_stage_->Run() != kTfLiteOk) return kTfLiteError;
  if (reference_stage_->Run() != kTfLiteOk) return kTfLiteError;

  // Calculate errors per output vector.
  for (int i = 0; i < model_info_->outputs.size(); ++i) {
    const TfLiteType model_output_type = model_info_->outputs[i]->type;
    void* reference_ptr = reference_stage_->GetOutputs()->at(i);
    void* test_ptr = test_stage_->GetOutputs()->at(i);
    float output_diff = 0;
    if (model_output_type == kTfLiteUInt8) {
      output_diff = CalculateAverageError(static_cast<uint8_t*>(reference_ptr),
                                          static_cast<uint8_t*>(test_ptr),
                                          output_num_elements_[i]);
    } else if (model_output_type == kTfLiteInt8) {
      output_diff = CalculateAverageError(static_cast<int8_t*>(reference_ptr),
                                          static_cast<int8_t*>(test_ptr),
                                          output_num_elements_[i]);
    } else if (model_output_type == kTfLiteFloat32) {
      output_diff = CalculateAverageError(static_cast<float*>(reference_ptr),
                                          static_cast<float*>(test_ptr),
                                          output_num_elements_[i]);
    }
    error_stats_[i].UpdateStat(output_diff);
  }

  return kTfLiteOk;
}

EvaluationStageMetrics InferenceProfilerStage::LatestMetrics() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSinference_profiler_stageDTcc mht_4(mht_4_v, 372, "", "./tensorflow/lite/tools/evaluation/stages/inference_profiler_stage.cc", "InferenceProfilerStage::LatestMetrics");

  EvaluationStageMetrics metrics;
  const auto& reference_metrics = reference_stage_->LatestMetrics();
  metrics.set_num_runs(reference_metrics.num_runs());
  auto* inference_profiler_metrics =
      metrics.mutable_process_metrics()->mutable_inference_profiler_metrics();

  *inference_profiler_metrics->mutable_reference_latency() =
      reference_metrics.process_metrics().total_latency();
  *inference_profiler_metrics->mutable_test_latency() =
      test_stage_->LatestMetrics().process_metrics().total_latency();

  for (int i = 0; i < error_stats_.size(); ++i) {
    AccuracyMetrics* diff = inference_profiler_metrics->add_output_errors();
    diff->set_avg_value(error_stats_[i].avg());
    diff->set_std_deviation(error_stats_[i].std_deviation());
    diff->set_min_value(error_stats_[i].min());
    // Avoiding the small positive values contained in max() even when avg() ==
    // 0.
    if (error_stats_[i].avg() != 0) {
      diff->set_max_value(error_stats_[i].max());
    } else {
      diff->set_max_value(0);
    }
  }

  return metrics;
}

}  // namespace evaluation
}  // namespace tflite
