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
class MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPStflite_inference_stageDTcc {
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
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPStflite_inference_stageDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPStflite_inference_stageDTcc() {
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
#include "tensorflow/lite/tools/evaluation/stages/tflite_inference_stage.h"

#include <cstring>
#include <fstream>

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/profiling/time.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_stages.pb.h"
#include "tensorflow/lite/tools/evaluation/utils.h"

namespace tflite {
namespace evaluation {
namespace {

TfLiteModelInfo GetTfliteModelInfo(const Interpreter& interpreter) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPStflite_inference_stageDTcc mht_0(mht_0_v, 199, "", "./tensorflow/lite/tools/evaluation/stages/tflite_inference_stage.cc", "GetTfliteModelInfo");

  TfLiteModelInfo model_info;
  for (int i : interpreter.inputs()) {
    model_info.inputs.push_back(interpreter.tensor(i));
  }
  for (int i : interpreter.outputs()) {
    model_info.outputs.push_back(interpreter.tensor(i));
  }
  return model_info;
}

}  // namespace

void TfliteInferenceStage::UpdateModelInfo() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPStflite_inference_stageDTcc mht_1(mht_1_v, 215, "", "./tensorflow/lite/tools/evaluation/stages/tflite_inference_stage.cc", "TfliteInferenceStage::UpdateModelInfo");

  model_info_ = GetTfliteModelInfo(*interpreter_);

  outputs_.clear();
  outputs_.reserve(interpreter_->outputs().size());
  for (int i : interpreter_->outputs()) {
    TfLiteTensor* tensor = interpreter_->tensor(i);
    outputs_.push_back(tensor->data.raw);
  }
}

TfLiteStatus TfliteInferenceStage::ResizeInputs(
    const std::vector<std::vector<int>>& shapes) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPStflite_inference_stageDTcc mht_2(mht_2_v, 230, "", "./tensorflow/lite/tools/evaluation/stages/tflite_inference_stage.cc", "TfliteInferenceStage::ResizeInputs");

  const std::vector<int>& intepreter_inputs = interpreter_->inputs();
  if (intepreter_inputs.size() != shapes.size()) {
    LOG(ERROR) << "New shape is not compatible";
    return kTfLiteError;
  }

  for (int j = 0; j < shapes.size(); ++j) {
    int i = intepreter_inputs[j];
    TfLiteTensor* t = interpreter_->tensor(i);
    if (t->type != kTfLiteString) {
      TF_LITE_ENSURE_STATUS(interpreter_->ResizeInputTensor(i, shapes[j]));
    }
  }

  TF_LITE_ENSURE_STATUS(interpreter_->AllocateTensors());
  UpdateModelInfo();
  return kTfLiteOk;
}

TfLiteStatus TfliteInferenceStage::ApplyCustomDelegate(
    Interpreter::TfLiteDelegatePtr delegate) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPStflite_inference_stageDTcc mht_3(mht_3_v, 254, "", "./tensorflow/lite/tools/evaluation/stages/tflite_inference_stage.cc", "TfliteInferenceStage::ApplyCustomDelegate");

  if (!interpreter_) {
    LOG(ERROR) << "Stage not initialized before calling ApplyCustomDelegate";
    return kTfLiteError;
  }
  // Skip if delegate is a nullptr.
  if (!delegate) {
    LOG(WARNING)
        << "Tried to apply null TfLiteDelegatePtr to TfliteInferenceStage";
    return kTfLiteOk;
  }
  delegates_.push_back(std::move(delegate));
  TF_LITE_ENSURE_STATUS(
      interpreter_->ModifyGraphWithDelegate(delegates_.back().get()));
  UpdateModelInfo();
  return kTfLiteOk;
}

TfLiteStatus TfliteInferenceStage::Init(
    const DelegateProviders* delegate_providers) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPStflite_inference_stageDTcc mht_4(mht_4_v, 276, "", "./tensorflow/lite/tools/evaluation/stages/tflite_inference_stage.cc", "TfliteInferenceStage::Init");

  if (!config_.specification().has_tflite_inference_params()) {
    LOG(ERROR) << "TfliteInferenceParams not provided";
    return kTfLiteError;
  }
  auto& params = config_.specification().tflite_inference_params();
  if (!params.has_model_file_path()) {
    LOG(ERROR) << "Model path not provided";
    return kTfLiteError;
  }
  std::ifstream model_check(params.model_file_path());
  if (!model_check.good()) {
    LOG(ERROR) << "Model file not found";
    return kTfLiteError;
  }

  model_ = FlatBufferModel::BuildFromFile(params.model_file_path().c_str());

  bool apply_default_delegates = true;
  if (delegate_providers != nullptr) {
    const auto& provider_params = delegate_providers->GetAllParams();
    // When --use_xnnpack is explicitly set to false, to honor this, skip
    // applying the XNNPACK delegate by default in TfLite runtime.
    if (provider_params.HasParam("use_xnnpack") &&
        provider_params.HasValueSet<bool>("use_xnnpack") &&
        !provider_params.Get<bool>("use_xnnpack")) {
      apply_default_delegates = false;
    }
  }

  resolver_.reset(
      apply_default_delegates
          ? new ops::builtin::BuiltinOpResolver()
          : new ops::builtin::BuiltinOpResolverWithoutDefaultDelegates());
  InterpreterBuilder(*model_, *resolver_)(&interpreter_);
  if (!interpreter_) {
    LOG(ERROR) << "Could not build interpreter";
    return kTfLiteError;
  }
  interpreter_->SetNumThreads(params.num_threads());

  if (!delegate_providers) {
    std::string error_message;
    auto delegate = CreateTfLiteDelegate(params, &error_message);
    if (delegate) {
      delegates_.push_back(std::move(delegate));
      LOG(INFO) << "Successfully created "
                << params.Delegate_Name(params.delegate()) << " delegate.";
    } else {
      LOG(WARNING) << error_message;
    }
  } else {
    auto delegates = delegate_providers->CreateAllDelegates(params);
    for (auto& one : delegates) delegates_.push_back(std::move(one.delegate));
  }

  for (int i = 0; i < delegates_.size(); ++i) {
    if (interpreter_->ModifyGraphWithDelegate(delegates_[i].get()) !=
        kTfLiteOk) {
      LOG(FATAL) << "Failed to apply delegate " << i;
    }
  }
  interpreter_->AllocateTensors();
  UpdateModelInfo();

  return kTfLiteOk;
}

TfLiteStatus TfliteInferenceStage::Run() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPStflite_inference_stageDTcc mht_5(mht_5_v, 347, "", "./tensorflow/lite/tools/evaluation/stages/tflite_inference_stage.cc", "TfliteInferenceStage::Run");

  if (!inputs_) {
    LOG(ERROR) << "Input data not set";
    return kTfLiteError;
  }

  // Copy input data.
  for (int i = 0; i < interpreter_->inputs().size(); ++i) {
    TfLiteTensor* tensor = interpreter_->tensor(interpreter_->inputs()[i]);
    tensor->data.raw = static_cast<char*>(inputs_->at(i));
  }

  // Invoke.
  auto& params = config_.specification().tflite_inference_params();
  for (int i = 0; i < params.invocations_per_run(); ++i) {
    int64_t start_us = profiling::time::NowMicros();
    if (interpreter_->Invoke() != kTfLiteOk) {
      LOG(ERROR) << "TFLite interpreter failed to invoke at run " << i;
      return kTfLiteError;
    }
    latency_stats_.UpdateStat(profiling::time::NowMicros() - start_us);
  }

  return kTfLiteOk;
}

EvaluationStageMetrics TfliteInferenceStage::LatestMetrics() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPStflite_inference_stageDTcc mht_6(mht_6_v, 376, "", "./tensorflow/lite/tools/evaluation/stages/tflite_inference_stage.cc", "TfliteInferenceStage::LatestMetrics");

  auto& params = config_.specification().tflite_inference_params();
  EvaluationStageMetrics metrics;
  auto* latency_metrics =
      metrics.mutable_process_metrics()->mutable_total_latency();
  latency_metrics->set_last_us(latency_stats_.newest());
  latency_metrics->set_max_us(latency_stats_.max());
  latency_metrics->set_min_us(latency_stats_.min());
  latency_metrics->set_sum_us(latency_stats_.sum());
  latency_metrics->set_avg_us(latency_stats_.avg());
  latency_metrics->set_std_deviation_us(latency_stats_.std_deviation());
  metrics.set_num_runs(
      static_cast<int>(latency_stats_.count() / params.invocations_per_run()));
  auto* inference_metrics =
      metrics.mutable_process_metrics()->mutable_tflite_inference_metrics();
  inference_metrics->set_num_inferences(latency_stats_.count());
  return metrics;
}

}  // namespace evaluation
}  // namespace tflite
