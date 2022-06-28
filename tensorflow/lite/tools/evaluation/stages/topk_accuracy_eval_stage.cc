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
class MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPStopk_accuracy_eval_stageDTcc {
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
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPStopk_accuracy_eval_stageDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPStopk_accuracy_eval_stageDTcc() {
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
#include "tensorflow/lite/tools/evaluation/stages/topk_accuracy_eval_stage.h"

#include <stdint.h>

#include <algorithm>
#include <numeric>

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_stages.pb.h"

namespace tflite {
namespace evaluation {
namespace {

std::vector<int> GetTopKIndices(const std::vector<float>& values, int k) {
  std::vector<int> indices(values.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::stable_sort(indices.begin(), indices.end(),
                   [&values](int a, int b) { return values[a] > values[b]; });
  indices.resize(k);
  return indices;
}

}  // namespace

TfLiteStatus TopkAccuracyEvalStage::Init() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPStopk_accuracy_eval_stageDTcc mht_0(mht_0_v, 209, "", "./tensorflow/lite/tools/evaluation/stages/topk_accuracy_eval_stage.cc", "TopkAccuracyEvalStage::Init");

  num_runs_ = 0;
  auto& params = config_.specification().topk_accuracy_eval_params();
  if (!params.has_k()) {
    LOG(ERROR) << "Value of k not provided for TopkAccuracyEvalStage";
    return kTfLiteError;
  }
  accuracy_counts_ = std::vector<int>(params.k(), 0);

  if (ground_truth_labels_.empty()) {
    LOG(ERROR) << "Ground-truth labels are empty";
    return kTfLiteError;
  }
  num_total_labels_ = ground_truth_labels_.size();
  if (params.k() > num_total_labels_) {
    LOG(ERROR) << "k is too large";
    return kTfLiteError;
  }

  if (!model_output_shape_) {
    LOG(ERROR) << "Model output details not correctly set";
    return kTfLiteError;
  }
  // Ensure model output is of shape (1, num_total_labels_).
  if (!(model_output_shape_->size == 2) ||
      !(model_output_shape_->data[0] == 1) ||
      !(model_output_shape_->data[1] == num_total_labels_)) {
    LOG(ERROR) << "Invalid model_output_shape_";
    return kTfLiteError;
  }
  if (model_output_type_ != kTfLiteFloat32 &&
      model_output_type_ != kTfLiteUInt8 && model_output_type_ != kTfLiteInt8) {
    LOG(ERROR) << "model_output_type_ not supported";
    return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus TopkAccuracyEvalStage::Run() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPStopk_accuracy_eval_stageDTcc mht_1(mht_1_v, 250, "", "./tensorflow/lite/tools/evaluation/stages/topk_accuracy_eval_stage.cc", "TopkAccuracyEvalStage::Run");

  if (!model_output_) {
    LOG(ERROR) << "model_output_ not set correctly";
    return kTfLiteError;
  }
  if (!ground_truth_label_) {
    LOG(ERROR) << "ground_truth_label_ not provided";
    return kTfLiteError;
  }
  auto& params = config_.specification().topk_accuracy_eval_params();

  std::vector<float> probabilities;
  probabilities.reserve(num_total_labels_);
  if (model_output_type_ == kTfLiteFloat32) {
    auto probs = static_cast<float*>(model_output_);
    for (size_t i = 0; i < num_total_labels_; i++) {
      probabilities.push_back(probs[i]);
    }
  } else if (model_output_type_ == kTfLiteUInt8) {
    auto probs = static_cast<uint8_t*>(model_output_);
    for (size_t i = 0; i < num_total_labels_; i++) {
      probabilities.push_back(probs[i]);
    }
  } else if (model_output_type_ == kTfLiteInt8) {
    auto probs = static_cast<int8_t*>(model_output_);
    for (size_t i = 0; i < num_total_labels_; i++) {
      probabilities.push_back(probs[i]);
    }
  }

  std::vector<int> top_k = GetTopKIndices(probabilities, params.k());
  UpdateCounts(top_k);
  return kTfLiteOk;
}

EvaluationStageMetrics TopkAccuracyEvalStage::LatestMetrics() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPStopk_accuracy_eval_stageDTcc mht_2(mht_2_v, 288, "", "./tensorflow/lite/tools/evaluation/stages/topk_accuracy_eval_stage.cc", "TopkAccuracyEvalStage::LatestMetrics");

  EvaluationStageMetrics metrics;
  if (num_runs_ == 0) return metrics;

  metrics.set_num_runs(num_runs_);
  auto* topk_metrics =
      metrics.mutable_process_metrics()->mutable_topk_accuracy_metrics();
  for (const auto& count : accuracy_counts_) {
    topk_metrics->add_topk_accuracies(static_cast<float>(count) / num_runs_);
  }
  return metrics;
}

void TopkAccuracyEvalStage::UpdateCounts(const std::vector<int>& topk_indices) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPStopk_accuracy_eval_stageDTcc mht_3(mht_3_v, 304, "", "./tensorflow/lite/tools/evaluation/stages/topk_accuracy_eval_stage.cc", "TopkAccuracyEvalStage::UpdateCounts");

  for (size_t i = 0; i < topk_indices.size(); ++i) {
    if (*ground_truth_label_ == ground_truth_labels_[topk_indices[i]]) {
      for (size_t j = i; j < topk_indices.size(); j++) {
        accuracy_counts_[j] += 1;
      }
      break;
    }
  }
  num_runs_++;
}

}  // namespace evaluation
}  // namespace tflite
