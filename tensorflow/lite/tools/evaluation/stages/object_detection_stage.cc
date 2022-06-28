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
class MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSobject_detection_stageDTcc {
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
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSobject_detection_stageDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSobject_detection_stageDTcc() {
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
#include "tensorflow/lite/tools/evaluation/stages/object_detection_stage.h"

#include <fstream>

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_config.pb.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_stages.pb.h"
#include "tensorflow/lite/tools/evaluation/utils.h"

namespace tflite {
namespace evaluation {

TfLiteStatus ObjectDetectionStage::Init(
    const DelegateProviders* delegate_providers) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSobject_detection_stageDTcc mht_0(mht_0_v, 198, "", "./tensorflow/lite/tools/evaluation/stages/object_detection_stage.cc", "ObjectDetectionStage::Init");

  // Ensure inference params are provided.
  if (!config_.specification().has_object_detection_params()) {
    LOG(ERROR) << "ObjectDetectionParams not provided";
    return kTfLiteError;
  }
  auto& params = config_.specification().object_detection_params();
  if (!params.has_inference_params()) {
    LOG(ERROR) << "inference_params not provided";
    return kTfLiteError;
  }
  if (all_labels_ == nullptr) {
    LOG(ERROR) << "Detection output labels not provided";
    return kTfLiteError;
  }

  // TfliteInferenceStage.
  EvaluationStageConfig tflite_inference_config;
  tflite_inference_config.set_name("tflite_inference");
  *tflite_inference_config.mutable_specification()
       ->mutable_tflite_inference_params() = params.inference_params();
  inference_stage_.reset(new TfliteInferenceStage(tflite_inference_config));
  TF_LITE_ENSURE_STATUS(inference_stage_->Init(delegate_providers));

  // Validate model inputs.
  const TfLiteModelInfo* model_info = inference_stage_->GetModelInfo();
  if (model_info->inputs.size() != 1 || model_info->outputs.size() != 4) {
    LOG(ERROR) << "Object detection model must have 1 input & 4 outputs";
    return kTfLiteError;
  }
  TfLiteType input_type = model_info->inputs[0]->type;
  auto* input_shape = model_info->inputs[0]->dims;
  // Input should be of the shape {1, height, width, 3}
  if (input_shape->size != 4 || input_shape->data[0] != 1 ||
      input_shape->data[3] != 3) {
    LOG(ERROR) << "Invalid input shape for model";
    return kTfLiteError;
  }

  // ImagePreprocessingStage
  tflite::evaluation::ImagePreprocessingConfigBuilder builder(
      "image_preprocessing", input_type);
  builder.AddResizingStep(input_shape->data[2], input_shape->data[1], false);
  builder.AddDefaultNormalizationStep();
  preprocessing_stage_.reset(new ImagePreprocessingStage(builder.build()));
  TF_LITE_ENSURE_STATUS(preprocessing_stage_->Init());

  // ObjectDetectionAveragePrecisionStage
  EvaluationStageConfig eval_config;
  eval_config.set_name("average_precision");
  *eval_config.mutable_specification()
       ->mutable_object_detection_average_precision_params() =
      params.ap_params();
  eval_config.mutable_specification()
      ->mutable_object_detection_average_precision_params()
      ->set_num_classes(all_labels_->size());
  eval_stage_.reset(new ObjectDetectionAveragePrecisionStage(eval_config));
  TF_LITE_ENSURE_STATUS(eval_stage_->Init());

  return kTfLiteOk;
}

TfLiteStatus ObjectDetectionStage::Run() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSobject_detection_stageDTcc mht_1(mht_1_v, 263, "", "./tensorflow/lite/tools/evaluation/stages/object_detection_stage.cc", "ObjectDetectionStage::Run");

  if (image_path_.empty()) {
    LOG(ERROR) << "Input image not set";
    return kTfLiteError;
  }

  // Preprocessing.
  preprocessing_stage_->SetImagePath(&image_path_);
  TF_LITE_ENSURE_STATUS(preprocessing_stage_->Run());

  // Inference.
  std::vector<void*> data_ptrs = {};
  data_ptrs.push_back(preprocessing_stage_->GetPreprocessedImageData());
  inference_stage_->SetInputs(data_ptrs);
  TF_LITE_ENSURE_STATUS(inference_stage_->Run());

  // Convert model output to ObjectsSet.
  predicted_objects_.Clear();
  const int class_offset =
      config_.specification().object_detection_params().class_offset();
  const std::vector<void*>* outputs = inference_stage_->GetOutputs();
  int num_detections = static_cast<int>(*static_cast<float*>(outputs->at(3)));
  float* detected_label_boxes = static_cast<float*>(outputs->at(0));
  float* detected_label_indices = static_cast<float*>(outputs->at(1));
  float* detected_label_probabilities = static_cast<float*>(outputs->at(2));
  for (int i = 0; i < num_detections; ++i) {
    const int bounding_box_offset = i * 4;
    auto* object = predicted_objects_.add_objects();
    // Bounding box
    auto* bbox = object->mutable_bounding_box();
    bbox->set_normalized_top(detected_label_boxes[bounding_box_offset + 0]);
    bbox->set_normalized_left(detected_label_boxes[bounding_box_offset + 1]);
    bbox->set_normalized_bottom(detected_label_boxes[bounding_box_offset + 2]);
    bbox->set_normalized_right(detected_label_boxes[bounding_box_offset + 3]);
    // Class.
    object->set_class_id(static_cast<int>(detected_label_indices[i]) +
                         class_offset);
    // Score
    object->set_score(detected_label_probabilities[i]);
  }

  // AP Evaluation.
  eval_stage_->SetEvalInputs(predicted_objects_, *ground_truth_objects_);
  TF_LITE_ENSURE_STATUS(eval_stage_->Run());

  return kTfLiteOk;
}

EvaluationStageMetrics ObjectDetectionStage::LatestMetrics() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSobject_detection_stageDTcc mht_2(mht_2_v, 314, "", "./tensorflow/lite/tools/evaluation/stages/object_detection_stage.cc", "ObjectDetectionStage::LatestMetrics");

  EvaluationStageMetrics metrics;
  auto* detection_metrics =
      metrics.mutable_process_metrics()->mutable_object_detection_metrics();

  *detection_metrics->mutable_pre_processing_latency() =
      preprocessing_stage_->LatestMetrics().process_metrics().total_latency();
  EvaluationStageMetrics inference_metrics = inference_stage_->LatestMetrics();
  *detection_metrics->mutable_inference_latency() =
      inference_metrics.process_metrics().total_latency();
  *detection_metrics->mutable_inference_metrics() =
      inference_metrics.process_metrics().tflite_inference_metrics();
  *detection_metrics->mutable_average_precision_metrics() =
      eval_stage_->LatestMetrics()
          .process_metrics()
          .object_detection_average_precision_metrics();
  metrics.set_num_runs(inference_metrics.num_runs());
  return metrics;
}

TfLiteStatus PopulateGroundTruth(
    const std::string& grouth_truth_proto_file,
    absl::flat_hash_map<std::string, ObjectDetectionResult>*
        ground_truth_mapping) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("grouth_truth_proto_file: \"" + grouth_truth_proto_file + "\"");
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSobject_detection_stageDTcc mht_3(mht_3_v, 341, "", "./tensorflow/lite/tools/evaluation/stages/object_detection_stage.cc", "PopulateGroundTruth");

  if (ground_truth_mapping == nullptr) {
    return kTfLiteError;
  }
  ground_truth_mapping->clear();

  // Read the ground truth dump.
  std::ifstream t(grouth_truth_proto_file);
  std::string proto_str((std::istreambuf_iterator<char>(t)),
                        std::istreambuf_iterator<char>());
  ObjectDetectionGroundTruth ground_truth_proto;
  ground_truth_proto.ParseFromString(proto_str);

  for (const auto& image_ground_truth :
       ground_truth_proto.detection_results()) {
    (*ground_truth_mapping)[image_ground_truth.image_name()] =
        image_ground_truth;
  }

  return kTfLiteOk;
}

}  // namespace evaluation
}  // namespace tflite
