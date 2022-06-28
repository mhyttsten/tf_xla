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
class MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSobject_detection_average_precision_stageDTcc {
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
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSobject_detection_average_precision_stageDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSobject_detection_average_precision_stageDTcc() {
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
#include "tensorflow/lite/tools/evaluation/stages/object_detection_average_precision_stage.h"

#include <stdint.h>

#include <numeric>

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_stages.pb.h"

namespace tflite {
namespace evaluation {
namespace {

image::Detection ConvertProtoToDetection(
    const ObjectDetectionResult::ObjectInstance& input, int image_id) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSobject_detection_average_precision_stageDTcc mht_0(mht_0_v, 198, "", "./tensorflow/lite/tools/evaluation/stages/object_detection_average_precision_stage.cc", "ConvertProtoToDetection");

  image::Detection detection;
  detection.box.x.min = input.bounding_box().normalized_left();
  detection.box.x.max = input.bounding_box().normalized_right();
  detection.box.y.min = input.bounding_box().normalized_top();
  detection.box.y.max = input.bounding_box().normalized_bottom();
  detection.imgid = image_id;
  detection.score = input.score();
  return detection;
}

}  // namespace

TfLiteStatus ObjectDetectionAveragePrecisionStage::Init() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSobject_detection_average_precision_stageDTcc mht_1(mht_1_v, 214, "", "./tensorflow/lite/tools/evaluation/stages/object_detection_average_precision_stage.cc", "ObjectDetectionAveragePrecisionStage::Init");

  num_classes_ = config_.specification()
                     .object_detection_average_precision_params()
                     .num_classes();
  if (num_classes_ <= 0) {
    LOG(ERROR) << "num_classes cannot be <= 0";
    return kTfLiteError;
  }

  // Initialize per-class data structures.
  for (int i = 0; i < num_classes_; ++i) {
    ground_truth_object_vectors_.emplace_back();
    predicted_object_vectors_.emplace_back();
  }
  return kTfLiteOk;
}

TfLiteStatus ObjectDetectionAveragePrecisionStage::Run() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSobject_detection_average_precision_stageDTcc mht_2(mht_2_v, 234, "", "./tensorflow/lite/tools/evaluation/stages/object_detection_average_precision_stage.cc", "ObjectDetectionAveragePrecisionStage::Run");

  for (int i = 0; i < ground_truth_objects_.objects_size(); ++i) {
    const int class_id = ground_truth_objects_.objects(i).class_id();
    if (class_id >= num_classes_) {
      LOG(ERROR) << "Encountered invalid class ID: " << class_id;
      return kTfLiteError;
    }

    ground_truth_object_vectors_[class_id].push_back(ConvertProtoToDetection(
        ground_truth_objects_.objects(i), current_image_index_));
  }

  for (int i = 0; i < predicted_objects_.objects_size(); ++i) {
    const int class_id = predicted_objects_.objects(i).class_id();
    if (class_id >= num_classes_) {
      LOG(ERROR) << "Encountered invalid class ID: " << class_id;
      return kTfLiteError;
    }

    predicted_object_vectors_[class_id].push_back(ConvertProtoToDetection(
        predicted_objects_.objects(i), current_image_index_));
  }

  current_image_index_++;
  return kTfLiteOk;
}

EvaluationStageMetrics ObjectDetectionAveragePrecisionStage::LatestMetrics() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSobject_detection_average_precision_stageDTcc mht_3(mht_3_v, 264, "", "./tensorflow/lite/tools/evaluation/stages/object_detection_average_precision_stage.cc", "ObjectDetectionAveragePrecisionStage::LatestMetrics");

  EvaluationStageMetrics metrics;
  if (current_image_index_ == 0) return metrics;

  metrics.set_num_runs(current_image_index_);
  auto* ap_metrics = metrics.mutable_process_metrics()
                         ->mutable_object_detection_average_precision_metrics();
  auto& ap_params =
      config_.specification().object_detection_average_precision_params();

  std::vector<float> iou_thresholds;
  if (ap_params.iou_thresholds_size() == 0) {
    // Default IoU thresholds as defined by COCO evaluation.
    // Refer: http://cocodataset.org/#detection-eval
    float threshold = 0.5;
    for (int i = 0; i < 10; ++i) {
      iou_thresholds.push_back(threshold + i * 0.05);
    }
  } else {
    for (auto& threshold : ap_params.iou_thresholds()) {
      iou_thresholds.push_back(threshold);
    }
  }

  image::AveragePrecision::Options opts;
  opts.num_recall_points = ap_params.num_recall_points();

  float ap_sum = 0;
  int num_total_aps = 0;
  for (float threshold : iou_thresholds) {
    float threshold_ap_sum = 0;
    int num_counted_classes = 0;

    for (int i = 0; i < num_classes_; ++i) {
      // Skip if this class wasn't encountered at all.
      // TODO(b/133772912): Investigate the validity of this snippet when a
      // subset of the classes is encountered in datasets.
      if (ground_truth_object_vectors_[i].empty() &&
          predicted_object_vectors_[i].empty())
        continue;

      // Output is NaN if there are no ground truth objects.
      // So we assume 0.
      float ap_value = 0.0;
      if (!ground_truth_object_vectors_[i].empty()) {
        opts.iou_threshold = threshold;
        ap_value = image::AveragePrecision(opts).FromBoxes(
            ground_truth_object_vectors_[i], predicted_object_vectors_[i]);
      }

      ap_sum += ap_value;
      num_total_aps += 1;
      threshold_ap_sum += ap_value;
      num_counted_classes += 1;
    }

    if (num_counted_classes == 0) continue;
    auto* threshold_ap = ap_metrics->add_individual_average_precisions();
    threshold_ap->set_average_precision(threshold_ap_sum / num_counted_classes);
    threshold_ap->set_iou_threshold(threshold);
  }

  if (num_total_aps == 0) return metrics;
  ap_metrics->set_overall_mean_average_precision(ap_sum / num_total_aps);
  return metrics;
}

}  // namespace evaluation
}  // namespace tflite
