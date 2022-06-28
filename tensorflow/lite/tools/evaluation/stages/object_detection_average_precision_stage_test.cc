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
class MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSobject_detection_average_precision_stage_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSobject_detection_average_precision_stage_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSobject_detection_average_precision_stage_testDTcc() {
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

#include <string>

#include <gtest/gtest.h>
#include "tensorflow/lite/tools/evaluation/proto/evaluation_config.pb.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_stages.pb.h"

namespace tflite {
namespace evaluation {
namespace {

constexpr char kAveragePrecisionStageName[] =
    "object_detection_average_precision";

EvaluationStageConfig GetAveragePrecisionStageConfig(int num_classes) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSobject_detection_average_precision_stage_testDTcc mht_0(mht_0_v, 201, "", "./tensorflow/lite/tools/evaluation/stages/object_detection_average_precision_stage_test.cc", "GetAveragePrecisionStageConfig");

  EvaluationStageConfig config;
  config.set_name(kAveragePrecisionStageName);
  auto* params = config.mutable_specification()
                     ->mutable_object_detection_average_precision_params();
  params->add_iou_thresholds(0.5);
  params->add_iou_thresholds(0.999);
  params->set_num_classes(num_classes);
  return config;
}

ObjectDetectionResult GetGroundTruthDetectionResult() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSobject_detection_average_precision_stage_testDTcc mht_1(mht_1_v, 215, "", "./tensorflow/lite/tools/evaluation/stages/object_detection_average_precision_stage_test.cc", "GetGroundTruthDetectionResult");

  ObjectDetectionResult ground_truth;
  ground_truth.set_image_name("some_image.jpg");

  auto* object_1 = ground_truth.add_objects();
  object_1->set_class_id(1);
  auto* object_1_bbox = object_1->mutable_bounding_box();
  object_1_bbox->set_normalized_top(0.5);
  object_1_bbox->set_normalized_bottom(1.0);
  object_1_bbox->set_normalized_left(0.5);
  object_1_bbox->set_normalized_right(1.0);

  auto* object_2 = ground_truth.add_objects();
  object_2->set_class_id(1);
  auto* object_2_bbox = object_2->mutable_bounding_box();
  object_2_bbox->set_normalized_top(0);
  object_2_bbox->set_normalized_bottom(1.0);
  object_2_bbox->set_normalized_left(0);
  object_2_bbox->set_normalized_right(1.0);

  auto* object_3 = ground_truth.add_objects();
  object_3->set_class_id(2);
  auto* object_3_bbox = object_3->mutable_bounding_box();
  object_3_bbox->set_normalized_top(0.5);
  object_3_bbox->set_normalized_bottom(1.0);
  object_3_bbox->set_normalized_left(0.5);
  object_3_bbox->set_normalized_right(1.0);

  return ground_truth;
}

ObjectDetectionResult GetPredictedDetectionResult() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSobject_detection_average_precision_stage_testDTcc mht_2(mht_2_v, 249, "", "./tensorflow/lite/tools/evaluation/stages/object_detection_average_precision_stage_test.cc", "GetPredictedDetectionResult");

  ObjectDetectionResult predicted;

  auto* object_1 = predicted.add_objects();
  object_1->set_class_id(1);
  object_1->set_score(0.8);
  auto* object_1_bbox = object_1->mutable_bounding_box();
  object_1_bbox->set_normalized_top(0.091);
  object_1_bbox->set_normalized_bottom(1.0);
  object_1_bbox->set_normalized_left(0.091);
  object_1_bbox->set_normalized_right(1.0);

  auto* object_2 = predicted.add_objects();
  object_2->set_class_id(1);
  object_2->set_score(0.9);
  auto* object_2_bbox = object_2->mutable_bounding_box();
  object_2_bbox->set_normalized_top(0.474);
  object_2_bbox->set_normalized_bottom(1.0);
  object_2_bbox->set_normalized_left(0.474);
  object_2_bbox->set_normalized_right(1.0);

  auto* object_3 = predicted.add_objects();
  object_3->set_class_id(1);
  object_3->set_score(0.95);
  auto* object_3_bbox = object_3->mutable_bounding_box();
  object_3_bbox->set_normalized_top(0.474);
  object_3_bbox->set_normalized_bottom(1.0);
  object_3_bbox->set_normalized_left(0.474);
  object_3_bbox->set_normalized_right(1.0);
  return predicted;
}

TEST(ObjectDetectionAveragePrecisionStage, ZeroClasses) {
  // Create stage.
  EvaluationStageConfig config = GetAveragePrecisionStageConfig(0);
  ObjectDetectionAveragePrecisionStage stage =
      ObjectDetectionAveragePrecisionStage(config);

  EXPECT_EQ(stage.Init(), kTfLiteError);
}

// Tests ObjectDetectionAveragePrecisionStage with sample inputs & outputs.
// The underlying library is tested extensively in utils/image_metrics_test.
TEST(ObjectDetectionAveragePrecisionStage, SampleInputs) {
  // Create & initialize stage.
  EvaluationStageConfig config = GetAveragePrecisionStageConfig(3);
  ObjectDetectionAveragePrecisionStage stage =
      ObjectDetectionAveragePrecisionStage(config);
  EXPECT_EQ(stage.Init(), kTfLiteOk);

  const ObjectDetectionResult ground_truth = GetGroundTruthDetectionResult();
  const ObjectDetectionResult predicted = GetPredictedDetectionResult();

  // Run with no predictions.
  stage.SetEvalInputs(ObjectDetectionResult(), ground_truth);
  EXPECT_EQ(stage.Run(), kTfLiteOk);

  EvaluationStageMetrics metrics = stage.LatestMetrics();
  ObjectDetectionAveragePrecisionMetrics detection_metrics =
      metrics.process_metrics().object_detection_average_precision_metrics();
  EXPECT_FLOAT_EQ(detection_metrics.overall_mean_average_precision(), 0.0);
  EXPECT_EQ(detection_metrics.individual_average_precisions_size(), 2);

  // Run with matching predictions.
  stage.SetEvalInputs(ground_truth, ground_truth);
  EXPECT_EQ(stage.Run(), kTfLiteOk);

  metrics = stage.LatestMetrics();
  detection_metrics =
      metrics.process_metrics().object_detection_average_precision_metrics();
  EXPECT_FLOAT_EQ(detection_metrics.overall_mean_average_precision(),
                  0.50495052);
  EXPECT_EQ(metrics.num_runs(), 2);

  // Run.
  stage.SetEvalInputs(predicted, ground_truth);
  EXPECT_EQ(stage.Run(), kTfLiteOk);

  metrics = stage.LatestMetrics();
  detection_metrics =
      metrics.process_metrics().object_detection_average_precision_metrics();
  EXPECT_FLOAT_EQ(
      detection_metrics.individual_average_precisions(0).iou_threshold(), 0.5);
  EXPECT_FLOAT_EQ(
      detection_metrics.individual_average_precisions(0).average_precision(),
      0.4841584);
  EXPECT_FLOAT_EQ(
      detection_metrics.individual_average_precisions(1).iou_threshold(),
      0.999);
  EXPECT_FLOAT_EQ(
      detection_metrics.individual_average_precisions(1).average_precision(),
      0.33663365);
  // Should be average of above two values.
  EXPECT_FLOAT_EQ(detection_metrics.overall_mean_average_precision(),
                  0.41039604);
}

TEST(ObjectDetectionAveragePrecisionStage, DefaultIoUThresholds) {
  // Create & initialize stage.
  EvaluationStageConfig config = GetAveragePrecisionStageConfig(3);
  auto* params = config.mutable_specification()
                     ->mutable_object_detection_average_precision_params();
  params->clear_iou_thresholds();
  ObjectDetectionAveragePrecisionStage stage =
      ObjectDetectionAveragePrecisionStage(config);
  EXPECT_EQ(stage.Init(), kTfLiteOk);

  const ObjectDetectionResult ground_truth = GetGroundTruthDetectionResult();
  const ObjectDetectionResult predicted = GetPredictedDetectionResult();

  // Run with matching predictions.
  stage.SetEvalInputs(ground_truth, ground_truth);
  EXPECT_EQ(stage.Run(), kTfLiteOk);

  EvaluationStageMetrics metrics = stage.LatestMetrics();
  ObjectDetectionAveragePrecisionMetrics detection_metrics =
      metrics.process_metrics().object_detection_average_precision_metrics();
  // Full AP, since ground-truth & predictions match.
  EXPECT_FLOAT_EQ(detection_metrics.overall_mean_average_precision(), 1.0);
  // Should be 10 IoU thresholds.
  EXPECT_EQ(detection_metrics.individual_average_precisions_size(), 10);
  EXPECT_FLOAT_EQ(
      detection_metrics.individual_average_precisions(0).iou_threshold(), 0.5);
  EXPECT_FLOAT_EQ(
      detection_metrics.individual_average_precisions(9).iou_threshold(), 0.95);
}

}  // namespace
}  // namespace evaluation
}  // namespace tflite
