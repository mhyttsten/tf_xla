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
#ifndef TENSORFLOW_LITE_TOOLS_EVALUATION_STAGES_OBJECT_DETECTION_STAGE_H_
#define TENSORFLOW_LITE_TOOLS_EVALUATION_STAGES_OBJECT_DETECTION_STAGE_H_
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
class MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSobject_detection_stageDTh {
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
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSobject_detection_stageDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSobject_detection_stageDTh() {
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


#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/lite/tools/evaluation/evaluation_delegate_provider.h"
#include "tensorflow/lite/tools/evaluation/evaluation_stage.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_config.pb.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_stages.pb.h"
#include "tensorflow/lite/tools/evaluation/stages/image_preprocessing_stage.h"
#include "tensorflow/lite/tools/evaluation/stages/object_detection_average_precision_stage.h"
#include "tensorflow/lite/tools/evaluation/stages/tflite_inference_stage.h"

namespace tflite {
namespace evaluation {

// An EvaluationStage to encapsulate the complete Object Detection task.
// Assumes that the object detection model's signature (number of
// inputs/outputs, ordering of outputs & what they denote) is same as the
// MobileNet SSD model:
// https://www.tensorflow.org/lite/models/object_detection/overview#output.
// Input size/type & number of detections could be different.
//
// This class will be extended to support other types of detection models, if
// required in the future.
class ObjectDetectionStage : public EvaluationStage {
 public:
  explicit ObjectDetectionStage(const EvaluationStageConfig& config)
      : EvaluationStage(config) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSobject_detection_stageDTh mht_0(mht_0_v, 215, "", "./tensorflow/lite/tools/evaluation/stages/object_detection_stage.h", "ObjectDetectionStage");
}

  TfLiteStatus Init() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSobject_detection_stageDTh mht_1(mht_1_v, 220, "", "./tensorflow/lite/tools/evaluation/stages/object_detection_stage.h", "Init");
 return Init(nullptr); }
  TfLiteStatus Init(const DelegateProviders* delegate_providers);

  TfLiteStatus Run() override;

  EvaluationStageMetrics LatestMetrics() override;

  // Call before Init(). all_labels should contain all possible object labels
  // that can be detected by the model, in the correct order. all_labels should
  // outlive the call to Init().
  void SetAllLabels(const std::vector<std::string>& all_labels) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSobject_detection_stageDTh mht_2(mht_2_v, 233, "", "./tensorflow/lite/tools/evaluation/stages/object_detection_stage.h", "SetAllLabels");

    all_labels_ = &all_labels;
  }

  // Call before Run().
  // ground_truth_objects instance should outlive the call to Run().
  void SetInputs(const std::string& image_path,
                 const ObjectDetectionResult& ground_truth_objects) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("image_path: \"" + image_path + "\"");
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSobject_detection_stageDTh mht_3(mht_3_v, 244, "", "./tensorflow/lite/tools/evaluation/stages/object_detection_stage.h", "SetInputs");

    image_path_ = image_path;
    ground_truth_objects_ = &ground_truth_objects;
  }

  // Provides a pointer to the underlying TfLiteInferenceStage.
  // Returns non-null value only if this stage has been initialized.
  TfliteInferenceStage* const GetInferenceStage() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSobject_detection_stageDTh mht_4(mht_4_v, 254, "", "./tensorflow/lite/tools/evaluation/stages/object_detection_stage.h", "GetInferenceStage");

    return inference_stage_.get();
  }

  // Returns a const pointer to the latest inference output.
  const ObjectDetectionResult* GetLatestPrediction() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSobject_detection_stageDTh mht_5(mht_5_v, 262, "", "./tensorflow/lite/tools/evaluation/stages/object_detection_stage.h", "GetLatestPrediction");

    return &predicted_objects_;
  }

 private:
  const std::vector<std::string>* all_labels_ = nullptr;
  std::unique_ptr<ImagePreprocessingStage> preprocessing_stage_;
  std::unique_ptr<TfliteInferenceStage> inference_stage_;
  std::unique_ptr<ObjectDetectionAveragePrecisionStage> eval_stage_;
  std::string image_path_;

  // Obtained from SetInputs(...).
  const ObjectDetectionResult* ground_truth_objects_;
  // Reflects the outputs generated from the latest call to Run().
  ObjectDetectionResult predicted_objects_;
};

// Reads a tflite::evaluation::ObjectDetectionGroundTruth instance from a
// textproto file and populates a mapping of image name to
// ObjectDetectionResult.
// File with ObjectDetectionGroundTruth can be generated using the
// preprocess_coco_minival.py script in evaluation/tasks/coco_object_detection.
// Useful for wrappers/scripts that use ObjectDetectionStage.
TfLiteStatus PopulateGroundTruth(
    const std::string& grouth_truth_proto_file,
    absl::flat_hash_map<std::string, ObjectDetectionResult>*
        ground_truth_mapping);

}  // namespace evaluation
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_EVALUATION_STAGES_OBJECT_DETECTION_STAGE_H_
