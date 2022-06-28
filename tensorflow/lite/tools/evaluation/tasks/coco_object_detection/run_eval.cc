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
class MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPStasksPScoco_object_detectionPSrun_evalDTcc {
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
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPStasksPScoco_object_detectionPSrun_evalDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPStasksPScoco_object_detectionPSrun_evalDTcc() {
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
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/types/optional.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/tools/command_line_flags.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_config.pb.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_stages.pb.h"
#include "tensorflow/lite/tools/evaluation/stages/object_detection_stage.h"
#include "tensorflow/lite/tools/evaluation/tasks/task_executor.h"
#include "tensorflow/lite/tools/evaluation/utils.h"
#include "tensorflow/lite/tools/logging.h"

namespace tflite {
namespace evaluation {

constexpr char kModelFileFlag[] = "model_file";
constexpr char kGroundTruthImagesPathFlag[] = "ground_truth_images_path";
constexpr char kModelOutputLabelsFlag[] = "model_output_labels";
constexpr char kOutputFilePathFlag[] = "output_file_path";
constexpr char kGroundTruthProtoFileFlag[] = "ground_truth_proto";
constexpr char kInterpreterThreadsFlag[] = "num_interpreter_threads";
constexpr char kDebugModeFlag[] = "debug_mode";
constexpr char kDelegateFlag[] = "delegate";

std::string GetNameFromPath(const std::string& str) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("str: \"" + str + "\"");
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPStasksPScoco_object_detectionPSrun_evalDTcc mht_0(mht_0_v, 213, "", "./tensorflow/lite/tools/evaluation/tasks/coco_object_detection/run_eval.cc", "GetNameFromPath");

  int pos = str.find_last_of("/\\");
  if (pos == std::string::npos) return "";
  return str.substr(pos + 1);
}

class CocoObjectDetection : public TaskExecutor {
 public:
  CocoObjectDetection() : debug_mode_(false), num_interpreter_threads_(1) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPStasksPScoco_object_detectionPSrun_evalDTcc mht_1(mht_1_v, 224, "", "./tensorflow/lite/tools/evaluation/tasks/coco_object_detection/run_eval.cc", "CocoObjectDetection");
}
  ~CocoObjectDetection() override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPStasksPScoco_object_detectionPSrun_evalDTcc mht_2(mht_2_v, 228, "", "./tensorflow/lite/tools/evaluation/tasks/coco_object_detection/run_eval.cc", "~CocoObjectDetection");
}

 protected:
  std::vector<Flag> GetFlags() final;

  // If the run is successful, the latest metrics will be returned.
  absl::optional<EvaluationStageMetrics> RunImpl() final;

 private:
  void OutputResult(const EvaluationStageMetrics& latest_metrics) const;
  std::string model_file_path_;
  std::string model_output_labels_path_;
  std::string ground_truth_images_path_;
  std::string ground_truth_proto_file_;
  std::string output_file_path_;
  bool debug_mode_;
  std::string delegate_;
  int num_interpreter_threads_;
};

std::vector<Flag> CocoObjectDetection::GetFlags() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPStasksPScoco_object_detectionPSrun_evalDTcc mht_3(mht_3_v, 251, "", "./tensorflow/lite/tools/evaluation/tasks/coco_object_detection/run_eval.cc", "CocoObjectDetection::GetFlags");

  std::vector<tflite::Flag> flag_list = {
      tflite::Flag::CreateFlag(kModelFileFlag, &model_file_path_,
                               "Path to test tflite model file."),
      tflite::Flag::CreateFlag(
          kModelOutputLabelsFlag, &model_output_labels_path_,
          "Path to labels that correspond to output of model."
          " E.g. in case of COCO-trained SSD model, this is the path to file "
          "where each line contains a class detected by the model in correct "
          "order, starting from background."),
      tflite::Flag::CreateFlag(
          kGroundTruthImagesPathFlag, &ground_truth_images_path_,
          "Path to ground truth images. These will be evaluated in "
          "alphabetical order of filenames"),
      tflite::Flag::CreateFlag(kGroundTruthProtoFileFlag,
                               &ground_truth_proto_file_,
                               "Path to file containing "
                               "tflite::evaluation::ObjectDetectionGroundTruth "
                               "proto in binary serialized format. If left "
                               "empty, mAP numbers are not output."),
      tflite::Flag::CreateFlag(
          kOutputFilePathFlag, &output_file_path_,
          "File to output to. Contains only metrics proto if debug_mode is "
          "off, and per-image predictions also otherwise."),
      tflite::Flag::CreateFlag(kDebugModeFlag, &debug_mode_,
                               "Whether to enable debug mode. Per-image "
                               "predictions are written to the output file "
                               "along with metrics."),
      tflite::Flag::CreateFlag(
          kInterpreterThreadsFlag, &num_interpreter_threads_,
          "Number of interpreter threads to use for inference."),
      tflite::Flag::CreateFlag(
          kDelegateFlag, &delegate_,
          "Delegate to use for inference, if available. "
          "Must be one of {'nnapi', 'gpu', 'xnnpack', 'hexagon'}"),
  };
  return flag_list;
}

absl::optional<EvaluationStageMetrics> CocoObjectDetection::RunImpl() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPStasksPScoco_object_detectionPSrun_evalDTcc mht_4(mht_4_v, 293, "", "./tensorflow/lite/tools/evaluation/tasks/coco_object_detection/run_eval.cc", "CocoObjectDetection::RunImpl");

  // Process images in filename-sorted order.
  std::vector<std::string> image_paths;
  if (GetSortedFileNames(StripTrailingSlashes(ground_truth_images_path_),
                         &image_paths) != kTfLiteOk) {
    return absl::nullopt;
  }

  std::vector<std::string> model_labels;
  if (!ReadFileLines(model_output_labels_path_, &model_labels)) {
    TFLITE_LOG(ERROR) << "Could not read model output labels file";
    return absl::nullopt;
  }

  EvaluationStageConfig eval_config;
  eval_config.set_name("object_detection");
  auto* detection_params =
      eval_config.mutable_specification()->mutable_object_detection_params();
  auto* inference_params = detection_params->mutable_inference_params();
  inference_params->set_model_file_path(model_file_path_);
  inference_params->set_num_threads(num_interpreter_threads_);
  inference_params->set_delegate(ParseStringToDelegateType(delegate_));

  // Get ground truth data.
  absl::flat_hash_map<std::string, ObjectDetectionResult> ground_truth_map;
  if (!ground_truth_proto_file_.empty()) {
    PopulateGroundTruth(ground_truth_proto_file_, &ground_truth_map);
  }

  ObjectDetectionStage eval(eval_config);

  eval.SetAllLabels(model_labels);
  if (eval.Init(&delegate_providers_) != kTfLiteOk) return absl::nullopt;

  const int step = image_paths.size() / 100;
  for (int i = 0; i < image_paths.size(); ++i) {
    if (step > 1 && i % step == 0) {
      TFLITE_LOG(INFO) << "Finished: " << i / step << "%";
    }

    const std::string image_name = GetNameFromPath(image_paths[i]);
    eval.SetInputs(image_paths[i], ground_truth_map[image_name]);
    if (eval.Run() != kTfLiteOk) return absl::nullopt;

    if (debug_mode_) {
      ObjectDetectionResult prediction = *eval.GetLatestPrediction();
      TFLITE_LOG(INFO) << "Image: " << image_name << "\n";
      for (int i = 0; i < prediction.objects_size(); ++i) {
        const auto& object = prediction.objects(i);
        TFLITE_LOG(INFO) << "Object [" << i << "]";
        TFLITE_LOG(INFO) << "  Score: " << object.score();
        TFLITE_LOG(INFO) << "  Class-ID: " << object.class_id();
        TFLITE_LOG(INFO) << "  Bounding Box:";
        const auto& bounding_box = object.bounding_box();
        TFLITE_LOG(INFO) << "    Normalized Top: "
                         << bounding_box.normalized_top();
        TFLITE_LOG(INFO) << "    Normalized Bottom: "
                         << bounding_box.normalized_bottom();
        TFLITE_LOG(INFO) << "    Normalized Left: "
                         << bounding_box.normalized_left();
        TFLITE_LOG(INFO) << "    Normalized Right: "
                         << bounding_box.normalized_right();
      }
      TFLITE_LOG(INFO)
          << "======================================================\n";
    }
  }

  // Write metrics to file.
  EvaluationStageMetrics latest_metrics = eval.LatestMetrics();
  if (ground_truth_proto_file_.empty()) {
    TFLITE_LOG(WARN) << "mAP metrics are meaningless w/o ground truth.";
    latest_metrics.mutable_process_metrics()
        ->mutable_object_detection_metrics()
        ->clear_average_precision_metrics();
  }

  OutputResult(latest_metrics);
  return absl::make_optional(latest_metrics);
}

void CocoObjectDetection::OutputResult(
    const EvaluationStageMetrics& latest_metrics) const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPStasksPScoco_object_detectionPSrun_evalDTcc mht_5(mht_5_v, 378, "", "./tensorflow/lite/tools/evaluation/tasks/coco_object_detection/run_eval.cc", "CocoObjectDetection::OutputResult");

  if (!output_file_path_.empty()) {
    std::ofstream metrics_ofile;
    metrics_ofile.open(output_file_path_, std::ios::out);
    metrics_ofile << latest_metrics.SerializeAsString();
    metrics_ofile.close();
  }
  TFLITE_LOG(INFO) << "Num evaluation runs: " << latest_metrics.num_runs();
  const auto object_detection_metrics =
      latest_metrics.process_metrics().object_detection_metrics();
  const auto& preprocessing_latency =
      object_detection_metrics.pre_processing_latency();
  TFLITE_LOG(INFO) << "Preprocessing latency: avg="
                   << preprocessing_latency.avg_us() << "(us), std_dev="
                   << preprocessing_latency.std_deviation_us() << "(us)";
  const auto& inference_latency = object_detection_metrics.inference_latency();
  TFLITE_LOG(INFO) << "Inference latency: avg=" << inference_latency.avg_us()
                   << "(us), std_dev=" << inference_latency.std_deviation_us()
                   << "(us)";
  const auto& precision_metrics =
      object_detection_metrics.average_precision_metrics();
  for (int i = 0; i < precision_metrics.individual_average_precisions_size();
       ++i) {
    const auto ap_metric = precision_metrics.individual_average_precisions(i);
    TFLITE_LOG(INFO) << "Average Precision [IOU Threshold="
                     << ap_metric.iou_threshold()
                     << "]: " << ap_metric.average_precision();
  }
  TFLITE_LOG(INFO) << "Overall mAP: "
                   << precision_metrics.overall_mean_average_precision();
}

std::unique_ptr<TaskExecutor> CreateTaskExecutor() {
  return std::unique_ptr<TaskExecutor>(new CocoObjectDetection());
}

}  // namespace evaluation
}  // namespace tflite
