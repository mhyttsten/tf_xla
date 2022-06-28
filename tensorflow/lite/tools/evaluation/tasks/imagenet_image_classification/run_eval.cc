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
class MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPStasksPSimagenet_image_classificationPSrun_evalDTcc {
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
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPStasksPSimagenet_image_classificationPSrun_evalDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPStasksPSimagenet_image_classificationPSrun_evalDTcc() {
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

#include "absl/types/optional.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/tools/command_line_flags.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_config.pb.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_stages.pb.h"
#include "tensorflow/lite/tools/evaluation/stages/image_classification_stage.h"
#include "tensorflow/lite/tools/evaluation/tasks/task_executor.h"
#include "tensorflow/lite/tools/evaluation/utils.h"
#include "tensorflow/lite/tools/logging.h"

namespace tflite {
namespace evaluation {

constexpr char kModelFileFlag[] = "model_file";
constexpr char kGroundTruthImagesPathFlag[] = "ground_truth_images_path";
constexpr char kGroundTruthLabelsFlag[] = "ground_truth_labels";
constexpr char kOutputFilePathFlag[] = "output_file_path";
constexpr char kModelOutputLabelsFlag[] = "model_output_labels";
constexpr char kDenylistFilePathFlag[] = "denylist_file_path";
constexpr char kNumImagesFlag[] = "num_images";
constexpr char kInterpreterThreadsFlag[] = "num_interpreter_threads";
constexpr char kDelegateFlag[] = "delegate";

template <typename T>
std::vector<T> GetFirstN(const std::vector<T>& v, int n) {
  if (n >= v.size()) return v;
  std::vector<T> result(v.begin(), v.begin() + n);
  return result;
}

class ImagenetClassification : public TaskExecutor {
 public:
  ImagenetClassification() : num_images_(0), num_interpreter_threads_(1) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPStasksPSimagenet_image_classificationPSrun_evalDTcc mht_0(mht_0_v, 221, "", "./tensorflow/lite/tools/evaluation/tasks/imagenet_image_classification/run_eval.cc", "ImagenetClassification");
}
  ~ImagenetClassification() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPStasksPSimagenet_image_classificationPSrun_evalDTcc mht_1(mht_1_v, 225, "", "./tensorflow/lite/tools/evaluation/tasks/imagenet_image_classification/run_eval.cc", "~ImagenetClassification");
}

 protected:
  std::vector<Flag> GetFlags() final;

  // If the run is successful, the latest metrics will be returned.
  absl::optional<EvaluationStageMetrics> RunImpl() final;

 private:
  void OutputResult(const EvaluationStageMetrics& latest_metrics) const;
  std::string model_file_path_;
  std::string ground_truth_images_path_;
  std::string ground_truth_labels_path_;
  std::string model_output_labels_path_;
  std::string denylist_file_path_;
  std::string output_file_path_;
  std::string delegate_;
  int num_images_;
  int num_interpreter_threads_;
};

std::vector<Flag> ImagenetClassification::GetFlags() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPStasksPSimagenet_image_classificationPSrun_evalDTcc mht_2(mht_2_v, 249, "", "./tensorflow/lite/tools/evaluation/tasks/imagenet_image_classification/run_eval.cc", "ImagenetClassification::GetFlags");

  std::vector<tflite::Flag> flag_list = {
      tflite::Flag::CreateFlag(kModelFileFlag, &model_file_path_,
                               "Path to test tflite model file."),
      tflite::Flag::CreateFlag(
          kModelOutputLabelsFlag, &model_output_labels_path_,
          "Path to labels that correspond to output of model."
          " E.g. in case of mobilenet, this is the path to label "
          "file where each label is in the same order as the output"
          " of the model."),
      tflite::Flag::CreateFlag(
          kGroundTruthImagesPathFlag, &ground_truth_images_path_,
          "Path to ground truth images. These will be evaluated in "
          "alphabetical order of filename"),
      tflite::Flag::CreateFlag(
          kGroundTruthLabelsFlag, &ground_truth_labels_path_,
          "Path to ground truth labels, corresponding to alphabetical ordering "
          "of ground truth images."),
      tflite::Flag::CreateFlag(
          kDenylistFilePathFlag, &denylist_file_path_,
          "Path to denylist file (optional) where each line is a single "
          "integer that is "
          "equal to index number of denylisted image."),
      tflite::Flag::CreateFlag(kOutputFilePathFlag, &output_file_path_,
                               "File to output metrics proto to."),
      tflite::Flag::CreateFlag(kNumImagesFlag, &num_images_,
                               "Number of examples to evaluate, pass 0 for all "
                               "examples. Default: 0"),
      tflite::Flag::CreateFlag(
          kInterpreterThreadsFlag, &num_interpreter_threads_,
          "Number of interpreter threads to use for inference."),
      tflite::Flag::CreateFlag(
          kDelegateFlag, &delegate_,
          "Delegate to use for inference, if available. "
          "Must be one of {'nnapi', 'gpu', 'hexagon', 'xnnpack'}"),
  };
  return flag_list;
}

absl::optional<EvaluationStageMetrics> ImagenetClassification::RunImpl() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPStasksPSimagenet_image_classificationPSrun_evalDTcc mht_3(mht_3_v, 291, "", "./tensorflow/lite/tools/evaluation/tasks/imagenet_image_classification/run_eval.cc", "ImagenetClassification::RunImpl");

  // Process images in filename-sorted order.
  std::vector<std::string> image_files, ground_truth_image_labels;
  if (GetSortedFileNames(StripTrailingSlashes(ground_truth_images_path_),
                         &image_files) != kTfLiteOk) {
    return absl::nullopt;
  }
  if (!ReadFileLines(ground_truth_labels_path_, &ground_truth_image_labels)) {
    TFLITE_LOG(ERROR) << "Could not read ground truth labels file";
    return absl::nullopt;
  }
  if (image_files.size() != ground_truth_image_labels.size()) {
    TFLITE_LOG(ERROR) << "Number of images and ground truth labels is not same";
    return absl::nullopt;
  }
  std::vector<ImageLabel> image_labels;
  image_labels.reserve(image_files.size());
  for (int i = 0; i < image_files.size(); i++) {
    image_labels.push_back({image_files[i], ground_truth_image_labels[i]});
  }

  // Filter out denylisted/unwanted images.
  if (FilterDenyListedImages(denylist_file_path_, &image_labels) != kTfLiteOk) {
    return absl::nullopt;
  }
  if (num_images_ > 0) {
    image_labels = GetFirstN(image_labels, num_images_);
  }

  std::vector<std::string> model_labels;
  if (!ReadFileLines(model_output_labels_path_, &model_labels)) {
    TFLITE_LOG(ERROR) << "Could not read model output labels file";
    return absl::nullopt;
  }

  EvaluationStageConfig eval_config;
  eval_config.set_name("image_classification");
  auto* classification_params = eval_config.mutable_specification()
                                    ->mutable_image_classification_params();
  auto* inference_params = classification_params->mutable_inference_params();
  inference_params->set_model_file_path(model_file_path_);
  inference_params->set_num_threads(num_interpreter_threads_);
  inference_params->set_delegate(ParseStringToDelegateType(delegate_));
  classification_params->mutable_topk_accuracy_eval_params()->set_k(10);

  ImageClassificationStage eval(eval_config);

  eval.SetAllLabels(model_labels);
  if (eval.Init(&delegate_providers_) != kTfLiteOk) return absl::nullopt;

  const int step = image_labels.size() / 100;
  for (int i = 0; i < image_labels.size(); ++i) {
    if (step > 1 && i % step == 0) {
      TFLITE_LOG(INFO) << "Evaluated: " << i / step << "%";
    }
    eval.SetInputs(image_labels[i].image, image_labels[i].label);
    if (eval.Run() != kTfLiteOk) return absl::nullopt;
  }

  const auto latest_metrics = eval.LatestMetrics();
  OutputResult(latest_metrics);
  return absl::make_optional(latest_metrics);
}

void ImagenetClassification::OutputResult(
    const EvaluationStageMetrics& latest_metrics) const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPStasksPSimagenet_image_classificationPSrun_evalDTcc mht_4(mht_4_v, 359, "", "./tensorflow/lite/tools/evaluation/tasks/imagenet_image_classification/run_eval.cc", "ImagenetClassification::OutputResult");

  if (!output_file_path_.empty()) {
    std::ofstream metrics_ofile;
    metrics_ofile.open(output_file_path_, std::ios::out);
    metrics_ofile << latest_metrics.SerializeAsString();
    metrics_ofile.close();
  }

  TFLITE_LOG(INFO) << "Num evaluation runs: " << latest_metrics.num_runs();
  const auto& metrics =
      latest_metrics.process_metrics().image_classification_metrics();
  const auto& preprocessing_latency = metrics.pre_processing_latency();
  TFLITE_LOG(INFO) << "Preprocessing latency: avg="
                   << preprocessing_latency.avg_us() << "(us), std_dev="
                   << preprocessing_latency.std_deviation_us() << "(us)";
  const auto& inference_latency = metrics.inference_latency();
  TFLITE_LOG(INFO) << "Inference latency: avg=" << inference_latency.avg_us()
                   << "(us), std_dev=" << inference_latency.std_deviation_us()
                   << "(us)";
  const auto& accuracy_metrics = metrics.topk_accuracy_metrics();
  for (int i = 0; i < accuracy_metrics.topk_accuracies_size(); ++i) {
    TFLITE_LOG(INFO) << "Top-" << i + 1
                     << " Accuracy: " << accuracy_metrics.topk_accuracies(i);
  }
}

std::unique_ptr<TaskExecutor> CreateTaskExecutor() {
  return std::unique_ptr<TaskExecutor>(new ImagenetClassification());
}

}  // namespace evaluation
}  // namespace tflite
