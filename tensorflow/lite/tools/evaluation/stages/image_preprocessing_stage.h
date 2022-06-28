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
#ifndef TENSORFLOW_LITE_TOOLS_EVALUATION_STAGES_IMAGE_PREPROCESSING_STAGE_H_
#define TENSORFLOW_LITE_TOOLS_EVALUATION_STAGES_IMAGE_PREPROCESSING_STAGE_H_
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
class MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSimage_preprocessing_stageDTh {
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
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSimage_preprocessing_stageDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSimage_preprocessing_stageDTh() {
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


#include <stdint.h>

#include <vector>

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/stats_calculator.h"
#include "tensorflow/lite/tools/evaluation/evaluation_stage.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_config.pb.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_stages.pb.h"
#include "tensorflow/lite/tools/evaluation/proto/preprocessing_steps.pb.h"

namespace tflite {
namespace evaluation {

// EvaluationStage to read contents of an image and preprocess it for inference.
// Currently only supports JPEGs.
class ImagePreprocessingStage : public EvaluationStage {
 public:
  explicit ImagePreprocessingStage(const EvaluationStageConfig& config)
      : EvaluationStage(config) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSimage_preprocessing_stageDTh mht_0(mht_0_v, 206, "", "./tensorflow/lite/tools/evaluation/stages/image_preprocessing_stage.h", "ImagePreprocessingStage");
}

  TfLiteStatus Init() override;

  TfLiteStatus Run() override;

  EvaluationStageMetrics LatestMetrics() override;

  ~ImagePreprocessingStage() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSimage_preprocessing_stageDTh mht_1(mht_1_v, 217, "", "./tensorflow/lite/tools/evaluation/stages/image_preprocessing_stage.h", "~ImagePreprocessingStage");
}

  // Call before Run().
  void SetImagePath(std::string* image_path) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSimage_preprocessing_stageDTh mht_2(mht_2_v, 223, "", "./tensorflow/lite/tools/evaluation/stages/image_preprocessing_stage.h", "SetImagePath");
 image_path_ = image_path; }

  // Provides preprocessing output.
  void* GetPreprocessedImageData();

 private:
  std::string* image_path_ = nullptr;
  TfLiteType output_type_;
  tensorflow::Stat<int64_t> latency_stats_;

  // One of the following 3 vectors will be populated based on output_type_.
  std::vector<float> float_preprocessed_image_;
  std::vector<int8_t> int8_preprocessed_image_;
  std::vector<uint8_t> uint8_preprocessed_image_;
};

// Helper class to build a new ImagePreprocessingParams.
class ImagePreprocessingConfigBuilder {
 public:
  ImagePreprocessingConfigBuilder(const std::string& name,
                                  TfLiteType output_type) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSimage_preprocessing_stageDTh mht_3(mht_3_v, 247, "", "./tensorflow/lite/tools/evaluation/stages/image_preprocessing_stage.h", "ImagePreprocessingConfigBuilder");

    config_.set_name(name);
    config_.mutable_specification()
        ->mutable_image_preprocessing_params()
        ->set_output_type(static_cast<int>(output_type));
  }

  // Adds a cropping step with cropping fraction.
  void AddCroppingStep(float cropping_fraction,
                       bool use_square_cropping = false) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSimage_preprocessing_stageDTh mht_4(mht_4_v, 259, "", "./tensorflow/lite/tools/evaluation/stages/image_preprocessing_stage.h", "AddCroppingStep");

    ImagePreprocessingStepParams params;
    params.mutable_cropping_params()->set_cropping_fraction(cropping_fraction);
    params.mutable_cropping_params()->set_square_cropping(use_square_cropping);
    config_.mutable_specification()
        ->mutable_image_preprocessing_params()
        ->mutable_steps()
        ->Add(std::move(params));
  }

  // Adds a cropping step with target size.
  void AddCroppingStep(uint32_t width, uint32_t height,
                       bool use_square_cropping = false) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSimage_preprocessing_stageDTh mht_5(mht_5_v, 274, "", "./tensorflow/lite/tools/evaluation/stages/image_preprocessing_stage.h", "AddCroppingStep");

    ImagePreprocessingStepParams params;
    params.mutable_cropping_params()->mutable_target_size()->set_height(height);
    params.mutable_cropping_params()->mutable_target_size()->set_width(width);
    params.mutable_cropping_params()->set_square_cropping(use_square_cropping);
    config_.mutable_specification()
        ->mutable_image_preprocessing_params()
        ->mutable_steps()
        ->Add(std::move(params));
  }

  // Adds a resizing step.
  void AddResizingStep(uint32_t width, uint32_t height,
                       bool aspect_preserving) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSimage_preprocessing_stageDTh mht_6(mht_6_v, 290, "", "./tensorflow/lite/tools/evaluation/stages/image_preprocessing_stage.h", "AddResizingStep");

    ImagePreprocessingStepParams params;
    params.mutable_resizing_params()->set_aspect_preserving(aspect_preserving);
    params.mutable_resizing_params()->mutable_target_size()->set_height(height);
    params.mutable_resizing_params()->mutable_target_size()->set_width(width);
    config_.mutable_specification()
        ->mutable_image_preprocessing_params()
        ->mutable_steps()
        ->Add(std::move(params));
  }

  // Adds a padding step.
  void AddPaddingStep(uint32_t width, uint32_t height, int value) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSimage_preprocessing_stageDTh mht_7(mht_7_v, 305, "", "./tensorflow/lite/tools/evaluation/stages/image_preprocessing_stage.h", "AddPaddingStep");

    ImagePreprocessingStepParams params;
    params.mutable_padding_params()->mutable_target_size()->set_height(height);
    params.mutable_padding_params()->mutable_target_size()->set_width(width);
    params.mutable_padding_params()->set_padding_value(value);
    config_.mutable_specification()
        ->mutable_image_preprocessing_params()
        ->mutable_steps()
        ->Add(std::move(params));
  }

  // Adds a square padding step.
  void AddSquarePaddingStep(int value) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSimage_preprocessing_stageDTh mht_8(mht_8_v, 320, "", "./tensorflow/lite/tools/evaluation/stages/image_preprocessing_stage.h", "AddSquarePaddingStep");

    ImagePreprocessingStepParams params;
    params.mutable_padding_params()->set_square_padding(true);
    params.mutable_padding_params()->set_padding_value(value);
    config_.mutable_specification()
        ->mutable_image_preprocessing_params()
        ->mutable_steps()
        ->Add(std::move(params));
  }

  // Adds a subtracting means step.
  void AddPerChannelNormalizationStep(float r_mean, float g_mean, float b_mean,
                                      float scale) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSimage_preprocessing_stageDTh mht_9(mht_9_v, 335, "", "./tensorflow/lite/tools/evaluation/stages/image_preprocessing_stage.h", "AddPerChannelNormalizationStep");

    ImagePreprocessingStepParams params;
    params.mutable_normalization_params()->mutable_means()->set_r_mean(r_mean);
    params.mutable_normalization_params()->mutable_means()->set_g_mean(g_mean);
    params.mutable_normalization_params()->mutable_means()->set_b_mean(b_mean);
    params.mutable_normalization_params()->set_scale(scale);
    config_.mutable_specification()
        ->mutable_image_preprocessing_params()
        ->mutable_steps()
        ->Add(std::move(params));
  }

  // Adds a normalization step.
  void AddNormalizationStep(float mean, float scale) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSimage_preprocessing_stageDTh mht_10(mht_10_v, 351, "", "./tensorflow/lite/tools/evaluation/stages/image_preprocessing_stage.h", "AddNormalizationStep");

    ImagePreprocessingStepParams params;
    params.mutable_normalization_params()->set_channelwise_mean(mean);
    params.mutable_normalization_params()->set_scale(scale);
    config_.mutable_specification()
        ->mutable_image_preprocessing_params()
        ->mutable_steps()
        ->Add(std::move(params));
  }

  // Adds a normalization step with default value.
  void AddDefaultNormalizationStep() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSimage_preprocessing_stageDTh mht_11(mht_11_v, 365, "", "./tensorflow/lite/tools/evaluation/stages/image_preprocessing_stage.h", "AddDefaultNormalizationStep");

    switch (
        config_.specification().image_preprocessing_params().output_type()) {
      case kTfLiteFloat32:
        AddNormalizationStep(127.5, 1.0 / 127.5);
        break;
      case kTfLiteUInt8:
        break;
      case kTfLiteInt8:
        AddNormalizationStep(128.0, 1.0);
        break;
      default:
        LOG(ERROR) << "Type not supported";
        break;
    }
  }

  EvaluationStageConfig build() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSimage_preprocessing_stageDTh mht_12(mht_12_v, 385, "", "./tensorflow/lite/tools/evaluation/stages/image_preprocessing_stage.h", "build");
 return std::move(config_); }

 private:
  EvaluationStageConfig config_;
};

}  // namespace evaluation
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_EVALUATION_STAGES_IMAGE_PREPROCESSING_STAGE_H_
