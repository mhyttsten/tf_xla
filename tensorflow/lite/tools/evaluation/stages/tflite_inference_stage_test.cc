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
class MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPStflite_inference_stage_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPStflite_inference_stage_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPStflite_inference_stage_testDTcc() {
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

#include <stdint.h>

#include <string>

#include <gtest/gtest.h>
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_config.pb.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_stages.pb.h"
#include "tensorflow/lite/tools/evaluation/utils.h"

namespace tflite {
namespace evaluation {
namespace {

constexpr char kTfliteInferenceStageName[] = "tflite_inference_stage";
constexpr char kModelPath[] =
    "tensorflow/lite/testdata/add_quantized.bin";
constexpr int kTotalElements = 1 * 8 * 8 * 3;

template <typename T>
T* SetValues(T array[], T value) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPStflite_inference_stage_testDTcc mht_0(mht_0_v, 208, "", "./tensorflow/lite/tools/evaluation/stages/tflite_inference_stage_test.cc", "SetValues");

  for (int i = 0; i < kTotalElements; i++) {
    array[i] = value;
  }
  return array;
}

EvaluationStageConfig GetTfliteInferenceStageConfig() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPStflite_inference_stage_testDTcc mht_1(mht_1_v, 218, "", "./tensorflow/lite/tools/evaluation/stages/tflite_inference_stage_test.cc", "GetTfliteInferenceStageConfig");

  EvaluationStageConfig config;
  config.set_name(kTfliteInferenceStageName);
  auto* params =
      config.mutable_specification()->mutable_tflite_inference_params();
  params->set_model_file_path(kModelPath);
  params->set_invocations_per_run(2);
  return config;
}

TEST(TfliteInferenceStage, NoParams) {
  EvaluationStageConfig config = GetTfliteInferenceStageConfig();
  config.mutable_specification()->clear_tflite_inference_params();
  TfliteInferenceStage stage(config);
  EXPECT_EQ(stage.Init(), kTfLiteError);
}

TEST(TfliteInferenceStage, NoModelPath) {
  EvaluationStageConfig config = GetTfliteInferenceStageConfig();
  config.mutable_specification()
      ->mutable_tflite_inference_params()
      ->clear_model_file_path();
  TfliteInferenceStage stage(config);
  EXPECT_EQ(stage.Init(), kTfLiteError);
}

TEST(TfliteInferenceStage, IncorrectModelPath) {
  EvaluationStageConfig config = GetTfliteInferenceStageConfig();
  config.mutable_specification()
      ->mutable_tflite_inference_params()
      ->set_model_file_path("xyz.tflite");
  TfliteInferenceStage stage(config);
  EXPECT_EQ(stage.Init(), kTfLiteError);
}

TEST(TfliteInferenceStage, NoInputData) {
  // Create stage.
  EvaluationStageConfig config = GetTfliteInferenceStageConfig();
  TfliteInferenceStage stage(config);

  // Initialize.
  EXPECT_EQ(stage.Init(), kTfLiteOk);

  // Run.
  EXPECT_EQ(stage.Run(), kTfLiteError);
}

TEST(TfliteInferenceStage, CorrectModelInfo) {
  // Create stage.
  EvaluationStageConfig config = GetTfliteInferenceStageConfig();
  TfliteInferenceStage stage(config);

  // Initialize.
  EXPECT_EQ(stage.Init(), kTfLiteOk);

  const TfLiteModelInfo* model_info = stage.GetModelInfo();
  // Verify Input
  EXPECT_EQ(model_info->inputs.size(), 1);
  const TfLiteTensor* tensor = model_info->inputs[0];
  EXPECT_EQ(tensor->type, kTfLiteUInt8);
  EXPECT_EQ(tensor->bytes, kTotalElements);
  const TfLiteIntArray* input_shape = tensor->dims;
  EXPECT_EQ(input_shape->data[0], 1);
  EXPECT_EQ(input_shape->data[1], 8);
  EXPECT_EQ(input_shape->data[2], 8);
  EXPECT_EQ(input_shape->data[3], 3);
  // Verify Output
  EXPECT_EQ(model_info->outputs.size(), 1);
  tensor = model_info->outputs[0];
  EXPECT_EQ(tensor->type, kTfLiteUInt8);
  EXPECT_EQ(tensor->bytes, kTotalElements);
  const TfLiteIntArray* output_shape = tensor->dims;
  EXPECT_EQ(output_shape->data[0], 1);
  EXPECT_EQ(output_shape->data[1], 8);
  EXPECT_EQ(output_shape->data[2], 8);
  EXPECT_EQ(output_shape->data[3], 3);
}

TEST(TfliteInferenceStage, TestResizeModel) {
  // Create stage.
  EvaluationStageConfig config = GetTfliteInferenceStageConfig();
  TfliteInferenceStage stage(config);

  // Initialize.
  EXPECT_EQ(stage.Init(), kTfLiteOk);

  // Resize.
  EXPECT_EQ(stage.ResizeInputs({{3, 8, 8, 3}}), kTfLiteOk);

  const TfLiteModelInfo* model_info = stage.GetModelInfo();
  // Verify Input
  EXPECT_EQ(model_info->inputs.size(), 1);
  const TfLiteTensor* tensor = model_info->inputs[0];
  EXPECT_EQ(tensor->type, kTfLiteUInt8);
  EXPECT_EQ(tensor->bytes, 3 * kTotalElements);
  const TfLiteIntArray* input_shape = tensor->dims;
  EXPECT_EQ(input_shape->data[0], 3);
  EXPECT_EQ(input_shape->data[1], 8);
  EXPECT_EQ(input_shape->data[2], 8);
  EXPECT_EQ(input_shape->data[3], 3);
  // Verify Output
  EXPECT_EQ(model_info->outputs.size(), 1);
  tensor = model_info->outputs[0];
  EXPECT_EQ(tensor->type, kTfLiteUInt8);
  EXPECT_EQ(tensor->bytes, 3 * kTotalElements);
  const TfLiteIntArray* output_shape = tensor->dims;
  EXPECT_EQ(output_shape->data[0], 3);
  EXPECT_EQ(output_shape->data[1], 8);
  EXPECT_EQ(output_shape->data[2], 8);
  EXPECT_EQ(output_shape->data[3], 3);
}

TEST(TfliteInferenceStage, CorrectOutput) {
  // Create stage.
  EvaluationStageConfig config = GetTfliteInferenceStageConfig();
  TfliteInferenceStage stage(config);

  // Initialize.
  EXPECT_EQ(stage.Init(), kTfLiteOk);

  // Set input data.
  uint8_t input_tensor[kTotalElements];
  SetValues(input_tensor, static_cast<uint8_t>(2));
  std::vector<void*> inputs;
  inputs.push_back(input_tensor);
  stage.SetInputs(inputs);

  // Run.
  EXPECT_EQ(stage.Run(), kTfLiteOk);

  // Verify outputs.
  uint8_t* output_tensor = static_cast<uint8_t*>(stage.GetOutputs()->at(0));
  for (int i = 0; i < kTotalElements; i++) {
    EXPECT_EQ(output_tensor[i], static_cast<uint8_t>(6));
  }

  // Verify metrics.
  EvaluationStageMetrics metrics = stage.LatestMetrics();
  EXPECT_EQ(metrics.num_runs(), 1);
  const auto& latency = metrics.process_metrics().total_latency();
  const auto max_latency = latency.max_us();
  EXPECT_GT(max_latency, 0);
  EXPECT_LT(max_latency, 1e7);
  EXPECT_LE(latency.last_us(), max_latency);
  EXPECT_LE(latency.min_us(), max_latency);
  EXPECT_GT(latency.sum_us(), max_latency);
  EXPECT_LE(latency.avg_us(), max_latency);
  EXPECT_TRUE(latency.has_std_deviation_us());
  EXPECT_EQ(
      metrics.process_metrics().tflite_inference_metrics().num_inferences(), 2);
}

TEST(TfliteInferenceStage, CustomDelegate) {
  // Create stage.
  EvaluationStageConfig config = GetTfliteInferenceStageConfig();
  TfliteInferenceStage stage(config);

  Interpreter::TfLiteDelegatePtr test_delegate = CreateNNAPIDelegate();

  // Delegate application should only work after initialization of stage.
  EXPECT_NE(stage.ApplyCustomDelegate(std::move(test_delegate)), kTfLiteOk);
  EXPECT_EQ(stage.Init(), kTfLiteOk);
  EXPECT_EQ(stage.ApplyCustomDelegate(std::move(test_delegate)), kTfLiteOk);
}

}  // namespace
}  // namespace evaluation
}  // namespace tflite
