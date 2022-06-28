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
class MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPStopk_accuracy_eval_stage_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPStopk_accuracy_eval_stage_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPStopk_accuracy_eval_stage_testDTcc() {
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
#include <string>

#include <gtest/gtest.h>
#include "tensorflow/lite/tools/evaluation/proto/evaluation_config.pb.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_stages.pb.h"

namespace tflite {
namespace evaluation {
namespace {

constexpr char kTopkAccuracyEvalStageName[] = "topk_accuracy_eval_stage";
constexpr int kNumCategories = 1001;

EvaluationStageConfig GetTopkAccuracyEvalStageConfig() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPStopk_accuracy_eval_stage_testDTcc mht_0(mht_0_v, 200, "", "./tensorflow/lite/tools/evaluation/stages/topk_accuracy_eval_stage_test.cc", "GetTopkAccuracyEvalStageConfig");

  EvaluationStageConfig config;
  config.set_name(kTopkAccuracyEvalStageName);
  auto* params =
      config.mutable_specification()->mutable_topk_accuracy_eval_params();
  params->set_k(5);
  return config;
}

template <typename T>
T* ResetOutputArray(T array[]) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPStopk_accuracy_eval_stage_testDTcc mht_1(mht_1_v, 213, "", "./tensorflow/lite/tools/evaluation/stages/topk_accuracy_eval_stage_test.cc", "ResetOutputArray");

  for (int i = 0; i < kNumCategories; i++) {
    array[i] = 0;
  }
  return array;
}

std::vector<std::string> CreateGroundTruthLabels() {
  std::vector<std::string> ground_truth_labels;
  ground_truth_labels.reserve(kNumCategories);
  for (int i = 0; i < kNumCategories; i++) {
    ground_truth_labels.push_back(std::to_string(i));
  }
  return ground_truth_labels;
}

TEST(TopkAccuracyEvalStage, NoInitializers) {
  // Create stage.
  EvaluationStageConfig config = GetTopkAccuracyEvalStageConfig();
  TopkAccuracyEvalStage stage = TopkAccuracyEvalStage(config);

  EXPECT_EQ(stage.Init(), kTfLiteError);
}

TEST(TopkAccuracyEvalStage, NoK) {
  // Create stage.
  EvaluationStageConfig config = GetTopkAccuracyEvalStageConfig();
  config.mutable_specification()
      ->mutable_topk_accuracy_eval_params()
      ->clear_k();
  TopkAccuracyEvalStage stage = TopkAccuracyEvalStage(config);

  // Initialize.
  std::vector<std::string> ground_truth_labels = CreateGroundTruthLabels();
  TfLiteIntArray* model_output_shape = TfLiteIntArrayCreate(2);
  model_output_shape->data[0] = 1;
  model_output_shape->data[1] = kNumCategories;
  TfLiteType model_output_type = kTfLiteFloat32;
  stage.SetTaskInfo(ground_truth_labels, model_output_type, model_output_shape);
  EXPECT_EQ(stage.Init(), kTfLiteError);
  TfLiteIntArrayFree(model_output_shape);
}

TEST(TopkAccuracyEvalStage, NoGroundTruthLabels) {
  // Create stage.
  EvaluationStageConfig config = GetTopkAccuracyEvalStageConfig();
  TopkAccuracyEvalStage stage = TopkAccuracyEvalStage(config);

  // Initialize.
  std::vector<std::string> ground_truth_labels = {};
  TfLiteIntArray* model_output_shape = TfLiteIntArrayCreate(2);
  model_output_shape->data[0] = 1;
  model_output_shape->data[1] = kNumCategories;
  TfLiteType model_output_type = kTfLiteFloat32;
  stage.SetTaskInfo(ground_truth_labels, model_output_type, model_output_shape);
  EXPECT_EQ(stage.Init(), kTfLiteError);
  TfLiteIntArrayFree(model_output_shape);
}

TEST(TopkAccuracyEvalStage, KTooLarge) {
  // Create stage.
  EvaluationStageConfig config = GetTopkAccuracyEvalStageConfig();
  config.mutable_specification()->mutable_topk_accuracy_eval_params()->set_k(
      10000);
  TopkAccuracyEvalStage stage = TopkAccuracyEvalStage(config);

  // Initialize.
  std::vector<std::string> ground_truth_labels = CreateGroundTruthLabels();
  TfLiteIntArray* model_output_shape = TfLiteIntArrayCreate(2);
  model_output_shape->data[0] = 1;
  model_output_shape->data[1] = kNumCategories;
  TfLiteType model_output_type = kTfLiteFloat32;
  stage.SetTaskInfo(ground_truth_labels, model_output_type, model_output_shape);
  EXPECT_EQ(stage.Init(), kTfLiteError);
  TfLiteIntArrayFree(model_output_shape);
}

TEST(TopkAccuracyEvalStage, WeirdModelOutputShape) {
  // Create stage.
  EvaluationStageConfig config = GetTopkAccuracyEvalStageConfig();
  TopkAccuracyEvalStage stage = TopkAccuracyEvalStage(config);

  // Initialize.
  std::vector<std::string> ground_truth_labels = CreateGroundTruthLabels();
  TfLiteIntArray* model_output_shape = TfLiteIntArrayCreate(2);
  model_output_shape->data[0] = 1;
  model_output_shape->data[1] = kNumCategories + 1;
  TfLiteType model_output_type = kTfLiteFloat32;
  stage.SetTaskInfo(ground_truth_labels, model_output_type, model_output_shape);
  EXPECT_EQ(stage.Init(), kTfLiteError);
  TfLiteIntArrayFree(model_output_shape);
}

TEST(TopkAccuracyEvalStage, UnsupportedModelOutputType) {
  // Create stage.
  EvaluationStageConfig config = GetTopkAccuracyEvalStageConfig();
  TopkAccuracyEvalStage stage = TopkAccuracyEvalStage(config);

  // Initialize.
  std::vector<std::string> ground_truth_labels = CreateGroundTruthLabels();
  TfLiteIntArray* model_output_shape = TfLiteIntArrayCreate(2);
  model_output_shape->data[0] = 1;
  model_output_shape->data[1] = kNumCategories + 1;
  TfLiteType model_output_type = kTfLiteComplex64;
  stage.SetTaskInfo(ground_truth_labels, model_output_type, model_output_shape);
  EXPECT_EQ(stage.Init(), kTfLiteError);
  TfLiteIntArrayFree(model_output_shape);
}

TEST(TopkAccuracyEvalStage, NoInputs) {
  // Create stage.
  EvaluationStageConfig config = GetTopkAccuracyEvalStageConfig();
  TopkAccuracyEvalStage stage = TopkAccuracyEvalStage(config);
  // Initialize.
  std::vector<std::string> ground_truth_labels = CreateGroundTruthLabels();
  TfLiteIntArray* model_output_shape = TfLiteIntArrayCreate(2);
  model_output_shape->data[0] = 1;
  model_output_shape->data[1] = kNumCategories;
  TfLiteType model_output_type = kTfLiteFloat32;
  stage.SetTaskInfo(ground_truth_labels, model_output_type, model_output_shape);
  EXPECT_EQ(stage.Init(), kTfLiteOk);
  TfLiteIntArrayFree(model_output_shape);

  EXPECT_EQ(stage.Run(), kTfLiteError);
}

TEST(TopkAccuracyEvalStage, InvalidGroundTruth) {
  // Create stage.
  EvaluationStageConfig config = GetTopkAccuracyEvalStageConfig();
  TopkAccuracyEvalStage stage = TopkAccuracyEvalStage(config);
  // Initialize.
  std::vector<std::string> ground_truth_labels = CreateGroundTruthLabels();
  TfLiteIntArray* model_output_shape = TfLiteIntArrayCreate(2);
  model_output_shape->data[0] = 1;
  model_output_shape->data[1] = kNumCategories;
  TfLiteType model_output_type = kTfLiteFloat32;
  stage.SetTaskInfo(ground_truth_labels, model_output_type, model_output_shape);
  EXPECT_EQ(stage.Init(), kTfLiteOk);
  TfLiteIntArrayFree(model_output_shape);

  float array[kNumCategories];
  float* tensor = ResetOutputArray(array);
  tensor[0] = 0.8;
  stage.SetEvalInputs(tensor, /*ground_truth_label=*/nullptr);
  EXPECT_EQ(stage.Run(), kTfLiteError);
}

TEST(TopkAccuracyEvalStage, FloatTest_CorrectLabelsAtLastIndices) {
  // Create stage.
  EvaluationStageConfig config = GetTopkAccuracyEvalStageConfig();
  TopkAccuracyEvalStage stage = TopkAccuracyEvalStage(config);
  // Initialize.
  std::vector<std::string> ground_truth_labels = CreateGroundTruthLabels();
  TfLiteIntArray* model_output_shape = TfLiteIntArrayCreate(2);
  model_output_shape->data[0] = 1;
  model_output_shape->data[1] = kNumCategories;
  TfLiteType model_output_type = kTfLiteFloat32;
  stage.SetTaskInfo(ground_truth_labels, model_output_type, model_output_shape);
  EXPECT_EQ(stage.Init(), kTfLiteOk);
  TfLiteIntArrayFree(model_output_shape);

  float array[kNumCategories];

  // The ground truth is index 0, but it is 5th most likely based on model's
  // output.
  float* tensor = ResetOutputArray(array);
  tensor[4] = 0.9;
  tensor[3] = 0.8;
  tensor[2] = 0.7;
  tensor[1] = 0.6;
  tensor[0] = 0.5;
  std::string ground_truth = "0";
  stage.SetEvalInputs(tensor, &ground_truth);
  EXPECT_EQ(stage.Run(), kTfLiteOk);
  EvaluationStageMetrics metrics = stage.LatestMetrics();
  EXPECT_EQ(1, metrics.num_runs());
  auto accuracy_metrics = metrics.process_metrics().topk_accuracy_metrics();
  // Only top-5 count is 1.0, rest are 0.0
  EXPECT_FLOAT_EQ(1.0, accuracy_metrics.topk_accuracies(4));
  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(0.0, accuracy_metrics.topk_accuracies(i));
  }

  // The ground truth is index 1, but it is 4th highest based on model's output.
  ground_truth = "1";
  stage.SetEvalInputs(tensor, &ground_truth);
  EXPECT_EQ(stage.Run(), kTfLiteOk);
  metrics = stage.LatestMetrics();
  EXPECT_EQ(2, metrics.num_runs());
  accuracy_metrics = metrics.process_metrics().topk_accuracy_metrics();
  // 1/2 images had the currect output in top-4, 2/2 has currect output in
  // top-5.
  EXPECT_FLOAT_EQ(1.0, accuracy_metrics.topk_accuracies(4));
  EXPECT_FLOAT_EQ(0.5, accuracy_metrics.topk_accuracies(3));
  for (int i = 0; i < 3; ++i) {
    EXPECT_FLOAT_EQ(0.0, accuracy_metrics.topk_accuracies(i));
  }
}

class CorrectTopkAccuracyEvalTest : public ::testing::Test {
 protected:
  template <typename T>
  void VerifyCorrectBehaviorForType(T ground_truth_0_value,
                                    T ground_truth_1_value,
                                    TfLiteType model_output_type) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPStopk_accuracy_eval_stage_testDTcc mht_2(mht_2_v, 420, "", "./tensorflow/lite/tools/evaluation/stages/topk_accuracy_eval_stage_test.cc", "VerifyCorrectBehaviorForType");

    // Create stage.
    EvaluationStageConfig config = GetTopkAccuracyEvalStageConfig();
    TopkAccuracyEvalStage stage = TopkAccuracyEvalStage(config);
    // Initialize.
    std::vector<std::string> ground_truth_labels = CreateGroundTruthLabels();
    TfLiteIntArray* model_output_shape = TfLiteIntArrayCreate(2);
    model_output_shape->data[0] = 1;
    model_output_shape->data[1] = kNumCategories;
    stage.SetTaskInfo(ground_truth_labels, model_output_type,
                      model_output_shape);
    EXPECT_EQ(stage.Init(), kTfLiteOk);
    TfLiteIntArrayFree(model_output_shape);

    // Pre-run state.
    EvaluationStageMetrics metrics = stage.LatestMetrics();
    EXPECT_EQ(0, metrics.num_runs());
    auto accuracy_metrics = metrics.process_metrics().topk_accuracy_metrics();
    EXPECT_EQ(0, accuracy_metrics.topk_accuracies_size());

    T array[kNumCategories];

    // First image was correctly identified as "0".
    T* tensor = ResetOutputArray(array);
    tensor[0] = ground_truth_0_value;
    std::string ground_truth = "0";
    stage.SetEvalInputs(tensor, &ground_truth);
    EXPECT_EQ(stage.Run(), kTfLiteOk);
    metrics = stage.LatestMetrics();
    EXPECT_EQ(1, metrics.num_runs());
    accuracy_metrics = metrics.process_metrics().topk_accuracy_metrics();
    for (int i = 0; i < accuracy_metrics.topk_accuracies_size(); ++i) {
      EXPECT_FLOAT_EQ(1.0, accuracy_metrics.topk_accuracies(i));
    }

    // Second image was also correctly identified as "1".
    // Hence, for the second image as well, the top output ("1") was correct.
    tensor[1] = ground_truth_1_value;
    ground_truth = "1";
    stage.SetEvalInputs(tensor, &ground_truth);
    EXPECT_EQ(stage.Run(), kTfLiteOk);
    metrics = stage.LatestMetrics();
    EXPECT_EQ(2, metrics.num_runs());
    accuracy_metrics = metrics.process_metrics().topk_accuracy_metrics();
    for (int i = 0; i < accuracy_metrics.topk_accuracies_size(); ++i) {
      EXPECT_FLOAT_EQ(1.0, accuracy_metrics.topk_accuracies(i));
    }
  }
};

TEST_F(CorrectTopkAccuracyEvalTest, FloatTest) {
  VerifyCorrectBehaviorForType(static_cast<float>(0.8), static_cast<float>(0.9),
                               kTfLiteFloat32);
}

TEST_F(CorrectTopkAccuracyEvalTest, Int8Test) {
  VerifyCorrectBehaviorForType(static_cast<int8_t>(1), static_cast<int8_t>(2),
                               kTfLiteInt8);
}

TEST_F(CorrectTopkAccuracyEvalTest, UInt8Test) {
  VerifyCorrectBehaviorForType(static_cast<uint8_t>(1), static_cast<uint8_t>(2),
                               kTfLiteUInt8);
}

}  // namespace
}  // namespace evaluation
}  // namespace tflite
