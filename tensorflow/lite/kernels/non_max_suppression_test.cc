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
class MHTracer_DTPStensorflowPSlitePSkernelsPSnon_max_suppression_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSnon_max_suppression_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSnon_max_suppression_testDTcc() {
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

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include <initializer_list>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

class BaseNMSOp : public SingleOpModel {
 public:
  void SetScores(std::initializer_list<float> data) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSnon_max_suppression_testDTcc mht_0(mht_0_v, 200, "", "./tensorflow/lite/kernels/non_max_suppression_test.cc", "SetScores");

    PopulateTensor(input_scores_, data);
  }

  void SetMaxOutputSize(int max_output_size) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSnon_max_suppression_testDTcc mht_1(mht_1_v, 207, "", "./tensorflow/lite/kernels/non_max_suppression_test.cc", "SetMaxOutputSize");

    PopulateTensor(input_max_output_size_, {max_output_size});
  }

  void SetScoreThreshold(float score_threshold) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSnon_max_suppression_testDTcc mht_2(mht_2_v, 214, "", "./tensorflow/lite/kernels/non_max_suppression_test.cc", "SetScoreThreshold");

    PopulateTensor(input_score_threshold_, {score_threshold});
  }

  std::vector<int> GetSelectedIndices() {
    return ExtractVector<int>(output_selected_indices_);
  }

  std::vector<float> GetSelectedScores() {
    return ExtractVector<float>(output_selected_scores_);
  }

  std::vector<int> GetNumSelectedIndices() {
    return ExtractVector<int>(output_num_selected_indices_);
  }

 protected:
  int input_boxes_;
  int input_scores_;
  int input_max_output_size_;
  int input_iou_threshold_;
  int input_score_threshold_;
  int input_sigma_;

  int output_selected_indices_;
  int output_selected_scores_;
  int output_num_selected_indices_;
};

class NonMaxSuppressionV4OpModel : public BaseNMSOp {
 public:
  explicit NonMaxSuppressionV4OpModel(const float iou_threshold,
                                      const bool static_shaped_outputs,
                                      const int max_output_size = -1) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSnon_max_suppression_testDTcc mht_3(mht_3_v, 250, "", "./tensorflow/lite/kernels/non_max_suppression_test.cc", "NonMaxSuppressionV4OpModel");

    const int num_boxes = 6;
    input_boxes_ = AddInput({TensorType_FLOAT32, {num_boxes, 4}});
    input_scores_ = AddInput({TensorType_FLOAT32, {num_boxes}});
    if (static_shaped_outputs) {
      input_max_output_size_ =
          AddConstInput(TensorType_INT32, {max_output_size});
    } else {
      input_max_output_size_ = AddInput(TensorType_INT32);
    }
    input_iou_threshold_ = AddConstInput(TensorType_FLOAT32, {iou_threshold});
    input_score_threshold_ = AddInput({TensorType_FLOAT32, {}});

    output_selected_indices_ = AddOutput(TensorType_INT32);

    output_num_selected_indices_ = AddOutput(TensorType_INT32);

    SetBuiltinOp(BuiltinOperator_NON_MAX_SUPPRESSION_V4,
                 BuiltinOptions_NonMaxSuppressionV4Options,
                 CreateNonMaxSuppressionV4Options(builder_).Union());
    BuildInterpreter({GetShape(input_boxes_), GetShape(input_scores_),
                      GetShape(input_max_output_size_),
                      GetShape(input_iou_threshold_),
                      GetShape(input_score_threshold_)});

    // Default data.
    PopulateTensor<float>(input_boxes_, {
                                            1, 1,     0, 0,     // Box 0
                                            0, 0.1,   1, 1.1,   // Box 1
                                            0, .9f,   1, -0.1,  // Box 2
                                            0, 10,    1, 11,    // Box 3
                                            1, 10.1f, 0, 11.1,  // Box 4
                                            1, 101,   0, 100    // Box 5
                                        });
  }
};

TEST(NonMaxSuppressionV4OpModel, TestOutput) {
  NonMaxSuppressionV4OpModel nms(/**iou_threshold=**/ 0.5,
                                 /**static_shaped_outputs=**/ true,
                                 /**max_output_size=**/ 6);
  nms.SetScores({0.9, 0.75, 0.6, 0.95, 0.5, 0.3});
  nms.SetScoreThreshold(0.4);
  ASSERT_EQ(nms.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(nms.GetNumSelectedIndices(), ElementsAreArray({2}));
  EXPECT_THAT(nms.GetSelectedIndices(), ElementsAreArray({3, 0, 0, 0, 0, 0}));

  nms.SetScoreThreshold(0.99);
  ASSERT_EQ(nms.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(nms.GetNumSelectedIndices(), ElementsAreArray({0}));
  // The first two indices should be zeroed-out.
  EXPECT_THAT(nms.GetSelectedIndices(), ElementsAreArray({0, 0, 0, 0, 0, 0}));
}

TEST(NonMaxSuppressionV4OpModel, TestDynamicOutput) {
  NonMaxSuppressionV4OpModel nms(/**iou_threshold=**/ 0.5,
                                 /**static_shaped_outputs=**/ false);
  nms.SetScores({0.9, 0.75, 0.6, 0.95, 0.5, 0.3});
  nms.SetScoreThreshold(0.4);

  nms.SetMaxOutputSize(1);
  ASSERT_EQ(nms.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(nms.GetNumSelectedIndices(), ElementsAreArray({1}));
  EXPECT_THAT(nms.GetSelectedIndices(), ElementsAreArray({3}));

  nms.SetMaxOutputSize(2);
  ASSERT_EQ(nms.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(nms.GetNumSelectedIndices(), ElementsAreArray({2}));
  EXPECT_THAT(nms.GetSelectedIndices(), ElementsAreArray({3, 0}));

  nms.SetScoreThreshold(0.99);
  ASSERT_EQ(nms.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(nms.GetNumSelectedIndices(), ElementsAreArray({0}));
  EXPECT_THAT(nms.GetSelectedIndices(), ElementsAreArray({0, 0}));
}

TEST(NonMaxSuppressionV4OpModel, TestOutputWithZeroMaxOutput) {
  NonMaxSuppressionV4OpModel nms(/**iou_threshold=**/ 0.5,
                                 /**static_shaped_outputs=**/ true,
                                 /**max_output_size=**/ 0);
  nms.SetScores({0.9, 0.75, 0.6, 0.95, 0.5, 0.3});
  nms.SetScoreThreshold(0.4);
  ASSERT_EQ(nms.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(nms.GetNumSelectedIndices(), ElementsAreArray({0}));
}

class NonMaxSuppressionV5OpModel : public BaseNMSOp {
 public:
  explicit NonMaxSuppressionV5OpModel(const float iou_threshold,
                                      const float sigma,
                                      const bool static_shaped_outputs,
                                      const int max_output_size = -1) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSnon_max_suppression_testDTcc mht_4(mht_4_v, 344, "", "./tensorflow/lite/kernels/non_max_suppression_test.cc", "NonMaxSuppressionV5OpModel");

    const int num_boxes = 6;
    input_boxes_ = AddInput({TensorType_FLOAT32, {num_boxes, 4}});
    input_scores_ = AddInput({TensorType_FLOAT32, {num_boxes}});
    if (static_shaped_outputs) {
      input_max_output_size_ =
          AddConstInput(TensorType_INT32, {max_output_size});
    } else {
      input_max_output_size_ = AddInput(TensorType_INT32);
    }
    input_iou_threshold_ = AddConstInput(TensorType_FLOAT32, {iou_threshold});
    input_score_threshold_ = AddInput({TensorType_FLOAT32, {}});
    input_sigma_ = AddConstInput(TensorType_FLOAT32, {sigma});

    output_selected_indices_ = AddOutput(TensorType_INT32);
    output_selected_scores_ = AddOutput(TensorType_FLOAT32);
    output_num_selected_indices_ = AddOutput(TensorType_INT32);

    SetBuiltinOp(BuiltinOperator_NON_MAX_SUPPRESSION_V5,
                 BuiltinOptions_NonMaxSuppressionV5Options,
                 CreateNonMaxSuppressionV5Options(builder_).Union());

    BuildInterpreter(
        {GetShape(input_boxes_), GetShape(input_scores_),
         GetShape(input_max_output_size_), GetShape(input_iou_threshold_),
         GetShape(input_score_threshold_), GetShape(input_sigma_)});

    // Default data.
    PopulateTensor<float>(input_boxes_, {
                                            1, 1,     0, 0,     // Box 0
                                            0, 0.1,   1, 1.1,   // Box 1
                                            0, .9f,   1, -0.1,  // Box 2
                                            0, 10,    1, 11,    // Box 3
                                            1, 10.1f, 0, 11.1,  // Box 4
                                            1, 101,   0, 100    // Box 5
                                        });
  }
};

TEST(NonMaxSuppressionV5OpModel, TestOutput) {
  NonMaxSuppressionV5OpModel nms(/**iou_threshold=**/ 0.5,
                                 /**sigma=**/ 0.5,
                                 /**static_shaped_outputs=**/ true,
                                 /**max_output_size=**/ 6);
  nms.SetScores({0.9, 0.75, 0.6, 0.95, 0.5, 0.3});
  nms.SetScoreThreshold(0.0);
  ASSERT_EQ(nms.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(nms.GetNumSelectedIndices(), ElementsAreArray({3}));
  EXPECT_THAT(nms.GetSelectedIndices(), ElementsAreArray({3, 0, 5, 0, 0, 0}));
  EXPECT_THAT(nms.GetSelectedScores(),
              ElementsAreArray({0.95, 0.9, 0.3, 0.0, 0.0, 0.0}));

  // No candidate gets selected. But the outputs should be zeroed out.
  nms.SetScoreThreshold(0.99);
  ASSERT_EQ(nms.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(nms.GetNumSelectedIndices(), ElementsAreArray({0}));
  EXPECT_THAT(nms.GetSelectedIndices(), ElementsAreArray({0, 0, 0, 0, 0, 0}));
  EXPECT_THAT(nms.GetSelectedScores(),
              ElementsAreArray({0.0, 0.0, 0.0, 0.0, 0.0, 0.0}));
}

TEST(NonMaxSuppressionV5OpModel, TestDynamicOutput) {
  NonMaxSuppressionV5OpModel nms(/**iou_threshold=**/ 0.5,
                                 /**sigma=**/ 0.5,
                                 /**static_shaped_outputs=**/ false,
                                 /**max_output_size=**/ 6);
  nms.SetScores({0.9, 0.75, 0.6, 0.95, 0.5, 0.3});
  nms.SetScoreThreshold(0.0);

  nms.SetMaxOutputSize(2);
  ASSERT_EQ(nms.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(nms.GetNumSelectedIndices(), ElementsAreArray({2}));
  EXPECT_THAT(nms.GetSelectedIndices(), ElementsAreArray({3, 0}));
  EXPECT_THAT(nms.GetSelectedScores(), ElementsAreArray({0.95, 0.9}));

  nms.SetMaxOutputSize(1);
  ASSERT_EQ(nms.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(nms.GetNumSelectedIndices(), ElementsAreArray({1}));
  EXPECT_THAT(nms.GetSelectedIndices(), ElementsAreArray({3}));
  EXPECT_THAT(nms.GetSelectedScores(), ElementsAreArray({0.95}));

  nms.SetMaxOutputSize(3);
  ASSERT_EQ(nms.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(nms.GetNumSelectedIndices(), ElementsAreArray({3}));
  EXPECT_THAT(nms.GetSelectedIndices(), ElementsAreArray({3, 0, 5}));
  EXPECT_THAT(nms.GetSelectedScores(), ElementsAreArray({0.95, 0.9, 0.3}));

  // No candidate gets selected. But the outputs should be zeroed out.
  nms.SetScoreThreshold(0.99);
  ASSERT_EQ(nms.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(nms.GetNumSelectedIndices(), ElementsAreArray({0}));
  EXPECT_THAT(nms.GetSelectedIndices(), ElementsAreArray({0, 0, 0}));
  EXPECT_THAT(nms.GetSelectedScores(), ElementsAreArray({0.0, 0.0, 0.0}));
}
}  // namespace
}  // namespace tflite
