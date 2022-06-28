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
class MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSnon_max_suppression_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSnon_max_suppression_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSnon_max_suppression_testDTcc() {
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

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/kernels/internal/reference/non_max_suppression.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/kernels/test_util.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

constexpr int kNumBoxes = 6;

void InitializeCandidates(std::vector<float>* boxes, std::vector<float>* scores,
                          bool flip_coordinates = false) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSnon_max_suppression_testDTcc mht_0(mht_0_v, 201, "", "./tensorflow/lite/kernels/internal/non_max_suppression_test.cc", "InitializeCandidates");

  if (!flip_coordinates) {
    *boxes = {
        0, 0,    1, 1,     // Box 0
        0, 0.1,  1, 1.1,   // Box 1
        0, -0.1, 1, 0.9,   // Box 2
        0, 10,   1, 11,    // Box 3
        0, 10.1, 1, 11.1,  // Box 4
        0, 100,  1, 101    // Box 5
    };
  } else {
    *boxes = {
        1, 1,     0, 0,     // Box 0
        0, 0.1,   1, 1.1,   // Box 1
        0, .9f,   1, -0.1,  // Box 2
        0, 10,    1, 11,    // Box 3
        1, 10.1f, 0, 11.1,  // Box 4
        1, 101,   0, 100    // Box 5
    };
  }
  *scores = {0.9, 0.75, 0.6, 0.95, 0.5, 0.3};
}

template <typename T>
void MatchFirstNElements(int num_elements, const std::vector<T>& test_values,
                         const std::vector<T>& reference_values) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSnon_max_suppression_testDTcc mht_1(mht_1_v, 229, "", "./tensorflow/lite/kernels/internal/non_max_suppression_test.cc", "MatchFirstNElements");

  EXPECT_LT(num_elements, test_values.size());
  EXPECT_EQ(num_elements, reference_values.size());

  for (int i = 0; i < num_elements; ++i) {
    EXPECT_EQ(test_values[i], reference_values[i]);
  }
}

TEST(NonMaxSuppression, TestZeroBoxes) {
  // Inputs
  std::vector<float> boxes(1);
  std::vector<float> scores(1);
  const float iou_threshold = 0.5;
  const float score_threshold = 0.4;
  const int max_output_size = 4;

  // Outputs
  std::vector<int> selected_indices(6);
  std::vector<float> selected_scores(6);
  int num_selected_indices = -1;

  reference_ops::NonMaxSuppression(
      boxes.data(), /**num_boxes=**/ 0, scores.data(), max_output_size,
      iou_threshold, score_threshold, /**sigma=**/ 0.0, selected_indices.data(),
      selected_scores.data(), &num_selected_indices);
  EXPECT_EQ(num_selected_indices, 0);
}

TEST(NonMaxSuppression, TestSelectFromIdenticalBoxes) {
  // Inputs
  std::vector<float> boxes(kNumBoxes * 4);
  std::vector<float> scores(kNumBoxes);
  for (int i = 0; i < kNumBoxes; ++i) {
    boxes[i * 4 + 0] = 0;
    boxes[i * 4 + 1] = 0;
    boxes[i * 4 + 2] = 1;
    boxes[i * 4 + 3] = 1;
    scores[i] = 0.75;
  }
  const float iou_threshold = 0.5;
  float score_threshold = 0.5;
  const int max_output_size = kNumBoxes;

  // Outputs
  std::vector<int> selected_indices(6);
  std::vector<float> selected_scores(6);
  int num_selected_indices = -1;

  reference_ops::NonMaxSuppression(
      boxes.data(), kNumBoxes, scores.data(), max_output_size, iou_threshold,
      score_threshold, /**sigma=**/ 0.0, selected_indices.data(),
      selected_scores.data(), &num_selected_indices);
  EXPECT_EQ(num_selected_indices, 1);
  MatchFirstNElements(1, selected_scores, {.75});

  score_threshold = 0.95;
  reference_ops::NonMaxSuppression(
      boxes.data(), kNumBoxes, scores.data(), max_output_size, iou_threshold,
      score_threshold, /**sigma=**/ 0.0, selected_indices.data(),
      selected_scores.data(), &num_selected_indices);
  EXPECT_EQ(num_selected_indices, 0);
}

TEST(NonMaxSuppression, TestSelectFromThreeClustersWithZeroScoreThreshold) {
  // Inputs
  std::vector<float> boxes;
  std::vector<float> scores;
  InitializeCandidates(&boxes, &scores);
  const float iou_threshold = 0.5;
  int max_output_size;

  // Outputs
  std::vector<int> selected_indices(6);
  std::vector<float> selected_scores(6);
  int num_selected_indices = -1;

  // Test a large max_output_size.
  max_output_size = 100;
  reference_ops::NonMaxSuppression(
      boxes.data(), kNumBoxes, scores.data(), max_output_size, iou_threshold,
      /**score_threshold=**/ 0.0, /**sigma=**/ 0.0, selected_indices.data(),
      selected_scores.data(), &num_selected_indices);
  EXPECT_EQ(num_selected_indices, 3);
  MatchFirstNElements(3, selected_indices, {3, 0, 5});
  MatchFirstNElements(3, selected_scores, {0.95, 0.9, 0.3});

  // Smaller max_output_size.
  max_output_size = 2;
  reference_ops::NonMaxSuppression(
      boxes.data(), kNumBoxes, scores.data(), max_output_size, iou_threshold,
      /**score_threshold=**/ 0.0, /**sigma=**/ 0.0, selected_indices.data(),
      selected_scores.data(), &num_selected_indices);
  EXPECT_EQ(num_selected_indices, max_output_size);
  MatchFirstNElements(max_output_size, selected_indices, {3, 0});
  MatchFirstNElements(max_output_size, selected_scores, {0.95, 0.9});

  // max_output_size = 0.
  max_output_size = 0;
  reference_ops::NonMaxSuppression(
      boxes.data(), kNumBoxes, scores.data(), max_output_size, iou_threshold,
      /**score_threshold=**/ 0.0, /**sigma=**/ 0.0, selected_indices.data(),
      selected_scores.data(), &num_selected_indices);
  EXPECT_EQ(num_selected_indices, 0);
}

TEST(NonMaxSuppression, TestSelectFromThreeClustersWithScoreThreshold) {
  // Inputs
  std::vector<float> boxes;
  std::vector<float> scores;
  InitializeCandidates(&boxes, &scores);
  const float iou_threshold = 0.5;
  const float score_threshold = 0.4;
  int max_output_size;

  // Outputs
  std::vector<int> selected_indices(6);
  std::vector<float> selected_scores(6);
  int num_selected_indices = -1;

  // Test a large max_output_size.
  max_output_size = 100;
  reference_ops::NonMaxSuppression(
      boxes.data(), kNumBoxes, scores.data(), max_output_size, iou_threshold,
      score_threshold, /**sigma=**/ 0.0, selected_indices.data(),
      selected_scores.data(), &num_selected_indices);
  EXPECT_EQ(num_selected_indices, 2);
  MatchFirstNElements(2, selected_indices, {3, 0});
  MatchFirstNElements(2, selected_scores, {0.95, 0.9});

  // max_output_size = 1.
  max_output_size = 1;
  reference_ops::NonMaxSuppression(
      boxes.data(), kNumBoxes, scores.data(), max_output_size, iou_threshold,
      score_threshold, /**sigma=**/ 0.0, selected_indices.data(),
      selected_scores.data(), &num_selected_indices);
  EXPECT_EQ(num_selected_indices, 1);
  MatchFirstNElements(1, selected_indices, {3});
  MatchFirstNElements(1, selected_scores, {0.95});
}

// This flips the (y1, x1) & (y2, x2) corners for each box. The output should
// match what we get without flipping.
TEST(NonMaxSuppression, TestSelectFromThreeClustersWithFlippedCoordinates) {
  // Inputs
  std::vector<float> boxes;
  std::vector<float> scores;
  InitializeCandidates(&boxes, &scores, /**flipped_coordinates=**/ true);
  const float iou_threshold = 0.5;
  const float score_threshold = 0.4;
  const int max_output_size = 3;

  // Outputs
  std::vector<int> selected_indices(6);
  std::vector<float> selected_scores(6);
  int num_selected_indices = -1;

  // Test a large max_output_size.
  reference_ops::NonMaxSuppression(
      boxes.data(), kNumBoxes, scores.data(), max_output_size, iou_threshold,
      score_threshold, /**sigma=**/ 0.0, selected_indices.data(),
      selected_scores.data(), &num_selected_indices);
  EXPECT_EQ(num_selected_indices, 2);
  MatchFirstNElements(2, selected_indices, {3, 0});
  MatchFirstNElements(2, selected_scores, {0.95, 0.9});

  // score_threshold = 0.
  reference_ops::NonMaxSuppression(
      boxes.data(), kNumBoxes, scores.data(), max_output_size, iou_threshold,
      /**score_threshold=**/ 0.0, /**sigma=**/ 0.0, selected_indices.data(),
      selected_scores.data(), &num_selected_indices);
  EXPECT_EQ(num_selected_indices, 3);
  MatchFirstNElements(3, selected_indices, {3, 0, 5});
  MatchFirstNElements(3, selected_scores, {0.95, 0.9, 0.3});
}

TEST(NonMaxSuppression, TestIoUThresholdBoundaryCases) {
  // Inputs
  std::vector<float> boxes;
  std::vector<float> scores;
  InitializeCandidates(&boxes, &scores);
  const float score_threshold = 0.4;
  const int max_output_size = 4;

  // Outputs
  std::vector<int> selected_indices(6);
  std::vector<float> selected_scores(6);
  int num_selected_indices = -1;

  // IoU threshold is zero. Only one index should get selected.
  reference_ops::NonMaxSuppression(
      boxes.data(), kNumBoxes, scores.data(), max_output_size,
      /**iou_threshold=**/ 0.0, score_threshold, /**sigma=**/ 0.0,
      selected_indices.data(), selected_scores.data(), &num_selected_indices);
  EXPECT_EQ(num_selected_indices, 1);
  MatchFirstNElements(1, selected_indices, {3});
  MatchFirstNElements(1, selected_scores, {0.95});

  // IoU threshold too high. max_output_size number of indices should be
  // selected.
  reference_ops::NonMaxSuppression(
      boxes.data(), kNumBoxes, scores.data(), max_output_size,
      /**iou_threshold=**/ 0.9999,
      /**score_threshold=**/ 0.0, /**sigma=**/ 0.0, selected_indices.data(),
      selected_scores.data(), &num_selected_indices);
  EXPECT_EQ(num_selected_indices, max_output_size);
  MatchFirstNElements(max_output_size, selected_indices, {3, 0, 1, 2});
  MatchFirstNElements(max_output_size, selected_scores, {0.95, 0.9, 0.75, 0.6});
}

TEST(NonMaxSuppression, TestSelectFromThreeClustersWithSoftNMS) {
  // Inputs
  std::vector<float> boxes;
  std::vector<float> scores;
  InitializeCandidates(&boxes, &scores);
  const float iou_threshold = 1.0;
  float score_threshold = 0.0;
  const float soft_nms_sigma = 0.5;
  int max_output_size = 6;

  // Outputs
  std::vector<int> selected_indices(6);
  std::vector<float> selected_scores(6);
  int num_selected_indices = -1;

  reference_ops::NonMaxSuppression(
      boxes.data(), kNumBoxes, scores.data(), max_output_size, iou_threshold,
      score_threshold, soft_nms_sigma, selected_indices.data(),
      selected_scores.data(), &num_selected_indices);
  EXPECT_EQ(num_selected_indices, 6);
  // Box 0 soft-suppresses box 1, but not enough to cause it to fall under
  // `score_threshold` (which is 0.0) or the score of box 5, so in this test,
  // box 1 ends up being selected before box 5.
  EXPECT_THAT(selected_indices, ElementsAreArray({3, 0, 1, 5, 4, 2}));
  EXPECT_THAT(selected_scores,
              ElementsAreArray(
                  ArrayFloatNear({0.95, 0.9, 0.384, 0.3, 0.256, 0.197}, 1e-3)));

  score_threshold = 0.299;
  reference_ops::NonMaxSuppression(
      boxes.data(), kNumBoxes, scores.data(), max_output_size, iou_threshold,
      score_threshold, soft_nms_sigma, selected_indices.data(),
      selected_scores.data(), &num_selected_indices);
  EXPECT_EQ(num_selected_indices, 4);
  MatchFirstNElements(4, selected_indices, {3, 0, 1, 5});
}

TEST(NonMaxSuppression, TestNullSelectedScoresOutput) {
  // Inputs
  std::vector<float> boxes;
  std::vector<float> scores;
  InitializeCandidates(&boxes, &scores);
  const float iou_threshold = 0.5;
  const float score_threshold = 0.4;
  int max_output_size;

  // Outputs
  std::vector<int> selected_indices(6);
  int num_selected_indices = -1;

  max_output_size = 100;
  reference_ops::NonMaxSuppression(
      boxes.data(), kNumBoxes, scores.data(), max_output_size, iou_threshold,
      score_threshold, /**sigma=**/ 0.0, selected_indices.data(),
      /**selected_scores=**/ nullptr, &num_selected_indices);
  EXPECT_EQ(num_selected_indices, 2);
}
}  // namespace
}  // namespace tflite
