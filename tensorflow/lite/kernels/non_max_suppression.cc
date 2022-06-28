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
class MHTracer_DTPStensorflowPSlitePSkernelsPSnon_max_suppressionDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSnon_max_suppressionDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSnon_max_suppressionDTcc() {
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
#include "tensorflow/lite/kernels/internal/reference/non_max_suppression.h"

#include <initializer_list>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace non_max_suppression {

// Boxes in format [y1, x1, y2, x2]. Shape: [num_boxes, 4]
// Type: Float.
constexpr int kInputTensorBoxes = 0;
// Shape: [num_boxes]
// Type: Float.
constexpr int kInputTensorScores = 1;
// Max number of boxes to output. Actual output can be smaller.
// The output tensors (indices/scores) are of this length.
// Type: Int32.
constexpr int kInputTensorMaxOutputSize = 2;
// Type: Float.
constexpr int kInputTensorIouThreshold = 3;
// Type: Float.
constexpr int kInputTensorScoreThreshold = 4;
// Only applies to NON_MAX_SUPPRESSION_V5.
// Type: Float.
constexpr int kInputTensorSigma = 5;

// Indices of selected boxes. Shape: [num_selected_indices]
// Type: Int32.
constexpr int kNMSOutputTensorSelectedIndices = 0;
// Type: Int32.
constexpr int kNMSOutputTensorNumSelectedIndices = 1;

// Indices of selected boxes. Shape: [num_selected_indices]
// Type: Int32.
constexpr int kSoftNMSOutputTensorSelectedIndices = 0;
// Scores of selected boxes. Shape: [num_selected_indices]
// Type: Float.
constexpr int kSoftNMSOutputTensorSelectedScores = 1;
// Type: Int32.
constexpr int kSoftNMSOutputTensorNumSelectedIndices = 2;

TfLiteStatus SetTensorSizes(TfLiteContext* context, TfLiteTensor* tensor,
                            std::initializer_list<int> values) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSnon_max_suppressionDTcc mht_0(mht_0_v, 232, "", "./tensorflow/lite/kernels/non_max_suppression.cc", "SetTensorSizes");

  TfLiteIntArray* size = TfLiteIntArrayCreate(values.size());
  int index = 0;
  for (const auto& v : values) {
    size->data[index++] = v;
  }
  return context->ResizeTensor(context, tensor, size);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSnon_max_suppressionDTcc mht_1(mht_1_v, 244, "", "./tensorflow/lite/kernels/non_max_suppression.cc", "Prepare");

  const int num_inputs = NumInputs(node);
  const bool is_soft_nms = num_inputs == 6;
  if (num_inputs != 5 && num_inputs != 6) {
    context->ReportError(context, "Found NMS op with invalid num inputs: %d",
                         NumInputs(node));
    return kTfLiteError;
  }

  // Boxes & Scores.
  const TfLiteTensor* input_boxes;
  TF_LITE_ENSURE_OK(
      context, GetInputSafe(context, node, kInputTensorBoxes, &input_boxes));
  TF_LITE_ENSURE_EQ(context, input_boxes->type, kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, NumDimensions(input_boxes), 2);
  TF_LITE_ENSURE_EQ(context, SizeOfDimension(input_boxes, 1), 4);
  const int num_boxes = SizeOfDimension(input_boxes, 0);
  const TfLiteTensor* input_scores;
  TF_LITE_ENSURE_OK(
      context, GetInputSafe(context, node, kInputTensorScores, &input_scores));
  TF_LITE_ENSURE_EQ(context, input_scores->type, kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, NumDimensions(input_scores), 1);
  TF_LITE_ENSURE_EQ(context, num_boxes, SizeOfDimension(input_scores, 0));

  // Max output size.
  const TfLiteTensor* input_max_output_size;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensorMaxOutputSize,
                                 &input_max_output_size));
  TF_LITE_ENSURE_EQ(context, input_max_output_size->type, kTfLiteInt32);
  TF_LITE_ENSURE_EQ(context, NumDimensions(input_max_output_size), 0);
  const bool is_max_output_size_const = IsConstantTensor(input_max_output_size);
  int max_output_size_value = 0;
  if (is_max_output_size_const) {
    max_output_size_value = *GetTensorData<int>(input_max_output_size);
    TF_LITE_ENSURE(context, (max_output_size_value >= 0));
  }

  // IoU & Score thresholds.
  const TfLiteTensor* input_iou_threshold;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensorIouThreshold,
                                 &input_iou_threshold));
  TF_LITE_ENSURE_EQ(context, input_iou_threshold->type, kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, NumDimensions(input_iou_threshold), 0);
  const TfLiteTensor* input_score_threshold;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensorScoreThreshold,
                                 &input_score_threshold));
  TF_LITE_ENSURE_EQ(context, input_iou_threshold->type, kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, NumDimensions(input_score_threshold), 0);

  if (is_soft_nms) {
    const TfLiteTensor* input_sigma;
    TF_LITE_ENSURE_OK(
        context, GetInputSafe(context, node, kInputTensorSigma, &input_sigma));
    TF_LITE_ENSURE_EQ(context, input_sigma->type, kTfLiteFloat32);
    TF_LITE_ENSURE_EQ(context, NumDimensions(input_sigma), 0);

    TF_LITE_ENSURE_EQ(context, NumOutputs(node), 3);
    TfLiteTensor* output_selected_indices;
    TF_LITE_ENSURE_OK(
        context,
        GetOutputSafe(context, node, kSoftNMSOutputTensorSelectedIndices,
                      &output_selected_indices));
    output_selected_indices->type = kTfLiteInt32;
    TfLiteTensor* output_selected_scores;
    TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node,
                                             kSoftNMSOutputTensorSelectedScores,
                                             &output_selected_scores));
    output_selected_scores->type = kTfLiteFloat32;
    TfLiteTensor* output_num_selected_indices;
    TF_LITE_ENSURE_OK(
        context,
        GetOutputSafe(context, node, kSoftNMSOutputTensorNumSelectedIndices,
                      &output_num_selected_indices));
    output_num_selected_indices->type = kTfLiteInt32;
    SetTensorSizes(context, output_num_selected_indices, {});

    if (is_max_output_size_const) {
      SetTensorSizes(context, output_selected_indices, {max_output_size_value});
      SetTensorSizes(context, output_selected_scores, {max_output_size_value});
    } else {
      SetTensorToDynamic(output_selected_indices);
      SetTensorToDynamic(output_selected_scores);
    }
  } else {
    TF_LITE_ENSURE_EQ(context, NumOutputs(node), 2);
    TfLiteTensor* output_selected_indices;
    TF_LITE_ENSURE_OK(
        context, GetOutputSafe(context, node, kNMSOutputTensorSelectedIndices,
                               &output_selected_indices));
    output_selected_indices->type = kTfLiteInt32;
    TfLiteTensor* output_num_selected_indices;
    TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node,
                                             kNMSOutputTensorNumSelectedIndices,
                                             &output_num_selected_indices));
    output_num_selected_indices->type = kTfLiteInt32;
    SetTensorSizes(context, output_num_selected_indices, {});

    if (is_max_output_size_const) {
      SetTensorSizes(context, output_selected_indices, {max_output_size_value});
    } else {
      SetTensorToDynamic(output_selected_indices);
    }
  }

  return kTfLiteOk;
}

// If num_selected_indices < max_output_size, the output tensor can contain
// garbage values initially present in memory. This causes segfault in
// downstream ops such as GATHER, since one of the outputs denotes indices and
// int garbage values can be pretty large. This method zeroes-out the remaining
// values.
// NOTE: We ensure memory being reset is valid, by setting pertinent output
// tensors to max_output_size length in Prepare.
void ResetUnusedElementsToZeroes(const int max_output_size,
                                 const int num_selected_indices,
                                 int* selected_indices,
                                 float* selected_scores) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSnon_max_suppressionDTcc mht_2(mht_2_v, 367, "", "./tensorflow/lite/kernels/non_max_suppression.cc", "ResetUnusedElementsToZeroes");

  for (int i = num_selected_indices; i < max_output_size; ++i) {
    selected_indices[i] = 0;
    if (selected_scores) {
      selected_scores[i] = 0.0;
    }
  }
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSnon_max_suppressionDTcc mht_3(mht_3_v, 379, "", "./tensorflow/lite/kernels/non_max_suppression.cc", "Eval");

  const bool is_soft_nms = NumInputs(node) == 6;

  const TfLiteTensor* input_boxes;
  TF_LITE_ENSURE_OK(
      context, GetInputSafe(context, node, kInputTensorBoxes, &input_boxes));
  const int num_boxes = SizeOfDimension(input_boxes, 0);
  const TfLiteTensor* input_scores;
  TF_LITE_ENSURE_OK(
      context, GetInputSafe(context, node, kInputTensorScores, &input_scores));
  const TfLiteTensor* input_max_output_size;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensorMaxOutputSize,
                                 &input_max_output_size));
  const int max_output_size_value = *GetTensorData<int>(input_max_output_size);
  TF_LITE_ENSURE(context, (max_output_size_value >= 0));
  const bool is_max_output_size_const = IsConstantTensor(input_max_output_size);
  const TfLiteTensor* input_iou_threshold;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensorIouThreshold,
                                 &input_iou_threshold));
  const float iou_threshold = *GetTensorData<float>(input_iou_threshold);
  const TfLiteTensor* input_score_threshold;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensorScoreThreshold,
                                 &input_score_threshold));
  const float score_threshold = *GetTensorData<float>(input_score_threshold);

  TfLiteTensor* output_selected_indices = nullptr;
  TfLiteTensor* output_selected_scores = nullptr;
  TfLiteTensor* output_num_selected_indices = nullptr;

  if (is_soft_nms) {
    const TfLiteTensor* input_sigma;
    TF_LITE_ENSURE_OK(
        context, GetInputSafe(context, node, kInputTensorSigma, &input_sigma));
    const float soft_nms_sigma = *GetTensorData<float>(input_sigma);
    if (soft_nms_sigma < 0) {
      context->ReportError(context, "Invalid sigma value for soft NMS: %f",
                           soft_nms_sigma);
      return kTfLiteError;
    }

    TF_LITE_ENSURE_OK(
        context,
        GetOutputSafe(context, node, kSoftNMSOutputTensorSelectedIndices,
                      &output_selected_indices));
    TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node,
                                             kSoftNMSOutputTensorSelectedScores,
                                             &output_selected_scores));
    TF_LITE_ENSURE_OK(
        context,
        GetOutputSafe(context, node, kSoftNMSOutputTensorNumSelectedIndices,
                      &output_num_selected_indices));
    if (!is_max_output_size_const) {
      SetTensorSizes(context, output_selected_indices, {max_output_size_value});
      SetTensorSizes(context, output_selected_scores, {max_output_size_value});
    }
    reference_ops::NonMaxSuppression(
        input_boxes->data.f, num_boxes, input_scores->data.f,
        max_output_size_value, iou_threshold, score_threshold, soft_nms_sigma,
        output_selected_indices->data.i32, output_selected_scores->data.f,
        output_num_selected_indices->data.i32);
    ResetUnusedElementsToZeroes(
        max_output_size_value, *output_num_selected_indices->data.i32,
        output_selected_indices->data.i32, output_selected_scores->data.f);
  } else {
    TF_LITE_ENSURE_OK(
        context, GetOutputSafe(context, node, kNMSOutputTensorSelectedIndices,
                               &output_selected_indices));
    TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node,
                                             kNMSOutputTensorNumSelectedIndices,
                                             &output_num_selected_indices));
    if (!is_max_output_size_const) {
      SetTensorSizes(context, output_selected_indices, {max_output_size_value});
    }
    reference_ops::NonMaxSuppression(
        input_boxes->data.f, num_boxes, input_scores->data.f,
        max_output_size_value, iou_threshold, score_threshold, /**sigma=**/ 0.0,
        output_selected_indices->data.i32, /**selected_scores=**/ nullptr,
        output_num_selected_indices->data.i32);
    ResetUnusedElementsToZeroes(max_output_size_value,
                                *output_num_selected_indices->data.i32,
                                output_selected_indices->data.i32, nullptr);
  }

  return kTfLiteOk;
}
}  // namespace non_max_suppression

TfLiteRegistration* Register_NON_MAX_SUPPRESSION_V4() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSnon_max_suppressionDTcc mht_4(mht_4_v, 472, "", "./tensorflow/lite/kernels/non_max_suppression.cc", "Register_NON_MAX_SUPPRESSION_V4");

  static TfLiteRegistration r = {nullptr, nullptr, non_max_suppression::Prepare,
                                 non_max_suppression::Eval};
  return &r;
}

TfLiteRegistration* Register_NON_MAX_SUPPRESSION_V5() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSnon_max_suppressionDTcc mht_5(mht_5_v, 481, "", "./tensorflow/lite/kernels/non_max_suppression.cc", "Register_NON_MAX_SUPPRESSION_V5");

  static TfLiteRegistration r = {nullptr, nullptr, non_max_suppression::Prepare,
                                 non_max_suppression::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
