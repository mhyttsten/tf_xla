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
class MHTracer_DTPStensorflowPSlitePSkernelsPSdetection_postprocessDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSdetection_postprocessDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSdetection_postprocessDTcc() {
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
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <initializer_list>
#include <numeric>
#include <vector>

#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace custom {
namespace detection_postprocess {

// Input tensors
constexpr int kInputTensorBoxEncodings = 0;
constexpr int kInputTensorClassPredictions = 1;
constexpr int kInputTensorAnchors = 2;

// Output tensors
// When max_classes_per_detection > 1, detection boxes will be replicated by the
// number of detected classes of that box. Dummy data will be appended if the
// number of classes is smaller than max_classes_per_detection.
constexpr int kOutputTensorDetectionBoxes = 0;
constexpr int kOutputTensorDetectionClasses = 1;
constexpr int kOutputTensorDetectionScores = 2;
constexpr int kOutputTensorNumDetections = 3;

constexpr int kNumCoordBox = 4;
constexpr int kBatchSize = 1;

constexpr int kNumDetectionsPerClass = 100;

// Object Detection model produces axis-aligned boxes in two formats:
// BoxCorner represents the lower left corner (xmin, ymin) and
// the upper right corner (xmax, ymax).
// CenterSize represents the center (xcenter, ycenter), height and width.
// BoxCornerEncoding and CenterSizeEncoding are related as follows:
// ycenter = y / y_scale * anchor.h + anchor.y;
// xcenter = x / x_scale * anchor.w + anchor.x;
// half_h = 0.5*exp(h/ h_scale)) * anchor.h;
// half_w = 0.5*exp(w / w_scale)) * anchor.w;
// ymin = ycenter - half_h
// ymax = ycenter + half_h
// xmin = xcenter - half_w
// xmax = xcenter + half_w
struct BoxCornerEncoding {
  float ymin;
  float xmin;
  float ymax;
  float xmax;
};

struct CenterSizeEncoding {
  float y;
  float x;
  float h;
  float w;
};
// We make sure that the memory allocations are contiguous with static assert.
static_assert(sizeof(BoxCornerEncoding) == sizeof(float) * kNumCoordBox,
              "Size of BoxCornerEncoding is 4 float values");
static_assert(sizeof(CenterSizeEncoding) == sizeof(float) * kNumCoordBox,
              "Size of CenterSizeEncoding is 4 float values");

struct OpData {
  int max_detections;
  int max_classes_per_detection;  // Fast Non-Max-Suppression
  int detections_per_class;       // Regular Non-Max-Suppression
  float non_max_suppression_score_threshold;
  float intersection_over_union_threshold;
  int num_classes;
  bool use_regular_non_max_suppression;
  CenterSizeEncoding scale_values;
  // Indices of Temporary tensors
  int decoded_boxes_index;
  int scores_index;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("buffer: \"" + (buffer == nullptr ? std::string("nullptr") : std::string((char*)buffer)) + "\"");
   MHTracer_DTPStensorflowPSlitePSkernelsPSdetection_postprocessDTcc mht_0(mht_0_v, 273, "", "./tensorflow/lite/kernels/detection_postprocess.cc", "Init");

  auto* op_data = new OpData;
  const uint8_t* buffer_t = reinterpret_cast<const uint8_t*>(buffer);
  const flexbuffers::Map& m = flexbuffers::GetRoot(buffer_t, length).AsMap();
  op_data->max_detections = m["max_detections"].AsInt32();
  op_data->max_classes_per_detection = m["max_classes_per_detection"].AsInt32();
  if (m["detections_per_class"].IsNull())
    op_data->detections_per_class = kNumDetectionsPerClass;
  else
    op_data->detections_per_class = m["detections_per_class"].AsInt32();
  if (m["use_regular_nms"].IsNull())
    op_data->use_regular_non_max_suppression = false;
  else
    op_data->use_regular_non_max_suppression = m["use_regular_nms"].AsBool();

  op_data->non_max_suppression_score_threshold =
      m["nms_score_threshold"].AsFloat();
  op_data->intersection_over_union_threshold = m["nms_iou_threshold"].AsFloat();
  op_data->num_classes = m["num_classes"].AsInt32();
  op_data->scale_values.y = m["y_scale"].AsFloat();
  op_data->scale_values.x = m["x_scale"].AsFloat();
  op_data->scale_values.h = m["h_scale"].AsFloat();
  op_data->scale_values.w = m["w_scale"].AsFloat();
  context->AddTensors(context, 1, &op_data->decoded_boxes_index);
  context->AddTensors(context, 1, &op_data->scores_index);
  return op_data;
}

void Free(TfLiteContext* context, void* buffer) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSdetection_postprocessDTcc mht_1(mht_1_v, 304, "", "./tensorflow/lite/kernels/detection_postprocess.cc", "Free");

  delete static_cast<OpData*>(buffer);
}

TfLiteStatus SetTensorSizes(TfLiteContext* context, TfLiteTensor* tensor,
                            std::initializer_list<int> values) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSdetection_postprocessDTcc mht_2(mht_2_v, 312, "", "./tensorflow/lite/kernels/detection_postprocess.cc", "SetTensorSizes");

  TfLiteIntArray* size = TfLiteIntArrayCreate(values.size());
  int index = 0;
  for (const auto& v : values) {
    size->data[index] = v;
    ++index;
  }
  return context->ResizeTensor(context, tensor, size);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSdetection_postprocessDTcc mht_3(mht_3_v, 325, "", "./tensorflow/lite/kernels/detection_postprocess.cc", "Prepare");

  auto* op_data = static_cast<OpData*>(node->user_data);
  // Inputs: box_encodings, scores, anchors
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 3);
  const TfLiteTensor* input_box_encodings;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensorBoxEncodings,
                                 &input_box_encodings));
  const TfLiteTensor* input_class_predictions;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensorClassPredictions,
                                 &input_class_predictions));
  const TfLiteTensor* input_anchors;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensorAnchors,
                                          &input_anchors));
  TF_LITE_ENSURE_EQ(context, NumDimensions(input_box_encodings), 3);
  TF_LITE_ENSURE_EQ(context, NumDimensions(input_class_predictions), 3);
  TF_LITE_ENSURE_EQ(context, NumDimensions(input_anchors), 2);
  // number of detected boxes
  const int num_detected_boxes =
      op_data->max_detections * op_data->max_classes_per_detection;

  // Outputs: detection_boxes, detection_scores, detection_classes,
  // num_detections
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 4);
  // Output Tensor detection_boxes: size is set to (1, num_detected_boxes, 4)
  TfLiteTensor* detection_boxes;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensorDetectionBoxes,
                                  &detection_boxes));
  detection_boxes->type = kTfLiteFloat32;
  SetTensorSizes(context, detection_boxes,
                 {kBatchSize, num_detected_boxes, kNumCoordBox});

  // Output Tensor detection_classes: size is set to (1, num_detected_boxes)
  TfLiteTensor* detection_classes;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensorDetectionClasses,
                                  &detection_classes));
  detection_classes->type = kTfLiteFloat32;
  SetTensorSizes(context, detection_classes, {kBatchSize, num_detected_boxes});

  // Output Tensor detection_scores: size is set to (1, num_detected_boxes)
  TfLiteTensor* detection_scores;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensorDetectionScores,
                                  &detection_scores));
  detection_scores->type = kTfLiteFloat32;
  SetTensorSizes(context, detection_scores, {kBatchSize, num_detected_boxes});

  // Output Tensor num_detections: size is set to 1
  TfLiteTensor* num_detections;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensorNumDetections,
                                  &num_detections));
  num_detections->type = kTfLiteFloat32;
  SetTensorSizes(context, num_detections, {1});

  // Temporary tensors
  TfLiteIntArrayFree(node->temporaries);
  node->temporaries = TfLiteIntArrayCreate(2);
  node->temporaries->data[0] = op_data->decoded_boxes_index;
  node->temporaries->data[1] = op_data->scores_index;

  // decoded_boxes
  TfLiteTensor* decoded_boxes = &context->tensors[op_data->decoded_boxes_index];
  decoded_boxes->type = kTfLiteFloat32;
  decoded_boxes->allocation_type = kTfLiteArenaRw;
  SetTensorSizes(context, decoded_boxes,
                 {input_box_encodings->dims->data[1], kNumCoordBox});

  // scores
  TfLiteTensor* scores = &context->tensors[op_data->scores_index];
  scores->type = kTfLiteFloat32;
  scores->allocation_type = kTfLiteArenaRw;
  SetTensorSizes(context, scores,
                 {input_class_predictions->dims->data[1],
                  input_class_predictions->dims->data[2]});

  return kTfLiteOk;
}

class Dequantizer {
 public:
  Dequantizer(int zero_point, float scale)
      : zero_point_(zero_point), scale_(scale) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSdetection_postprocessDTcc mht_4(mht_4_v, 413, "", "./tensorflow/lite/kernels/detection_postprocess.cc", "Dequantizer");
}
  float operator()(uint8 x) {
    return (static_cast<float>(x) - zero_point_) * scale_;
  }

 private:
  int zero_point_;
  float scale_;
};

void DequantizeBoxEncodings(const TfLiteTensor* input_box_encodings, int idx,
                            float quant_zero_point, float quant_scale,
                            int length_box_encoding,
                            CenterSizeEncoding* box_centersize) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSdetection_postprocessDTcc mht_5(mht_5_v, 429, "", "./tensorflow/lite/kernels/detection_postprocess.cc", "DequantizeBoxEncodings");

  const uint8* boxes =
      GetTensorData<uint8>(input_box_encodings) + length_box_encoding * idx;
  Dequantizer dequantize(quant_zero_point, quant_scale);
  // See definition of the KeyPointBoxCoder at
  // https://github.com/tensorflow/models/blob/master/research/object_detection/box_coders/keypoint_box_coder.py
  // The first four elements are the box coordinates, which is the same as the
  // FastRnnBoxCoder at
  // https://github.com/tensorflow/models/blob/master/research/object_detection/box_coders/faster_rcnn_box_coder.py
  box_centersize->y = dequantize(boxes[0]);
  box_centersize->x = dequantize(boxes[1]);
  box_centersize->h = dequantize(boxes[2]);
  box_centersize->w = dequantize(boxes[3]);
}

template <class T>
T ReInterpretTensor(const TfLiteTensor* tensor) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSdetection_postprocessDTcc mht_6(mht_6_v, 448, "", "./tensorflow/lite/kernels/detection_postprocess.cc", "ReInterpretTensor");

  const float* tensor_base = GetTensorData<float>(tensor);
  return reinterpret_cast<T>(tensor_base);
}

template <class T>
T ReInterpretTensor(TfLiteTensor* tensor) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSdetection_postprocessDTcc mht_7(mht_7_v, 457, "", "./tensorflow/lite/kernels/detection_postprocess.cc", "ReInterpretTensor");

  float* tensor_base = GetTensorData<float>(tensor);
  return reinterpret_cast<T>(tensor_base);
}

TfLiteStatus DecodeCenterSizeBoxes(TfLiteContext* context, TfLiteNode* node,
                                   OpData* op_data) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSdetection_postprocessDTcc mht_8(mht_8_v, 466, "", "./tensorflow/lite/kernels/detection_postprocess.cc", "DecodeCenterSizeBoxes");

  // Parse input tensor boxencodings
  const TfLiteTensor* input_box_encodings;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensorBoxEncodings,
                                 &input_box_encodings));
  TF_LITE_ENSURE_EQ(context, input_box_encodings->dims->data[0], kBatchSize);
  const int num_boxes = input_box_encodings->dims->data[1];
  TF_LITE_ENSURE(context, input_box_encodings->dims->data[2] >= kNumCoordBox);
  const TfLiteTensor* input_anchors;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensorAnchors,
                                          &input_anchors));

  // Decode the boxes to get (ymin, xmin, ymax, xmax) based on the anchors
  CenterSizeEncoding box_centersize;
  CenterSizeEncoding scale_values = op_data->scale_values;
  CenterSizeEncoding anchor;
  for (int idx = 0; idx < num_boxes; ++idx) {
    switch (input_box_encodings->type) {
        // Quantized
      case kTfLiteUInt8:
        DequantizeBoxEncodings(
            input_box_encodings, idx,
            static_cast<float>(input_box_encodings->params.zero_point),
            static_cast<float>(input_box_encodings->params.scale),
            input_box_encodings->dims->data[2], &box_centersize);
        DequantizeBoxEncodings(
            input_anchors, idx,
            static_cast<float>(input_anchors->params.zero_point),
            static_cast<float>(input_anchors->params.scale), kNumCoordBox,
            &anchor);
        break;
        // Float
      case kTfLiteFloat32: {
        // Please see DequantizeBoxEncodings function for the support detail.
        const int box_encoding_idx = idx * input_box_encodings->dims->data[2];
        const float* boxes =
            &(GetTensorData<float>(input_box_encodings)[box_encoding_idx]);
        box_centersize = *reinterpret_cast<const CenterSizeEncoding*>(boxes);
        TF_LITE_ENSURE_EQ(context, input_anchors->type, kTfLiteFloat32);
        anchor =
            ReInterpretTensor<const CenterSizeEncoding*>(input_anchors)[idx];
        break;
      }
      default:
        // Unsupported type.
        return kTfLiteError;
    }

    float ycenter = static_cast<float>(static_cast<double>(box_centersize.y) /
                                           static_cast<double>(scale_values.y) *
                                           static_cast<double>(anchor.h) +
                                       static_cast<double>(anchor.y));

    float xcenter = static_cast<float>(static_cast<double>(box_centersize.x) /
                                           static_cast<double>(scale_values.x) *
                                           static_cast<double>(anchor.w) +
                                       static_cast<double>(anchor.x));

    float half_h =
        static_cast<float>(0.5 *
                           (std::exp(static_cast<double>(box_centersize.h) /
                                     static_cast<double>(scale_values.h))) *
                           static_cast<double>(anchor.h));
    float half_w =
        static_cast<float>(0.5 *
                           (std::exp(static_cast<double>(box_centersize.w) /
                                     static_cast<double>(scale_values.w))) *
                           static_cast<double>(anchor.w));

    TfLiteTensor* decoded_boxes =
        &context->tensors[op_data->decoded_boxes_index];
    TF_LITE_ENSURE_EQ(context, decoded_boxes->type, kTfLiteFloat32);
    auto& box = ReInterpretTensor<BoxCornerEncoding*>(decoded_boxes)[idx];
    box.ymin = ycenter - half_h;
    box.xmin = xcenter - half_w;
    box.ymax = ycenter + half_h;
    box.xmax = xcenter + half_w;
  }
  return kTfLiteOk;
}

void DecreasingPartialArgSort(const float* values, int num_values,
                              int num_to_sort, int* indices) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSdetection_postprocessDTcc mht_9(mht_9_v, 552, "", "./tensorflow/lite/kernels/detection_postprocess.cc", "DecreasingPartialArgSort");

  if (num_to_sort == 1) {
    indices[0] = optimized_ops::ArgMaxVector(values, num_values);
  } else {
    std::iota(indices, indices + num_values, 0);
    std::partial_sort(
        indices, indices + num_to_sort, indices + num_values,
        [&values](const int i, const int j) { return values[i] > values[j]; });
  }
}

void DecreasingArgSort(const float* values, int num_values, int* indices) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSdetection_postprocessDTcc mht_10(mht_10_v, 566, "", "./tensorflow/lite/kernels/detection_postprocess.cc", "DecreasingArgSort");

  std::iota(indices, indices + num_values, 0);

  // We want here a stable sort, in order to get completely defined output.
  // In this way TFL and TFLM can be bit-exact.
  std::stable_sort(
      indices, indices + num_values,
      [&values](const int i, const int j) { return values[i] > values[j]; });
}

void SelectDetectionsAboveScoreThreshold(const std::vector<float>& values,
                                         const float threshold,
                                         std::vector<float>* keep_values,
                                         std::vector<int>* keep_indices) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSdetection_postprocessDTcc mht_11(mht_11_v, 582, "", "./tensorflow/lite/kernels/detection_postprocess.cc", "SelectDetectionsAboveScoreThreshold");

  for (int i = 0; i < values.size(); i++) {
    if (values[i] >= threshold) {
      keep_values->emplace_back(values[i]);
      keep_indices->emplace_back(i);
    }
  }
}

bool ValidateBoxes(const TfLiteTensor* decoded_boxes, const int num_boxes) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSdetection_postprocessDTcc mht_12(mht_12_v, 594, "", "./tensorflow/lite/kernels/detection_postprocess.cc", "ValidateBoxes");

  for (int i = 0; i < num_boxes; ++i) {
    auto& box = ReInterpretTensor<const BoxCornerEncoding*>(decoded_boxes)[i];
    // Note: `ComputeIntersectionOverUnion` properly handles degenerated boxes
    // (xmin == xmax and/or ymin == ymax) as it just returns 0 in case the box
    // area is <= 0.
    if (box.ymin > box.ymax || box.xmin > box.xmax) {
      return false;
    }
  }
  return true;
}

float ComputeIntersectionOverUnion(const TfLiteTensor* decoded_boxes,
                                   const int i, const int j) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSdetection_postprocessDTcc mht_13(mht_13_v, 611, "", "./tensorflow/lite/kernels/detection_postprocess.cc", "ComputeIntersectionOverUnion");

  auto& box_i = ReInterpretTensor<const BoxCornerEncoding*>(decoded_boxes)[i];
  auto& box_j = ReInterpretTensor<const BoxCornerEncoding*>(decoded_boxes)[j];
  const float area_i = (box_i.ymax - box_i.ymin) * (box_i.xmax - box_i.xmin);
  const float area_j = (box_j.ymax - box_j.ymin) * (box_j.xmax - box_j.xmin);
  if (area_i <= 0 || area_j <= 0) return 0.0;
  const float intersection_ymin = std::max<float>(box_i.ymin, box_j.ymin);
  const float intersection_xmin = std::max<float>(box_i.xmin, box_j.xmin);
  const float intersection_ymax = std::min<float>(box_i.ymax, box_j.ymax);
  const float intersection_xmax = std::min<float>(box_i.xmax, box_j.xmax);
  const float intersection_area =
      std::max<float>(intersection_ymax - intersection_ymin, 0.0) *
      std::max<float>(intersection_xmax - intersection_xmin, 0.0);
  return intersection_area / (area_i + area_j - intersection_area);
}

// NonMaxSuppressionSingleClass() prunes out the box locations with high overlap
// before selecting the highest scoring boxes (max_detections in number)
// It assumes all boxes are good in beginning and sorts based on the scores.
// If lower-scoring box has too much overlap with a higher-scoring box,
// we get rid of the lower-scoring box.
// Complexity is O(N^2) pairwise comparison between boxes
TfLiteStatus NonMaxSuppressionSingleClassHelper(
    TfLiteContext* context, TfLiteNode* node, OpData* op_data,
    const std::vector<float>& scores, int max_detections,
    std::vector<int>* selected) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSdetection_postprocessDTcc mht_14(mht_14_v, 639, "", "./tensorflow/lite/kernels/detection_postprocess.cc", "NonMaxSuppressionSingleClassHelper");

  const TfLiteTensor* input_box_encodings;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensorBoxEncodings,
                                 &input_box_encodings));
  const TfLiteTensor* decoded_boxes =
      &context->tensors[op_data->decoded_boxes_index];
  const int num_boxes = input_box_encodings->dims->data[1];
  const float non_max_suppression_score_threshold =
      op_data->non_max_suppression_score_threshold;
  const float intersection_over_union_threshold =
      op_data->intersection_over_union_threshold;
  // Maximum detections should be positive.
  TF_LITE_ENSURE(context, (max_detections >= 0));
  // intersection_over_union_threshold should be positive
  // and should be less than 1.
  TF_LITE_ENSURE(context, (intersection_over_union_threshold > 0.0f) &&
                              (intersection_over_union_threshold <= 1.0f));
  // Validate boxes
  TF_LITE_ENSURE_EQ(context, decoded_boxes->type, kTfLiteFloat32);
  TF_LITE_ENSURE(context, ValidateBoxes(decoded_boxes, num_boxes));

  // threshold scores
  std::vector<int> keep_indices;
  // TODO(b/177068807): Remove the dynamic allocation and replace it
  // with temporaries, esp for std::vector<float>
  std::vector<float> keep_scores;
  SelectDetectionsAboveScoreThreshold(
      scores, non_max_suppression_score_threshold, &keep_scores, &keep_indices);

  int num_scores_kept = keep_scores.size();
  std::vector<int> sorted_indices;
  sorted_indices.resize(num_scores_kept);
  DecreasingArgSort(keep_scores.data(), num_scores_kept, sorted_indices.data());

  const int num_boxes_kept = num_scores_kept;
  const int output_size = std::min(num_boxes_kept, max_detections);
  selected->clear();
  int num_active_candidate = num_boxes_kept;
  std::vector<uint8_t> active_box_candidate(num_boxes_kept, 1);

  for (int i = 0; i < num_boxes_kept; ++i) {
    if (num_active_candidate == 0 || selected->size() >= output_size) break;
    if (active_box_candidate[i] == 1) {
      selected->push_back(keep_indices[sorted_indices[i]]);
      active_box_candidate[i] = 0;
      num_active_candidate--;
    } else {
      continue;
    }
    for (int j = i + 1; j < num_boxes_kept; ++j) {
      if (active_box_candidate[j] == 1) {
        TF_LITE_ENSURE_EQ(context, decoded_boxes->type, kTfLiteFloat32);
        float intersection_over_union = ComputeIntersectionOverUnion(
            decoded_boxes, keep_indices[sorted_indices[i]],
            keep_indices[sorted_indices[j]]);

        if (intersection_over_union > intersection_over_union_threshold) {
          active_box_candidate[j] = 0;
          num_active_candidate--;
        }
      }
    }
  }
  return kTfLiteOk;
}

struct BoxInfo {
  int index;
  float score;
};

struct NMSTaskParam {
  // Caller retains the ownership of `context`, `node`, `op_data` and `scores`.
  // Caller should ensure their lifetime is longer than NMSTaskParam instance.
  TfLiteContext* context;
  TfLiteNode* node;
  OpData* op_data;
  const float* scores;

  int num_classes;
  int num_boxes;
  int label_offset;
  int num_classes_with_background;
  int num_detections_per_class;
  int max_detections;
  std::vector<int>& num_selected;
};

void InplaceMergeBoxInfo(std::vector<BoxInfo>& boxes, int mid_index,
                         int end_index) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSdetection_postprocessDTcc mht_15(mht_15_v, 732, "", "./tensorflow/lite/kernels/detection_postprocess.cc", "InplaceMergeBoxInfo");

  std::inplace_merge(
      boxes.begin(), boxes.begin() + mid_index, boxes.begin() + end_index,
      [](const BoxInfo& a, const BoxInfo& b) { return a.score >= b.score; });
}

TfLiteStatus ComputeNMSResult(const NMSTaskParam& nms_task_param, int col_begin,
                              int col_end, int& sorted_indices_size,
                              std::vector<BoxInfo>& resulted_sorted_box_info) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSdetection_postprocessDTcc mht_16(mht_16_v, 743, "", "./tensorflow/lite/kernels/detection_postprocess.cc", "ComputeNMSResult");

  std::vector<float> class_scores(nms_task_param.num_boxes);
  std::vector<int> selected;
  selected.reserve(nms_task_param.num_detections_per_class);

  for (int col = col_begin; col <= col_end; ++col) {
    const float* scores_base =
        nms_task_param.scores + col + nms_task_param.label_offset;
    for (int row = 0; row < nms_task_param.num_boxes; row++) {
      // Get scores of boxes corresponding to all anchors for single class
      class_scores[row] = *scores_base;
      scores_base += nms_task_param.num_classes_with_background;
    }

    // Perform non-maximal suppression on single class
    selected.clear();
    TF_LITE_ENSURE_OK(
        nms_task_param.context,
        NonMaxSuppressionSingleClassHelper(
            nms_task_param.context, nms_task_param.node, nms_task_param.op_data,
            class_scores, nms_task_param.num_detections_per_class, &selected));
    if (selected.empty()) {
      continue;
    }

    for (int i = 0; i < selected.size(); ++i) {
      resulted_sorted_box_info[sorted_indices_size + i].score =
          class_scores[selected[i]];
      resulted_sorted_box_info[sorted_indices_size + i].index =
          (selected[i] * nms_task_param.num_classes_with_background + col +
           nms_task_param.label_offset);
    }

    // In-place merge the original boxes and new selected boxes which are both
    // sorted by scores.
    InplaceMergeBoxInfo(resulted_sorted_box_info, sorted_indices_size,
                        sorted_indices_size + selected.size());

    sorted_indices_size =
        std::min(sorted_indices_size + static_cast<int>(selected.size()),
                 nms_task_param.max_detections);
  }
  return kTfLiteOk;
}

struct NonMaxSuppressionWorkerTask : cpu_backend_threadpool::Task {
  NonMaxSuppressionWorkerTask(NMSTaskParam& nms_task_param,
                              std::atomic<int>& next_col, int col_begin)
      : nms_task_param(nms_task_param),
        next_col(next_col),
        col_begin(col_begin),
        sorted_indices_size(0) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSdetection_postprocessDTcc mht_17(mht_17_v, 797, "", "./tensorflow/lite/kernels/detection_postprocess.cc", "NonMaxSuppressionWorkerTask");
}
  void Run() override {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSdetection_postprocessDTcc mht_18(mht_18_v, 801, "", "./tensorflow/lite/kernels/detection_postprocess.cc", "Run");

    sorted_box_info.resize(nms_task_param.num_detections_per_class +
                           nms_task_param.max_detections);
    for (int col = col_begin; col < nms_task_param.num_classes;
         col = (++next_col)) {
      if (ComputeNMSResult(nms_task_param, col, col, sorted_indices_size,
                           sorted_box_info) != kTfLiteOk) {
        break;
      }
    }
  }
  NMSTaskParam& nms_task_param;
  // A shared atomic variable across threads, representing the next col this
  // task will work on after completing the work for 'col_begin'
  std::atomic<int>& next_col;
  const int col_begin;
  int sorted_indices_size;
  std::vector<BoxInfo> sorted_box_info;
};

// This function implements a regular version of Non Maximal Suppression (NMS)
// for multiple classes where
// 1) we do NMS separately for each class across all anchors and
// 2) keep only the highest anchor scores across all classes
// 3) The worst runtime of the regular NMS is O(K*N^2)
// where N is the number of anchors and K the number of
// classes.
TfLiteStatus NonMaxSuppressionMultiClassRegularHelper(TfLiteContext* context,
                                                      TfLiteNode* node,
                                                      OpData* op_data,
                                                      const float* scores) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSdetection_postprocessDTcc mht_19(mht_19_v, 834, "", "./tensorflow/lite/kernels/detection_postprocess.cc", "NonMaxSuppressionMultiClassRegularHelper");

  const TfLiteTensor* input_box_encodings;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensorBoxEncodings,
                                 &input_box_encodings));
  const TfLiteTensor* input_class_predictions;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensorClassPredictions,
                                 &input_class_predictions));
  const TfLiteTensor* decoded_boxes =
      &context->tensors[op_data->decoded_boxes_index];

  TfLiteTensor* detection_boxes;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensorDetectionBoxes,
                                  &detection_boxes));
  TfLiteTensor* detection_classes;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensorDetectionClasses,
                                  &detection_classes));
  TfLiteTensor* detection_scores;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensorDetectionScores,
                                  &detection_scores));
  TfLiteTensor* num_detections;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensorNumDetections,
                                  &num_detections));

  const int num_boxes = input_box_encodings->dims->data[1];
  const int num_classes = op_data->num_classes;
  const int num_detections_per_class =
      std::min(op_data->detections_per_class, op_data->max_detections);
  const int max_detections = op_data->max_detections;
  const int num_classes_with_background =
      input_class_predictions->dims->data[2];
  // The row index offset is 1 if background class is included and 0 otherwise.
  int label_offset = num_classes_with_background - num_classes;
  TF_LITE_ENSURE(context, num_detections_per_class > 0);

  int sorted_indices_size = 0;
  std::vector<BoxInfo> box_info_after_regular_non_max_suppression(
      max_detections + num_detections_per_class);
  std::vector<int> num_selected(num_classes);

  NMSTaskParam nms_task_param{context,
                              node,
                              op_data,
                              scores,
                              num_classes,
                              num_boxes,
                              label_offset,
                              num_classes_with_background,
                              num_detections_per_class,
                              max_detections,
                              num_selected};

  int num_threads =
      CpuBackendContext::GetFromContext(context)->max_num_threads();
  if (num_threads == 1) {
    // For each class, perform non-max suppression.
    TF_LITE_ENSURE_OK(
        context, ComputeNMSResult(nms_task_param, /* col_begin= */ 0,
                                  num_classes - 1, sorted_indices_size,
                                  box_info_after_regular_non_max_suppression));
  } else {
    std::atomic<int> next_col(num_threads);
    std::vector<NonMaxSuppressionWorkerTask> tasks;
    tasks.reserve(num_threads);
    for (int i = 0; i < num_threads; ++i) {
      tasks.emplace_back(
          NonMaxSuppressionWorkerTask(nms_task_param, next_col, i));
    }
    cpu_backend_threadpool::Execute(tasks.size(), tasks.data(),
                                    CpuBackendContext::GetFromContext(context));

    // Merge results from tasks.
    for (int j = 0; j < tasks.size(); ++j) {
      if (tasks[j].sorted_indices_size == 0) {
        continue;
      }
      memcpy(&box_info_after_regular_non_max_suppression[sorted_indices_size],
             &tasks[j].sorted_box_info[0],
             sizeof(BoxInfo) * tasks[j].sorted_indices_size);
      InplaceMergeBoxInfo(box_info_after_regular_non_max_suppression,
                          sorted_indices_size,
                          sorted_indices_size + tasks[j].sorted_indices_size);
      sorted_indices_size = std::min(
          sorted_indices_size + tasks[j].sorted_indices_size, max_detections);
    }
  }

  // Allocate output tensors
  for (int output_box_index = 0; output_box_index < max_detections;
       output_box_index++) {
    if (output_box_index < sorted_indices_size) {
      const int anchor_index = floor(
          box_info_after_regular_non_max_suppression[output_box_index].index /
          num_classes_with_background);
      const int class_index =
          box_info_after_regular_non_max_suppression[output_box_index].index -
          anchor_index * num_classes_with_background - label_offset;
      const float selected_score =
          box_info_after_regular_non_max_suppression[output_box_index].score;
      // detection_boxes
      TF_LITE_ENSURE_EQ(context, detection_boxes->type, kTfLiteFloat32);
      TF_LITE_ENSURE_EQ(context, decoded_boxes->type, kTfLiteFloat32);
      ReInterpretTensor<BoxCornerEncoding*>(detection_boxes)[output_box_index] =
          ReInterpretTensor<const BoxCornerEncoding*>(
              decoded_boxes)[anchor_index];
      // detection_classes
      GetTensorData<float>(detection_classes)[output_box_index] = class_index;
      // detection_scores
      GetTensorData<float>(detection_scores)[output_box_index] = selected_score;
    } else {
      TF_LITE_ENSURE_EQ(context, detection_boxes->type, kTfLiteFloat32);
      ReInterpretTensor<BoxCornerEncoding*>(
          detection_boxes)[output_box_index] = {0.0f, 0.0f, 0.0f, 0.0f};
      // detection_classes
      GetTensorData<float>(detection_classes)[output_box_index] = 0.0f;
      // detection_scores
      GetTensorData<float>(detection_scores)[output_box_index] = 0.0f;
    }
  }
  GetTensorData<float>(num_detections)[0] = sorted_indices_size;
  box_info_after_regular_non_max_suppression.clear();
  return kTfLiteOk;
}

// This function implements a fast version of Non Maximal Suppression for
// multiple classes where
// 1) we keep the top-k scores for each anchor and
// 2) during NMS, each anchor only uses the highest class score for sorting.
// 3) Compared to standard NMS, the worst runtime of this version is O(N^2)
// instead of O(KN^2) where N is the number of anchors and K the number of
// classes.
TfLiteStatus NonMaxSuppressionMultiClassFastHelper(TfLiteContext* context,
                                                   TfLiteNode* node,
                                                   OpData* op_data,
                                                   const float* scores) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSdetection_postprocessDTcc mht_20(mht_20_v, 976, "", "./tensorflow/lite/kernels/detection_postprocess.cc", "NonMaxSuppressionMultiClassFastHelper");

  const TfLiteTensor* input_box_encodings;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensorBoxEncodings,
                                 &input_box_encodings));
  const TfLiteTensor* input_class_predictions;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensorClassPredictions,
                                 &input_class_predictions));
  const TfLiteTensor* decoded_boxes =
      &context->tensors[op_data->decoded_boxes_index];

  TfLiteTensor* detection_boxes;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensorDetectionBoxes,
                                  &detection_boxes));
  TfLiteTensor* detection_classes;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensorDetectionClasses,
                                  &detection_classes));
  TfLiteTensor* detection_scores;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensorDetectionScores,
                                  &detection_scores));
  TfLiteTensor* num_detections;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensorNumDetections,
                                  &num_detections));

  const int num_boxes = input_box_encodings->dims->data[1];
  const int num_classes = op_data->num_classes;
  const int max_categories_per_anchor = op_data->max_classes_per_detection;
  const int num_classes_with_background =
      input_class_predictions->dims->data[2];
  // The row index offset is 1 if background class is included and 0 otherwise.
  int label_offset = num_classes_with_background - num_classes;
  TF_LITE_ENSURE(context, (max_categories_per_anchor > 0));
  const int num_categories_per_anchor =
      std::min(max_categories_per_anchor, num_classes);
  std::vector<float> max_scores;
  max_scores.resize(num_boxes);
  std::vector<int> sorted_class_indices;
  sorted_class_indices.resize(num_boxes * num_classes);
  for (int row = 0; row < num_boxes; row++) {
    const float* box_scores =
        scores + row * num_classes_with_background + label_offset;
    int* class_indices = sorted_class_indices.data() + row * num_classes;
    DecreasingPartialArgSort(box_scores, num_classes, num_categories_per_anchor,
                             class_indices);
    max_scores[row] = box_scores[class_indices[0]];
  }
  // Perform non-maximal suppression on max scores
  std::vector<int> selected;
  TF_LITE_ENSURE_STATUS(NonMaxSuppressionSingleClassHelper(
      context, node, op_data, max_scores, op_data->max_detections, &selected));
  // Allocate output tensors
  int output_box_index = 0;
  for (const auto& selected_index : selected) {
    const float* box_scores =
        scores + selected_index * num_classes_with_background + label_offset;
    const int* class_indices =
        sorted_class_indices.data() + selected_index * num_classes;

    for (int col = 0; col < num_categories_per_anchor; ++col) {
      int box_offset = max_categories_per_anchor * output_box_index + col;
      // detection_boxes
      TF_LITE_ENSURE_EQ(context, detection_boxes->type, kTfLiteFloat32);
      TF_LITE_ENSURE_EQ(context, decoded_boxes->type, kTfLiteFloat32);
      ReInterpretTensor<BoxCornerEncoding*>(detection_boxes)[box_offset] =
          ReInterpretTensor<const BoxCornerEncoding*>(
              decoded_boxes)[selected_index];
      // detection_classes
      GetTensorData<float>(detection_classes)[box_offset] = class_indices[col];
      // detection_scores
      GetTensorData<float>(detection_scores)[box_offset] =
          box_scores[class_indices[col]];
    }
    output_box_index++;
  }
  GetTensorData<float>(num_detections)[0] = output_box_index;
  return kTfLiteOk;
}

void DequantizeClassPredictions(const TfLiteTensor* input_class_predictions,
                                const int num_boxes,
                                const int num_classes_with_background,
                                TfLiteTensor* scores) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSdetection_postprocessDTcc mht_21(mht_21_v, 1065, "", "./tensorflow/lite/kernels/detection_postprocess.cc", "DequantizeClassPredictions");

  float quant_zero_point =
      static_cast<float>(input_class_predictions->params.zero_point);
  float quant_scale = static_cast<float>(input_class_predictions->params.scale);
  tflite::DequantizationParams op_params;
  op_params.zero_point = quant_zero_point;
  op_params.scale = quant_scale;
  const auto shape = RuntimeShape(1, num_boxes * num_classes_with_background);
  optimized_ops::Dequantize(op_params, shape,
                            GetTensorData<uint8>(input_class_predictions),
                            shape, GetTensorData<float>(scores));
}

TfLiteStatus NonMaxSuppressionMultiClass(TfLiteContext* context,
                                         TfLiteNode* node, OpData* op_data) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSdetection_postprocessDTcc mht_22(mht_22_v, 1082, "", "./tensorflow/lite/kernels/detection_postprocess.cc", "NonMaxSuppressionMultiClass");

  // Get the input tensors
  const TfLiteTensor* input_box_encodings;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensorBoxEncodings,
                                 &input_box_encodings));
  const TfLiteTensor* input_class_predictions;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensorClassPredictions,
                                 &input_class_predictions));
  const int num_boxes = input_box_encodings->dims->data[1];
  const int num_classes = op_data->num_classes;
  TF_LITE_ENSURE_EQ(context, input_class_predictions->dims->data[0],
                    kBatchSize);
  TF_LITE_ENSURE_EQ(context, input_class_predictions->dims->data[1], num_boxes);
  const int num_classes_with_background =
      input_class_predictions->dims->data[2];

  TF_LITE_ENSURE(context, (num_classes_with_background - num_classes <= 1));
  TF_LITE_ENSURE(context, (num_classes_with_background >= num_classes));

  const TfLiteTensor* scores;
  switch (input_class_predictions->type) {
    case kTfLiteUInt8: {
      TfLiteTensor* temporary_scores = &context->tensors[op_data->scores_index];
      DequantizeClassPredictions(input_class_predictions, num_boxes,
                                 num_classes_with_background, temporary_scores);
      scores = temporary_scores;
    } break;
    case kTfLiteFloat32:
      scores = input_class_predictions;
      break;
    default:
      // Unsupported type.
      return kTfLiteError;
  }
  if (op_data->use_regular_non_max_suppression)
    TF_LITE_ENSURE_STATUS(NonMaxSuppressionMultiClassRegularHelper(
        context, node, op_data, GetTensorData<float>(scores)));
  else
    TF_LITE_ENSURE_STATUS(NonMaxSuppressionMultiClassFastHelper(
        context, node, op_data, GetTensorData<float>(scores)));

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSdetection_postprocessDTcc mht_23(mht_23_v, 1131, "", "./tensorflow/lite/kernels/detection_postprocess.cc", "Eval");

  // TODO(b/177068051):  Generalize for any batch size.
  TF_LITE_ENSURE(context, (kBatchSize == 1));
  auto* op_data = static_cast<OpData*>(node->user_data);
  // These two functions correspond to two blocks in the Object Detection model.
  // In future, we would like to break the custom op in two blocks, which is
  // currently not feasible because we would like to input quantized inputs
  // and do all calculations in float. Mixed quantized/float calculations are
  // currently not supported in TFLite.

  // This fills in temporary decoded_boxes
  // by transforming input_box_encodings and input_anchors from
  // CenterSizeEncodings to BoxCornerEncoding
  TF_LITE_ENSURE_STATUS(DecodeCenterSizeBoxes(context, node, op_data));
  // This fills in the output tensors
  // by choosing effective set of decoded boxes
  // based on Non Maximal Suppression, i.e. selecting
  // highest scoring non-overlapping boxes.
  TF_LITE_ENSURE_STATUS(NonMaxSuppressionMultiClass(context, node, op_data));

  return kTfLiteOk;
}
}  // namespace detection_postprocess

TfLiteRegistration* Register_DETECTION_POSTPROCESS() {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSdetection_postprocessDTcc mht_24(mht_24_v, 1158, "", "./tensorflow/lite/kernels/detection_postprocess.cc", "Register_DETECTION_POSTPROCESS");

  static TfLiteRegistration r = {
      detection_postprocess::Init, detection_postprocess::Free,
      detection_postprocess::Prepare, detection_postprocess::Eval};
  return &r;
}

// Since the op is named "TFLite_Detection_PostProcess", the selective build
// tool will assume the register function is named
// "Register_TFLITE_DETECTION_POST_PROCESS".
TfLiteRegistration* Register_TFLITE_DETECTION_POST_PROCESS() {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSdetection_postprocessDTcc mht_25(mht_25_v, 1171, "", "./tensorflow/lite/kernels/detection_postprocess.cc", "Register_TFLITE_DETECTION_POST_PROCESS");

  return Register_DETECTION_POSTPROCESS();
}

}  // namespace custom
}  // namespace ops
}  // namespace tflite
