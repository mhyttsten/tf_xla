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
class MHTracer_DTPStensorflowPSlitePSkernelsPSperceptionPSdense_image_warpDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSperceptionPSdense_image_warpDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSperceptionPSdense_image_warpDTcc() {
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

/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include <cmath>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"

namespace tflite {
namespace ops {
namespace custom {
namespace dense_image_warp {

constexpr int kInputTensor = 0;
constexpr int kFlowTensor = 1;
constexpr int kOutputTensor = 0;

inline void DenseImageWarp(const RuntimeShape& input_shape,
                           const float* input_data,
                           const RuntimeShape& flow_shape,
                           const float* flow_data, float* output_data) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSperceptionPSdense_image_warpDTcc mht_0(mht_0_v, 206, "", "./tensorflow/lite/kernels/perception/dense_image_warp.cc", "DenseImageWarp");

  const int batches = MatchingDim(input_shape, 0, flow_shape, 0);
  const int height = MatchingDim(input_shape, 1, flow_shape, 1);
  const int width = MatchingDim(input_shape, 2, flow_shape, 2);
  const int channels = input_shape.Dims(3);
  TFLITE_CHECK_EQ(flow_shape.Dims(3), 2);

  // Max values to make sure the indexes are not out of bound.
  const int max_floor_y = height - 2;
  const int max_floor_x = width - 2;

  for (int batch = 0; batch < batches; ++batch) {
    for (int in_y = 0; in_y < height; ++in_y) {
      for (int in_x = 0; in_x < width; ++in_x) {
        float querry_point_y =
            in_y - flow_data[Offset(flow_shape, batch, in_y, in_x, 0)];
        float querry_point_x =
            in_x - flow_data[Offset(flow_shape, batch, in_y, in_x, 1)];

        int floor_y =
            std::min(std::max(0, static_cast<int>(std::floor(querry_point_y))),
                     max_floor_y);
        int floor_x =
            std::min(std::max(0, static_cast<int>(std::floor(querry_point_x))),
                     max_floor_x);
        float alpha_y =
            std::min(std::max(0.0f, querry_point_y - floor_y), 1.0f);
        float alpha_x =
            std::min(std::max(0.0f, querry_point_x - floor_x), 1.0f);

        for (int c = 0; c < channels; ++c) {
          float top_left =
              input_data[Offset(input_shape, batch, floor_y, floor_x, c)];
          float top_right =
              input_data[Offset(input_shape, batch, floor_y, floor_x + 1, c)];
          float bottom_left =
              input_data[Offset(input_shape, batch, floor_y + 1, floor_x, c)];
          float bottom_right = input_data[Offset(input_shape, batch,
                                                 floor_y + 1, floor_x + 1, c)];

          float interp_top = alpha_x * (top_right - top_left) + top_left;
          float interp_bottom =
              alpha_x * (bottom_right - bottom_left) + bottom_left;
          float interp = alpha_y * (interp_bottom - interp_top) + interp_top;
          output_data[Offset(input_shape, batch, in_y, in_x, c)] = interp;
        }
      }
    }
  }
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSperceptionPSdense_image_warpDTcc mht_1(mht_1_v, 260, "", "./tensorflow/lite/kernels/perception/dense_image_warp.cc", "Prepare");

  // Check inputs and output.
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  const TfLiteTensor* flow = GetInput(context, node, kFlowTensor);
  TF_LITE_ENSURE(context, flow != nullptr);

  // Check types.
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, kTfLiteFloat32);
  TF_LITE_ENSURE_TYPES_EQ(context, output->type, kTfLiteFloat32);
  TF_LITE_ENSURE_TYPES_EQ(context, flow->type, kTfLiteFloat32);

  // Check shapes. If input has shape of [b, h, w, c], flow must have shape of
  // [b, h, w, 2].
  TF_LITE_ENSURE_EQ(context, NumDimensions(flow), 4);
  TF_LITE_ENSURE_EQ(context, NumDimensions(input), 4);
  const RuntimeShape input_shape = GetTensorShape(input);
  const RuntimeShape flow_shape = GetTensorShape(flow);
  TF_LITE_ENSURE_EQ(context, input_shape.Dims(0), flow_shape.Dims(0));
  TF_LITE_ENSURE_EQ(context, input_shape.Dims(1), flow_shape.Dims(1));
  TF_LITE_ENSURE_EQ(context, input_shape.Dims(2), flow_shape.Dims(2));
  TF_LITE_ENSURE_MSG(context, input_shape.Dims(1) >= 2,
                     "Image height must be at least 2.");
  TF_LITE_ENSURE_MSG(context, input_shape.Dims(2) >= 2,
                     "Image width must be at least 2.");
  TF_LITE_ENSURE_MSG(context, flow_shape.Dims(3) == 2,
                     "The last dimension of flow tensor must be 2.");

  TfLiteIntArray* output_size = TfLiteIntArrayCopy(input->dims);
  return context->ResizeTensor(context, output, output_size);
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSperceptionPSdense_image_warpDTcc mht_2(mht_2_v, 299, "", "./tensorflow/lite/kernels/perception/dense_image_warp.cc", "Eval");

  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  const TfLiteTensor* flow = GetInput(context, node, kFlowTensor);
  TF_LITE_ENSURE(context, flow != nullptr);

  DenseImageWarp(GetTensorShape(input), GetTensorData<float>(input),
                 GetTensorShape(flow), GetTensorData<float>(flow),
                 GetTensorData<float>(output));
  return kTfLiteOk;
}

}  // namespace dense_image_warp

TfLiteRegistration* RegisterDenseImageWarp() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSperceptionPSdense_image_warpDTcc mht_3(mht_3_v, 318, "", "./tensorflow/lite/kernels/perception/dense_image_warp.cc", "RegisterDenseImageWarp");

  static TfLiteRegistration reg = {/*init=*/nullptr,
                                   /*free=*/nullptr, dense_image_warp::Prepare,
                                   dense_image_warp::Eval};
  return &reg;
}

// Alias for selective build.
TfLiteRegistration* Register_DENSE_IMAGE_WARP() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSperceptionPSdense_image_warpDTcc mht_4(mht_4_v, 329, "", "./tensorflow/lite/kernels/perception/dense_image_warp.cc", "Register_DENSE_IMAGE_WARP");

  return RegisterDenseImageWarp();
}

}  // namespace custom
}  // namespace ops
}  // namespace tflite
