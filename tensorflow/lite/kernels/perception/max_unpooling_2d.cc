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
class MHTracer_DTPStensorflowPSlitePSkernelsPSperceptionPSmax_unpooling_2dDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSperceptionPSmax_unpooling_2dDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSperceptionPSmax_unpooling_2dDTcc() {
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

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"
namespace tflite {
namespace ops {
namespace custom {

namespace max_unpooling_2d {

constexpr int kDataInputTensor = 0;
constexpr int kIndicesTensor = 1;
constexpr int kOutputTensor = 0;

// TODO(b/175003241): Move this logic to lite/kernels/internal when promoting
// this op to a builtin op.
inline void MaxUnpooling(const RuntimeShape& input_shape,
                         const float* input_data, const int32_t* indices_data,
                         const RuntimeShape& output_shape, float* output_data) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSperceptionPSmax_unpooling_2dDTcc mht_0(mht_0_v, 204, "", "./tensorflow/lite/kernels/perception/max_unpooling_2d.cc", "MaxUnpooling");

  std::memset(output_data, 0, output_shape.FlatSize() * sizeof(float));
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int depth = MatchingDim(input_shape, 3, output_shape, 3);
  const int batch_stride =
      output_shape.Dims(1) * output_shape.Dims(2) * output_shape.Dims(3);
  for (int batch = 0; batch < batches; ++batch) {
    for (int in_y = 0; in_y < input_shape.Dims(1); ++in_y) {
      for (int in_x = 0; in_x < input_shape.Dims(2); ++in_x) {
        for (int channel = 0; channel < depth; ++channel) {
          const auto input_offset =
              Offset(input_shape, batch, in_y, in_x, channel);
          int idx = indices_data[input_offset];
          output_data[batch * batch_stride + idx] = input_data[input_offset];
        }
      }
    }
  }
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSperceptionPSmax_unpooling_2dDTcc mht_1(mht_1_v, 227, "", "./tensorflow/lite/kernels/perception/max_unpooling_2d.cc", "Prepare");

  auto* params =
      reinterpret_cast<const TfLitePoolParams*>(node->custom_initial_data);

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);
  const TfLiteTensor* input = GetInput(context, node, kDataInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  const TfLiteTensor* indices = GetInput(context, node, kIndicesTensor);
  TF_LITE_ENSURE(context, indices != nullptr);
  TF_LITE_ENSURE_EQ(context, NumDimensions(indices), 4);
  TF_LITE_ENSURE_EQ(context, NumDimensions(input), 4);
  TF_LITE_ENSURE_EQ(context, input->type, kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, output->type, kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, indices->type, kTfLiteInt32);
  TF_LITE_ENSURE(context, params->padding != kTfLitePaddingUnknown);

  // Size of input and indices tensor must match.
  const RuntimeShape input_shape = GetTensorShape(input);
  const RuntimeShape indices_shape = GetTensorShape(indices);
  TF_LITE_ENSURE_MSG(
      context, input_shape.DimensionsCount() == indices_shape.DimensionsCount(),
      "Input and indices must have the same shape.");
  for (int i = 0; i < input_shape.DimensionsCount(); ++i) {
    TF_LITE_ENSURE_MSG(context, input_shape.Dims(i) == indices_shape.Dims(i),
                       "Input and indices must have the same shape.");
  }

  int batches = input->dims->data[0];
  int height = input->dims->data[1];
  int width = input->dims->data[2];
  int channels_out = input->dims->data[3];

  int out_width, out_height;
  if (params->padding == kTfLitePaddingSame) {
    out_width = width * params->stride_width;
    out_height = height * params->stride_height;
  } else {
    out_width = (width - 1) * params->stride_width + params->filter_width;
    out_height = (height - 1) * params->stride_height + params->filter_height;
  }

  TfLiteIntArray* output_size = TfLiteIntArrayCreate(4);
  output_size->data[0] = batches;
  output_size->data[1] = out_height;
  output_size->data[2] = out_width;
  output_size->data[3] = channels_out;
  return context->ResizeTensor(context, output, output_size);
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSperceptionPSmax_unpooling_2dDTcc mht_2(mht_2_v, 282, "", "./tensorflow/lite/kernels/perception/max_unpooling_2d.cc", "Eval");

  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);
  const TfLiteTensor* input = GetInput(context, node, kDataInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  const TfLiteTensor* indices = GetInput(context, node, kIndicesTensor);
  TF_LITE_ENSURE(context, indices != nullptr);

  MaxUnpooling(GetTensorShape(input), GetTensorData<float>(input),
               GetTensorData<int32_t>(indices), GetTensorShape(output),
               GetTensorData<float>(output));
  return kTfLiteOk;
}

}  // namespace max_unpooling_2d

TfLiteRegistration* RegisterMaxUnpooling2D() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSperceptionPSmax_unpooling_2dDTcc mht_3(mht_3_v, 301, "", "./tensorflow/lite/kernels/perception/max_unpooling_2d.cc", "RegisterMaxUnpooling2D");

  static TfLiteRegistration reg = {/*init=*/nullptr,
                                   /*free=*/nullptr, max_unpooling_2d::Prepare,
                                   max_unpooling_2d::Eval};
  return &reg;
}

// Alias for selective build.
TfLiteRegistration* Register_MAX_UNPOOLING2D() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSperceptionPSmax_unpooling_2dDTcc mht_4(mht_4_v, 312, "", "./tensorflow/lite/kernels/perception/max_unpooling_2d.cc", "Register_MAX_UNPOOLING2D");

  return RegisterMaxUnpooling2D();
}

}  // namespace custom
}  // namespace ops
}  // namespace tflite
