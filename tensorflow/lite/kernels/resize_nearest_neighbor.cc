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
class MHTracer_DTPStensorflowPSlitePSkernelsPSresize_nearest_neighborDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSresize_nearest_neighborDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSresize_nearest_neighborDTcc() {
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
#include "tensorflow/lite/kernels/internal/reference/resize_nearest_neighbor.h"

#include <stdint.h>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/optimized/neon_check.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace resize_nearest_neighbor {

// This file has three implementations of RESIZE_NEAREST_NEIGHBOR.
enum KernelType {
  kReference,
  kGenericOptimized,
  kNeonOptimized,
};

constexpr int kInputTensor = 0;
constexpr int kSizeTensor = 1;
constexpr int kOutputTensor = 0;

TfLiteStatus ResizeOutputTensor(TfLiteContext* context,
                                const TfLiteTensor* input,
                                const TfLiteTensor* size,
                                TfLiteTensor* output) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSresize_nearest_neighborDTcc mht_0(mht_0_v, 218, "", "./tensorflow/lite/kernels/resize_nearest_neighbor.cc", "ResizeOutputTensor");

  TfLiteIntArray* output_size = TfLiteIntArrayCreate(4);
  output_size->data[0] = input->dims->data[0];
  const int32* size_data = GetTensorData<int32>(size);
  output_size->data[1] = size_data[0];
  output_size->data[2] = size_data[1];
  output_size->data[3] = input->dims->data[3];
  return context->ResizeTensor(context, output, output_size);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSresize_nearest_neighborDTcc mht_1(mht_1_v, 231, "", "./tensorflow/lite/kernels/resize_nearest_neighbor.cc", "Prepare");

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  const TfLiteTensor* size;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kSizeTensor, &size));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  // Our current implementations relies on the input being 4D,
  // and the size being 1D tensor with exactly 2 elements.
  TF_LITE_ENSURE_EQ(context, NumDimensions(input), 4);
  TF_LITE_ENSURE_EQ(context, NumDimensions(size), 1);
  TF_LITE_ENSURE_TYPES_EQ(context, size->type, kTfLiteInt32);
  TF_LITE_ENSURE_EQ(context, size->dims->data[0], 2);

  output->type = input->type;

  if (!IsConstantTensor(size)) {
    SetTensorToDynamic(output);
    return kTfLiteOk;
  }
  return ResizeOutputTensor(context, input, size, output);
}

template <KernelType kernel_type>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSresize_nearest_neighborDTcc mht_2(mht_2_v, 263, "", "./tensorflow/lite/kernels/resize_nearest_neighbor.cc", "Eval");

  auto* params =
      reinterpret_cast<TfLiteResizeNearestNeighborParams*>(node->builtin_data);

  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  const TfLiteTensor* size;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kSizeTensor, &size));

  if (IsDynamicTensor(output)) {
    TF_LITE_ENSURE_OK(context,
                      ResizeOutputTensor(context, input, size, output));
  }

  tflite::ResizeNearestNeighborParams op_params;
  op_params.align_corners = params->align_corners;
  op_params.half_pixel_centers = params->half_pixel_centers;

  if (output->type == kTfLiteFloat32) {
    reference_ops::ResizeNearestNeighbor(
        op_params, GetTensorShape(input), GetTensorData<int32>(input),
        GetTensorShape(size), GetTensorData<int32>(size),
        GetTensorShape(output), GetTensorData<int32>(output));
  } else if (output->type == kTfLiteUInt8) {
    if (kernel_type == kReference) {
      reference_ops::ResizeNearestNeighbor(
          op_params, GetTensorShape(input), GetTensorData<uint8_t>(input),
          GetTensorShape(size), GetTensorData<int32>(size),
          GetTensorShape(output), GetTensorData<uint8_t>(output));
    }
    if (kernel_type == kGenericOptimized || kernel_type == kNeonOptimized) {
      optimized_ops::ResizeNearestNeighbor(
          op_params, GetTensorShape(input), GetTensorData<uint8_t>(input),
          GetTensorShape(size), GetTensorData<int32>(size),
          GetTensorShape(output), GetTensorData<uint8_t>(output));
    }
  } else if (output->type == kTfLiteInt8) {
    reference_ops::ResizeNearestNeighbor(
        op_params, GetTensorShape(input), GetTensorData<int8_t>(input),
        GetTensorShape(size), GetTensorData<int32>(size),
        GetTensorShape(output), GetTensorData<int8_t>(output));
  } else if (output->type == kTfLiteInt16) {
    reference_ops::ResizeNearestNeighbor(
        op_params, GetTensorShape(input), GetTensorData<int16_t>(input),
        GetTensorShape(size), GetTensorData<int32>(size),
        GetTensorShape(output), GetTensorData<int16_t>(output));
  } else {
    TF_LITE_KERNEL_LOG(
        context, "Output type is %s, requires float, uint8, int8 or int16.",
        TfLiteTypeGetName(output->type));
    return kTfLiteError;
  }

  return kTfLiteOk;
}

}  // namespace resize_nearest_neighbor

TfLiteRegistration* Register_RESIZE_NEAREST_NEIGHBOR_REF() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSresize_nearest_neighborDTcc mht_3(mht_3_v, 327, "", "./tensorflow/lite/kernels/resize_nearest_neighbor.cc", "Register_RESIZE_NEAREST_NEIGHBOR_REF");

  static TfLiteRegistration r = {
      nullptr, nullptr, resize_nearest_neighbor::Prepare,
      resize_nearest_neighbor::Eval<resize_nearest_neighbor::kReference>};
  return &r;
}

TfLiteRegistration* Register_RESIZE_NEAREST_NEIGHBOR_GENERIC_OPT() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSresize_nearest_neighborDTcc mht_4(mht_4_v, 337, "", "./tensorflow/lite/kernels/resize_nearest_neighbor.cc", "Register_RESIZE_NEAREST_NEIGHBOR_GENERIC_OPT");

  static TfLiteRegistration r = {
      nullptr, nullptr, resize_nearest_neighbor::Prepare,
      resize_nearest_neighbor::Eval<
          resize_nearest_neighbor::kGenericOptimized>};
  return &r;
}

TfLiteRegistration* Register_RESIZE_NEAREST_NEIGHBOR_NEON_OPT() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSresize_nearest_neighborDTcc mht_5(mht_5_v, 348, "", "./tensorflow/lite/kernels/resize_nearest_neighbor.cc", "Register_RESIZE_NEAREST_NEIGHBOR_NEON_OPT");

  static TfLiteRegistration r = {
      nullptr, nullptr, resize_nearest_neighbor::Prepare,
      resize_nearest_neighbor::Eval<resize_nearest_neighbor::kNeonOptimized>};
  return &r;
}

TfLiteRegistration* Register_RESIZE_NEAREST_NEIGHBOR() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSresize_nearest_neighborDTcc mht_6(mht_6_v, 358, "", "./tensorflow/lite/kernels/resize_nearest_neighbor.cc", "Register_RESIZE_NEAREST_NEIGHBOR");

#ifdef USE_NEON
  return Register_RESIZE_NEAREST_NEIGHBOR_NEON_OPT();
#else
  return Register_RESIZE_NEAREST_NEIGHBOR_GENERIC_OPT();
#endif
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
