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
class MHTracer_DTPStensorflowPSlitePSkernelsPSwhereDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSwhereDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSwhereDTcc() {
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
#include <stdint.h>

#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace where {

constexpr int kInputConditionTensor = 0;
constexpr int kOutputTensor = 0;

template <typename T>
TfLiteStatus ResizeOutputTensor(TfLiteContext* context,
                                const TfLiteTensor* cond_tensor,
                                TfLiteTensor* output_tensor) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSwhereDTcc mht_0(mht_0_v, 205, "", "./tensorflow/lite/kernels/where.cc", "ResizeOutputTensor");

  // Output tensor should have shape:
  // (num_true, cond_rank), where num_true denotes the number of true values
  // in condition.
  const RuntimeShape& cond_shape = GetTensorShape(cond_tensor);
  const int size = cond_shape.FlatSize();
  const int cond_rank = cond_shape.DimensionsCount();
  const T* cond_data = GetTensorData<T>(cond_tensor);

  int true_count = 0;
  for (int i = 0; i < size; ++i) {
    if (cond_data[i] != T(0)) {
      true_count++;
    }
  }
  TfLiteIntArray* output_dims = TfLiteIntArrayCreate(2);
  output_dims->data[0] = true_count;
  output_dims->data[1] = cond_rank;
  return context->ResizeTensor(context, output_tensor, output_dims);
}

template <typename T>
TfLiteStatus PrepareOutput(TfLiteContext* context,
                           const TfLiteTensor* cond_tensor,
                           TfLiteTensor* output) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSwhereDTcc mht_1(mht_1_v, 232, "", "./tensorflow/lite/kernels/where.cc", "PrepareOutput");

  // As output will be a 2D tensor of indices, use int64 to be consistent with
  // tensorflow.
  output->type = kTfLiteInt64;

  // Exit early if cond is a non-const tensor. Set output tensor to dynamic so
  // output size can be determined in Eval.
  if (!IsConstantTensor(cond_tensor)) {
    SetTensorToDynamic(output);
    return kTfLiteOk;
  }
  return ResizeOutputTensor<T>(context, cond_tensor, output);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSwhereDTcc mht_2(mht_2_v, 249, "", "./tensorflow/lite/kernels/where.cc", "Prepare");

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* cond_tensor;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputConditionTensor,
                                          &cond_tensor));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  switch (cond_tensor->type) {
    case kTfLiteBool:
      return PrepareOutput<bool>(context, cond_tensor, output);
    case kTfLiteFloat32:
      return PrepareOutput<float>(context, cond_tensor, output);
    case kTfLiteInt64:
      return PrepareOutput<int64_t>(context, cond_tensor, output);
    case kTfLiteInt32:
      return PrepareOutput<int32_t>(context, cond_tensor, output);
    case kTfLiteInt8:
      return PrepareOutput<int8_t>(context, cond_tensor, output);
    case kTfLiteUInt8:
      return PrepareOutput<uint8_t>(context, cond_tensor, output);
    case kTfLiteUInt32:
      return PrepareOutput<uint32_t>(context, cond_tensor, output);
    default:
      TF_LITE_KERNEL_LOG(context,
                         "Condition tensor has unsupported type: '%s'.",
                         TfLiteTypeGetName(cond_tensor->type));
  }
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSwhereDTcc mht_3(mht_3_v, 286, "", "./tensorflow/lite/kernels/where.cc", "Eval");

  const TfLiteTensor* cond_tensor;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputConditionTensor,
                                          &cond_tensor));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  if (IsDynamicTensor(output)) {
    switch (cond_tensor->type) {
      case kTfLiteBool:
        TF_LITE_ENSURE_OK(
            context, ResizeOutputTensor<bool>(context, cond_tensor, output));
        break;
      case kTfLiteFloat32:
        TF_LITE_ENSURE_OK(
            context, ResizeOutputTensor<float>(context, cond_tensor, output));
        break;
      case kTfLiteInt64:
        TF_LITE_ENSURE_OK(
            context, ResizeOutputTensor<int64_t>(context, cond_tensor, output));
        break;
      case kTfLiteInt32:
        TF_LITE_ENSURE_OK(
            context, ResizeOutputTensor<int32_t>(context, cond_tensor, output));
        break;
      case kTfLiteInt8:
        TF_LITE_ENSURE_OK(
            context, ResizeOutputTensor<int8_t>(context, cond_tensor, output));
        break;
      case kTfLiteUInt8:
        TF_LITE_ENSURE_OK(
            context, ResizeOutputTensor<uint8_t>(context, cond_tensor, output));
        break;
      case kTfLiteUInt32:
        TF_LITE_ENSURE_OK(context, ResizeOutputTensor<uint32_t>(
                                       context, cond_tensor, output));
        break;
      default:
        TF_LITE_KERNEL_LOG(context,
                           "Condition tensor has unsupported type: '%s'.",
                           TfLiteTypeGetName(cond_tensor->type));
    }
  }

  TfLiteIntArray* dims = cond_tensor->dims;
  if (dims->size == 0) {
    // Scalar tensors are not supported.
    TF_LITE_KERNEL_LOG(context, "Where op requires condition w/ rank > 0");
    return kTfLiteError;
  }

  switch (cond_tensor->type) {
    case kTfLiteBool:
      reference_ops::SelectTrueCoords(GetTensorShape(cond_tensor),
                                      GetTensorData<bool>(cond_tensor),
                                      GetTensorData<int64_t>(output));
      break;
    case kTfLiteFloat32:
      reference_ops::SelectTrueCoords(GetTensorShape(cond_tensor),
                                      GetTensorData<float>(cond_tensor),
                                      GetTensorData<int64_t>(output));
      break;
    case kTfLiteInt64:
      reference_ops::SelectTrueCoords(GetTensorShape(cond_tensor),
                                      GetTensorData<int64_t>(cond_tensor),
                                      GetTensorData<int64_t>(output));
      break;
    case kTfLiteInt32:
      reference_ops::SelectTrueCoords(GetTensorShape(cond_tensor),
                                      GetTensorData<int32_t>(cond_tensor),
                                      GetTensorData<int64_t>(output));
      break;
    case kTfLiteInt8:
      reference_ops::SelectTrueCoords(GetTensorShape(cond_tensor),
                                      GetTensorData<int8_t>(cond_tensor),
                                      GetTensorData<int64_t>(output));
      break;
    case kTfLiteUInt8:
      reference_ops::SelectTrueCoords(GetTensorShape(cond_tensor),
                                      GetTensorData<uint8_t>(cond_tensor),
                                      GetTensorData<int64_t>(output));
      break;
    case kTfLiteUInt32:
      reference_ops::SelectTrueCoords(GetTensorShape(cond_tensor),
                                      GetTensorData<uint32_t>(cond_tensor),
                                      GetTensorData<int64_t>(output));
      break;
    default:
      TF_LITE_KERNEL_LOG(context,
                         "Condition tensor has unsupported type: '%s'.",
                         TfLiteTypeGetName(cond_tensor->type));
  }
  return kTfLiteOk;
}
}  // namespace where

TfLiteRegistration* Register_WHERE() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSwhereDTcc mht_4(mht_4_v, 386, "", "./tensorflow/lite/kernels/where.cc", "Register_WHERE");

  static TfLiteRegistration r = {/*init*/ nullptr, /*free*/ nullptr,
                                 where::Prepare, where::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
