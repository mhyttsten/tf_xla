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
class MHTracer_DTPStensorflowPSlitePSkernelsPSfillDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSfillDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSfillDTcc() {
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

#include <stdint.h>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/string_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace fill {

namespace {

constexpr int kDimsTensor = 0;
constexpr int kValueTensor = 1;
constexpr int kOutputTensor = 0;

template <typename T>
TfLiteStatus ResizeOutputImpl(TfLiteContext* context, const TfLiteTensor* dims,
                              TfLiteTensor* output) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSfillDTcc mht_0(mht_0_v, 207, "", "./tensorflow/lite/kernels/fill.cc", "ResizeOutputImpl");

  TfLiteIntArray* output_shape = TfLiteIntArrayCreate(dims->dims->data[0]);
  for (int i = 0; i < output_shape->size; ++i) {
    T data = GetTensorData<T>(dims)[i];
    if (data < 0) {
      TfLiteIntArrayFree(output_shape);
      context->ReportError(context, "Fill dimensions must be >= 0", dims->type);
      return kTfLiteError;
    }
    output_shape->data[i] = data;
  }
  return context->ResizeTensor(context, output, output_shape);
}

TfLiteStatus ResizeOutput(TfLiteContext* context, const TfLiteTensor* dims,
                          TfLiteTensor* output) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSfillDTcc mht_1(mht_1_v, 225, "", "./tensorflow/lite/kernels/fill.cc", "ResizeOutput");

  switch (dims->type) {
    case kTfLiteInt32:
      return ResizeOutputImpl<int32_t>(context, dims, output);
    case kTfLiteInt64:
      return ResizeOutputImpl<int64_t>(context, dims, output);
    default:
      context->ReportError(
          context,
          "Fill only currently supports int32, int64 for input 0, "
          "got %d.",
          dims->type);
      return kTfLiteError;
  }
}

}  // namespace

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSfillDTcc mht_2(mht_2_v, 246, "", "./tensorflow/lite/kernels/fill.cc", "Prepare");

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* dims;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kDimsTensor, &dims));
  const TfLiteTensor* value;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kValueTensor, &value));

  // Make sure the 1st input tensor is 1-D.
  TF_LITE_ENSURE_EQ(context, NumDimensions(dims), 1);

  // Make sure the 1st input tensor is int32 or int64.
  const auto dtype = dims->type;
  TF_LITE_ENSURE(context, dtype == kTfLiteInt32 || dtype == kTfLiteInt64);

  // Make sure the 2nd input tensor is a scalar.
  TF_LITE_ENSURE_EQ(context, NumDimensions(value), 0);

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  output->type = value->type;

  TF_LITE_ENSURE_EQ(context, output->params.scale, value->params.scale);
  TF_LITE_ENSURE_EQ(context, output->params.zero_point,
                    value->params.zero_point);

  if (value->type == kTfLiteInt16) {
    TF_LITE_ENSURE_EQ(context, value->params.zero_point, 0);
  }

  if (IsConstantTensor(dims)) {
    TF_LITE_ENSURE_OK(context, ResizeOutput(context, dims, output));
  } else {
    SetTensorToDynamic(output);
  }
  return kTfLiteOk;
}

TfLiteStatus FillString(const TfLiteTensor* value, TfLiteTensor* output) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSfillDTcc mht_3(mht_3_v, 289, "", "./tensorflow/lite/kernels/fill.cc", "FillString");

  DynamicBuffer buffer;
  const auto string_ref = GetString(value, 0);
  int n = 1;
  for (int i = 0; i < output->dims->size; ++i) {
    n *= output->dims->data[i];
  }
  for (int i = 0; i < n; ++i) {
    buffer.AddString(string_ref.str, string_ref.len);
  }
  buffer.WriteToTensor(output, /*new_shape=*/nullptr);
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSfillDTcc mht_4(mht_4_v, 306, "", "./tensorflow/lite/kernels/fill.cc", "Eval");

  const TfLiteTensor* value;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kValueTensor, &value));

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  if (IsDynamicTensor(output)) {
    const TfLiteTensor* dims;
    TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kDimsTensor, &dims));
    TF_LITE_ENSURE_OK(context, ResizeOutput(context, dims, output));
  }
#define TF_LITE_FILL(data_type)                                               \
  reference_ops::Fill(GetTensorShape(value), GetTensorData<data_type>(value), \
                      GetTensorShape(output),                                 \
                      GetTensorData<data_type>(output))
  switch (output->type) {
    case kTfLiteInt8:
      TF_LITE_FILL(int8_t);
      break;
    case kTfLiteInt16:
      TF_LITE_FILL(int16_t);
      break;
    case kTfLiteInt32:
      TF_LITE_FILL(int32_t);
      break;
    case kTfLiteInt64:
      TF_LITE_FILL(int64_t);
      break;
    case kTfLiteFloat32:
      TF_LITE_FILL(float);
      break;
    case kTfLiteBool:
      TF_LITE_FILL(bool);
      break;
    case kTfLiteString:
      FillString(value, output);
      break;
    default:
      context->ReportError(
          context,
          "Fill only currently supports int8, int16, int32, int64, float32, "
          "bool, string for input 1, got %d.",
          value->type);
      return kTfLiteError;
  }
#undef TF_LITE_FILL
  return kTfLiteOk;
}

}  // namespace fill

TfLiteRegistration* Register_FILL() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSfillDTcc mht_5(mht_5_v, 362, "", "./tensorflow/lite/kernels/fill.cc", "Register_FILL");

  static TfLiteRegistration r = {/*init=*/nullptr, /*free=*/nullptr,
                                 fill::Prepare, fill::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
