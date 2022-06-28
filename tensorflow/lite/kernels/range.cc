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
class MHTracer_DTPStensorflowPSlitePSkernelsPSrangeDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSrangeDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSrangeDTcc() {
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
#include <stdint.h>
#include <stdlib.h>

#include <functional>
#include <type_traits>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace range {
namespace {

constexpr int kStartTensor = 0;
constexpr int kLimitTensor = 1;
constexpr int kDeltaTensor = 2;
constexpr int kOutputTensor = 0;

template <typename T>
TfLiteStatus GetSize(TfLiteContext* context, T start, T limit, T delta,
                     int* size) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSrangeDTcc mht_0(mht_0_v, 211, "", "./tensorflow/lite/kernels/range.cc", "GetSize");

  TF_LITE_ENSURE(context, !std::equal_to<T>()(delta, 0));
  TF_LITE_ENSURE(
      context, (start >= limit && delta < 0) || (start <= limit && delta > 0));
  *size =
      (std::is_integral<T>::value
           ? ((std::abs(limit - start) + std::abs(delta) - 1) / std::abs(delta))
           : std::ceil(std::abs((limit - start) / delta)));
  return kTfLiteOk;
}

TfLiteStatus ResizeOutput(TfLiteContext* context, const TfLiteTensor* start,
                          const TfLiteTensor* limit, const TfLiteTensor* delta,
                          TfLiteTensor* output) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSrangeDTcc mht_1(mht_1_v, 227, "", "./tensorflow/lite/kernels/range.cc", "ResizeOutput");

  // The output will always be a 1-d array.
  int size = 0;
  switch (start->type) {
    case kTfLiteInt32: {
      TF_LITE_ENSURE_OK(context,
                        GetSize(context, *GetTensorData<int32_t>(start),
                                *GetTensorData<int32_t>(limit),
                                *GetTensorData<int32_t>(delta), &size));
      break;
    }
    case kTfLiteFloat32: {
      TF_LITE_ENSURE_OK(context, GetSize(context, *GetTensorData<float>(start),
                                         *GetTensorData<float>(limit),
                                         *GetTensorData<float>(delta), &size));
      break;
    }
    default: {
      context->ReportError(context, "Unknown data type: %d", start->type);
      return kTfLiteError;
    }
  }
  TfLiteIntArray* output_shape_array = TfLiteIntArrayCreate(1);
  output_shape_array->data[0] = size;
  return context->ResizeTensor(context, output, output_shape_array);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSrangeDTcc mht_2(mht_2_v, 257, "", "./tensorflow/lite/kernels/range.cc", "Prepare");

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 3);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* start;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kStartTensor, &start));
  const TfLiteTensor* limit;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kLimitTensor, &limit));
  const TfLiteTensor* delta;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kDeltaTensor, &delta));
  // Make sure all the inputs are scalars.
  TF_LITE_ENSURE_EQ(context, NumDimensions(start), 0);
  TF_LITE_ENSURE_EQ(context, NumDimensions(limit), 0);
  TF_LITE_ENSURE_EQ(context, NumDimensions(delta), 0);

  // Currently only supports int32 and float.
  // TODO(b/117912892): Support quantization as well.
  const auto dtype = start->type;
  if (dtype != kTfLiteFloat32 && dtype != kTfLiteInt32) {
    context->ReportError(context, "Unknown index output data type: %s",
                         TfLiteTypeGetName(dtype));
    return kTfLiteError;
  }

  TF_LITE_ENSURE_TYPES_EQ(context, limit->type, dtype);
  TF_LITE_ENSURE_TYPES_EQ(context, delta->type, dtype);

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  output->type = dtype;

  if (IsConstantTensor(start) && IsConstantTensor(limit) &&
      IsConstantTensor(delta)) {
    return ResizeOutput(context, start, limit, delta, output);
  }

  SetTensorToDynamic(output);
  return kTfLiteOk;
}

template <typename T>
void EvalImpl(const TfLiteTensor* start, const TfLiteTensor* delta,
              TfLiteTensor* output) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSrangeDTcc mht_3(mht_3_v, 303, "", "./tensorflow/lite/kernels/range.cc", "EvalImpl");

  const T start_value = *GetTensorData<T>(start);
  const T delta_value = *GetTensorData<T>(delta);
  T* output_data = GetTensorData<T>(output);
  const int num_elements = NumElements(output);
  T value = start_value;
  for (int i = 0; i < num_elements; ++i) {
    output_data[i] = value;
    value += delta_value;
  }
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSrangeDTcc mht_4(mht_4_v, 318, "", "./tensorflow/lite/kernels/range.cc", "Eval");

  const TfLiteTensor* start;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kStartTensor, &start));
  const TfLiteTensor* limit;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kLimitTensor, &limit));
  const TfLiteTensor* delta;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kDeltaTensor, &delta));

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  if (IsDynamicTensor(output)) {
    TF_LITE_ENSURE_OK(context,
                      ResizeOutput(context, start, limit, delta, output));
  }

  switch (output->type) {
    case kTfLiteInt32: {
      EvalImpl<int32_t>(start, delta, output);
      break;
    }
    case kTfLiteFloat32: {
      EvalImpl<float>(start, delta, output);
      break;
    }
    default: {
      context->ReportError(context, "Unsupported data type: %d", output->type);
      return kTfLiteError;
    }
  }
  return kTfLiteOk;
}

}  // namespace
}  // namespace range

TfLiteRegistration* Register_RANGE() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSrangeDTcc mht_5(mht_5_v, 358, "", "./tensorflow/lite/kernels/range.cc", "Register_RANGE");

  static TfLiteRegistration r = {nullptr, nullptr, range::Prepare, range::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
