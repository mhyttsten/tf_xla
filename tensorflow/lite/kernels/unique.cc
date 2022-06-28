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
class MHTracer_DTPStensorflowPSlitePSkernelsPSuniqueDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSuniqueDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSuniqueDTcc() {
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

#include <stddef.h>
#include <stdint.h>

#include <map>
#include <memory>
#include <vector>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace unique {

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("buffer: \"" + (buffer == nullptr ? std::string("nullptr") : std::string((char*)buffer)) + "\"");
   MHTracer_DTPStensorflowPSlitePSkernelsPSuniqueDTcc mht_0(mht_0_v, 205, "", "./tensorflow/lite/kernels/unique.cc", "Init");

  return nullptr;
}

void Free(TfLiteContext* context, void* buffer) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSuniqueDTcc mht_1(mht_1_v, 212, "", "./tensorflow/lite/kernels/unique.cc", "Free");
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSuniqueDTcc mht_2(mht_2_v, 217, "", "./tensorflow/lite/kernels/unique.cc", "Prepare");

  static const int kOutputUniqueTensor = 0;
  static const int kOutputIndexTensor = 1;

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 2);
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
  TfLiteTensor* output_unique_tensor;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, kOutputUniqueTensor,
                                           &output_unique_tensor));
  TfLiteTensor* output_index_tensor;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, kOutputIndexTensor,
                                           &output_index_tensor));

  // The op only supports 1D input.
  TF_LITE_ENSURE_EQ(context, NumDimensions(input), 1);
  TfLiteIntArray* output_index_shape = TfLiteIntArrayCopy(input->dims);
  // The unique values are determined during evaluation, so we don't know yet
  // the size of the output tensor.
  SetTensorToDynamic(output_unique_tensor);
  return context->ResizeTensor(context, output_index_tensor,
                               output_index_shape);
}

namespace {

// Actual evaluation for the unique op.
template <typename T, typename I>
TfLiteStatus EvalImpl(TfLiteContext* context, const TfLiteTensor* input,
                      TfLiteNode* node) {
  // Map from value, to index in the unique elements vector.
  // Note that we prefer to use map than unordered_map as it showed less
  // increase in the binary size.
  std::map<T, int> unique_values;
  TfLiteTensor* output_indexes;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 1, &output_indexes));
  std::vector<T> output_values;
  I* indexes = GetTensorData<I>(output_indexes);
  const T* data = GetTensorData<T>(input);
  const int num_elements = NumElements(input);

  for (int i = 0; i < num_elements; ++i) {
    const auto element_it = unique_values.find(data[i]);
    if (element_it != unique_values.end()) {
      indexes[i] = element_it->second;
    } else {
      const int unique_index = unique_values.size();
      unique_values[data[i]] = unique_index;
      indexes[i] = unique_index;
      output_values.push_back(data[i]);
    }
  }
  // Allocate output tensor.
  TfLiteTensor* unique_output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &unique_output));
  std::unique_ptr<TfLiteIntArray, void (*)(TfLiteIntArray*)> shape(
      TfLiteIntArrayCreate(NumDimensions(input)), TfLiteIntArrayFree);
  shape->data[0] = unique_values.size();
  TF_LITE_ENSURE_STATUS(
      context->ResizeTensor(context, unique_output, shape.release()));
  // Set the values in the output tensor.
  T* output_unique_values = GetTensorData<T>(unique_output);
  for (int i = 0; i < output_values.size(); ++i) {
    output_unique_values[i] = output_values[i];
  }
  return kTfLiteOk;
}

template <typename T>
TfLiteStatus EvalImpl(TfLiteContext* context, const TfLiteTensor* input,
                      TfLiteNode* node) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSuniqueDTcc mht_3(mht_3_v, 291, "", "./tensorflow/lite/kernels/unique.cc", "EvalImpl");

  auto* params = reinterpret_cast<TfLiteUniqueParams*>(node->builtin_data);
  if (params == nullptr) {
    context->ReportError(context, "Null params passed");
    return kTfLiteError;
  }
  switch (params->index_out_type) {
    case kTfLiteInt32:
      return EvalImpl<T, int32_t>(context, input, node);
    case kTfLiteInt64:
      return EvalImpl<T, int64_t>(context, input, node);
    default:
      context->ReportError(
          context,
          "Unique index output array can only be Int32 or In64, requested: %s",
          TfLiteTypeGetName(params->index_out_type));
  }
  return kTfLiteError;
}

}  // namespace

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSuniqueDTcc mht_4(mht_4_v, 316, "", "./tensorflow/lite/kernels/unique.cc", "Eval");

  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
  TfLiteTensor* output_index_tensor;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, 1, &output_index_tensor));
  TF_LITE_ENSURE_EQ(context, NumElements(output_index_tensor),
                    NumElements(input));

  switch (input->type) {
    case kTfLiteInt8:
      TF_LITE_ENSURE_STATUS(EvalImpl<int8_t>(context, input, node));
      break;
    case kTfLiteInt16:
      TF_LITE_ENSURE_STATUS(EvalImpl<int16_t>(context, input, node));
      break;
    case kTfLiteInt32:
      TF_LITE_ENSURE_STATUS(EvalImpl<int32_t>(context, input, node));
      break;
    case kTfLiteInt64:
      TF_LITE_ENSURE_STATUS(EvalImpl<int64_t>(context, input, node));
      break;
    case kTfLiteFloat32:
      TF_LITE_ENSURE_STATUS(EvalImpl<float>(context, input, node));
      break;
    case kTfLiteUInt8:
      TF_LITE_ENSURE_STATUS(EvalImpl<uint8_t>(context, input, node));
      break;
    default:
      context->ReportError(context, "Currently Unique doesn't support type: %s",
                           TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace unique

TfLiteRegistration* Register_UNIQUE() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSuniqueDTcc mht_5(mht_5_v, 357, "", "./tensorflow/lite/kernels/unique.cc", "Register_UNIQUE");

  static TfLiteRegistration r = {unique::Init, unique::Free, unique::Prepare,
                                 unique::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
