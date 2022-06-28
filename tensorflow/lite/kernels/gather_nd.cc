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
class MHTracer_DTPStensorflowPSlitePSkernelsPSgather_ndDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSgather_ndDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSgather_ndDTcc() {
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

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace gather_nd {
constexpr int kParams = 0;
constexpr int kIndices = 1;
constexpr int kOutputTensor = 0;

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSgather_ndDTcc mht_0(mht_0_v, 201, "", "./tensorflow/lite/kernels/gather_nd.cc", "Prepare");

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* params;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kParams, &params));
  const TfLiteTensor* indices;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kIndices, &indices));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  switch (params->type) {
    case kTfLiteFloat32:
    case kTfLiteUInt8:
    case kTfLiteInt8:
    case kTfLiteInt16:
    case kTfLiteInt64:
    case kTfLiteInt32:
    case kTfLiteString:
      break;
    default:
      context->ReportError(
          context, "Params of type '%s' are not supported by gather_nd.",
          TfLiteTypeGetName(params->type));
      return kTfLiteError;
  }
  switch (indices->type) {
    case kTfLiteInt64:
    case kTfLiteInt32:
      break;
    default:
      context->ReportError(
          context, "Indices of type '%s' are not supported by gather_nd.",
          TfLiteTypeGetName(indices->type));
      return kTfLiteError;
  }

  const int params_rank = NumDimensions(params);
  const int indices_rank = NumDimensions(indices);
  const int indices_nd = SizeOfDimension(indices, indices_rank - 1);
  if (params_rank < 1) {
    context->ReportError(context, "Params must be at least a vector.");
    return kTfLiteError;
  }
  if (indices_rank < 1) {
    context->ReportError(context, "Indices must be at least a vector.");
    return kTfLiteError;
  }
  if (indices_nd > params_rank) {
    context->ReportError(
        context, "Index innermost dimension length must be <= params rank.");
    return kTfLiteError;
  }

  // Assign to output the input type.
  output->type = params->type;

  // The result shape is
  // indices.shape[:-1] + params.shape[indices.shape[-1]:]
  const int output_rank = indices_rank + params_rank - indices_nd - 1;
  TfLiteIntArray* output_shape = TfLiteIntArrayCreate(output_rank);
  int output_index = 0;
  for (int i = 0; i < indices_rank - 1; ++i) {
    output_shape->data[output_index++] = indices->dims->data[i];
  }
  for (int i = indices_nd; i < params_rank; ++i) {
    output_shape->data[output_index++] = params->dims->data[i];
  }
  return context->ResizeTensor(context, output, output_shape);
}

template <typename ParamsT, typename IndicesT>
TfLiteStatus GatherNd(const TfLiteTensor* params, const TfLiteTensor* indices,
                      TfLiteTensor* output) {
  reference_ops::GatherNd(
      GetTensorShape(params), GetTensorData<ParamsT>(params),
      GetTensorShape(indices), GetTensorData<IndicesT>(indices),
      GetTensorShape(output), GetTensorData<ParamsT>(output));
  return kTfLiteOk;
}

template <typename IndicesT>
TfLiteStatus GatherNdString(const TfLiteTensor* params,
                            const TfLiteTensor* indices, TfLiteTensor* output) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSgather_ndDTcc mht_1(mht_1_v, 288, "", "./tensorflow/lite/kernels/gather_nd.cc", "GatherNdString");

  reference_ops::GatherNdString(
      GetTensorShape(params), params, GetTensorShape(indices),
      GetTensorData<IndicesT>(indices), GetTensorShape(output), output);
  return kTfLiteOk;
}

template <typename IndicesT>
TfLiteStatus EvalGatherNd(TfLiteContext* context, const TfLiteTensor* params,
                          const TfLiteTensor* indices, TfLiteTensor* output) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSgather_ndDTcc mht_2(mht_2_v, 300, "", "./tensorflow/lite/kernels/gather_nd.cc", "EvalGatherNd");

  bool indices_has_only_positive_elements = true;
  const auto* indices_values = GetTensorData<IndicesT>(indices);
  const size_t num_indices = indices->bytes / sizeof(IndicesT);
  for (size_t i = 0; i < num_indices; i++) {
    if (indices_values[i] < 0) {
      indices_has_only_positive_elements = false;
      break;
    }
  }
  TF_LITE_ENSURE(context, indices_has_only_positive_elements);

  switch (params->type) {
    case kTfLiteFloat32:
      return GatherNd<float, IndicesT>(params, indices, output);
    case kTfLiteUInt8:
      return GatherNd<uint8_t, IndicesT>(params, indices, output);
    case kTfLiteInt8:
      return GatherNd<int8_t, IndicesT>(params, indices, output);
    case kTfLiteInt16:
      return GatherNd<int16_t, IndicesT>(params, indices, output);
    case kTfLiteInt32:
      return GatherNd<int32_t, IndicesT>(params, indices, output);
    case kTfLiteInt64:
      return GatherNd<int64_t, IndicesT>(params, indices, output);
    case kTfLiteString:
      return GatherNdString<IndicesT>(params, indices, output);
    default:
      context->ReportError(context,
                           "Params type '%s' are not supported by gather_nd.",
                           TfLiteTypeGetName(params->type));
      return kTfLiteError;
  }
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSgather_ndDTcc mht_3(mht_3_v, 338, "", "./tensorflow/lite/kernels/gather_nd.cc", "Eval");

  const TfLiteTensor* params;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kParams, &params));
  const TfLiteTensor* indices;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kIndices, &indices));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  // Prevent division by 0 in the helper.
  // In TF, GatherND supports empty `params` only when `indices` is also empty.
  TF_LITE_ENSURE(context,
                 (NumElements(params) == 0 && NumElements(indices) == 0) ||
                     NumElements(params) > 0);

  switch (indices->type) {
    case kTfLiteInt32:
      return EvalGatherNd<int32_t>(context, params, indices, output);
    case kTfLiteInt64:
      return EvalGatherNd<int64_t>(context, params, indices, output);
    default:
      context->ReportError(
          context, "Indices of type '%s' are not supported by gather_nd.",
          TfLiteTypeGetName(indices->type));
      return kTfLiteError;
  }
}
}  // namespace gather_nd

TfLiteRegistration* Register_GATHER_ND() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSgather_ndDTcc mht_4(mht_4_v, 370, "", "./tensorflow/lite/kernels/gather_nd.cc", "Register_GATHER_ND");

  static TfLiteRegistration r = {/*init*/ nullptr, /*free*/ nullptr,
                                 gather_nd::Prepare, gather_nd::Eval};
  return &r;
}
}  // namespace builtin
}  // namespace ops
}  // namespace tflite
