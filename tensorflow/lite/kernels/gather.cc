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
class MHTracer_DTPStensorflowPSlitePSkernelsPSgatherDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSgatherDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSgatherDTcc() {
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
#include <stdint.h>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/string_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace gather {
constexpr int kInputTensor = 0;
constexpr int kInputPositions = 1;
constexpr int kOutputTensor = 0;

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSgatherDTcc mht_0(mht_0_v, 204, "", "./tensorflow/lite/kernels/gather.cc", "Prepare");

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const auto* params =
      reinterpret_cast<const TfLiteGatherParams*>(node->builtin_data);
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  const TfLiteTensor* positions;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputPositions, &positions));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  switch (positions->type) {
    case kTfLiteInt64:
    case kTfLiteInt32:
      break;
    default:
      context->ReportError(
          context, "Positions of type '%s' are not supported by gather.",
          TfLiteTypeGetName(positions->type));
      return kTfLiteError;
  }

  // Assign to output the input type.
  output->type = input->type;

  // Check conditions for different types.
  switch (input->type) {
    case kTfLiteFloat32:
    case kTfLiteUInt8:
    case kTfLiteInt8:
    case kTfLiteInt16:
    case kTfLiteInt64:
    case kTfLiteInt32:
    case kTfLiteBool:
      break;
    case kTfLiteString: {
      // Only 1D input is supported.
      TF_LITE_ENSURE_EQ(context, NumDimensions(input), 1);
    } break;
    default:
      context->ReportError(context, "Type '%s' is not supported by gather.",
                           TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }

  int axis = params->axis;
  if (axis < 0) {
    axis += NumDimensions(input);
  }
  TF_LITE_ENSURE(context, 0 <= axis && axis < NumDimensions(input));

  int batch_dims = params->batch_dims;
  // batch_dims should be in range: [-rank(positions), rank(positions)].
  // Negative batch_dims is added with rank of positions.
  if (batch_dims < 0) {
    batch_dims += NumDimensions(positions);
  }
  TF_LITE_ENSURE(context, batch_dims <= axis);
  TF_LITE_ENSURE(context, 0 <= batch_dims && batch_dims < NumDimensions(input));
  TF_LITE_ENSURE(context, batch_dims <= NumDimensions(positions));
  for (int i = 0; i < batch_dims; ++i) {
    TF_LITE_ENSURE_EQ(context, input->dims->data[i], positions->dims->data[i]);
  }

  const int num_dimensions =
      NumDimensions(input) + NumDimensions(positions) - 1 - batch_dims;
  TfLiteIntArray* output_shape = TfLiteIntArrayCreate(num_dimensions);
  int output_index = 0;
  for (int i = 0; i < axis; ++i) {
    output_shape->data[output_index++] = input->dims->data[i];
  }
  for (int i = batch_dims; i < positions->dims->size; ++i) {
    output_shape->data[output_index++] = positions->dims->data[i];
  }
  for (int i = axis + 1; i < input->dims->size; ++i) {
    output_shape->data[output_index++] = input->dims->data[i];
  }
  return context->ResizeTensor(context, output, output_shape);
}

template <typename InputT, typename PositionsT>
TfLiteStatus Gather(TfLiteContext* context, const TfLiteGatherParams& params,
                    const TfLiteTensor* input, const TfLiteTensor* positions,
                    TfLiteTensor* output) {
  const PositionsT* indexes = GetTensorData<PositionsT>(positions);
  bool indices_has_only_positive_elements = true;
  const size_t num_indices = positions->bytes / sizeof(PositionsT);
  for (size_t i = 0; i < num_indices; i++) {
    if (indexes[i] < 0) {
      indices_has_only_positive_elements = false;
      break;
    }
  }
  TF_LITE_ENSURE(context, indices_has_only_positive_elements);

  tflite::GatherParams op_params;
  op_params.axis = params.axis;
  op_params.batch_dims = params.batch_dims;
  optimized_ops::Gather(op_params, GetTensorShape(input),
                        GetTensorData<InputT>(input), GetTensorShape(positions),
                        GetTensorData<PositionsT>(positions),
                        GetTensorShape(output), GetTensorData<InputT>(output));
  return kTfLiteOk;
}

template <typename PositionT>
TfLiteStatus GatherStrings(TfLiteContext* context, const TfLiteTensor* input,
                           const TfLiteTensor* positions,
                           TfLiteTensor* output) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSgatherDTcc mht_1(mht_1_v, 319, "", "./tensorflow/lite/kernels/gather.cc", "GatherStrings");

  DynamicBuffer buffer;

  const PositionT* indexes = GetTensorData<PositionT>(positions);
  bool indices_has_only_positive_elements = true;
  const size_t num_indices = positions->bytes / sizeof(PositionT);
  for (size_t i = 0; i < num_indices; i++) {
    if (indexes[i] < 0) {
      indices_has_only_positive_elements = false;
      break;
    }
  }
  TF_LITE_ENSURE(context, indices_has_only_positive_elements);

  const PositionT num_strings = GetStringCount(input);
  const int num_indexes = NumElements(positions);

  for (int i = 0; i < num_indexes; ++i) {
    const PositionT pos = indexes[i];
    TF_LITE_ENSURE(context, pos < num_strings);
    const auto string_ref = GetString(input, pos);
    buffer.AddString(string_ref.str, string_ref.len);
  }
  buffer.WriteToTensor(output, /*new_shape=*/nullptr);
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSgatherDTcc mht_2(mht_2_v, 349, "", "./tensorflow/lite/kernels/gather.cc", "Eval");

  const auto* params =
      reinterpret_cast<const TfLiteGatherParams*>(node->builtin_data);
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  const TfLiteTensor* positions;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputPositions, &positions));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  if (positions->type == kTfLiteInt32) {
    switch (input->type) {
      case kTfLiteFloat32:
        return Gather<float, int32_t>(context, *params, input, positions,
                                      output);
      case kTfLiteUInt8:
        return Gather<uint8_t, int32_t>(context, *params, input, positions,
                                        output);
      case kTfLiteInt8:
        return Gather<int8_t, int32_t>(context, *params, input, positions,
                                       output);
      case kTfLiteInt16:
        return Gather<int16_t, int32_t>(context, *params, input, positions,
                                        output);
      case kTfLiteInt32:
        return Gather<int32_t, int32_t>(context, *params, input, positions,
                                        output);
      case kTfLiteInt64:
        return Gather<int64_t, int32_t>(context, *params, input, positions,
                                        output);
      case kTfLiteBool:
        return Gather<bool, int32_t>(context, *params, input, positions,
                                     output);
      case kTfLiteString:
        return GatherStrings<int32_t>(context, input, positions, output);
      default:
        context->ReportError(context, "Type '%s' is not supported by gather.",
                             TfLiteTypeGetName(input->type));
        return kTfLiteError;
    }
  }
  if (positions->type == kTfLiteInt64) {
    switch (input->type) {
      case kTfLiteFloat32:
        return Gather<float, int64_t>(context, *params, input, positions,
                                      output);
      case kTfLiteUInt8:
        return Gather<uint8_t, int64_t>(context, *params, input, positions,
                                        output);
      case kTfLiteInt8:
        return Gather<int8_t, int64_t>(context, *params, input, positions,
                                       output);
      case kTfLiteInt16:
        return Gather<int16_t, int64_t>(context, *params, input, positions,
                                        output);
      case kTfLiteInt32:
        return Gather<int32_t, int64_t>(context, *params, input, positions,
                                        output);
      case kTfLiteInt64:
        return Gather<int64_t, int64_t>(context, *params, input, positions,
                                        output);
      case kTfLiteBool:
        return Gather<bool, int64_t>(context, *params, input, positions,
                                     output);
      case kTfLiteString:
        return GatherStrings<int64_t>(context, input, positions, output);
      default:
        context->ReportError(context, "Type '%s' is not supported by gather.",
                             TfLiteTypeGetName(input->type));
        return kTfLiteError;
    }
  }
  context->ReportError(context,
                       "Positions of type '%s' are not supported by gather.",
                       TfLiteTypeGetName(positions->type));
  return kTfLiteError;
}
}  // namespace gather

TfLiteRegistration* Register_GATHER() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSgatherDTcc mht_3(mht_3_v, 433, "", "./tensorflow/lite/kernels/gather.cc", "Register_GATHER");

  static TfLiteRegistration r = {nullptr, nullptr, gather::Prepare,
                                 gather::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
