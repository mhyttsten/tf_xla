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
class MHTracer_DTPStensorflowPSlitePSkernelsPSsplitDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSsplitDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSsplitDTcc() {
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

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace split {

struct OpContext {
  OpContext(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSsplitDTcc mht_0(mht_0_v, 201, "", "./tensorflow/lite/kernels/split.cc", "OpContext");

    params = reinterpret_cast<TfLiteSplitParams*>(node->builtin_data);
    axis = GetInput(context, node, 0);
    input = GetInput(context, node, 1);
  }
  TfLiteSplitParams* params;
  const TfLiteTensor* axis;
  const TfLiteTensor* input;
};

TfLiteStatus UseDynamicOutputTensors(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSsplitDTcc mht_1(mht_1_v, 214, "", "./tensorflow/lite/kernels/split.cc", "UseDynamicOutputTensors");

  for (int i = 0; i < NumOutputs(node); ++i) {
    TfLiteTensor* tensor;
    TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, i, &tensor));
    SetTensorToDynamic(tensor);
  }
  return kTfLiteOk;
}

TfLiteStatus ResizeOutputTensors(TfLiteContext* context, TfLiteNode* node,
                                 const TfLiteTensor* axis,
                                 const TfLiteTensor* input, int num_splits) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSsplitDTcc mht_2(mht_2_v, 228, "", "./tensorflow/lite/kernels/split.cc", "ResizeOutputTensors");

  int axis_value = GetTensorData<int>(axis)[0];
  if (axis_value < 0) {
    axis_value += NumDimensions(input);
  }

  TF_LITE_ENSURE(context, axis_value >= 0);
  TF_LITE_ENSURE(context, axis_value < NumDimensions(input));

  const int input_size = SizeOfDimension(input, axis_value);
  TF_LITE_ENSURE(context, num_splits != 0);
  TF_LITE_ENSURE_MSG(context, input_size % num_splits == 0,
                     "Not an even split");
  const int slice_size = input_size / num_splits;

  for (int i = 0; i < NumOutputs(node); ++i) {
    TfLiteIntArray* output_dims = TfLiteIntArrayCopy(input->dims);
    output_dims->data[axis_value] = slice_size;
    TfLiteTensor* output;
    TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, i, &output));
    TF_LITE_ENSURE_STATUS(context->ResizeTensor(context, output, output_dims));
  }

  return kTfLiteOk;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSsplitDTcc mht_3(mht_3_v, 257, "", "./tensorflow/lite/kernels/split.cc", "Prepare");

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);

  OpContext op_context(context, node);

  TF_LITE_ENSURE_EQ(context, NumOutputs(node), op_context.params->num_splits);

  auto input_type = op_context.input->type;
  TF_LITE_ENSURE(context,
                 input_type == kTfLiteFloat32 || input_type == kTfLiteUInt8 ||
                     input_type == kTfLiteInt8 || input_type == kTfLiteInt16 ||
                     input_type == kTfLiteInt32);
  for (int i = 0; i < NumOutputs(node); ++i) {
    TfLiteTensor* tensor;
    TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, i, &tensor));
    tensor->type = input_type;
  }

  // If we know the contents of the 'axis' tensor, resize all outputs.
  // Otherwise, wait until Eval().
  if (IsConstantTensor(op_context.axis)) {
    return ResizeOutputTensors(context, node, op_context.axis, op_context.input,
                               op_context.params->num_splits);
  } else {
    return UseDynamicOutputTensors(context, node);
  }
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSsplitDTcc mht_4(mht_4_v, 288, "", "./tensorflow/lite/kernels/split.cc", "Eval");

  OpContext op_context(context, node);

  // When the 'axis' tensor is non-const we can't resize output tensors in
  // Prepare(), and we have to do it now.
  if (!IsConstantTensor(op_context.axis)) {
    TF_LITE_ENSURE_OK(
        context,
        ResizeOutputTensors(context, node, op_context.axis, op_context.input,
                            op_context.params->num_splits));
  }

  int axis_value = GetTensorData<int>(op_context.axis)[0];
  if (axis_value < 0) {
    axis_value += NumDimensions(op_context.input);
  }

  TF_LITE_ENSURE(context, axis_value >= 0);
  TF_LITE_ENSURE(context, axis_value < NumDimensions(op_context.input));

  // TODO(b/173221795): Our usage of VectorOfTensors could be optimized by
  // calculating it in Prepare, unless we defer shape calculation.
  // We can improve the optimized_ops version to handle other
  // cases too.
#define TF_LITE_SPLIT(scalar)                                       \
  VectorOfTensors<scalar> all_outputs(*context, *node->outputs);    \
  tflite::SplitParams op_params;                                    \
  op_params.num_split = NumOutputs(node);                           \
  op_params.axis = axis_value;                                      \
  reference_ops::Split(op_params, GetTensorShape(op_context.input), \
                       GetTensorData<scalar>(op_context.input),     \
                       all_outputs.shapes(), all_outputs.data());

  switch (op_context.input->type) {
    case kTfLiteFloat32: {
      TF_LITE_SPLIT(float);
      break;
    }
    case kTfLiteUInt8: {
      TF_LITE_SPLIT(uint8_t);
      break;
    }
    case kTfLiteInt8: {
      TF_LITE_SPLIT(int8_t);
      break;
    }
    case kTfLiteInt16: {
      TF_LITE_SPLIT(int16_t);
      break;
    }
    case kTfLiteInt32: {
      TF_LITE_SPLIT(int32_t);
      break;
    }
    default:
      context->ReportError(context, "Type %s currently not supported.",
                           TfLiteTypeGetName(op_context.input->type));
      return kTfLiteError;
  }
#undef TF_LITE_SPLIT

  return kTfLiteOk;
}

}  // namespace split

TfLiteRegistration* Register_SPLIT() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSsplitDTcc mht_5(mht_5_v, 357, "", "./tensorflow/lite/kernels/split.cc", "Register_SPLIT");

  static TfLiteRegistration r = {nullptr, nullptr, split::Prepare, split::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
