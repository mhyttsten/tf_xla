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
class MHTracer_DTPStensorflowPSlitePSkernelsPStransposeDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPStransposeDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPStransposeDTcc() {
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

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace transpose {

// This file has two implementations of Transpose.
enum KernelType {
  kReference,
  kGenericOptimized,
};

struct TransposeContext {
  TransposeContext(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPStransposeDTcc mht_0(mht_0_v, 207, "", "./tensorflow/lite/kernels/transpose.cc", "TransposeContext");

    input = GetInput(context, node, 0);
    perm = GetInput(context, node, 1);
    output = GetOutput(context, node, 0);
  }
  const TfLiteTensor* input;
  const TfLiteTensor* perm;
  TfLiteTensor* output;
};

TfLiteStatus ResizeOutputTensor(TfLiteContext* context,
                                TransposeContext* op_context) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPStransposeDTcc mht_1(mht_1_v, 221, "", "./tensorflow/lite/kernels/transpose.cc", "ResizeOutputTensor");

  int dims = NumDimensions(op_context->input);
  const int* perm_data = GetTensorData<int32_t>(op_context->perm);

  // Ensure validity of the permutations tensor as a 1D tensor.
  TF_LITE_ENSURE_EQ(context, NumDimensions(op_context->perm), 1);
  TF_LITE_ENSURE_EQ(context, op_context->perm->dims->data[0], dims);
  for (int idx = 0; idx < dims; ++idx) {
    TF_LITE_ENSURE_MSG(context, (perm_data[idx] >= 0 && perm_data[idx] < dims),
                       "Transpose op permutations array is out of bounds.");
  }

  // Determine size of output tensor.
  TfLiteIntArray* input_size = op_context->input->dims;
  TfLiteIntArray* output_size = TfLiteIntArrayCopy(input_size);
  for (int idx = 0; idx < dims; ++idx) {
    output_size->data[idx] = input_size->data[perm_data[idx]];
  }

  return context->ResizeTensor(context, op_context->output, output_size);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPStransposeDTcc mht_2(mht_2_v, 246, "", "./tensorflow/lite/kernels/transpose.cc", "Prepare");

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  TransposeContext op_context(context, node);

  // Ensure validity of input tensor.
  TF_LITE_ENSURE_MSG(context, NumDimensions(op_context.input) <= 5,
                     "Transpose op only supports 1D-5D input arrays.");
  TF_LITE_ENSURE_TYPES_EQ(context, op_context.input->type,
                          op_context.output->type);

  if (!IsConstantTensor(op_context.perm)) {
    SetTensorToDynamic(op_context.output);
    return kTfLiteOk;
  }
  return ResizeOutputTensor(context, &op_context);
}

template <KernelType kernel_type>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPStransposeDTcc mht_3(mht_3_v, 269, "", "./tensorflow/lite/kernels/transpose.cc", "Eval");

  TransposeContext op_context(context, node);

  // Resize the output tensor if the output tensor is dynamic.
  if (IsDynamicTensor(op_context.output)) {
    TF_LITE_ENSURE_OK(context, ResizeOutputTensor(context, &op_context));
  }

  const int* perm_data = GetTensorData<int32_t>(op_context.perm);
  const int size = op_context.perm->dims->data[0];
  TransposeParams params;
  params.perm_count = size;
  for (int i = 0; i < size; ++i) {
    params.perm[i] = perm_data[i];
  }

#define TF_LITE_TRANSPOSE(type, scalar)                     \
  type::Transpose(params, GetTensorShape(op_context.input), \
                  GetTensorData<scalar>(op_context.input),  \
                  GetTensorShape(op_context.output),        \
                  GetTensorData<scalar>(op_context.output))

  // Transpose kernel only does rearranging values not numeric evaluations on
  // each cell. It's safe to implement per size of scalar type and this trick
  // keeps the total code size in a reasonable range.
  switch (op_context.input->type) {
    case kTfLiteFloat32:
    case kTfLiteInt32:
      if (kernel_type == kGenericOptimized) {
        TF_LITE_TRANSPOSE(optimized_ops, int32_t);
      } else {
        TF_LITE_TRANSPOSE(reference_ops, int32_t);
      }
      break;
    case kTfLiteUInt8:
    case kTfLiteInt8:
      if (kernel_type == kGenericOptimized) {
        TF_LITE_TRANSPOSE(optimized_ops, int8_t);
      } else {
        TF_LITE_TRANSPOSE(reference_ops, int8_t);
      }
      break;
    case kTfLiteInt16:
      TF_LITE_TRANSPOSE(reference_ops, int16_t);
      break;
    case kTfLiteInt64:
      TF_LITE_TRANSPOSE(reference_ops, int64_t);
      break;
    case kTfLiteBool:
      if (sizeof(bool) == 1) {
        if (kernel_type == kGenericOptimized) {
          TF_LITE_TRANSPOSE(optimized_ops, int8_t);
        } else {
          TF_LITE_TRANSPOSE(reference_ops, int8_t);
        }
      } else {
        TF_LITE_TRANSPOSE(reference_ops, bool);
      }
      break;
    default:
      TF_LITE_KERNEL_LOG(context,
                         "Type %s is currently not supported by Transpose.",
                         TfLiteTypeGetName(op_context.input->type));
      return kTfLiteError;
  }
#undef TF_LITE_TRANSPOSE

  return kTfLiteOk;
}

}  // namespace transpose

TfLiteRegistration* Register_TRANSPOSE_REF() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPStransposeDTcc mht_4(mht_4_v, 344, "", "./tensorflow/lite/kernels/transpose.cc", "Register_TRANSPOSE_REF");

  static TfLiteRegistration r = {nullptr, nullptr, transpose::Prepare,
                                 transpose::Eval<transpose::kReference>};
  return &r;
}

TfLiteRegistration* Register_TRANSPOSE_GENERIC_OPTIMIZED() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPStransposeDTcc mht_5(mht_5_v, 353, "", "./tensorflow/lite/kernels/transpose.cc", "Register_TRANSPOSE_GENERIC_OPTIMIZED");

  static TfLiteRegistration r = {nullptr, nullptr, transpose::Prepare,
                                 transpose::Eval<transpose::kGenericOptimized>};
  return &r;
}

TfLiteRegistration* Register_TRANSPOSE() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPStransposeDTcc mht_6(mht_6_v, 362, "", "./tensorflow/lite/kernels/transpose.cc", "Register_TRANSPOSE");

  return Register_TRANSPOSE_GENERIC_OPTIMIZED();
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
