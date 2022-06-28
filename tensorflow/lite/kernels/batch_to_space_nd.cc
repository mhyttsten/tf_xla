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
class MHTracer_DTPStensorflowPSlitePSkernelsPSbatch_to_space_ndDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSbatch_to_space_ndDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSbatch_to_space_ndDTcc() {
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
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace batch_to_space_nd {

// This file has two implementations of BatchToSpaceND.
enum KernelType {
  kReference,
  kGenericOptimized,
};

struct BatchToSpaceNDContext {
  BatchToSpaceNDContext(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSbatch_to_space_ndDTcc mht_0(mht_0_v, 206, "", "./tensorflow/lite/kernels/batch_to_space_nd.cc", "BatchToSpaceNDContext");

    input = GetInput(context, node, 0);
    block_shape = GetInput(context, node, 1);
    crops = GetInput(context, node, 2);
    output = GetOutput(context, node, 0);
  }
  const TfLiteTensor* input;
  const TfLiteTensor* block_shape;
  const TfLiteTensor* crops;
  TfLiteTensor* output;
};

// Currently, only 3D NHC or 4D NHWC input/output op_context are supported.
// In case of 3D input,it will be converted to 4D by adding W=1 to be NH1C.
// The 4D array need to have exactly 2 spatial dimensions.
// TODO(ycling): Support arbitrary dimension in BatchToSpaceND.
const int kInputMinDimensionNum = 3;
const int kInputMaxDimensionNum = 4;

TfLiteStatus ResizeOutputTensor(TfLiteContext* context,
                                BatchToSpaceNDContext* op_context) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSbatch_to_space_ndDTcc mht_1(mht_1_v, 229, "", "./tensorflow/lite/kernels/batch_to_space_nd.cc", "ResizeOutputTensor");

  TfLiteIntArray* input_size = op_context->input->dims;
  const int* block_shape = GetTensorData<int32>(op_context->block_shape);
  const int* crops = GetTensorData<int32>(op_context->crops);

  int spatial_dims_num = input_size->size - 2;
  // Block_shape should be a 1D tensor with dimension [spatial_dims_num].
  TF_LITE_ENSURE_EQ(context, NumDimensions(op_context->block_shape), 1);
  TF_LITE_ENSURE_EQ(context, op_context->block_shape->dims->data[0],
                    spatial_dims_num);
  // Crops should be a 2D tensor with dimension [spatial_dims_num, 2].
  TF_LITE_ENSURE_EQ(context, NumDimensions(op_context->crops), 2);
  TF_LITE_ENSURE_EQ(context, op_context->crops->dims->data[0],
                    spatial_dims_num);
  TF_LITE_ENSURE_EQ(context, op_context->crops->dims->data[1], 2);

  for (int i = 0; i < spatial_dims_num * 2; ++i) {
    TF_LITE_ENSURE(context, crops[i] >= 0);
  }

  TfLiteIntArray* output_size = TfLiteIntArrayCopy(input_size);
  int output_batch_size = input_size->data[0];
  for (int dim = 0; dim < spatial_dims_num; ++dim) {
    // Number of batch must be multiple of (block_shape[dim]).
    TF_LITE_ENSURE(context, block_shape[dim] != 0);
    TF_LITE_ENSURE_EQ(context, output_batch_size % block_shape[dim], 0);
    output_batch_size = output_batch_size / block_shape[dim];
    output_size->data[dim + 1] = input_size->data[dim + 1] * block_shape[dim] -
                                 crops[dim * 2] - crops[dim * 2 + 1];
  }

  output_size->data[0] = output_batch_size;
  output_size->data[input_size->size - 1] =
      input_size->data[input_size->size - 1];

  return context->ResizeTensor(context, op_context->output, output_size);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSbatch_to_space_ndDTcc mht_2(mht_2_v, 270, "", "./tensorflow/lite/kernels/batch_to_space_nd.cc", "Prepare");

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 3);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  BatchToSpaceNDContext op_context(context, node);
  TF_LITE_ENSURE(context,
                 NumDimensions(op_context.input) >= kInputMinDimensionNum);
  TF_LITE_ENSURE(context,
                 NumDimensions(op_context.input) <= kInputMaxDimensionNum);
  TF_LITE_ENSURE_EQ(context, op_context.input->type, op_context.output->type);

  if (!IsConstantTensor(op_context.block_shape) ||
      !IsConstantTensor(op_context.crops)) {
    SetTensorToDynamic(op_context.output);
    return kTfLiteOk;
  }
  return ResizeOutputTensor(context, &op_context);
}

template <KernelType kernel_type>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSbatch_to_space_ndDTcc mht_3(mht_3_v, 293, "", "./tensorflow/lite/kernels/batch_to_space_nd.cc", "Eval");

  BatchToSpaceNDContext op_context(context, node);

  // Resize the output tensor if the output tensor is dynamic.
  if (IsDynamicTensor(op_context.output)) {
    TF_LITE_ENSURE_OK(context, ResizeOutputTensor(context, &op_context));
  }

#define TF_LITE_BATCH_TO_SPACE_ND(type, scalar)                        \
  type::BatchToSpaceND(GetTensorShape(op_context.input),               \
                       GetTensorData<scalar>(op_context.input),        \
                       GetTensorShape(op_context.block_shape),         \
                       GetTensorData<int32_t>(op_context.block_shape), \
                       GetTensorShape(op_context.crops),               \
                       GetTensorData<int32_t>(op_context.crops),       \
                       GetTensorShape(op_context.output),              \
                       GetTensorData<scalar>(op_context.output))
  switch (op_context.input->type) {  // Already know in/out types are same.
    case kTfLiteFloat32:
      if (kernel_type == kReference) {
        TF_LITE_BATCH_TO_SPACE_ND(reference_ops, float);
      } else {
        TF_LITE_BATCH_TO_SPACE_ND(optimized_ops, float);
      }
      break;
    case kTfLiteUInt8:
      if (kernel_type == kReference) {
        TF_LITE_BATCH_TO_SPACE_ND(reference_ops, uint8_t);
      } else {
        TF_LITE_BATCH_TO_SPACE_ND(optimized_ops, uint8_t);
      }
      break;
    case kTfLiteInt8:
      if (kernel_type == kReference) {
        TF_LITE_BATCH_TO_SPACE_ND(reference_ops, int8_t);
      } else {
        TF_LITE_BATCH_TO_SPACE_ND(optimized_ops, int8_t);
      }
      break;
    case kTfLiteInt32:
      if (kernel_type == kReference) {
        TF_LITE_BATCH_TO_SPACE_ND(reference_ops, int32_t);
      } else {
        TF_LITE_BATCH_TO_SPACE_ND(optimized_ops, int32_t);
      }
      break;
    case kTfLiteInt64:
      if (kernel_type == kReference) {
        TF_LITE_BATCH_TO_SPACE_ND(reference_ops, int64_t);
      } else {
        TF_LITE_BATCH_TO_SPACE_ND(optimized_ops, int64_t);
      }
      break;
    default:
      context->ReportError(
          context, "Type %d is currently not supported by BatchToSpace.",
          op_context.input->type);
      return kTfLiteError;
  }
#undef TF_LITE_BATCH_TO_SPACE_ND
  return kTfLiteOk;
}

}  // namespace batch_to_space_nd

TfLiteRegistration* Register_BATCH_TO_SPACE_ND_REF() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSbatch_to_space_ndDTcc mht_4(mht_4_v, 361, "", "./tensorflow/lite/kernels/batch_to_space_nd.cc", "Register_BATCH_TO_SPACE_ND_REF");

  static TfLiteRegistration r = {
      nullptr, nullptr, batch_to_space_nd::Prepare,
      batch_to_space_nd::Eval<batch_to_space_nd::kReference>};
  return &r;
}

TfLiteRegistration* Register_BATCH_TO_SPACE_ND_GENERIC_OPT() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSbatch_to_space_ndDTcc mht_5(mht_5_v, 371, "", "./tensorflow/lite/kernels/batch_to_space_nd.cc", "Register_BATCH_TO_SPACE_ND_GENERIC_OPT");

  static TfLiteRegistration r = {
      nullptr, nullptr, batch_to_space_nd::Prepare,
      batch_to_space_nd::Eval<batch_to_space_nd::kGenericOptimized>};
  return &r;
}

TfLiteRegistration* Register_BATCH_TO_SPACE_ND() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSbatch_to_space_ndDTcc mht_6(mht_6_v, 381, "", "./tensorflow/lite/kernels/batch_to_space_nd.cc", "Register_BATCH_TO_SPACE_ND");

  return Register_BATCH_TO_SPACE_ND_GENERIC_OPT();
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
