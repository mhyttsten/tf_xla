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
class MHTracer_DTPStensorflowPSlitePSkernelsPSconcatenationDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSconcatenationDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSconcatenationDTcc() {
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
#include "tensorflow/lite/kernels/internal/reference/concatenation.h"

#include <stdint.h>
#include <limits>

#include "tensorflow/lite/c/builtin_op_data.h"
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
namespace concatenation {

// This file has two implementation of Concatenation.
enum KernelType {
  kReference,
  kGenericOptimized,
};

template <KernelType kernel_type>
TfLiteStatus EvalImpl(TfLiteContext* context, TfLiteNode* node, int axis,
                      TfLiteTensor* output) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSconcatenationDTcc mht_0(mht_0_v, 212, "", "./tensorflow/lite/kernels/concatenation.cc", "EvalImpl");

// TODO(ahentz): Creating 'all_inputs' below is not very efficient. We should
// allocate and populate these during Prepare().
// TODO(ycling): Activation function parameter is ignored. For now we don't have
// a model with a Concatenation with fused activation function.
#define TF_LITE_CONCATENATION(scalar)                                         \
  {                                                                           \
    VectorOfTensors<scalar> all_inputs(*context, *node->inputs);              \
    tflite::ConcatenationParams op_params;                                    \
    op_params.axis = axis;                                                    \
    op_params.inputs_count = node->inputs->size;                              \
    if (kernel_type == kReference) {                                          \
      reference_ops::Concatenation(op_params, all_inputs.shapes(),            \
                                   all_inputs.data(), GetTensorShape(output), \
                                   GetTensorData<scalar>(output));            \
    } else {                                                                  \
      optimized_ops::Concatenation(op_params, all_inputs.shapes(),            \
                                   all_inputs.data(), GetTensorShape(output), \
                                   GetTensorData<scalar>(output));            \
    }                                                                         \
  }

#define TF_LITE_CONCATENATION_QUANTIZED()                         \
  {                                                               \
    VectorOfQuantizedTensors all_inputs(*context, *node->inputs); \
    tflite::ConcatenationParams op_params;                        \
    op_params.axis = axis;                                        \
    op_params.input_zeropoint = all_inputs.zero_point();          \
    op_params.input_scale = all_inputs.scale();                   \
    op_params.inputs_count = node->inputs->size;                  \
    op_params.output_zeropoint = output->params.zero_point;       \
    op_params.output_scale = output->params.scale;                \
    if (kernel_type == kReference) {                              \
      reference_ops::ConcatenationWithScaling(                    \
          op_params, all_inputs.shapes(), all_inputs.data(),      \
          GetTensorShape(output), GetTensorData<uint8>(output));  \
    } else {                                                      \
      optimized_ops::ConcatenationWithScaling(                    \
          op_params, all_inputs.shapes(), all_inputs.data(),      \
          GetTensorShape(output), GetTensorData<uint8>(output));  \
    }                                                             \
  }

  switch (output->type) {  // Already know in/outtypes are same.
    case kTfLiteFloat32:
      TF_LITE_CONCATENATION(float);
      break;
    case kTfLiteInt32:
      TF_LITE_CONCATENATION(int32);
      break;
    case kTfLiteUInt8:
      TF_LITE_CONCATENATION_QUANTIZED();
      break;
    case kTfLiteInt8:
      TF_LITE_CONCATENATION(int8_t);
      break;
    case kTfLiteInt64:
      TF_LITE_CONCATENATION(int64_t);
      break;
    case kTfLiteInt16:
      TF_LITE_CONCATENATION(int16_t);
      break;
    case kTfLiteBool:
      TF_LITE_CONCATENATION(bool);
      break;
    default:
      context->ReportError(context, "Type '%s' is not supported currently.",
                           TfLiteTypeGetName(output->type));
      return kTfLiteError;
  }

#undef TF_LITE_CONCATENATION_QUANTIZED
#undef TF_LITE_CONCATENATION

  return kTfLiteOk;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSconcatenationDTcc mht_1(mht_1_v, 292, "", "./tensorflow/lite/kernels/concatenation.cc", "Prepare");

  auto* params =
      reinterpret_cast<TfLiteConcatenationParams*>(node->builtin_data);
  int axis = params->axis;
  int num_inputs = node->inputs->size;

  // The number of dimensions of the input tensors must match, and all
  // dimensions except 'axis' must be equal.
  const TfLiteTensor* t0;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &t0));
  TfLiteType input_type = t0->type;
  if (axis < 0) axis += t0->dims->size;
  TF_LITE_ENSURE(context, axis >= 0);
  TF_LITE_ENSURE(context, axis < t0->dims->size);

  TF_LITE_ENSURE_EQ(context, params->activation, kTfLiteActNone);
  TF_LITE_ENSURE(context,
                 input_type == kTfLiteFloat32 || input_type == kTfLiteUInt8 ||
                     input_type == kTfLiteInt8 || input_type == kTfLiteInt16 ||
                     input_type == kTfLiteInt32 || input_type == kTfLiteInt64 ||
                     input_type == kTfLiteBool);

  // Output dimensions will match input dimensions, except 'axis', which
  // will be the sum of inputs
  int sum_axis = t0->dims->data[axis];
  for (int i = 1; i < num_inputs; ++i) {
    const TfLiteTensor* t;
    TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, i, &t));
    TF_LITE_ENSURE_EQ(context, t->dims->size, t0->dims->size);
    TF_LITE_ENSURE_EQ(context, t->type, input_type);
    for (int d = 0; d < t0->dims->size; ++d) {
      if (d == axis) {
        // Avoid integer overflow in sum_axis below
        TF_LITE_ENSURE(context, t->dims->data[axis] >= 0);
        TF_LITE_ENSURE(context, t->dims->data[axis] <=
                                    std::numeric_limits<int>::max() - sum_axis);
        sum_axis += t->dims->data[axis];
      } else {
        TF_LITE_ENSURE_EQ(context, t->dims->data[d], t0->dims->data[d]);
      }
    }
  }

  TfLiteIntArray* output_size = TfLiteIntArrayCreate(t0->dims->size);
  for (int d = 0; d < t0->dims->size; ++d) {
    output_size->data[d] = (d == axis) ? sum_axis : t0->dims->data[d];
  }

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  TF_LITE_ENSURE_TYPES_EQ(context, output->type, input_type);

  if (input_type == kTfLiteInt8) {
    // Make sure there is no re-scaling needed for Int8 quantized kernel. This
    // is a restriction we introduced to Int8 kernels.
    VectorOfTensors<int8_t> all_inputs(*context, *node->inputs);
    for (int i = 0; i < node->inputs->size; ++i) {
      const TfLiteTensor* t;
      TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, i, &t));
      TF_LITE_ENSURE_EQ(context, t->params.scale, output->params.scale);
      TF_LITE_ENSURE_EQ(context, t->params.zero_point,
                        output->params.zero_point);
    }
  }

  if (input_type == kTfLiteInt16) {
    // Make sure that all Int16 inputs have a null zero-point.
    for (int i = 0; i < node->inputs->size; ++i) {
      const TfLiteTensor* t = GetInput(context, node, i);
      TF_LITE_ENSURE_EQ(context, t->params.zero_point, 0);
    }
    TF_LITE_ENSURE_EQ(context, output->params.zero_point, 0);
  }

  // Check to see if we can calculate the output now.
  bool all_inputs_at_prepare = true;
  for (int i = 0; i < num_inputs; ++i) {
    const TfLiteTensor* t;
    TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, i, &t));
    if (!IsConstantOrPersistentTensor(t)) {
      all_inputs_at_prepare = false;
      break;
    }
  }
  if (all_inputs_at_prepare) {
    SetTensorToPersistentRo(output);
    context->ResizeTensor(context, output, output_size);
    return EvalImpl<kReference>(context, node, axis, output);
  }
  return context->ResizeTensor(context, output, output_size);
}

template <KernelType kernel_type>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSconcatenationDTcc mht_2(mht_2_v, 388, "", "./tensorflow/lite/kernels/concatenation.cc", "Eval");

  auto* params =
      reinterpret_cast<TfLiteConcatenationParams*>(node->builtin_data);
  int axis = params->axis;
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  if (IsConstantOrPersistentTensor(output)) {
    // Output is computed in Prepare.
    return kTfLiteOk;
  }
  if (axis < 0) axis += output->dims->size;

  return EvalImpl<kernel_type>(context, node, axis, output);
}

#undef TF_LITE_MACRO_DISPATCH

}  // namespace concatenation

TfLiteRegistration* Register_CONCATENATION_REF() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSconcatenationDTcc mht_3(mht_3_v, 410, "", "./tensorflow/lite/kernels/concatenation.cc", "Register_CONCATENATION_REF");

  static TfLiteRegistration r = {
      nullptr, nullptr, concatenation::Prepare,
      concatenation::Eval<concatenation::kReference>};
  return &r;
}

TfLiteRegistration* Register_CONCATENATION_GENERIC_OPT() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSconcatenationDTcc mht_4(mht_4_v, 420, "", "./tensorflow/lite/kernels/concatenation.cc", "Register_CONCATENATION_GENERIC_OPT");

  static TfLiteRegistration r = {
      nullptr, nullptr, concatenation::Prepare,
      concatenation::Eval<concatenation::kGenericOptimized>};
  return &r;
}

TfLiteRegistration* Register_CONCATENATION() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSconcatenationDTcc mht_5(mht_5_v, 430, "", "./tensorflow/lite/kernels/concatenation.cc", "Register_CONCATENATION");

  // TODO(ahentz): It turns out the two versions of Concatenation are almost
  // identical, so we should consider removing one.
  return Register_CONCATENATION_GENERIC_OPT();
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
