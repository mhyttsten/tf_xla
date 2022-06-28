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
class MHTracer_DTPStensorflowPSlitePSkernelsPSl2normDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSl2normDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSl2normDTcc() {
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
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/l2normalization.h"
#include "tensorflow/lite/kernels/internal/reference/l2normalization.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace l2norm {

// This file has two implementation of L2Norm.
enum KernelType {
  kReference,
  kGenericOptimized,
};

constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSl2normDTcc mht_0(mht_0_v, 210, "", "./tensorflow/lite/kernels/l2norm.cc", "Prepare");

  auto* params = reinterpret_cast<TfLiteL2NormParams*>(node->builtin_data);

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  TF_LITE_ENSURE(context, NumDimensions(input) <= 4);

  TF_LITE_ENSURE(context, output->type == kTfLiteFloat32 ||
                              output->type == kTfLiteUInt8 ||
                              output->type == kTfLiteInt8);
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);

  if (output->type == kTfLiteUInt8 || output->type == kTfLiteInt8) {
    TF_LITE_ENSURE_EQ(context, output->params.scale, (1. / 128.));
    if (output->type == kTfLiteUInt8) {
      TF_LITE_ENSURE_EQ(context, output->params.zero_point, 128);
    }
    if (output->type == kTfLiteInt8) {
      TF_LITE_ENSURE_EQ(context, output->params.zero_point, 0);
    }
  }

  // TODO(ahentz): For some reason our implementations don't support
  // activations.
  TF_LITE_ENSURE_EQ(context, params->activation, kTfLiteActNone);

  TfLiteIntArray* output_size = TfLiteIntArrayCopy(input->dims);
  return context->ResizeTensor(context, output, output_size);
}

template <KernelType kernel_type>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSl2normDTcc mht_1(mht_1_v, 251, "", "./tensorflow/lite/kernels/l2norm.cc", "Eval");

  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  // TODO(b/143912164): instead of hardcode the epsilon here, we should read it
  // from tensorflow, i.e., adding a params.
  // We don't compute epsilon for quantized kernel:
  //
  // epsilon_float = (epsilon_quant - zp) * scale
  // so
  // espsilon_quant = epsilon_float / scale + zp
  // We know epsilon_float is just a very small number to avoid division by
  // zero error, and scale is > 1, so the integer value of epsilon for quant
  // is just dominated by the zero point.
  // Also, GetInvSqrtQuantizedMultiplierExp handles the scenario where the sum
  // of input value squared is zero case well.
  // So we don't even need to do handle the epsilon for quantized kernel case.
  const float epsilon = 1e-6f;
  if (output->type == kTfLiteFloat32) {
#define TF_LITE_L2NORM(type)                                                 \
  tflite::L2NormalizationParams op_params;                                   \
  op_params.input_zero_point = 0;                                            \
  type::L2Normalization(op_params, GetTensorShape(input),                    \
                        GetTensorData<float>(input), GetTensorShape(output), \
                        GetTensorData<float>(output), epsilon)

    if (kernel_type == kReference) {
      TF_LITE_L2NORM(reference_ops);
    }
    if (kernel_type == kGenericOptimized) {
      TF_LITE_L2NORM(optimized_ops);
    }
#undef TF_LITE_L2NORM
  } else if (output->type == kTfLiteUInt8) {
#define TF_LITE_L2NORM(type)                                                 \
  tflite::L2NormalizationParams op_params;                                   \
  op_params.input_zero_point = input->params.zero_point;                     \
  type::L2Normalization(op_params, GetTensorShape(input),                    \
                        GetTensorData<uint8>(input), GetTensorShape(output), \
                        GetTensorData<uint8>(output))

    if (kernel_type == kReference) {
      TF_LITE_L2NORM(reference_ops);
    }
    if (kernel_type == kGenericOptimized) {
      TF_LITE_L2NORM(optimized_ops);
    }
#undef TF_LITE_L2NORM
  } else if (output->type == kTfLiteInt8) {
    const auto input_shape = GetTensorShape(input);
    const auto output_shape = GetTensorShape(output);
    const int trailing_dim = input_shape.DimensionsCount() - 1;
    const int depth =
        MatchingDim(input_shape, trailing_dim, output_shape, trailing_dim);
    const int outer_size =
        MatchingFlatSizeSkipDim(input_shape, trailing_dim, output_shape);
    reference_integer_ops::L2Normalization(input->params.zero_point, outer_size,
                                           depth, GetTensorData<int8>(input),
                                           GetTensorData<int8>(output));
  } else {
    TF_LITE_KERNEL_LOG(context, "Output type is %s, requires float.",
                       TfLiteTypeGetName(output->type));
    return kTfLiteError;
  }

  return kTfLiteOk;
}

}  // namespace l2norm

TfLiteRegistration* Register_L2NORM_REF() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSl2normDTcc mht_2(mht_2_v, 327, "", "./tensorflow/lite/kernels/l2norm.cc", "Register_L2NORM_REF");

  static TfLiteRegistration r = {nullptr, nullptr, l2norm::Prepare,
                                 l2norm::Eval<l2norm::kReference>};
  return &r;
}

TfLiteRegistration* Register_L2NORM_GENERIC_OPT() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSl2normDTcc mht_3(mht_3_v, 336, "", "./tensorflow/lite/kernels/l2norm.cc", "Register_L2NORM_GENERIC_OPT");

  static TfLiteRegistration r = {nullptr, nullptr, l2norm::Prepare,
                                 l2norm::Eval<l2norm::kGenericOptimized>};
  return &r;
}

TfLiteRegistration* Register_L2_NORMALIZATION() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSl2normDTcc mht_4(mht_4_v, 345, "", "./tensorflow/lite/kernels/l2norm.cc", "Register_L2_NORMALIZATION");

  return Register_L2NORM_GENERIC_OPT();
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
