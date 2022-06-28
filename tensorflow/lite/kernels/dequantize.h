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
#ifndef TENSORFLOW_LITE_KERNELS_DEQUANTIZE_H_
#define TENSORFLOW_LITE_KERNELS_DEQUANTIZE_H_
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
class MHTracer_DTPStensorflowPSlitePSkernelsPSdequantizeDTh {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSdequantizeDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSdequantizeDTh() {
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


#include <stdint.h>

#include "third_party/eigen3/Eigen/Core"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/reference/dequantize.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/dequantize.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace dequantize {

// This file has two implementation of Dequantize.
enum KernelType {
  kReference,
  kGenericOptimized,
};

inline bool IsQuantizedPerChannel(const TfLiteTensor* input) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSdequantizeDTh mht_0(mht_0_v, 210, "", "./tensorflow/lite/kernels/dequantize.h", "IsQuantizedPerChannel");

  if (input->quantization.type == kTfLiteAffineQuantization &&
      input->quantization.params) {
    auto* quant_params =
        reinterpret_cast<TfLiteAffineQuantization*>(input->quantization.params);
    return (quant_params->scale && quant_params->scale->size > 1);
  }
  return false;
}

inline TfLiteStatus PerChannelDequantizeImpl(TfLiteContext* context,
                                             TfLiteNode* node,
                                             const TfLiteTensor* input,
                                             TfLiteTensor* output) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSdequantizeDTh mht_1(mht_1_v, 226, "", "./tensorflow/lite/kernels/dequantize.h", "PerChannelDequantizeImpl");

  const auto* quantization_params =
      reinterpret_cast<const TfLiteAffineQuantization*>(
          input->quantization.params);
  PerChannelDequantizationParams per_channel_op_params;
  per_channel_op_params.quantized_dimension =
      quantization_params->quantized_dimension;
  per_channel_op_params.scale = quantization_params->scale->data;
  per_channel_op_params.zero_point = quantization_params->zero_point->data;
  switch (input->type) {
    case kTfLiteUInt8:
      reference_ops::PerChannelDequantize<uint8_t>(
          per_channel_op_params, GetTensorShape(input),
          GetTensorData<uint8_t>(input), GetTensorShape(output),
          GetTensorData<float>(output));
      break;
    case kTfLiteInt8:
      reference_ops::PerChannelDequantize<int8_t>(
          per_channel_op_params, GetTensorShape(input),
          GetTensorData<int8_t>(input), GetTensorShape(output),
          GetTensorData<float>(output));
      break;
    default:
      TF_LITE_KERNEL_LOG(context, "Type %d not supported for per-channel.",
                         input->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

template <KernelType kernel_type>
TfLiteStatus DequantizeImpl(TfLiteContext* context, TfLiteNode* node,
                            const TfLiteTensor* input, TfLiteTensor* output) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSdequantizeDTh mht_2(mht_2_v, 261, "", "./tensorflow/lite/kernels/dequantize.h", "DequantizeImpl");

  if (IsQuantizedPerChannel(input)) {
    return PerChannelDequantizeImpl(context, node, input, output);
  }
  DequantizationParams op_params;
  op_params.zero_point = input->params.zero_point;
  op_params.scale = input->params.scale;
  switch (input->type) {
    case kTfLiteUInt8:
      if (kernel_type == kReference) {
        reference_ops::Dequantize(
            op_params, GetTensorShape(input), GetTensorData<uint8_t>(input),
            GetTensorShape(output), GetTensorData<float>(output));
      } else {
        optimized_ops::Dequantize(
            op_params, GetTensorShape(input), GetTensorData<uint8_t>(input),
            GetTensorShape(output), GetTensorData<float>(output));
      }
      break;
    case kTfLiteInt8:
      if (kernel_type == kReference) {
        reference_integer_ops::Dequantize<int8_t>(
            op_params, GetTensorShape(input), GetTensorData<int8_t>(input),
            GetTensorShape(output), GetTensorData<float>(output));
      } else {
        optimized_ops::Dequantize(
            op_params, GetTensorShape(input), GetTensorData<int8_t>(input),
            GetTensorShape(output), GetTensorData<float>(output));
      }
      break;
    case kTfLiteInt16:
      if (kernel_type == kReference) {
        reference_integer_ops::Dequantize<int16_t>(
            op_params, GetTensorShape(input), GetTensorData<int16_t>(input),
            GetTensorShape(output), GetTensorData<float>(output));
      } else {
        optimized_ops::Dequantize(
            op_params, GetTensorShape(input), GetTensorData<int16_t>(input),
            GetTensorShape(output), GetTensorData<float>(output));
      }
      break;
    case kTfLiteFloat16: {
      const Eigen::half* half_data = reinterpret_cast<const Eigen::half*>(
          GetTensorData<TfLiteFloat16>(input));
      reference_ops::Dequantize(GetTensorShape(input), half_data,
                                GetTensorShape(output),
                                GetTensorData<float>(output));
      break;
    }
    default:
      TF_LITE_KERNEL_LOG(context, "Type %d not supported.", input->type);
      return kTfLiteError;
  }

  return kTfLiteOk;
}

}  // namespace dequantize
}  // namespace builtin
}  // namespace ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_DEQUANTIZE_H_
