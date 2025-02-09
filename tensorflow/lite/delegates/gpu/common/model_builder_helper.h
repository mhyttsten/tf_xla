/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_MODEL_BUILDER_HELPER_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_MODEL_BUILDER_HELPER_H_
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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSmodel_builder_helperDTh {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSmodel_builder_helperDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSmodel_builder_helperDTh() {
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


#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <string>

#include "absl/strings/str_cat.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"
#include "tensorflow/lite/kernels/internal/reference/dequantize.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {
namespace gpu {

absl::Status GetNodeAndRegistration(TfLiteContext* context, int node_id,
                                    TfLiteNode** tflite_node,
                                    TfLiteRegistration** registration);

DataType ToDataType(TfLiteType type);

absl::Status ExtractTensorShape(const TfLiteTensor& tflite_tensor, BHWC* bhwc);

absl::Status ExtractAxisFromIndex(const TfLiteTensor& tflite_tensor, int index,
                                  Axis* axis);

absl::Status ConvertTfLiteTensorToTensorRef(const TfLiteTensor& tflite_tensor,
                                            TensorRef<BHWC>* tensor_ref);

// Populates quantization parameters for non-constant UInt8/Int8 tensors.
// This helps the delegate emulate quantized inference with
// QuantizeAndDequantize.
absl::Status PopulateQuantParams(const TfLiteTensor& tensor,
                                 QuantizationParams* quant_params);

int GetNumberOfRuntimeInputsForNode(const TfLiteContext* context,
                                    const TfLiteNode* tflite_node);

int GetNumberOfConstInputsForNode(const TfLiteContext* context,
                                  const TfLiteNode* tflite_node);

absl::Status CheckInputsOutputs(const TfLiteContext* context,
                                const TfLiteNode* tflite_node,
                                int runtime_inputs, int outputs);

absl::Status CheckInputsConstsOutputs(const TfLiteContext* context,
                                      const TfLiteNode* tflite_node,
                                      int runtime_inputs, int const_inputs,
                                      int outputs);

void ConvertFloat16ToFloat32(size_t num_elements, const uint16_t* src,
                             float* dst);

template <typename T>
inline void DequantizeConstantTensor(const TfLiteTensor& tensor,
                                     const T* source_data,
                                     float* dequantized_data) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSmodel_builder_helperDTh mht_0(mht_0_v, 250, "", "./tensorflow/lite/delegates/gpu/common/model_builder_helper.h", "DequantizeConstantTensor");

  TfLiteAffineQuantization* quant_params =
      static_cast<TfLiteAffineQuantization*>(tensor.quantization.params);
  if (quant_params->scale->size > 1) {
    // Tensor is per-channel quantized.
    PerChannelDequantizationParams op_params;
    op_params.zero_point = quant_params->zero_point->data;
    op_params.scale = quant_params->scale->data;
    op_params.quantized_dimension = quant_params->quantized_dimension;
    reference_ops::PerChannelDequantize(op_params, GetTensorShape(&tensor),
                                        source_data, GetTensorShape(&tensor),
                                        dequantized_data);
  } else {
    DequantizationParams op_params;
    op_params.zero_point = tensor.params.zero_point;
    op_params.scale = tensor.params.scale;
    reference_ops::Dequantize(op_params, GetTensorShape(&tensor), source_data,
                              GetTensorShape(&tensor), dequantized_data);
  }
}

template <typename T>
absl::Status CreateVectorCopyData(const TfLiteTensor& tensor, T* tensor_data) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSmodel_builder_helperDTh mht_1(mht_1_v, 275, "", "./tensorflow/lite/delegates/gpu/common/model_builder_helper.h", "CreateVectorCopyData");

  if (tensor.bytes % sizeof(T) != 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("Input data size ", tensor.bytes,
                     " is not aligned to expected type: ", sizeof(T)));
  }
  std::memcpy(tensor_data, tensor.data.uint8, tensor.bytes);
  return absl::OkStatus();
}

template <>
absl::Status CreateVectorCopyData<float>(const TfLiteTensor& tensor,
                                         float* tensor_data);

absl::Status SetAllDimensions(const TfLiteIntArray* dimensions, Scalar* shape);

absl::Status CheckIfLinearConvertible(const TfLiteIntArray* dimensions);

absl::Status SetAllDimensions(const TfLiteIntArray* dimensions, Linear* shape);

absl::Status SetAllDimensions(const TfLiteIntArray* dimensions, HWC* shape);

absl::Status SetAllDimensions(const TfLiteIntArray* dimensions, HW* shape);

absl::Status SetAllDimensions(const TfLiteIntArray* dimensions, OHWI* shape);

absl::Status SetAllDimensions(const TfLiteIntArray* dimensions, BHWC* shape);

// If there is fused activation present, then there will be another node created
// that will have identical output as the given node. New operation node will
// depend on the given node output.
absl::Status MaybeFuseActivation(TfLiteFusedActivation fused_activation,
                                 GraphFloat32* graph, Node* node);

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_MODEL_BUILDER_HELPER_H_
