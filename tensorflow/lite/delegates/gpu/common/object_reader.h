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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_OBJECT_READER_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_OBJECT_READER_H_
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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSobject_readerDTh {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSobject_readerDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSobject_readerDTh() {
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


#include <cstdint>
#include <vector>

#include "fp16.h"  // from @FP16
#include "absl/container/flat_hash_map.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/model_builder_helper.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/kernels/internal/utils/sparsity_format_converter.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace gpu {

// If quantized tensors exist in the graph & quant_conversion_map is non-null,
// the mapping between the original tensors (fixed-point) & GPU values (fp) is
// stored in quant_conversion_map.
class ObjectReader {
 public:
  static absl::Status ReadNonConstantTensor(
      TfLiteContext* context, absl::flat_hash_map<int, Value*>* tensor_to_value,
      absl::flat_hash_map<int, int>* quant_conversion_map, GraphFloat32* graph,
      uint32_t tensor_idx, Value** value = nullptr);

  ObjectReader(GraphFloat32* graph, TfLiteContext* context,
               const TfLiteNode* node,
               absl::flat_hash_map<int, Value*>* tensor_to_value,
               absl::flat_hash_map<int, int>* quant_conversion_map = nullptr)
      : graph_(graph),
        context_(context),
        node_(node),
        tensor_to_value_(tensor_to_value),
        quant_conversion_map_(quant_conversion_map) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSobject_readerDTh mht_0(mht_0_v, 221, "", "./tensorflow/lite/delegates/gpu/common/object_reader.h", "ObjectReader");
}

  absl::Status ReadValue(uint32_t idx, Value** value);

  absl::Status ReadValueByTensorIdx(uint32_t tensor_idx, Value** value);

  int GetNumberOfRuntimeInputs() const;

  absl::Status GetTensorId(uint32_t input_id, int* tensor_id) const;

  absl::Status GetTensorDims(uint32_t idx, TfLiteIntArray* dimensions) const;

  template <typename TensorT>
  absl::Status ReadTensor(uint32_t index, TensorT* tensor) const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSobject_readerDTh mht_1(mht_1_v, 237, "", "./tensorflow/lite/delegates/gpu/common/object_reader.h", "ReadTensor");

    if (index < 0 || index >= node_->inputs->size) {
      // If larger, this can be an older model with fewer input tensors than the
      // current implementation.
      return absl::OutOfRangeError("Invalid data index found.");
    }
    const int32_t tensor_id = node_->inputs->data[index];
    if (tensor_id < 0) {
      return absl::InvalidArgumentError(
          "Invalid data index found. Possibly an unset optional tensor is "
          "being read.");
    }
    const TfLiteTensor* tflite_tensor = context_->tensors + tensor_id;
    tensor->data.resize(NumElements(tflite_tensor));
    if (tflite_tensor->sparsity) {
      std::vector<int> dims;
      dims.reserve(tflite_tensor->dims->size);
      for (int i = 0; i < tflite_tensor->dims->size; ++i) {
        dims.push_back(tflite_tensor->dims->data[i]);
      }
      switch (tflite_tensor->type) {
        case kTfLiteFloat32: {
          internal::sparsity::FormatConverter<float> converter(
              dims, *tflite_tensor->sparsity);
          converter.SparseToDense(
              static_cast<const float*>(tflite_tensor->data.data));
          const std::vector<float> out = converter.GetData();
          std::memcpy(&tensor->data[0], out.data(), out.size() * sizeof(float));
          break;
        }
        case kTfLiteFloat16: {
          internal::sparsity::FormatConverter<Eigen::half> converter(
              dims, *tflite_tensor->sparsity);
          converter.SparseToDense(
              static_cast<const Eigen::half*>(tflite_tensor->data.data));
          const std::vector<Eigen::half> out = converter.GetData();
          std::transform(out.begin(), out.end(), tensor->data.begin(),
                         [](const Eigen::half& x) {
                           return fp16_ieee_to_fp32_value(
                               Eigen::numext::bit_cast<uint16_t>(x));
                         });
          break;
        }
        default: {
          return absl::InvalidArgumentError(
              "Unexpected data type in sparse tensor");
        }
      }
    } else {
      RETURN_IF_ERROR(CreateVectorCopyData(*tflite_tensor, &tensor->data[0]));
    }

    // Axis and data layout depend on operation this tensor is used in. So,
    // postpone resolutions until operations are parsed.
    tensor->id = tensor_id;
    return SetAllDimensions(tflite_tensor->dims, &tensor->shape);
  }

  absl::Status AddOutput(const Node* node, int id);

  absl::Status AddOutputs(const Node* node);

  absl::Status AddInput(const Node* node, uint32_t idx);

  absl::Status AddUpdate(const Node* node, uint32_t idx);

  TfLiteTensor* GetInputTensor(int index) const;

  TfLiteTensor* GetOutputTensor(int index) const;

  absl::Status VerifyInputsConstsOutputs(const TfLiteNode* node,
                                         int runtime_inputs, int const_inputs,
                                         int outputs);

 private:
  GraphFloat32* graph_;
  TfLiteContext* context_;
  const TfLiteNode* node_;
  absl::flat_hash_map<int, Value*>* tensor_to_value_;
  absl::flat_hash_map<int, int>* quant_conversion_map_;
};

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_OBJECT_READER_H_
