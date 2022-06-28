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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_CONVOLUTION_TRANSPOSED_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_CONVOLUTION_TRANSPOSED_H_
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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconvolution_transposedDTh {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconvolution_transposedDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconvolution_transposedDTh() {
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
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/buffer_desc.h"
#include "tensorflow/lite/delegates/gpu/common/task/gpu_operation.h"
#include "tensorflow/lite/delegates/gpu/common/task/tensor_desc.h"
#include "tensorflow/lite/delegates/gpu/common/task/tensor_linear_desc.h"
#include "tensorflow/lite/delegates/gpu/common/task/texture2d_desc.h"
#include "tensorflow/lite/delegates/gpu/common/task/weights_conversion.h"
#include "tensorflow/lite/delegates/gpu/common/task/weights_layout.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {

class ConvolutionTransposed : public GPUOperation {
 public:
  ConvolutionTransposed() = default;
  void GetPossibleKernelWorkGroups(
      TuningType tuning_type, const GpuInfo& gpu_info,
      const KernelInfo& kernel_info,
      std::vector<int3>* work_groups) const override;
  absl::Status BindArguments(ArgumentsBinder* args) override;
  int3 GetGridSize() const override;

  // Move only
  ConvolutionTransposed(ConvolutionTransposed&& operation) = default;
  ConvolutionTransposed& operator=(ConvolutionTransposed&& operation) = default;
  ConvolutionTransposed(const ConvolutionTransposed&) = delete;
  ConvolutionTransposed& operator=(const ConvolutionTransposed&) = delete;

  WeightsDescription GetWeightsDescription() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconvolution_transposedDTh mht_0(mht_0_v, 226, "", "./tensorflow/lite/delegates/gpu/common/tasks/convolution_transposed.h", "GetWeightsDescription");

    WeightsDescription desc;
    desc.type = DeduceDataTypeFromPrecision(definition_.precision);
    desc.layout = weights_layout_;
    desc.output_group_size = block_size_.w;
    return desc;
  }

 private:
  friend ConvolutionTransposed CreateConvolutionTransposed(
      const GpuInfo& gpu_info, const OperationDef& definition,
      const ConvolutionTransposedAttributes& attr);
  friend ConvolutionTransposed CreateConvolutionTransposed3D(
      const GpuInfo& gpu_info, const OperationDef& definition,
      const ConvolutionTransposed3DAttributes& attr);
  friend ConvolutionTransposed CreateConvolutionTransposedDynamicWeights(
      const GpuInfo& gpu_info, const OperationDef& definition,
      const ConvolutionTransposedAttributes& attr);

  ConvolutionTransposed(const OperationDef& definition,
                        const ConvolutionTransposedAttributes& attr,
                        const GpuInfo& gpu_info, bool weights_are_buffer);
  ConvolutionTransposed(const OperationDef& definition,
                        const ConvolutionTransposed3DAttributes& attr,
                        const GpuInfo& gpu_info, bool weights_are_buffer);

  template <DataType T>
  void UploadWeights(const tflite::gpu::Tensor<OHWI, T>& weights,
                     bool weights_are_buffer);

  template <DataType T>
  void UploadWeights(const tflite::gpu::Tensor<OHWDI, T>& weights,
                     bool weights_are_buffer);

  std::string GenerateConvolutionTransposedCode(const OperationDef& op_def,
                                                const GpuInfo& gpu_info,
                                                bool weights_are_buffer,
                                                const int4& block_size);
  int4 stride_;
  int4 block_size_ = int4(1, 1, 1, 1);  // WHDS
  WeightsLayout weights_layout_;
};

template <DataType T>
void ConvolutionTransposed::UploadWeights(
    const tflite::gpu::Tensor<OHWI, T>& weights, bool weights_are_buffer) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconvolution_transposedDTh mht_1(mht_1_v, 274, "", "./tensorflow/lite/delegates/gpu/common/tasks/convolution_transposed.h", "ConvolutionTransposed::UploadWeights");

  const auto weights_desc = GetWeightsDescription();
  const int flt_count =
      GetTotalElementsCountForLayout(weights_desc, weights.shape);

  std::vector<uint8_t> weights_data(flt_count * SizeOf(weights_desc.type));
  RearrangeWeights(weights, weights_desc, absl::MakeSpan(weights_data));

  if (weights_are_buffer) {
    BufferDescriptor desc;
    desc.element_type = weights_desc.type;
    desc.element_size = 16;
    desc.size = weights_data.size();
    desc.data = std::move(weights_data);
    args_.AddObject("weights",
                    absl::make_unique<BufferDescriptor>(std::move(desc)));
  } else {
    uint2 tex_size = Get2dResourceSize(weights_desc, weights.shape);
    int sub_size = SizeOf(weights_desc.type) * 4 * tex_size.x * tex_size.y;
    for (int i = 0; i < 4; ++i) {
      Texture2DDescriptor desc;
      desc.element_type = weights_desc.type;
      desc.size = int2(tex_size.x, tex_size.y);
      desc.data.resize(sub_size);
      memcpy(desc.data.data(), weights_data.data() + sub_size * i, sub_size);
      const std::string name = "weights" + std::to_string(i);
      args_.AddObject(name,
                      absl::make_unique<Texture2DDescriptor>(std::move(desc)));
    }
  }
}

template <DataType T>
void ConvolutionTransposed::UploadWeights(
    const tflite::gpu::Tensor<OHWDI, T>& weights, bool weights_are_buffer) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconvolution_transposedDTh mht_2(mht_2_v, 311, "", "./tensorflow/lite/delegates/gpu/common/tasks/convolution_transposed.h", "ConvolutionTransposed::UploadWeights");

  const auto weights_desc = GetWeightsDescription();
  const int flt_count =
      GetTotalElementsCountForLayout(weights_desc, weights.shape);

  std::vector<uint8_t> weights_data(flt_count * SizeOf(weights_desc.type));
  RearrangeWeights(weights, weights_desc, absl::MakeSpan(weights_data));

  if (weights_are_buffer) {
    BufferDescriptor desc;
    desc.element_type = weights_desc.type;
    desc.element_size = 16;
    desc.size = weights_data.size();
    desc.data = std::move(weights_data);
    args_.AddObject("weights",
                    absl::make_unique<BufferDescriptor>(std::move(desc)));
  } else {
    uint2 tex_size = Get2dResourceSize(weights_desc, weights.shape);
    int sub_size = SizeOf(weights_desc.type) * 4 * tex_size.x * tex_size.y;
    for (int i = 0; i < 4; ++i) {
      Texture2DDescriptor desc;
      desc.element_type = weights_desc.type;
      desc.size = int2(tex_size.x, tex_size.y);
      desc.data.resize(sub_size);
      memcpy(desc.data.data(), weights_data.data() + sub_size * i, sub_size);
      const std::string name = "weights" + std::to_string(i);
      args_.AddObject(name,
                      absl::make_unique<Texture2DDescriptor>(std::move(desc)));
    }
  }
}

ConvolutionTransposed CreateConvolutionTransposed(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const ConvolutionTransposedAttributes& attr);

ConvolutionTransposed CreateConvolutionTransposed3D(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const ConvolutionTransposed3DAttributes& attr);

ConvolutionTransposed CreateConvolutionTransposedDynamicWeights(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const ConvolutionTransposedAttributes& attr);

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_CONVOLUTION_TRANSPOSED_H_
