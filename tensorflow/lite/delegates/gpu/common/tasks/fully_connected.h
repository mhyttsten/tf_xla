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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_FULLY_CONNECTED_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_FULLY_CONNECTED_H_
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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSfully_connectedDTh {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSfully_connectedDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSfully_connectedDTh() {
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

#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/task/buffer_desc.h"
#include "tensorflow/lite/delegates/gpu/common/task/gpu_operation.h"
#include "tensorflow/lite/delegates/gpu/common/task/texture2d_desc.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"

namespace tflite {
namespace gpu {

template <DataType T, typename S>
void RearrangeFCWeightsToIOO4I4(const tflite::gpu::Tensor<OHWI, T>& weights,
                                S* dst) {
  const int src_channels = weights.shape.i;
  const int padded_src_channels = AlignByN(src_channels, 4);
  const int dst_channels = weights.shape.o;
  const int padded_dst_channels = AlignByN(dst_channels, 4);

  // Change the travelsal order of the weight matrix in the following way:
  // The matrix is segmented to blocks of 4x4. If (any) dimension of the matrix
  // size is not divisible by 4, then pad with zeros. Each block is stored
  // contigously. The 16 elements within a block are ordered as 4 elements of
  // the first column, 4 elems of the second, etc. Blocks then traversed as
  // columns first, rows last. As an example, an 8x8 matrix would be traversed
  // as below.
  //
  //  |  0  4  8 12 32 36 40 44 |
  //  |  1  5  9 13 33 37 41 45 |
  //  |  2  6 10 14 34 38 42 46 |
  //  |  3  7 11 15 35 39 43 47 |
  //  | 16 20 24 28 48 52 56 60 |
  //  | 17 21 25 29 49 53 57 61 |
  //  | 18 22 26 30 50 54 58 62 |
  //  | 19 23 27 31 51 55 59 63 |
  //
  // The benefit of doing this is that reading contigous 16 elements gives a 4x4
  // block of the matrix, where the first 4 elements is the first row of the
  // block, second 4 elements is the second row of the block, etc. Subsequent
  // blocks contain elements of the same 4 columns.

  for (int block_y = 0; 4 * block_y < padded_dst_channels; block_y++) {
    for (int y_in_block = 0; y_in_block < 4; y_in_block++) {
      for (int block_x = 0; 4 * block_x < padded_src_channels; block_x++) {
        for (int x_in_block = 0; x_in_block < 4; x_in_block++) {
          int y = 4 * block_y + y_in_block;
          int x = 4 * block_x + x_in_block;
          // Consider destination as an array with extents
          // [padded_src_channels/4][padded_dst_channels/4][4][4]
          int dst_index = block_x * padded_dst_channels * 4 + block_y * 16 +
                          x_in_block * 4 + y_in_block;
          if (x < src_channels && y < dst_channels) {
            dst[dst_index] = weights.data[src_channels * y + x];
          } else {
            dst[dst_index] = 0.0f;
          }
        }
      }
    }
  }
}

template <DataType T, typename S>
void RearrangeFCWeightsToOIO4I4(const tflite::gpu::Tensor<OHWI, T>& weights,
                                S* dst) {
  const int src_channels = weights.shape.i;
  const int src_depth = DivideRoundUp(src_channels, 4);
  const int dst_channels = weights.shape.o;
  const int dst_depth = DivideRoundUp(dst_channels, 4);

  int counter = 0;
  for (int d = 0; d < dst_depth; ++d) {
    for (int s = 0; s < src_depth; ++s) {
      for (int i = 0; i < 4; ++i) {
        const int src_ch = s * 4 + i;
        for (int j = 0; j < 4; ++j) {
          const int dst_ch = d * 4 + j;
          if (src_ch < src_channels && dst_ch < dst_channels) {
            dst[counter++] = weights.data[dst_ch * src_channels + src_ch];
          } else {
            dst[counter++] = 0.0f;
          }
        }
      }
    }
  }
}

class FullyConnected : public GPUOperation {
 public:
  FullyConnected() = default;
  void GetPossibleKernelWorkGroups(
      TuningType tuning_type, const GpuInfo& gpu_info,
      const KernelInfo& kernel_info,
      std::vector<int3>* work_groups) const override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSfully_connectedDTh mht_0(mht_0_v, 291, "", "./tensorflow/lite/delegates/gpu/common/tasks/fully_connected.h", "GetPossibleKernelWorkGroups");

    work_groups->push_back(work_group_size_);
  }
  int3 GetGridSize() const override;

  // Move only
  FullyConnected(FullyConnected&& kernel);
  FullyConnected& operator=(FullyConnected&& kernel);
  FullyConnected(const FullyConnected&) = delete;
  FullyConnected& operator=(const FullyConnected&) = delete;

 private:
  FullyConnected(const OperationDef& definition, const GpuInfo& gpu_info);
  friend FullyConnected CreateFullyConnected(
      const GpuInfo& gpu_info, const OperationDef& definition,
      const FullyConnectedAttributes& attr);
  friend FullyConnected CreateFullyConnected(
      const GpuInfo& gpu_info, const OperationDef& definition,
      const FullyConnectedInt8Attributes& attr);

  void UploadQuantizedWeights(
      const tflite::gpu::Tensor<OHWI, DataType::INT8>& weights, float scale,
      float zero_point);
  template <DataType T>
  void UploadWeights(const tflite::gpu::Tensor<OHWI, T>& weights,
                     bool weights_are_buffer);

  std::string GetFullyConnectedKernelCode(const OperationDef& op_def,
                                          const GpuInfo& gpu_info,
                                          bool weights_are_buffer,
                                          bool quantized);
};

template <DataType T>
void FullyConnected::UploadWeights(const tflite::gpu::Tensor<OHWI, T>& weights,
                                   bool weights_are_buffer) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSfully_connectedDTh mht_1(mht_1_v, 329, "", "./tensorflow/lite/delegates/gpu/common/tasks/fully_connected.h", "FullyConnected::UploadWeights");

  const int src_depth = DivideRoundUp(weights.shape.i, 4);
  const int dst_depth = DivideRoundUp(weights.shape.o, 4);

  const int elements_count = src_depth * dst_depth * 4;
  const bool f32_weights = definition_.precision == CalculationsPrecision::F32;

  const int float4_size = f32_weights ? 16 : 8;

  if (weights_are_buffer) {
    BufferDescriptor desc;
    desc.element_type = f32_weights ? DataType::FLOAT32 : DataType::FLOAT16;
    desc.element_size = 16;
    desc.size = float4_size * elements_count;
    desc.data.resize(desc.size);

    if (f32_weights) {
      float* ptr = reinterpret_cast<float*>(desc.data.data());
      RearrangeFCWeightsToIOO4I4(weights, ptr);
    } else {
      half* ptr = reinterpret_cast<half*>(desc.data.data());
      RearrangeFCWeightsToIOO4I4(weights, ptr);
    }

    args_.AddObject("weights",
                    absl::make_unique<BufferDescriptor>(std::move(desc)));
  } else {
    Texture2DDescriptor desc;
    desc.element_type = f32_weights ? DataType::FLOAT32 : DataType::FLOAT16;
    desc.size = int2(src_depth * 4, dst_depth);
    desc.data.resize(float4_size * elements_count);

    if (f32_weights) {
      float* ptr = reinterpret_cast<float*>(desc.data.data());
      RearrangeFCWeightsToOIO4I4(weights, ptr);
    } else {
      half* ptr = reinterpret_cast<half*>(desc.data.data());
      RearrangeFCWeightsToOIO4I4(weights, ptr);
    }

    args_.AddObject("weights",
                    absl::make_unique<Texture2DDescriptor>(std::move(desc)));
  }
}

FullyConnected CreateFullyConnected(const GpuInfo& gpu_info,
                                    const OperationDef& definition,
                                    const FullyConnectedAttributes& attr);

FullyConnected CreateFullyConnected(const GpuInfo& gpu_info,
                                    const OperationDef& definition,
                                    const FullyConnectedInt8Attributes& attr);

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_FULLY_CONNECTED_H_
