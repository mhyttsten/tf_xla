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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSkernelsPStest_utilDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSkernelsPStest_utilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSkernelsPStest_utilDTcc() {
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

#include "tensorflow/lite/delegates/gpu/metal/kernels/test_util.h"

#import <Metal/Metal.h>

#include <functional>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/substitute.h"
#include "tensorflow/lite/delegates/gpu/common/convert.h"
#include "tensorflow/lite/delegates/gpu/common/gpu_info.h"
#include "tensorflow/lite/delegates/gpu/common/precision.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"
#include "tensorflow/lite/delegates/gpu/metal/compute_task.h"
#include "tensorflow/lite/delegates/gpu/metal/metal_spatial_tensor.h"

namespace tflite {
namespace gpu {
namespace metal {

std::vector<CalculationsPrecision>
MetalExecutionEnvironment::GetSupportedPrecisions() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSkernelsPStest_utilDTcc mht_0(mht_0_v, 212, "", "./tensorflow/lite/delegates/gpu/metal/kernels/test_util.cc", "MetalExecutionEnvironment::GetSupportedPrecisions");

  return {CalculationsPrecision::F32, CalculationsPrecision::F32_F16,
          CalculationsPrecision::F16};
}

std::vector<TensorStorageType> MetalExecutionEnvironment::GetSupportedStorages(
    DataType data_type) const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSkernelsPStest_utilDTcc mht_1(mht_1_v, 221, "", "./tensorflow/lite/delegates/gpu/metal/kernels/test_util.cc", "MetalExecutionEnvironment::GetSupportedStorages");

  return {TensorStorageType::BUFFER, TensorStorageType::IMAGE_BUFFER,
          TensorStorageType::TEXTURE_2D, TensorStorageType::TEXTURE_3D,
          TensorStorageType::TEXTURE_ARRAY};
}

// returns storage types that support zero clamping when reading OOB in HW
// (Height/Width) dimensions.
std::vector<TensorStorageType>
MetalExecutionEnvironment::GetSupportedStoragesWithHWZeroClampSupport(
    DataType data_type) const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSkernelsPStest_utilDTcc mht_2(mht_2_v, 234, "", "./tensorflow/lite/delegates/gpu/metal/kernels/test_util.cc", "MetalExecutionEnvironment::GetSupportedStoragesWithHWZeroClampSupport");

  return {TensorStorageType::TEXTURE_2D, TensorStorageType::TEXTURE_3D,
          TensorStorageType::TEXTURE_ARRAY};
}

absl::Status MetalExecutionEnvironment::ExecuteGPUOperation(
    const std::vector<TensorFloat32>& src_cpu,
    std::unique_ptr<GPUOperation>&& operation,
    const std::vector<BHWC>& dst_sizes,
    const std::vector<TensorFloat32*>& dst_cpu) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSkernelsPStest_utilDTcc mht_3(mht_3_v, 246, "", "./tensorflow/lite/delegates/gpu/metal/kernels/test_util.cc", "MetalExecutionEnvironment::ExecuteGPUOperation");

  const OperationDef op_def = operation->GetDefinition();
  std::vector<MetalSpatialTensor> src(src_cpu.size());
  for (int i = 0; i < src_cpu.size(); ++i) {
    auto src_shape = src_cpu[i].shape;
    if (src_shape.b != 1 && !op_def.IsBatchSupported()) {
      return absl::InvalidArgumentError(
          "Layout doesn't have Batch dimension, but shape.b != 1");
    }
    RETURN_IF_ERROR(CreateTensor(device_.device(), src_shape,
                                 op_def.src_tensors[i], &src[i]));
    RETURN_IF_ERROR(src[i].WriteData(device_.device(), src_cpu[i]));
  }

  std::vector<MetalSpatialTensor> dst(dst_cpu.size());
  for (int i = 0; i < dst_cpu.size(); ++i) {
    auto dst_shape = dst_sizes[i];
    if (dst_shape.b != 1 && !op_def.IsBatchSupported()) {
      return absl::InvalidArgumentError(
          "Layout doesn't have Batch dimension, but shape.b != 1");
    }
    RETURN_IF_ERROR(CreateTensor(device_.device(), dst_shape,
                                 op_def.dst_tensors[i], &dst[i]));
  }
  RETURN_IF_ERROR(operation->AssembleCode(GetGpuInfo()));

  ComputeTask gpu_task;
  gpu_task.Init(std::move(operation));
  RETURN_IF_ERROR(gpu_task.Compile(&device_));
  for (int i = 0; i < src_cpu.size(); ++i) {
    gpu_task.SetSrcTensor(&src[i], i);
  }
  for (int i = 0; i < dst_cpu.size(); ++i) {
    gpu_task.SetDstTensor(&dst[i], i);
  }
  RETURN_IF_ERROR(gpu_task.UpdateParams());

  id<MTLCommandQueue> command_queue = [device_.device() newCommandQueue];
  id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
  gpu_task.Encode(encoder);
  [encoder endEncoding];
  [command_buffer commit];
  [command_buffer waitUntilCompleted];

  for (int i = 0; i < dst_cpu.size(); ++i) {
    dst_cpu[i]->shape = dst_sizes[i];
    dst_cpu[i]->data = std::vector<float>(dst_sizes[i].DimensionsProduct(), 0);
    RETURN_IF_ERROR(dst[i].ReadData(device_.device(), dst_cpu[i]));
  }

  return absl::OkStatus();
}

absl::Status MetalExecutionEnvironment::ExecuteGPUOperation(
    const std::vector<Tensor5DFloat32>& src_cpu,
    std::unique_ptr<GPUOperation>&& operation,
    const std::vector<BHWDC>& dst_sizes,
    const std::vector<Tensor5DFloat32*>& dst_cpu) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSkernelsPStest_utilDTcc mht_4(mht_4_v, 307, "", "./tensorflow/lite/delegates/gpu/metal/kernels/test_util.cc", "MetalExecutionEnvironment::ExecuteGPUOperation");

  const OperationDef op_def = operation->GetDefinition();
  std::vector<MetalSpatialTensor> src(src_cpu.size());
  for (int i = 0; i < src_cpu.size(); ++i) {
    auto src_shape = src_cpu[i].shape;
    if (src_shape.b != 1 && !op_def.IsBatchSupported()) {
      return absl::InvalidArgumentError(
          "Layout doesn't have Batch dimension, but shape.b != 1");
    }
    RETURN_IF_ERROR(CreateTensor(device_.device(), src_shape,
                                 op_def.src_tensors[i], &src[i]));
    RETURN_IF_ERROR(src[i].WriteData(device_.device(), src_cpu[i]));
  }

  std::vector<MetalSpatialTensor> dst(dst_cpu.size());
  for (int i = 0; i < dst_cpu.size(); ++i) {
    auto dst_shape = dst_sizes[i];
    if (dst_shape.b != 1 && !op_def.IsBatchSupported()) {
      return absl::InvalidArgumentError(
          "Layout doesn't have Batch dimension, but shape.b != 1");
    }
    RETURN_IF_ERROR(CreateTensor(device_.device(), dst_shape,
                                 op_def.dst_tensors[i], &dst[i]));
  }
  RETURN_IF_ERROR(operation->AssembleCode(GetGpuInfo()));

  ComputeTask gpu_task;
  gpu_task.Init(std::move(operation));
  RETURN_IF_ERROR(gpu_task.Compile(&device_));
  for (int i = 0; i < src_cpu.size(); ++i) {
    gpu_task.SetSrcTensor(&src[i], i);
  }
  for (int i = 0; i < dst_cpu.size(); ++i) {
    gpu_task.SetDstTensor(&dst[i], i);
  }
  RETURN_IF_ERROR(gpu_task.UpdateParams());

  bool use_icb = false;
  if (use_icb) {
    if (@available(macOS 11.00, iOS 13.0, tvOS 13.0, *)) {
      MTLIndirectCommandBufferDescriptor* icb_desc =
          [[MTLIndirectCommandBufferDescriptor alloc] init];
      icb_desc.commandTypes = MTLIndirectCommandTypeConcurrentDispatch;
      icb_desc.inheritBuffers = NO;
      icb_desc.inheritPipelineState = NO;
      icb_desc.maxKernelBufferBindCount = 1;

      id<MTLIndirectCommandBuffer> icb =
          [device_.device() newIndirectCommandBufferWithDescriptor:icb_desc
                                                   maxCommandCount:1
                                                           options:0];

      id<MTLIndirectComputeCommand> icb_command =
          [icb indirectComputeCommandAtIndex:0];
      gpu_task.EncodeToICB(icb_command);
      [icb_command setBarrier];

      id<MTLCommandQueue> command_queue = [device_.device() newCommandQueue];
      id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
      id<MTLComputeCommandEncoder> encoder =
          [command_buffer computeCommandEncoder];
      gpu_task.AddResourcesToEncoder(encoder);
      [encoder executeCommandsInBuffer:icb withRange:NSMakeRange(0, 1)];
      [encoder endEncoding];
      [command_buffer commit];
      [command_buffer waitUntilCompleted];
    } else {
      return absl::InternalError(
          "Indirect compute command buffer available since ios 13");
    }
  } else {
    id<MTLCommandQueue> command_queue = [device_.device() newCommandQueue];
    id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder =
        [command_buffer computeCommandEncoder];
    gpu_task.Encode(encoder);
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
  }

  for (int i = 0; i < dst_cpu.size(); ++i) {
    dst_cpu[i]->shape = dst_sizes[i];
    dst_cpu[i]->data = std::vector<float>(dst_sizes[i].DimensionsProduct(), 0);
    RETURN_IF_ERROR(dst[i].ReadData(device_.device(), dst_cpu[i]));
  }

  return absl::OkStatus();
}

absl::Status MetalExecutionEnvironment::ExecuteGPUOperation(
    const std::vector<TensorDescriptor*>& src_cpu,
    const std::vector<TensorDescriptor*>& dst_cpu,
    std::unique_ptr<GPUOperation>&& operation) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSkernelsPStest_utilDTcc mht_5(mht_5_v, 403, "", "./tensorflow/lite/delegates/gpu/metal/kernels/test_util.cc", "MetalExecutionEnvironment::ExecuteGPUOperation");

  const OperationDef& op_def = operation->GetDefinition();
  std::vector<MetalSpatialTensor> src(src_cpu.size());
  for (int i = 0; i < src_cpu.size(); ++i) {
    auto src_shape = src_cpu[i]->GetBHWDCShape();
    if (src_shape.b != 1 && !op_def.IsBatchSupported()) {
      return absl::InvalidArgumentError(
          "Layout doesn't have Batch dimension, but shape.b != 1");
    }
    RETURN_IF_ERROR(src[i].CreateFromDescriptor(*src_cpu[i], device_.device()));
    operation->SetSrc(&src[i], i);
  }

  std::vector<MetalSpatialTensor> dst(dst_cpu.size());
  for (int i = 0; i < dst_cpu.size(); ++i) {
    auto dst_shape = dst_cpu[i]->GetBHWDCShape();
    if (dst_shape.b != 1 && !op_def.IsBatchSupported()) {
      return absl::InvalidArgumentError(
          "Layout doesn't have Batch dimension, but shape.b != 1");
    }
    RETURN_IF_ERROR(CreateTensor(device_.device(), dst_shape,
                                 op_def.dst_tensors[i], &dst[i]));

    operation->SetDst(&dst[i], i);
  }
  RETURN_IF_ERROR(operation->AssembleCode(GetGpuInfo()));

  ComputeTask gpu_task;
  gpu_task.Init(std::move(operation));
  RETURN_IF_ERROR(gpu_task.Compile(&device_));
  for (int i = 0; i < src_cpu.size(); ++i) {
    gpu_task.SetSrcTensor(&src[i], i);
  }
  for (int i = 0; i < dst_cpu.size(); ++i) {
    gpu_task.SetDstTensor(&dst[i], i);
  }
  RETURN_IF_ERROR(gpu_task.UpdateParams());

  id<MTLCommandQueue> command_queue = [device_.device() newCommandQueue];
  id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
  gpu_task.Encode(encoder);
  [encoder endEncoding];
  [command_buffer commit];
  [command_buffer waitUntilCompleted];

  for (int i = 0; i < dst_cpu.size(); ++i) {
    RETURN_IF_ERROR(dst[i].ToDescriptor(dst_cpu[i], device_.device()));
  }
  return absl::OkStatus();
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
