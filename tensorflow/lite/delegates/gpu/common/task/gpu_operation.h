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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASK_GPU_OPERATION_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASK_GPU_OPERATION_H_
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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSgpu_operationDTh {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSgpu_operationDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSgpu_operationDTh() {
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


#include <string>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/gpu_info.h"
#include "tensorflow/lite/delegates/gpu/common/kernel_info.h"
#include "tensorflow/lite/delegates/gpu/common/precision.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/arguments.h"
#include "tensorflow/lite/delegates/gpu/common/task/buffer_desc.h"
#include "tensorflow/lite/delegates/gpu/common/task/compiler_options.h"
#include "tensorflow/lite/delegates/gpu/common/task/gpu_tensor.h"
#include "tensorflow/lite/delegates/gpu/common/task/serialization_base_generated.h"
#include "tensorflow/lite/delegates/gpu/common/task/tensor_desc.h"
#include "tensorflow/lite/delegates/gpu/common/task/texture2d_desc.h"
#include "tensorflow/lite/delegates/gpu/common/task/tuning_type.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {
// kCustom: default value
//   GPUOperation::GetGridSize must be overloaded
// kWBToX_HDToY_SToZ:
//   grid_x = dst_[0]->Width() * dst_[0]->Batch();
//   grid_y = dst_[0]->Height() * dst_[0]->Depth();
//   grid_z = dst_[0]->Slices();
// kWBToX_HDToY_ZIs1:
//   grid_x = dst_[0]->Width() * dst_[0]->Batch();
//   grid_y = dst_[0]->Height() * dst_[0]->Depth();
//   grid_z = 1;
// kWBToX_HToY_DToZ:
//   grid_x = dst_[0]->Width() * dst_[0]->Batch();
//   grid_y = dst_[0]->Height();
//   grid_z = dst_[0]->Depth();
// kBToX_YIs1_ZIs1:
//   grid_x = dst_[0]->Batch();
//   grid_y = 1;
//   grid_z = 1;
enum class TensorToGrid {
  kCustom,
  kWBToX_HDToY_SToZ,
  kWBToX_HDToY_ZIs1,
  kWBToX_HToY_DToZ,
  kBToX_YIs1_ZIs1
};

struct OperationDef {
  CalculationsPrecision precision;
  std::vector<TensorDescriptor> src_tensors;
  std::vector<TensorDescriptor> dst_tensors;

  // returns FLOAT32 for F32 precision and FLOAT16 for F16 precision
  DataType GetDataType() const;
  // Primary means the first src tensor, because first tensor usually defines
  // the structure of kernel, all other resources(biases) types and etc.
  DataType GetPrimaryDataType() const;
  TensorStorageType GetPrimaryStorageType() const;
  bool IsBatchSupported() const;
};

// GPUOperation represents some implementation of neural network operation on
// GPU. GPUOperation can contain another GPU operations with flag elementwise_.
// When GPUOperation contains another GPU ops, this GPUoperation replaces
// some sequence of operations Op + op0 + op1 + ...
// Because of this abilities of GPUOperation, usage scenario is next:
// Create instance of GPUOperation.
// Create all instances of GPUOperations that we will(probably) attach
// to GPUOperation. Attach all GPUOperations to GPUOperation. Call
// GPUOperation.Compile(). Don't call GPUOperations.Compile() if it
// attached, it useless(and may be error)
class GPUOperation {
 public:
  GPUOperation() = default;
  explicit GPUOperation(const OperationDef& definition);
  virtual ~GPUOperation() = default;
  // Move only
  GPUOperation(GPUOperation&& operation);
  GPUOperation& operator=(GPUOperation&& operation);
  GPUOperation(const GPUOperation&) = delete;
  GPUOperation& operator=(const GPUOperation&) = delete;

  absl::Status AddOperation(GPUOperation* operation);

  void SetSrc(GpuSpatialTensor* ptr, int index = 0);
  void SetDst(GpuSpatialTensor* ptr, int index = 0);

  struct DispatchInfo {
    int3 work_group_size;
    int3 work_groups_count;
  };
  void GetPossibleDispatches(TuningType tuning_type, const GpuInfo& gpu_info,
                             const KernelInfo& kernel_info,
                             std::vector<DispatchInfo>* dispatches) const;

  const std::vector<std::string>& GetSrcTensorsNames() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSgpu_operationDTh mht_0(mht_0_v, 282, "", "./tensorflow/lite/delegates/gpu/common/task/gpu_operation.h", "GetSrcTensorsNames");

    return src_tensors_names_;
  }
  const std::vector<std::string>& GetDstTensorsNames() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSgpu_operationDTh mht_1(mht_1_v, 288, "", "./tensorflow/lite/delegates/gpu/common/task/gpu_operation.h", "GetDstTensorsNames");

    return dst_tensors_names_;
  }
  const std::vector<GpuSpatialTensor*>& GetSrcTensors() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSgpu_operationDTh mht_2(mht_2_v, 294, "", "./tensorflow/lite/delegates/gpu/common/task/gpu_operation.h", "GetSrcTensors");
 return src_; }
  const std::vector<GpuSpatialTensor*>& GetDstTensors() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSgpu_operationDTh mht_3(mht_3_v, 298, "", "./tensorflow/lite/delegates/gpu/common/task/gpu_operation.h", "GetDstTensors");
 return dst_; }
  const int3& GetWorkGroupsCount() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSgpu_operationDTh mht_4(mht_4_v, 302, "", "./tensorflow/lite/delegates/gpu/common/task/gpu_operation.h", "GetWorkGroupsCount");
 return work_groups_count_; }

  absl::Status AssembleCode(const GpuInfo& gpu_info);

  virtual absl::Status PostCompileCheck(const GpuInfo& gpu_info,
                                        const KernelInfo& kernel_info) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSgpu_operationDTh mht_5(mht_5_v, 310, "", "./tensorflow/lite/delegates/gpu/common/task/gpu_operation.h", "PostCompileCheck");

    return absl::OkStatus();
  }

  const OperationDef& GetDefinition() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSgpu_operationDTh mht_6(mht_6_v, 317, "", "./tensorflow/lite/delegates/gpu/common/task/gpu_operation.h", "GetDefinition");
 return definition_; }

  void AddSrcTensor(const std::string& tensor_name,
                    const TensorDescriptor& desc);
  void AddSrcBuffer(const std::string& buffer_name,
                    const BufferDescriptor& desc);
  void AddSrcTexture2D(const std::string& texture_name,
                       const Texture2DDescriptor& desc);
  void AddDstTensor(const std::string& tensor_name,
                    const TensorDescriptor& desc);

  bool IsLinkable() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSgpu_operationDTh mht_7(mht_7_v, 331, "", "./tensorflow/lite/delegates/gpu/common/task/gpu_operation.h", "IsLinkable");
 return elementwise_ && linkable_; }

  // for linking
  void AddUniquePostfix(const std::string& unique_postfix);

  virtual absl::Status BindArguments(ArgumentsBinder* args) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSgpu_operationDTh mht_8(mht_8_v, 339, "", "./tensorflow/lite/delegates/gpu/common/task/gpu_operation.h", "BindArguments");

    return absl::OkStatus();
  }
  void RecalculateGridSize() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSgpu_operationDTh mht_9(mht_9_v, 345, "", "./tensorflow/lite/delegates/gpu/common/task/gpu_operation.h", "RecalculateGridSize");
 grid_size_ = GetGridSize(); }
  void RecalculateWorkGroupsCount();

  Arguments args_;
  std::string code_;
  int3 work_group_size_ = int3(8, 4, 1);
  std::vector<CompilerOptions> compiler_options_;
  // not applicable to elementwise
  TensorToGrid tensor_to_grid_ = TensorToGrid::kCustom;

  bool elementwise_ = false;
  // applicable only with elementwise_ = true;
  bool linkable_ = true;  // by default every elementwise is linkable
  // applicable only with elementwise_ = true;
  bool check_src_channels_size_ = false;

  // for profiling
  uint64_t flops_ = 0;
  // size in bytes of constant gpu_objects inside args_
  uint64_t const_args_size_ = 0;

  // Must be called before const generic objects in args_ released.
  void CalculateConstArgsSize();

 protected:
  friend flatbuffers::Offset<tflite::gpu::data::GPUOperation> Encode(
      const GPUOperation& op, flatbuffers::FlatBufferBuilder* builder);
  friend absl::Status Decode(const tflite::gpu::data::GPUOperation* fb_op,
                             GPUOperation* op);

  virtual int3 GetGridSize() const;
  virtual void GetPossibleKernelWorkGroups(
      TuningType tuning_type, const GpuInfo& gpu_info,
      const KernelInfo& kernel_info, std::vector<int3>* work_groups) const;

  // Defines operation calculation precision and format of src/dst tensors.
  OperationDef definition_;
  std::vector<GpuSpatialTensor*> src_;
  std::vector<GpuSpatialTensor*> dst_;
  int grid_dimension_ = 3;  // can be 1, 2 or 3
  int3 work_group_launch_order_ = int3(0, 1, 2);
  int3 grid_size_ = int3(0, 0, 0);
  std::vector<std::string> src_tensors_names_;
  std::vector<std::string> dst_tensors_names_;

 private:
  int3 work_groups_count_ = int3(0, 0, 0);
  int linkable_count_ = 0;
  std::string elementwise_code_;  // temporary, used during op construction
};

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASK_GPU_OPERATION_H_
