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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSgpu_operationDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSgpu_operationDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSgpu_operationDTcc() {
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

#include "tensorflow/lite/delegates/gpu/common/task/gpu_operation.h"

#include <string>
#include <utility>

#include "absl/strings/substitute.h"
#include "tensorflow/lite/delegates/gpu/common/access_type.h"
#include "tensorflow/lite/delegates/gpu/common/task/work_group_picking.h"

namespace tflite {
namespace gpu {
namespace {
int3 GetWorkGroupsCountInternal(int grid_dimension, const int3& grid_size,
                                const int3& work_group_size,
                                const int3& work_group_launch_order) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSgpu_operationDTcc mht_0(mht_0_v, 199, "", "./tensorflow/lite/delegates/gpu/common/task/gpu_operation.cc", "GetWorkGroupsCountInternal");

  int3 work_groups_count;
  if (grid_dimension == 1) {
    work_groups_count.x = DivideRoundUp(grid_size.x, work_group_size.x);
    work_groups_count.y = 1;
    work_groups_count.z = 1;
  } else if (grid_dimension == 2) {
    int3 wgs;
    wgs.x = DivideRoundUp(grid_size.x, work_group_size.x);
    wgs.y = DivideRoundUp(grid_size.y, work_group_size.y);
    work_groups_count.x = wgs[work_group_launch_order[0]];
    work_groups_count.y = wgs[work_group_launch_order[1]];
    work_groups_count.z = 1;
  } else {  // grid_dimension == 3
    int3 wgs;
    wgs.x = DivideRoundUp(grid_size.x, work_group_size.x);
    wgs.y = DivideRoundUp(grid_size.y, work_group_size.y);
    wgs.z = DivideRoundUp(grid_size.z, work_group_size.z);
    work_groups_count.x = wgs[work_group_launch_order[0]];
    work_groups_count.y = wgs[work_group_launch_order[1]];
    work_groups_count.z = wgs[work_group_launch_order[2]];
  }
  return work_groups_count;
}

std::string GetElementWiseCode(const OperationDef& op_def,
                               bool check_src_slices) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSgpu_operationDTcc mht_1(mht_1_v, 228, "", "./tensorflow/lite/delegates/gpu/common/task/gpu_operation.cc", "GetElementWiseCode");

  std::string c;
  c += "MAIN_FUNCTION(\n";
  c += "$0) {\n";
  c += "  int X = GLOBAL_ID_0;\n";
  c += "  int Y = GLOBAL_ID_1;\n";
  c += "  int Z = GLOBAL_ID_2;\n";
  c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height() || "
       "Z >= args.dst_tensor.Slices()) return; \n";
  if (check_src_slices) {
    c += "  args.src_tensor::type src = args.src_tensor::zero_value;\n";
    c += "  if (Z < args.src_tensor.Slices()) {\n";
    c += "    src = args.src_tensor.Read(X, Y, Z);\n";
    c += "  }\n";
  } else {
    c += "  args.src_tensor::type src = args.src_tensor.Read(X, Y, Z);\n";
  }
  c += "  args.dst_tensor.Write(src, X, Y, Z);\n";
  c += "} \n";
  return c;
}

}  // namespace

DataType OperationDef::GetDataType() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSgpu_operationDTcc mht_2(mht_2_v, 255, "", "./tensorflow/lite/delegates/gpu/common/task/gpu_operation.cc", "OperationDef::GetDataType");

  return DeduceDataTypeFromPrecision(precision);
}

DataType OperationDef::GetPrimaryDataType() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSgpu_operationDTcc mht_3(mht_3_v, 262, "", "./tensorflow/lite/delegates/gpu/common/task/gpu_operation.cc", "OperationDef::GetPrimaryDataType");

  return src_tensors[0].data_type;
}
TensorStorageType OperationDef::GetPrimaryStorageType() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSgpu_operationDTcc mht_4(mht_4_v, 268, "", "./tensorflow/lite/delegates/gpu/common/task/gpu_operation.cc", "OperationDef::GetPrimaryStorageType");

  return src_tensors[0].storage_type;
}

bool OperationDef::IsBatchSupported() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSgpu_operationDTcc mht_5(mht_5_v, 275, "", "./tensorflow/lite/delegates/gpu/common/task/gpu_operation.cc", "OperationDef::IsBatchSupported");

  for (const auto& src : src_tensors) {
    if (HasAxis(src.layout, Axis::BATCH)) {
      return true;
    }
  }
  for (const auto& dst : dst_tensors) {
    if (HasAxis(dst.layout, Axis::BATCH)) {
      return true;
    }
  }
  return false;
}

GPUOperation::GPUOperation(const OperationDef& definition)
    : definition_(definition) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSgpu_operationDTcc mht_6(mht_6_v, 293, "", "./tensorflow/lite/delegates/gpu/common/task/gpu_operation.cc", "GPUOperation::GPUOperation");
}

void GPUOperation::SetSrc(GpuSpatialTensor* ptr, int index) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSgpu_operationDTcc mht_7(mht_7_v, 298, "", "./tensorflow/lite/delegates/gpu/common/task/gpu_operation.cc", "GPUOperation::SetSrc");

  if (index >= src_.size()) {
    src_.resize(index + 1, nullptr);
  }
  src_[index] = ptr;
}

void GPUOperation::SetDst(GpuSpatialTensor* ptr, int index) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSgpu_operationDTcc mht_8(mht_8_v, 308, "", "./tensorflow/lite/delegates/gpu/common/task/gpu_operation.cc", "GPUOperation::SetDst");

  if (index >= dst_.size()) {
    dst_.resize(index + 1, nullptr);
  }
  dst_[index] = ptr;
}

GPUOperation::GPUOperation(GPUOperation&& operation)
    : args_(std::move(operation.args_)),
      code_(std::move(operation.code_)),
      work_group_size_(operation.work_group_size_),
      compiler_options_(std::move(operation.compiler_options_)),
      tensor_to_grid_(operation.tensor_to_grid_),
      elementwise_(operation.elementwise_),
      linkable_(operation.linkable_),
      check_src_channels_size_(operation.check_src_channels_size_),
      flops_(operation.flops_),
      const_args_size_(operation.const_args_size_),
      definition_(std::move(operation.definition_)),
      src_(std::move(operation.src_)),
      dst_(std::move(operation.dst_)),
      grid_dimension_(operation.grid_dimension_),
      work_group_launch_order_(operation.work_group_launch_order_),
      grid_size_(operation.grid_size_),
      src_tensors_names_(std::move(operation.src_tensors_names_)),
      dst_tensors_names_(std::move(operation.dst_tensors_names_)),
      work_groups_count_(operation.work_groups_count_),
      linkable_count_(operation.linkable_count_),
      elementwise_code_(std::move(operation.elementwise_code_)) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSgpu_operationDTcc mht_9(mht_9_v, 339, "", "./tensorflow/lite/delegates/gpu/common/task/gpu_operation.cc", "GPUOperation::GPUOperation");
}

GPUOperation& GPUOperation::operator=(GPUOperation&& operation) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSgpu_operationDTcc mht_10(mht_10_v, 344, "", "./tensorflow/lite/delegates/gpu/common/task/gpu_operation.cc", "=");

  if (this != &operation) {
    args_ = std::move(operation.args_);
    code_ = std::move(operation.code_);
    std::swap(work_group_size_, operation.work_group_size_);
    compiler_options_ = std::move(operation.compiler_options_);
    tensor_to_grid_ = operation.tensor_to_grid_;
    elementwise_ = operation.elementwise_;
    linkable_ = operation.linkable_;
    check_src_channels_size_ = operation.check_src_channels_size_;
    flops_ = operation.flops_;
    const_args_size_ = operation.const_args_size_;
    definition_ = std::move(operation.definition_);
    src_ = std::move(operation.src_);
    dst_ = std::move(operation.dst_);
    std::swap(grid_dimension_, operation.grid_dimension_);
    std::swap(work_group_launch_order_, operation.work_group_launch_order_);
    std::swap(grid_size_, operation.grid_size_);
    src_tensors_names_ = std::move(operation.src_tensors_names_);
    dst_tensors_names_ = std::move(operation.dst_tensors_names_);
    std::swap(work_groups_count_, operation.work_groups_count_);
    std::swap(linkable_count_, operation.linkable_count_);
    elementwise_code_ = std::move(operation.elementwise_code_);
  }
  return *this;
}

absl::Status GPUOperation::AddOperation(GPUOperation* operation) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSgpu_operationDTcc mht_11(mht_11_v, 374, "", "./tensorflow/lite/delegates/gpu/common/task/gpu_operation.cc", "GPUOperation::AddOperation");

  linkable_count_ += 1;
  std::string code = operation->code_;
  std::string unique_postfix = absl::StrCat("_link", linkable_count_);
  operation->args_.RenameArgs(unique_postfix, &code);
  elementwise_code_ += "{\n" + code + "\n}\n";
  RETURN_IF_ERROR(args_.Merge(std::move(operation->args_), unique_postfix));
  for (int i = 0; i < operation->src_tensors_names_.size(); ++i) {
    definition_.src_tensors.push_back(
        operation->definition_.src_tensors[i + 1]);
    src_tensors_names_.push_back(operation->src_tensors_names_[i] +
                                 unique_postfix);
  }
  for (int i = 0; i < operation->dst_tensors_names_.size(); ++i) {
    dst_tensors_names_.push_back(operation->dst_tensors_names_[i] +
                                 unique_postfix);
  }
  return absl::OkStatus();
}

void GPUOperation::AddSrcTensor(const std::string& tensor_name,
                                const TensorDescriptor& desc) {
   std::vector<std::string> mht_12_v;
   mht_12_v.push_back("tensor_name: \"" + tensor_name + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSgpu_operationDTcc mht_12(mht_12_v, 399, "", "./tensorflow/lite/delegates/gpu/common/task/gpu_operation.cc", "GPUOperation::AddSrcTensor");

  src_tensors_names_.push_back(tensor_name);
  auto desc_new = absl::make_unique<TensorDescriptor>(desc);
  args_.AddObjectRef(tensor_name, AccessType::READ, std::move(desc_new));
}

void GPUOperation::AddSrcBuffer(const std::string& buffer_name,
                                const BufferDescriptor& desc) {
   std::vector<std::string> mht_13_v;
   mht_13_v.push_back("buffer_name: \"" + buffer_name + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSgpu_operationDTcc mht_13(mht_13_v, 410, "", "./tensorflow/lite/delegates/gpu/common/task/gpu_operation.cc", "GPUOperation::AddSrcBuffer");

  src_tensors_names_.push_back(buffer_name);
  auto desc_new = absl::make_unique<BufferDescriptor>(desc);
  args_.AddObjectRef(buffer_name, AccessType::READ, std::move(desc_new));
}

void GPUOperation::AddSrcTexture2D(const std::string& texture_name,
                                   const Texture2DDescriptor& desc) {
   std::vector<std::string> mht_14_v;
   mht_14_v.push_back("texture_name: \"" + texture_name + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSgpu_operationDTcc mht_14(mht_14_v, 421, "", "./tensorflow/lite/delegates/gpu/common/task/gpu_operation.cc", "GPUOperation::AddSrcTexture2D");

  src_tensors_names_.push_back(texture_name);
  auto desc_new = absl::make_unique<Texture2DDescriptor>(desc);
  args_.AddObjectRef(texture_name, AccessType::READ, std::move(desc_new));
}

void GPUOperation::AddDstTensor(const std::string& tensor_name,
                                const TensorDescriptor& desc) {
   std::vector<std::string> mht_15_v;
   mht_15_v.push_back("tensor_name: \"" + tensor_name + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSgpu_operationDTcc mht_15(mht_15_v, 432, "", "./tensorflow/lite/delegates/gpu/common/task/gpu_operation.cc", "GPUOperation::AddDstTensor");

  dst_tensors_names_.push_back(tensor_name);
  auto desc_new = absl::make_unique<TensorDescriptor>(desc);
  args_.AddObjectRef(tensor_name, AccessType::WRITE, std::move(desc_new));
}

absl::Status GPUOperation::AssembleCode(const GpuInfo& gpu_info) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSgpu_operationDTcc mht_16(mht_16_v, 441, "", "./tensorflow/lite/delegates/gpu/common/task/gpu_operation.cc", "GPUOperation::AssembleCode");

  if (elementwise_) {
    auto src_desc =
        absl::make_unique<TensorDescriptor>(definition_.src_tensors[0]);
    if (definition_.IsBatchSupported()) {
      src_desc->SetStateVar("BatchedWidth", "true");
    }
    src_tensors_names_.insert(src_tensors_names_.begin(), "src_tensor");
    args_.AddObjectRef("src_tensor", AccessType::READ, std::move(src_desc));

    auto dst_desc =
        absl::make_unique<TensorDescriptor>(definition_.dst_tensors[0]);
    if (definition_.IsBatchSupported()) {
      dst_desc->SetStateVar("BatchedWidth", "true");
    }
    dst_tensors_names_.insert(dst_tensors_names_.begin(), "dst_tensor");
    args_.AddObjectRef("dst_tensor", AccessType::WRITE, std::move(dst_desc));

    elementwise_code_ = "{\n" + code_ + "\n}\n" + elementwise_code_;
    code_ = GetElementWiseCode(definition_, check_src_channels_size_);
  }
  RETURN_IF_ERROR(args_.Compile(
      gpu_info, {{dst_tensors_names_[0], elementwise_code_}}, &code_));
  CalculateConstArgsSize();
  return absl::OkStatus();
}

void GPUOperation::RecalculateWorkGroupsCount() {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSgpu_operationDTcc mht_17(mht_17_v, 471, "", "./tensorflow/lite/delegates/gpu/common/task/gpu_operation.cc", "GPUOperation::RecalculateWorkGroupsCount");

  work_groups_count_ = GetWorkGroupsCountInternal(
      grid_dimension_, grid_size_, work_group_size_, work_group_launch_order_);
}

void GPUOperation::CalculateConstArgsSize() {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSgpu_operationDTcc mht_18(mht_18_v, 479, "", "./tensorflow/lite/delegates/gpu/common/task/gpu_operation.cc", "GPUOperation::CalculateConstArgsSize");

  const_args_size_ = 0;
  for (const auto& obj : args_.GetObjects()) {
    const_args_size_ += obj.second->GetSizeInBytes();
  }
}

void GPUOperation::GetPossibleDispatches(
    TuningType tuning_type, const GpuInfo& gpu_info,
    const KernelInfo& kernel_info,
    std::vector<DispatchInfo>* dispatches) const {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSgpu_operationDTcc mht_19(mht_19_v, 492, "", "./tensorflow/lite/delegates/gpu/common/task/gpu_operation.cc", "GPUOperation::GetPossibleDispatches");

  std::vector<int3> work_group_sizes;
  GetPossibleKernelWorkGroups(tuning_type, gpu_info, kernel_info,
                              &work_group_sizes);
  dispatches->resize(work_group_sizes.size());
  for (int i = 0; i < work_group_sizes.size(); ++i) {
    auto& dispatch_info = (*dispatches)[i];
    dispatch_info.work_group_size = work_group_sizes[i];
    dispatch_info.work_groups_count = GetWorkGroupsCountInternal(
        grid_dimension_, grid_size_, work_group_sizes[i],
        work_group_launch_order_);
  }
}

void GPUOperation::GetPossibleKernelWorkGroups(
    TuningType tuning_type, const GpuInfo& gpu_info,
    const KernelInfo& kernel_info, std::vector<int3>* work_groups) const {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSgpu_operationDTcc mht_20(mht_20_v, 511, "", "./tensorflow/lite/delegates/gpu/common/task/gpu_operation.cc", "GPUOperation::GetPossibleKernelWorkGroups");

  GetPossibleWorkGroups(tuning_type, gpu_info, kernel_info, grid_size_,
                        work_groups);
}

int3 GPUOperation::GetGridSize() const {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSgpu_operationDTcc mht_21(mht_21_v, 519, "", "./tensorflow/lite/delegates/gpu/common/task/gpu_operation.cc", "GPUOperation::GetGridSize");

  if (elementwise_ || tensor_to_grid_ == TensorToGrid::kWBToX_HDToY_SToZ) {
    const int grid_x = dst_[0]->Width() * dst_[0]->Batch();
    const int grid_y = dst_[0]->Height() * dst_[0]->Depth();
    const int grid_z = dst_[0]->Slices();
    return int3(grid_x, grid_y, grid_z);
  }
  if (tensor_to_grid_ == TensorToGrid::kWBToX_HDToY_ZIs1) {
    const int grid_x = dst_[0]->Width() * dst_[0]->Batch();
    const int grid_y = dst_[0]->Height() * dst_[0]->Depth();
    const int grid_z = 1;
    return int3(grid_x, grid_y, grid_z);
  }
  if (tensor_to_grid_ == TensorToGrid::kWBToX_HToY_DToZ) {
    const int grid_x = dst_[0]->Width() * dst_[0]->Batch();
    const int grid_y = dst_[0]->Height();
    const int grid_z = dst_[0]->Depth();
    return int3(grid_x, grid_y, grid_z);
  }
  if (tensor_to_grid_ == TensorToGrid::kBToX_YIs1_ZIs1) {
    const int grid_x = dst_[0]->Batch();
    const int grid_y = 1;
    const int grid_z = 1;
    return int3(grid_x, grid_y, grid_z);
  }
  return grid_size_;
}

void GPUOperation::AddUniquePostfix(const std::string& unique_postfix) {
   std::vector<std::string> mht_22_v;
   mht_22_v.push_back("unique_postfix: \"" + unique_postfix + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSgpu_operationDTcc mht_22(mht_22_v, 551, "", "./tensorflow/lite/delegates/gpu/common/task/gpu_operation.cc", "GPUOperation::AddUniquePostfix");

  for (int i = 0; i < src_tensors_names_.size(); ++i) {
    src_tensors_names_[i] += unique_postfix;
  }
  for (int i = 0; i < dst_tensors_names_.size(); ++i) {
    dst_tensors_names_[i] += unique_postfix;
  }
}

}  // namespace gpu
}  // namespace tflite
