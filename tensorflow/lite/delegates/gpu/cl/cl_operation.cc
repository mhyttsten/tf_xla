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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_operationDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_operationDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_operationDTcc() {
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

#include "tensorflow/lite/delegates/gpu/cl/cl_operation.h"

#include <string>

namespace tflite {
namespace gpu {
namespace cl {
namespace {
std::string GetCommonOpenCLDefines(CalculationsPrecision precision) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_operationDTcc mht_0(mht_0_v, 193, "", "./tensorflow/lite/delegates/gpu/cl/cl_operation.cc", "GetCommonOpenCLDefines");

  std::string result;

  result += "#define FLT16_0123(V) V.s0123\n";
  result += "#define FLT16_4567(V) V.s4567\n";
  result += "#define FLT16_89ab(V) V.s89ab\n";
  result += "#define FLT16_cdef(V) V.scdef\n";
  result += "#define GLOBAL_ID_0 get_global_id(0)\n";
  result += "#define GLOBAL_ID_1 get_global_id(1)\n";
  result += "#define GLOBAL_ID_2 get_global_id(2)\n";
  result += "#define LOCAL_ID_0 get_local_id(0)\n";
  result += "#define LOCAL_ID_1 get_local_id(1)\n";
  result += "#define LOCAL_ID_2 get_local_id(2)\n";
  result += "#define GROUP_ID_0 get_group_id(0)\n";
  result += "#define GROUP_ID_1 get_group_id(1)\n";
  result += "#define GROUP_ID_2 get_group_id(2)\n";
  result += "#define GROUP_SIZE_0 get_local_size(0)\n";
  result += "#define GROUP_SIZE_1 get_local_size(1)\n";
  result += "#define GROUP_SIZE_2 get_local_size(2)\n";
  result += "#define SUB_GROUP_LOCAL_ID get_sub_group_local_id()\n";
  result += "#define SUB_GROUP_BROADCAST(V, ID) sub_group_broadcast(V, ID)\n";
  result += "#define SIMD_LOCAL_MEM_BARRIER barrier(CLK_LOCAL_MEM_FENCE)\n";
  result += "#define LOCAL_MEM_BARRIER barrier(CLK_LOCAL_MEM_FENCE)\n";
  result += "#define MAIN_FUNCTION __kernel void main_function\n";
  result += "#define INIT_FLOAT(value) (float)(value)\n";
  result += "#define INIT_FLOAT2(value) (float2)(value)\n";
  result += "#define INIT_FLOAT2v2(v0, v1) (float2)(v0, v1)\n";
  result += "#define INIT_FLOAT3(value) (float3)(value)\n";
  result += "#define INIT_FLOAT3v3(v0, v1, v2) (float3)(v0, v1, v2)\n";
  result += "#define INIT_FLOAT4(value) (float4)(value)\n";
  result += "#define INIT_FLOAT4v4(v0, v1, v2, v3) (float4)(v0, v1, v2, v3)\n";
  result += "#define INIT_INT(value) (int)(value)\n";
  result += "#define INIT_INT2v2(v0, v1) (int2)(v0, v1)\n";
  result += "#define INIT_INT4v4(v0, v1, v2, v3) (int4)(v0, v1, v2, v3)\n";
  result += "#define CONVERT_TO_INT4(value) convert_int4(value)\n";
  switch (precision) {
    case CalculationsPrecision::F32:
      result += "#pragma OPENCL EXTENSION cl_khr_3d_image_writes : enable\n";
      result += "#define ACCUM_FLT4 float4\n";
      result += "#define INIT_ACCUM_FLT4(value) (float4)(value)\n";
      result += "#define FLT float\n";
      result += "#define FLT2 float2\n";
      result += "#define FLT3 float3\n";
      result += "#define FLT4 float4\n";
      result += "#define TO_FLT4 convert_float4\n";
      result += "#define TO_ACCUM_TYPE convert_float4\n";
      result += "#define TO_ACCUM_FLT convert_float\n";
      result += "#define TO_ACCUM_FLT2 convert_float2\n";
      result += "#define TO_ACCUM_FLT3 convert_float3\n";
      result += "#define TO_ACCUM_FLT4 convert_float4\n";
      result += "#define INIT_FLT(value) (float)(value)\n";
      result += "#define INIT_FLT4(value) (float4)(value)\n";
      result +=
          "#define INIT_FLT4v4(v0, v1, v2, v3) (float4)(v0, v1, v2, v3)\n";
      break;
    case CalculationsPrecision::F16:
      result += "#pragma OPENCL EXTENSION cl_khr_3d_image_writes : enable\n";
      result += "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n";
      result += "#define ACCUM_FLT4 half4\n";
      result += "#define INIT_ACCUM_FLT4(value) (half4)(value)\n";
      result += "#define FLT half\n";
      result += "#define FLT2 half2\n";
      result += "#define FLT3 half3\n";
      result += "#define FLT4 half4\n";
      result += "#define TO_FLT4 convert_half4\n";
      result += "#define TO_ACCUM_TYPE convert_half4\n";
      result += "#define TO_ACCUM_FLT convert_half\n";
      result += "#define TO_ACCUM_FLT2 convert_half2\n";
      result += "#define TO_ACCUM_FLT3 convert_half3\n";
      result += "#define TO_ACCUM_FLT4 convert_half4\n";
      result += "#define INIT_FLT(value) (half)(value)\n";
      result += "#define INIT_FLT4(value) (half4)(value)\n";
      result += "#define INIT_FLT4v4(v0, v1, v2, v3) (half4)(v0, v1, v2, v3)\n";
      break;
    case CalculationsPrecision::F32_F16:
      result += "#pragma OPENCL EXTENSION cl_khr_3d_image_writes : enable\n";
      result += "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n";
      result += "#define ACCUM_FLT4 float4\n";
      result += "#define INIT_ACCUM_FLT4(value) (float4)(value)\n";
      result += "#define FLT half\n";
      result += "#define FLT2 half2\n";
      result += "#define FLT3 half3\n";
      result += "#define FLT4 half4\n";
      result += "#define TO_FLT4 convert_half4\n";
      result += "#define TO_ACCUM_TYPE convert_float4\n";
      result += "#define TO_ACCUM_FLT convert_float\n";
      result += "#define TO_ACCUM_FLT2 convert_float2\n";
      result += "#define TO_ACCUM_FLT3 convert_float3\n";
      result += "#define TO_ACCUM_FLT4 convert_float4\n";
      result += "#define INIT_FLT(value) (half)(value)\n";
      result += "#define INIT_FLT4(value) (half4)(value)\n";
      result += "#define INIT_FLT4v4(v0, v1, v2, v3) (half4)(v0, v1, v2, v3)\n";
      break;
  }
  return result;
}
}  // namespace

absl::Status ClOperation::AddOperation(ClOperation* operation) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_operationDTcc mht_1(mht_1_v, 294, "", "./tensorflow/lite/delegates/gpu/cl/cl_operation.cc", "ClOperation::AddOperation");

  return operation_->AddOperation(operation->operation_.get());
}

absl::Status ClOperation::UpdateParams() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_operationDTcc mht_2(mht_2_v, 301, "", "./tensorflow/lite/delegates/gpu/cl/cl_operation.cc", "ClOperation::UpdateParams");

  for (int i = 0; i < operation_->GetSrcTensorsNames().size(); ++i) {
    const auto* cl_spatial_tensor =
        dynamic_cast<const Tensor*>(operation_->GetSrcTensors()[i]);
    if (!cl_spatial_tensor) {
      return absl::InvalidArgumentError("Expected CLSpatialTensor.");
    }
    RETURN_IF_ERROR(cl_args_.SetObjectRef(operation_->GetSrcTensorsNames()[i],
                                          cl_spatial_tensor));
  }
  for (int i = 0; i < operation_->GetDstTensorsNames().size(); ++i) {
    const auto* cl_spatial_tensor =
        dynamic_cast<const Tensor*>(operation_->GetDstTensors()[i]);
    if (!cl_spatial_tensor) {
      return absl::InvalidArgumentError("Expected CLSpatialTensor.");
    }
    RETURN_IF_ERROR(cl_args_.SetObjectRef(operation_->GetDstTensorsNames()[i],
                                          cl_spatial_tensor));
  }
  RETURN_IF_ERROR(operation_->BindArguments(&cl_args_));
  operation_->RecalculateGridSize();
  operation_->RecalculateWorkGroupsCount();
  return absl::OkStatus();
}

absl::Status ClOperation::SetSrcTensor(int index, Tensor* tensor) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_operationDTcc mht_3(mht_3_v, 329, "", "./tensorflow/lite/delegates/gpu/cl/cl_operation.cc", "ClOperation::SetSrcTensor");

  operation_->SetSrc(tensor, index);
  return cl_args_.SetObjectRef(operation_->GetSrcTensorsNames()[index], tensor);
}

absl::Status ClOperation::SetDstTensor(int index, Tensor* tensor) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_operationDTcc mht_4(mht_4_v, 337, "", "./tensorflow/lite/delegates/gpu/cl/cl_operation.cc", "ClOperation::SetDstTensor");

  operation_->SetDst(tensor, index);
  return cl_args_.SetObjectRef(operation_->GetDstTensorsNames()[index], tensor);
}

absl::Status ClOperation::Compile(const CreationContext& creation_context) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_operationDTcc mht_5(mht_5_v, 345, "", "./tensorflow/lite/delegates/gpu/cl/cl_operation.cc", "ClOperation::Compile");

  operation_->code_ =
      GetCommonOpenCLDefines(operation_->GetDefinition().precision) +
      operation_->code_;
  RETURN_IF_ERROR(cl_args_.Init(
      creation_context.GetGpuInfo(),
      creation_context.context, &operation_->args_, &operation_->code_));
  operation_->args_.ReleaseCPURepresentation();
  RETURN_IF_ERROR(creation_context.cache->GetOrCreateCLKernel(
      operation_->code_, "main_function", operation_->compiler_options_,
      *creation_context.context, *creation_context.device, &kernel_,
      &kernel_fingerprint_));
  return operation_->PostCompileCheck(creation_context.GetGpuInfo(),
                                      kernel_.info_);
}

absl::Status ClOperation::RestoreDeserialized(const ProgramCache& program_cache,
                                              uint64_t fingerprint,
                                              const GpuInfo& gpu_info,
                                              const int3& work_group_size,
                                              CLContext* context) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_operationDTcc mht_6(mht_6_v, 368, "", "./tensorflow/lite/delegates/gpu/cl/cl_operation.cc", "ClOperation::RestoreDeserialized");

  kernel_fingerprint_ = fingerprint;
  RETURN_IF_ERROR(
      program_cache.GetKernel(kernel_fingerprint_, "main_function", &kernel_));
  operation_->work_group_size_ = work_group_size;
  operation_->RecalculateWorkGroupsCount();
  RETURN_IF_ERROR(cl_args_.Init(gpu_info, &operation_->args_, context));
  operation_->args_.ReleaseCPURepresentation();
  return absl::OkStatus();
}

absl::Status ClOperation::Tune(TuningType tuning_type, const GpuInfo& gpu_info,
                               ProfilingCommandQueue* profiling_queue) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_operationDTcc mht_7(mht_7_v, 383, "", "./tensorflow/lite/delegates/gpu/cl/cl_operation.cc", "ClOperation::Tune");

  std::vector<GPUOperation::DispatchInfo> possible_dispatches;
  operation_->GetPossibleDispatches(tuning_type, gpu_info, kernel_.info_,
                                    &possible_dispatches);
  if (possible_dispatches.empty()) {
    return absl::NotFoundError("No dispatch parameters to launch kernel");
  }
  if (possible_dispatches.size() == 1) {
    operation_->work_group_size_ = possible_dispatches[0].work_group_size;
    operation_->RecalculateWorkGroupsCount();
    return absl::OkStatus();
  } else {
    std::vector<int3> work_group_sizes(possible_dispatches.size());
    std::vector<int3> work_groups_counts(possible_dispatches.size());
    for (int i = 0; i < possible_dispatches.size(); ++i) {
      work_group_sizes[i] = possible_dispatches[i].work_group_size;
      work_groups_counts[i] = possible_dispatches[i].work_groups_count;
    }
    RETURN_IF_ERROR(cl_args_.Bind(kernel_.kernel()));
    int best_work_group_index;
    RETURN_IF_ERROR(profiling_queue->GetBestWorkGroupIndex(
        kernel_, gpu_info, work_groups_counts, work_group_sizes,
        &best_work_group_index));
    operation_->work_group_size_ = work_group_sizes[best_work_group_index];
    operation_->RecalculateWorkGroupsCount();
    return absl::OkStatus();
  }
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
