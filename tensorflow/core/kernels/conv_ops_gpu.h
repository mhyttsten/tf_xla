/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_CONV_OPS_GPU_H_
#define TENSORFLOW_CORE_KERNELS_CONV_OPS_GPU_H_
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
class MHTracer_DTPStensorflowPScorePSkernelsPSconv_ops_gpuDTh {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSconv_ops_gpuDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSconv_ops_gpuDTh() {
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


#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include <tuple>
#include <unordered_map>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/gpu_utils.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/util/autotune_maps/conv_parameters.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {

// Get the Dnn workspace limit from the environment variable, which is in MB.
// Return the workspace memory limit in bytes. If no value is set, return the
// default value.
int64 GetDnnWorkspaceLimit(const string& envvar_in_mb,
                           int64_t default_value_in_bytes);

// A class to provide scratch-space allocator for Stream-Executor Cudnn
// callback. TensorFlow is responsible for releasing the temporary buffers after
// the kernel finishes.
class DnnScratchAllocator : public se::ScratchAllocator {
 public:
  virtual ~DnnScratchAllocator() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSconv_ops_gpuDTh mht_0(mht_0_v, 215, "", "./tensorflow/core/kernels/conv_ops_gpu.h", "~DnnScratchAllocator");
}
  DnnScratchAllocator(int64_t memory_limit, OpKernelContext* context)
      : memory_limit_(memory_limit), total_byte_size_(0), context_(context) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSconv_ops_gpuDTh mht_1(mht_1_v, 220, "", "./tensorflow/core/kernels/conv_ops_gpu.h", "DnnScratchAllocator");
}
  int64 GetMemoryLimitInBytes() override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSconv_ops_gpuDTh mht_2(mht_2_v, 224, "", "./tensorflow/core/kernels/conv_ops_gpu.h", "GetMemoryLimitInBytes");
 return memory_limit_; }
  se::port::StatusOr<se::DeviceMemory<uint8>> AllocateBytes(
      int64_t byte_size) override {
    Tensor temporary_memory;
    if (byte_size < 0) {
      return se::port::Status{se::port::error::INVALID_ARGUMENT,
                              "Requested negative byte size!"};
    }
    if (byte_size > memory_limit_) {
      return se::port::Status{se::port::error::UNAVAILABLE,
                              absl::StrCat("Requested memory size (", byte_size,
                                           ") exceeds the max memory limit (",
                                           memory_limit_, ").")};
    }
    AllocationAttributes allocation_attr;
    allocation_attr.retry_on_failure = false;
    Status allocation_status(context_->allocate_temp(
        DT_UINT8, TensorShape({byte_size}), &temporary_memory,
        AllocatorAttributes(), allocation_attr));
    if (!allocation_status.ok()) {
      return se::port::Status{
          se::port::error::UNAVAILABLE,
          absl::StrCat("Failed to allocate the requested memory size (",
                       byte_size, ").")};
    }
    // Hold the reference of the allocated tensors until the end of the
    // allocator.
    allocated_tensors_.push_back(temporary_memory);
    total_byte_size_ += byte_size;
    return se::port::StatusOr<se::DeviceMemory<uint8>>(
        AsDeviceMemory(temporary_memory.flat<uint8>().data(),
                       temporary_memory.flat<uint8>().size()));
  }
  int64 TotalByteSize() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSconv_ops_gpuDTh mht_3(mht_3_v, 260, "", "./tensorflow/core/kernels/conv_ops_gpu.h", "TotalByteSize");
 return total_byte_size_; }

 private:
  int64 memory_limit_;
  int64 total_byte_size_;
  OpKernelContext* context_;
  std::vector<Tensor> allocated_tensors_;
};

typedef Eigen::GpuDevice GPUDevice;

// Select an algorithm for the given convolution, either by running actual
// autotuning with a cache, or by falling back to a default if
// 'cudnn_use_autotune' is true and cuDNN is the statically-chosen DNN backend.
template <typename T>
StatusOr<AutotuneEntry<se::dnn::FusedConvOp>> AutotuneFusedConv(
    bool cudnn_use_autotune,
    AutotuneMap<ConvParameters, AutotuneEntry<se::dnn::FusedConvOp>>*
        autotune_map,
    const ConvParameters& params, OpKernelContext* ctx,
    const se::dnn::BatchDescriptor& input_desc,
    const se::dnn::FilterDescriptor& filter_desc,
    const se::dnn::BatchDescriptor& bias_desc,
    const se::dnn::BatchDescriptor& output_desc,
    const se::dnn::ConvolutionDescriptor& conv_desc,
    const se::dnn::ActivationMode activation_mode, double conv_input_scale,
    double side_input_scale, se::DeviceMemory<T> input_ptr,
    se::DeviceMemory<T> filter_ptr, se::DeviceMemory<T> output_ptr,
    se::DeviceMemory<T> bias_ptr, se::DeviceMemory<T> side_input_ptr,
    int64_t scratch_size);

template <typename T>
StatusOr<AutotuneEntry<se::dnn::ConvOp>> AutotuneUnfusedConv(
    bool cudnn_use_autotune,
    AutotuneMap<ConvParameters, AutotuneEntry<se::dnn::ConvOp>>* autotune_map,
    const ConvParameters& conv_parameters, OpKernelContext* ctx,
    se::dnn::ConvolutionKind kind, const se::dnn::BatchDescriptor& input_desc,
    se::DeviceMemory<T> input_ptr, const se::dnn::FilterDescriptor& filter_desc,
    se::DeviceMemory<T> filter_ptr,
    const se::dnn::ConvolutionDescriptor& conv_desc,
    const se::dnn::BatchDescriptor& output_desc, se::DeviceMemory<T> output_ptr,
    int64_t scratch_size_limit);

// Returns a pointer to the primary 'OpRunner' of 'runners' and allocated
// scratch memory if allocatable; else a pointer to its fallback
// no-scratch-space runner, and a null 'DeviceMemoryBase'.
template <typename Sig>
StatusOr<std::tuple<const se::dnn::OpRunner<Sig>*, se::DeviceMemoryBase>>
AllocateScratchOrFallback(se::ScratchAllocator* scratch_allocator,
                          const se::dnn::OpRunner<Sig>* primary,
                          const se::dnn::OpRunner<Sig>* no_scratch_fallback) {
  const se::dnn::OpRunner<Sig>* selected_runner = primary;

  auto workspace_size = selected_runner->GetWorkspaceSize();

  se::DeviceMemoryBase scratch_memory;
  if (workspace_size > 0) {
    auto scratch_or = scratch_allocator->AllocateBytes(workspace_size);
    if (scratch_or.ok()) {
      scratch_memory = scratch_or.ValueOrDie();
    } else if ((selected_runner = no_scratch_fallback)) {
      if (selected_runner->GetWorkspaceSize() > 0) {
        return errors::Internal(
            "No-scratch fallback runner requires nonzero scratch space");
      }
    } else {
      return errors::Unknown(
          "CUDNN failed to allocate the scratch space for the runner or to "
          "find a working no-scratch runner.");
    }
  }

  return std::make_tuple(selected_runner, scratch_memory);
}

template <typename T>
Status LaunchAutotunedConv(const AutotuneEntry<se::dnn::ConvOp>& autotune_entry,
                           DnnScratchAllocator* scratch_allocator,
                           se::dnn::ConvolutionKind kind, se::Stream* stream,
                           const se::dnn::BatchDescriptor& input_desc,
                           se::DeviceMemory<T> in_ptr,
                           const se::dnn::FilterDescriptor& filter_desc,
                           se::DeviceMemory<T> filter_ptr,
                           const se::dnn::ConvolutionDescriptor& conv_desc,
                           const se::dnn::BatchDescriptor& output_desc,
                           se::DeviceMemory<T> out_ptr) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSconv_ops_gpuDTh mht_4(mht_4_v, 348, "", "./tensorflow/core/kernels/conv_ops_gpu.h", "LaunchAutotunedConv");

  if (!autotune_entry.is_algorithm_config()) {
    const auto& runners = autotune_entry.GetOpRunners();
    se::dnn::DataType element_type = se::dnn::ToDataType<T>::value;
    se::dnn::ConvOp::Config config{kind,       element_type, element_type,
                                   input_desc, filter_desc,  output_desc,
                                   conv_desc};
    TF_ASSIGN_OR_RETURN(auto* primary,
                        runners.primary->GetOrCreateRunner(config, stream));

    const se::dnn::ConvRunner* no_scratch_fallback = nullptr;
    if (runners.no_scratch_fallback) {
      TF_ASSIGN_OR_RETURN(
          no_scratch_fallback,
          runners.no_scratch_fallback->GetOrCreateRunner(config, stream));
    }

    TF_ASSIGN_OR_RETURN(auto runner_and_scratch,
                        AllocateScratchOrFallback<se::dnn::ConvOp::Signature>(
                            scratch_allocator, primary, no_scratch_fallback));
    auto& runner = *std::get<const se::dnn::ConvRunner*>(runner_and_scratch);
    return runner(stream, nullptr,
                  std::get<se::DeviceMemoryBase>(runner_and_scratch), in_ptr,
                  filter_ptr, out_ptr);
  } else {
    return stream->ConvolveWithAlgorithm(
        kind, input_desc, in_ptr, filter_desc, filter_ptr, output_desc, out_ptr,
        conv_desc, scratch_allocator, autotune_entry.GetAlgorithmConfig(),
        nullptr);
  }
}

}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#endif  // TENSORFLOW_CORE_KERNELS_CONV_OPS_GPU_H_
