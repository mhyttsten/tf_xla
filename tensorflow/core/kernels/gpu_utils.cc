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
class MHTracer_DTPStensorflowPScorePSkernelsPSgpu_utilsDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSgpu_utilsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSgpu_utilsDTcc() {
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

#include "tensorflow/core/kernels/gpu_utils.h"

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include <iterator>

#include "google/protobuf/any.pb.h"
#include "absl/algorithm/container.h"
#include "absl/base/call_once.h"
#include "tensorflow/core/platform/logger.h"
#include "tensorflow/core/protobuf/autotuning.pb.h"
#include "tensorflow/core/protobuf/conv_autotuning.pb.h"
#include "tensorflow/core/util/determinism.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/core/util/proto/proto_utils.h"
#include "tensorflow/stream_executor/gpu/asm_compiler.h"
#include "tensorflow/stream_executor/gpu/redzone_allocator.h"

namespace tensorflow {

bool RedzoneCheckDisabled() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSgpu_utilsDTcc mht_0(mht_0_v, 205, "", "./tensorflow/core/kernels/gpu_utils.cc", "RedzoneCheckDisabled");

  const char* disable_rz_str = std::getenv("TF_DISABLE_RZ_CHECK");
  return disable_rz_str != nullptr && std::strcmp(disable_rz_str, "1") == 0;
}

se::DeviceMemoryBase WrapRedzoneBestEffort(se::RedzoneAllocator* rz_allocator,
                                           se::DeviceMemoryBase buffer) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSgpu_utilsDTcc mht_1(mht_1_v, 214, "", "./tensorflow/core/kernels/gpu_utils.cc", "WrapRedzoneBestEffort");

  if (RedzoneCheckDisabled()) {
    return buffer;
  }
  auto output_rz_or = rz_allocator->AllocateBytes(buffer.size());
  if (!output_rz_or.ok()) {
    static absl::once_flag rz_allocation_failure_logged;
    absl::call_once(rz_allocation_failure_logged, []() {
      LOG(WARNING) << "Failed to allocate memory for convolution redzone "
                   << "checking; skipping this check. This is benign and only "
                   << "means that we won't check cudnn for out-of-bounds reads "
                   << "and writes. This message will only be printed once.";
    });
    return buffer;
  }
  return se::DeviceMemoryBase(output_rz_or.ValueOrDie());
}

void CheckRedzones(const se::RedzoneAllocator& rz_allocator,
                   tensorflow::AutotuneResult* autotune_result) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSgpu_utilsDTcc mht_2(mht_2_v, 236, "", "./tensorflow/core/kernels/gpu_utils.cc", "CheckRedzones");

  if (RedzoneCheckDisabled()) {
    return;
  }
  se::port::StatusOr<se::RedzoneAllocator::RedzoneCheckStatus> rz_status =
      rz_allocator.CheckRedzones();
  if (!rz_status.ok()) {
    static absl::once_flag failure_logged;
    absl::call_once(failure_logged, [&]() {
      LOG(WARNING) << "Failed to check cudnn convolutions for out-of-bounds "
                   << "reads and writes with an error message: '"
                   << rz_status.status().error_message()
                   << "'; skipping this check. This only means that we won't "
                   << "check cudnn for out-of-bounds reads and writes. This "
                   << "message will only be printed once.";
    });
    return;
  }
  auto rz_check_status = rz_status.ValueOrDie();
  if (!rz_check_status.ok()) {
    auto* fail = autotune_result->mutable_failure();
    fail->set_msg(rz_check_status.RedzoneFailureMsg());
    fail->set_kind(AutotuneResult::REDZONE_MODIFIED);
    fail->set_buffer_address(
        reinterpret_cast<uint64>(rz_check_status.user_buffer_address));
    LOG(ERROR)
        << "Detected cudnn out-of-bounds write in convolution buffer! This is "
           "likely a cudnn bug. We will skip this algorithm in the future, but "
           "your GPU state may already be corrupted, leading to incorrect "
           "results. Within Google, no action is needed on your part. Outside "
           "of Google, please ensure you're running the latest version of "
           "cudnn. If that doesn't fix the problem, please file a bug with "
           "this full error message and we'll contact nvidia.";
    LOG(ERROR) << rz_check_status.RedzoneFailureMsg();
  }
}

namespace {

tensorflow::CudnnVersion GetCudnnVersion(se::StreamExecutor* stream_executor) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSgpu_utilsDTcc mht_3(mht_3_v, 278, "", "./tensorflow/core/kernels/gpu_utils.cc", "GetCudnnVersion");

  tensorflow::CudnnVersion cudnn_version;
  if (auto* dnn = stream_executor->AsDnn()) {
    se::port::StatusOr<se::dnn::VersionInfo> version_or = dnn->GetVersion();
    if (version_or.ok()) {
      const auto& version = version_or.ValueOrDie();
      cudnn_version.set_major(version.major_version());
      cudnn_version.set_minor(version.minor_version());
      cudnn_version.set_patch(version.patch());
    }
  }
  return cudnn_version;
}

tensorflow::ComputeCapability GetComputeCapability(
    se::StreamExecutor* stream_executor) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSgpu_utilsDTcc mht_4(mht_4_v, 296, "", "./tensorflow/core/kernels/gpu_utils.cc", "GetComputeCapability");

  tensorflow::ComputeCapability cc_proto;
  se::CudaComputeCapability cc =
      stream_executor->GetDeviceDescription().cuda_compute_capability();
  cc_proto.set_major(cc.major);
  cc_proto.set_minor(cc.minor);
  return cc_proto;
}

}  // namespace

void LogConvAutotuneResults(se::dnn::ConvolutionKind kind,
                            se::dnn::DataType element_type,
                            se::DeviceMemoryBase input_buffer,
                            se::DeviceMemoryBase filter_buffer,
                            se::DeviceMemoryBase output_buffer,
                            const se::dnn::BatchDescriptor& input_desc,
                            const se::dnn::FilterDescriptor& filter_desc,
                            const se::dnn::BatchDescriptor& output_desc,
                            const se::dnn::ConvolutionDescriptor& conv_desc,
                            se::StreamExecutor* stream_exec,
                            absl::Span<const AutotuneResult> results) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSgpu_utilsDTcc mht_5(mht_5_v, 320, "", "./tensorflow/core/kernels/gpu_utils.cc", "LogConvAutotuneResults");

  AutotuningLog log;
  {
    ConvolutionProto instr;
    instr.set_kind(kind);
    *instr.mutable_input() = input_desc.ToProto(element_type);
    *instr.mutable_filter() = filter_desc.ToProto(element_type);
    *instr.mutable_output() = output_desc.ToProto(element_type);
    *instr.mutable_conv_desc() = conv_desc.ToProto();
    instr.set_conv_scale(1);
    instr.set_side_value_scale(0);
    instr.set_input_address(reinterpret_cast<uint64>(input_buffer.opaque()));
    instr.set_filter_address(reinterpret_cast<uint64>(filter_buffer.opaque()));
    instr.set_output_address(reinterpret_cast<uint64>(output_buffer.opaque()));
    log.mutable_instr()->PackFrom(std::move(instr));
  }
  *log.mutable_cudnn_version() = GetCudnnVersion(stream_exec);
  *log.mutable_compute_capability() = GetComputeCapability(stream_exec);
  log.set_device_pci_bus_id(stream_exec->GetDeviceDescription().pci_bus_id());
  {
    string blas_version;
    if (auto* blas = stream_exec->AsBlas()) {
      if (blas->GetVersion(&blas_version).ok()) {
        log.set_blas_version(blas_version);
      }
    }
  }
  for (const auto& result : results) {
    *log.add_results() = result;
  }
  VLOG(2) << log.DebugString();
  Logger::GetSingleton()->LogProto(log);
}

void LogFusedConvForwardAutotuneResults(
    se::dnn::DataType element_type, se::DeviceMemoryBase input_buffer,
    se::DeviceMemoryBase filter_buffer, se::DeviceMemoryBase output_buffer,
    se::DeviceMemoryBase bias_buffer, se::DeviceMemoryBase side_input_buffer,
    const se::dnn::BatchDescriptor& input_desc,
    const se::dnn::FilterDescriptor& filter_desc,
    const se::dnn::BatchDescriptor& output_desc,
    const se::dnn::ConvolutionDescriptor& conv_desc, double conv_scale,
    double side_value_scale, se::dnn::ActivationMode activation_mode,
    se::StreamExecutor* stream_exec, absl::Span<const AutotuneResult> results) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSgpu_utilsDTcc mht_6(mht_6_v, 366, "", "./tensorflow/core/kernels/gpu_utils.cc", "LogFusedConvForwardAutotuneResults");

  AutotuningLog log;
  {
    ConvolutionProto instr;
    instr.set_kind(se::dnn::ConvolutionKind::FORWARD_BIAS_ACTIVATION);
    *instr.mutable_input() = input_desc.ToProto(element_type);
    *instr.mutable_filter() = filter_desc.ToProto(element_type);
    *instr.mutable_output() = output_desc.ToProto(element_type);
    *instr.mutable_conv_desc() = conv_desc.ToProto();
    instr.set_conv_scale(conv_scale);
    instr.set_side_value_scale(side_value_scale);
    instr.set_activation(activation_mode);
    instr.set_input_address(reinterpret_cast<uint64>(input_buffer.opaque()));
    instr.set_filter_address(reinterpret_cast<uint64>(filter_buffer.opaque()));
    instr.set_output_address(reinterpret_cast<uint64>(output_buffer.opaque()));
    instr.set_bias_address(reinterpret_cast<uint64>(bias_buffer.opaque()));
    instr.set_side_input_address(
        reinterpret_cast<uint64>(side_input_buffer.opaque()));
    log.mutable_instr()->PackFrom(std::move(instr));
  }
  *log.mutable_cudnn_version() = GetCudnnVersion(stream_exec);
  *log.mutable_compute_capability() = GetComputeCapability(stream_exec);
  log.set_device_pci_bus_id(stream_exec->GetDeviceDescription().pci_bus_id());
  {
    string blas_version;
    if (auto* blas = stream_exec->AsBlas()) {
      if (blas->GetVersion(&blas_version).ok()) {
        log.set_blas_version(blas_version);
      }
    }
  }
  for (const auto& result : results) {
    *log.add_results() = result;
  }
  VLOG(2) << log.DebugString();
  Logger::GetSingleton()->LogProto(log);
}

namespace {
StatusOr<std::tuple<int, int>> BestCudnnConvAlgorithmIndices(
    absl::Span<const AutotuneResult> results) {
  auto compare_run_times = [](const AutotuneResult& lhs,
                              const AutotuneResult& rhs) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSgpu_utilsDTcc mht_7(mht_7_v, 411, "", "./tensorflow/core/kernels/gpu_utils.cc", "lambda");

    return proto_utils::FromDurationProto(lhs.run_time()) <
           proto_utils::FromDurationProto(rhs.run_time());
  };
  int idx = -1;
  int idx_no_scratch = -1;
  for (int i = 0; i < results.size(); i++) {
    if (!results[i].has_failure()) {
      if (OpDeterminismRequired()) {
        // When determinism is enabled, choose first working algorithm, and
        // don't choose a no_scratch algorithm.
        idx = i;
        break;
      }
      if (idx == -1 || compare_run_times(results[i], results[idx])) {
        idx = i;
      }
      if (results[i].scratch_bytes() == 0 &&
          (idx_no_scratch == -1 ||
           compare_run_times(results[i], results[idx_no_scratch]))) {
        idx_no_scratch = i;
      }
    }
  }

  if (idx == -1) {
    std::ostringstream msg;
    msg << "No algorithm worked!  Error messages:";
    // TODO(awpr): identify the algorithm as part of this error message, too.
    for (const auto& result : results) {
      msg << "\n  " << result.failure().msg();
    }
    return errors::NotFound(msg.str());
  }

  return std::make_tuple(idx, idx_no_scratch);
}
}  // namespace

StatusOr<se::dnn::AlgorithmConfig> BestCudnnConvAlgorithm(
    absl::Span<const AutotuneResult> results) {
  int idx;
  int idx_no_scratch;
  TF_ASSIGN_OR_RETURN(std::tie(idx, idx_no_scratch),
                      BestCudnnConvAlgorithmIndices(results));
  VLOG(2) << "fastest algorithm: "
          << proto_utils::FromDurationProto(results[idx].run_time())
          << " with algo " << results[idx].algorithm().algo_id()
          << ", workspace bytes " << results[idx].scratch_bytes();

  se::dnn::AlgorithmConfig result(
      se::dnn::AlgorithmDesc(results[idx].algorithm()),
      results[idx].scratch_bytes());

  if (idx_no_scratch != -1) {
    result.set_algorithm_no_scratch(
        se::dnn::AlgorithmDesc(results[idx_no_scratch].algorithm()));
  }
  return result;
}

template <typename Op>
StatusOr<AutotuneEntry<Op>> BestCudnnConvAlgorithm(
    absl::Span<const AutotuneResult> results,
    std::vector<
        std::unique_ptr<const se::dnn::OpRunner<typename Op::Signature>>>
        runners) {
  if (runners.size() != results.size()) {
    return errors::Internal(
        "Mismatched size of autotune results and runners vectors.");
  }
  int idx;
  int idx_no_scratch;
  TF_ASSIGN_OR_RETURN(std::tie(idx, idx_no_scratch),
                      BestCudnnConvAlgorithmIndices(results));
  VLOG(2) << "fastest algorithm: "
          << proto_utils::FromDurationProto(results[idx].run_time())
          << " with algo " << runners[idx]->ToString() << ", workspace bytes "
          << results[idx].scratch_bytes();
  return AutotuneEntry<Op>::FromOpRunners(
      std::move(runners[idx]), idx_no_scratch == -1 || idx_no_scratch == idx
                                   ? nullptr
                                   : std::move(runners[idx_no_scratch]));
}

template StatusOr<AutotuneEntry<se::dnn::ConvOp>>
BestCudnnConvAlgorithm<se::dnn::ConvOp>(
    absl::Span<const AutotuneResult> results,
    std::vector<
        std::unique_ptr<const se::dnn::OpRunner<se::dnn::ConvSignature>>>
        runners);

template StatusOr<AutotuneEntry<se::dnn::FusedConvOp>>
BestCudnnConvAlgorithm<se::dnn::FusedConvOp>(
    absl::Span<const AutotuneResult> results,
    std::vector<
        std::unique_ptr<const se::dnn::OpRunner<se::dnn::FusedConvSignature>>>
        runners);

}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
