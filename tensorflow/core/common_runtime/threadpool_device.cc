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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSthreadpool_deviceDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSthreadpool_deviceDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSthreadpool_deviceDTcc() {
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

#if defined(ENABLE_ONEDNN_OPENMP) && defined(ENABLE_MKL) && defined(_OPENMP)
#ifndef DNNL_AARCH64_USE_ACL
// Using LLVM's OpenMP header
#include "external/llvm_openmp/include/omp.h"
/* Added EIGEN_DONT_PARALLELIZE to avoid duplicating omp.h, please refer to
this link https://eigen.tuxfamily.org/dox/TopicMultiThreading.html for more
info. It does not have any negative impact on performance. */
#define EIGEN_DONT_PARALLELIZE
#else
#include "omp.h"  // NOLINT
#endif
#endif  // ENABLE_ONEDNN_OPENMP && ENABLE_MKL &&_OPENMP

#include "absl/base/call_once.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/common_runtime/local_device.h"
#include "tensorflow/core/common_runtime/scoped_allocator.h"
#include "tensorflow/core/common_runtime/scoped_allocator_mgr.h"
#include "tensorflow/core/common_runtime/threadpool_device.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/allocator_registry.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/types.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/util/util.h"

#ifdef INTEL_MKL
#include "tensorflow/core/common_runtime/mkl_cpu_allocator.h"
#include "tensorflow/core/platform/cpu_info.h"
#endif  // INTEL_MKL

namespace tensorflow {

ThreadPoolDevice::ThreadPoolDevice(const SessionOptions& options,
                                   const string& name, Bytes memory_limit,
                                   const DeviceLocality& locality,
                                   Allocator* allocator)
    : LocalDevice(options, Device::BuildDeviceAttributes(
                               name, DEVICE_CPU, memory_limit, locality)),
      allocator_(allocator),
      scoped_allocator_mgr_(new ScopedAllocatorMgr(name)) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSthreadpool_deviceDTcc mht_0(mht_0_v, 233, "", "./tensorflow/core/common_runtime/threadpool_device.cc", "ThreadPoolDevice::ThreadPoolDevice");

  auto s = NodeFileWriter::GetNodeFileWriterIfEnabled(name, env());
  if (!s.ok()) {
    LOG(ERROR) << s.status();
  } else {
    node_file_writer_ = *s;
    if (node_file_writer_) {
      LOG(INFO) << "Writing NodeDefs to file: "
                << node_file_writer_->filename();
    }
  }

#if defined(ENABLE_ONEDNN_OPENMP) && defined(INTEL_MKL)
  // Early return when MKL is disabled
  if (!IsMKLEnabled()) return;
#ifdef _OPENMP
  const char* user_omp_threads = getenv("OMP_NUM_THREADS");
  static absl::once_flag num_threads_setting_flag;
  if (user_omp_threads == nullptr) {
    // OMP_NUM_THREADS controls MKL's intra-op parallelization
    // Default to available physical cores
    const int mkl_intra_op = port::NumSchedulableCPUs();
    const int ht = port::NumHyperthreadsPerCore();
    absl::call_once(num_threads_setting_flag, omp_set_num_threads,
                    (mkl_intra_op + ht - 1) / ht);
  }

#ifndef DNNL_AARCH64_USE_ACL
  const char* user_kmp_blocktime = getenv("KMP_BLOCKTIME");
  static absl::once_flag blocktime_setting_flag;
  if (user_kmp_blocktime == nullptr) {
    // Sets the time, in milliseconds, that a thread should wait,
    // after completing the execution of a parallel region, before sleeping.
    absl::call_once(blocktime_setting_flag, kmp_set_blocktime, 1);
  }
#endif

#endif  // _OPENMP
#endif  // defined(ENABLE_ONEDNN_OPENMP) && defined(INTEL_MKL)
}

ThreadPoolDevice::~ThreadPoolDevice() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSthreadpool_deviceDTcc mht_1(mht_1_v, 277, "", "./tensorflow/core/common_runtime/threadpool_device.cc", "ThreadPoolDevice::~ThreadPoolDevice");
}

Allocator* ThreadPoolDevice::GetAllocator(AllocatorAttributes attr) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSthreadpool_deviceDTcc mht_2(mht_2_v, 282, "", "./tensorflow/core/common_runtime/threadpool_device.cc", "ThreadPoolDevice::GetAllocator");

  return allocator_;
}

Allocator* ThreadPoolDevice::GetScopedAllocator(AllocatorAttributes attr,
                                                int64_t step_id) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSthreadpool_deviceDTcc mht_3(mht_3_v, 290, "", "./tensorflow/core/common_runtime/threadpool_device.cc", "ThreadPoolDevice::GetScopedAllocator");

  if (attr.scope_id > 0) {
    return scoped_allocator_mgr_->GetContainer(step_id)->GetInstance(
        attr.scope_id);
  }
  LOG(FATAL) << "Unexpected call to ThreadPoolDevice::GetScopedAllocator "
             << "attr.scope_id = " << attr.scope_id;
  return allocator_;
}

Status ThreadPoolDevice::MakeTensorFromProto(
    const TensorProto& tensor_proto, const AllocatorAttributes alloc_attrs,
    Tensor* tensor) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSthreadpool_deviceDTcc mht_4(mht_4_v, 305, "", "./tensorflow/core/common_runtime/threadpool_device.cc", "ThreadPoolDevice::MakeTensorFromProto");

  if (tensor_proto.dtype() > 0 && tensor_proto.dtype() <= DataType_MAX) {
    Tensor parsed(tensor_proto.dtype());
    if (parsed.FromProto(allocator_, tensor_proto)) {
      *tensor = std::move(parsed);
      return Status::OK();
    }
  }
  return errors::InvalidArgument("Cannot parse tensor from proto: ",
                                 tensor_proto.DebugString());
}

void ThreadPoolDevice::CopyTensorInSameDevice(
    const Tensor* input_tensor, Tensor* output_tensor,
    const DeviceContext* device_context, StatusCallback done) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSthreadpool_deviceDTcc mht_5(mht_5_v, 322, "", "./tensorflow/core/common_runtime/threadpool_device.cc", "ThreadPoolDevice::CopyTensorInSameDevice");

  if (input_tensor->NumElements() != output_tensor->NumElements()) {
    done(errors::Internal(
        "CPU->CPU copy shape mismatch: input=", input_tensor->shape(),
        ", output=", output_tensor->shape()));
    return;
  }
  tensor::DeepCopy(*input_tensor, output_tensor);
  done(Status::OK());
}

namespace {
const absl::flat_hash_set<std::string>* GetOpsToLogFromEnv() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSthreadpool_deviceDTcc mht_6(mht_6_v, 337, "", "./tensorflow/core/common_runtime/threadpool_device.cc", "GetOpsToLogFromEnv");

  auto* result = new absl::flat_hash_set<std::string>;
  const char* env = getenv("TF_CPU_DEBUG_OPS_TO_LOG");
  if (!env) {
    return result;
  }

  std::vector<absl::string_view> ops = absl::StrSplit(env, ',');
  LOG(INFO) << "Will log inputs & outputs from the following ops: ";
  for (absl::string_view op : ops) {
    result->insert(std::string(op));
    LOG(INFO) << "  |" << op << "|";
  }

  return result;
}

bool ShouldLogInputsAndOutputs(OpKernel* op_kernel) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSthreadpool_deviceDTcc mht_7(mht_7_v, 357, "", "./tensorflow/core/common_runtime/threadpool_device.cc", "ShouldLogInputsAndOutputs");

  static const absl::flat_hash_set<std::string>& ops_to_log =
      *GetOpsToLogFromEnv();
  static const bool is_empty = ops_to_log.empty();
  if (is_empty) {
    return false;
  }
  return ops_to_log.count(op_kernel->type_string());
}
}  // namespace

void ThreadPoolDevice::Compute(OpKernel* op_kernel, OpKernelContext* context) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSthreadpool_deviceDTcc mht_8(mht_8_v, 371, "", "./tensorflow/core/common_runtime/threadpool_device.cc", "ThreadPoolDevice::Compute");

  bool should_log_inputs_and_outputs = ShouldLogInputsAndOutputs(op_kernel);

  if (should_log_inputs_and_outputs) {
    LogInputs(op_kernel, context);
  }

  op_kernel->Compute(context);

  if (context->status().ok() && node_file_writer_) {
    Status s = node_file_writer_->RecordNodeExecution(op_kernel, context);
    if (!s.ok()) {
      LOG(ERROR) << s;
      context->SetStatus(s);
    }
  }

  if (should_log_inputs_and_outputs) {
    LogOutputs(op_kernel, context);
  }
}

void ThreadPoolDevice::ComputeAsync(AsyncOpKernel* op_kernel,
                                    OpKernelContext* context,
                                    AsyncOpKernel::DoneCallback done) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSthreadpool_deviceDTcc mht_9(mht_9_v, 398, "", "./tensorflow/core/common_runtime/threadpool_device.cc", "ThreadPoolDevice::ComputeAsync");

  bool should_log_inputs_and_outputs = ShouldLogInputsAndOutputs(op_kernel);

  if (should_log_inputs_and_outputs) {
    LogInputs(op_kernel, context);
    AsyncOpKernel::DoneCallback parent_done = done;
    done = [this, parent_done, op_kernel, context]() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSthreadpool_deviceDTcc mht_10(mht_10_v, 407, "", "./tensorflow/core/common_runtime/threadpool_device.cc", "lambda");

      LogOutputs(op_kernel, context);
      parent_done();
    };
  }

  op_kernel->ComputeAsync(context, done);
}

void ThreadPoolDevice::LogInputs(OpKernel* op_kernel,
                                 OpKernelContext* context) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSthreadpool_deviceDTcc mht_11(mht_11_v, 420, "", "./tensorflow/core/common_runtime/threadpool_device.cc", "ThreadPoolDevice::LogInputs");

  LOG(INFO) << "Inputs for " << op_kernel->name() << " (total "
            << context->num_inputs() << "):";
  for (int i = 0; i < context->num_inputs(); i++) {
    if (!context->has_input(i)) {
      LOG(INFO) << "input # " << i << " is absent";
      continue;
    }
    LOG(INFO) << "input # " << i;
    LOG(INFO) << context->input(i).DebugString(-1);
  }
  LOG(INFO) << "";
}

void ThreadPoolDevice::LogOutputs(OpKernel* op_kernel,
                                  OpKernelContext* context) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSthreadpool_deviceDTcc mht_12(mht_12_v, 438, "", "./tensorflow/core/common_runtime/threadpool_device.cc", "ThreadPoolDevice::LogOutputs");

  if (!context->status().ok()) {
    LOG(INFO) << op_kernel->name()
              << " failed: " << context->status().error_message();
    return;
  }

  LOG(INFO) << "Outputs for " << op_kernel->name() << " (total "
            << context->num_inputs() << "):";
  for (int i = 0; i < context->num_outputs(); i++) {
    Tensor* output = context->mutable_output(i);
    if (output == nullptr) {
      LOG(INFO) << "output # " << i << " is null";
    } else {
      LOG(INFO) << "output # " << i;
      LOG(INFO) << output->DebugString(-1);
    }
  }
  LOG(INFO) << "";
}

#ifdef INTEL_MKL
namespace {
class MklCPUAllocatorFactory : public AllocatorFactory {
 public:
  bool NumaEnabled() override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSthreadpool_deviceDTcc mht_13(mht_13_v, 466, "", "./tensorflow/core/common_runtime/threadpool_device.cc", "NumaEnabled");
 return false; }

  Allocator* CreateAllocator() override {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSthreadpool_deviceDTcc mht_14(mht_14_v, 471, "", "./tensorflow/core/common_runtime/threadpool_device.cc", "CreateAllocator");
 return new MklCPUAllocator; }

  // Note: Ignores numa_node, for now.
  virtual SubAllocator* CreateSubAllocator(int numa_node) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSthreadpool_deviceDTcc mht_15(mht_15_v, 477, "", "./tensorflow/core/common_runtime/threadpool_device.cc", "CreateSubAllocator");

    return new MklSubAllocator;
  }
};

REGISTER_MEM_ALLOCATOR("MklCPUAllocator", (IsMKLEnabled() ? 200 : 50),
                       MklCPUAllocatorFactory);

}  // namespace
#endif  // INTEL_MKL

}  // namespace tensorflow
