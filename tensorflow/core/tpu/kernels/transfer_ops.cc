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
class MHTracer_DTPStensorflowPScorePStpuPSkernelsPStransfer_opsDTcc {
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
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStransfer_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePStpuPSkernelsPStransfer_opsDTcc() {
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

#include "tensorflow/core/tpu/kernels/transfer_ops.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/profiler/lib/connected_traceme.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/profiler/lib/traceme_encode.h"
#include "tensorflow/stream_executor/multi_platform_manager.h"
#include "tensorflow/stream_executor/tpu/tpu_node_context.h"
#include "tensorflow/stream_executor/tpu/tpu_platform_interface.h"
#include "tensorflow/stream_executor/tpu/tpu_transfer_manager_interface.h"

namespace tensorflow {

TpuTransferAsyncOpKernelBase::TpuTransferAsyncOpKernelBase(
    OpKernelConstruction* ctx, const string& transfer_type,
    int number_of_threads, std::unique_ptr<TpuTransferOpInterface> transfer_op)
    : AsyncOpKernel(ctx),
      transfer_type_(transfer_type),
      transfer_op_(std::move(transfer_op)),
      thread_pool_(new thread::ThreadPool(
          ctx->env(),
          strings::StrCat(transfer_type, "_thread_",
                          SanitizeThreadSuffix(def().name())),
          /*num_threads=*/8)) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("transfer_type: \"" + transfer_type + "\"");
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStransfer_opsDTcc mht_0(mht_0_v, 212, "", "./tensorflow/core/tpu/kernels/transfer_ops.cc", "TpuTransferAsyncOpKernelBase::TpuTransferAsyncOpKernelBase");
}

void TpuTransferAsyncOpKernelBase::ComputeAsync(OpKernelContext* ctx,
                                                DoneCallback done) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStransfer_opsDTcc mht_1(mht_1_v, 218, "", "./tensorflow/core/tpu/kernels/transfer_ops.cc", "TpuTransferAsyncOpKernelBase::ComputeAsync");

  profiler::TraceMeProducer schedule_activity(
      "TpuTransferAsyncOpKernelBase::ComputeAsync");
  CancellationToken token =
      ctx->cancellation_manager()->get_cancellation_token();
  bool already_cancelled;
  {
    // Only protect registering the cancellation callback as mu_ cannot be held
    // at a point where `done` could be called.
    mutex_lock lock(mu_);
    already_cancelled =
        !ctx->cancellation_manager()->RegisterCallback(token, [this]() {
          mutex_lock lock(mu_);
          transfer_op_->Cancel();
        });
  }
  OP_REQUIRES_ASYNC(ctx, !already_cancelled,
                    errors::Cancelled("Infeed was cancelled."), done);
  thread_pool_->Schedule(
      [this, ctx, done, token,
       traceme_context_id = schedule_activity.GetContextId()]() {
        profiler::TraceMeConsumer compute_activity(
            [this] { return profiler::TraceMeOp(name(), type_string()); },
            traceme_context_id);
        Status s = RunTransfer(ctx);
        ctx->cancellation_manager()->DeregisterCallback(token);
        OP_REQUIRES_OK_ASYNC(ctx, s, done);
        done();
      });
}

Status TpuTransferAsyncOpKernelBase::RunTransferWithOrdinal(
    OpKernelContext* ctx, int device_ordinal) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStransfer_opsDTcc mht_2(mht_2_v, 253, "", "./tensorflow/core/tpu/kernels/transfer_ops.cc", "TpuTransferAsyncOpKernelBase::RunTransferWithOrdinal");


  int real_device_ordinal = device_ordinal;
  if (real_device_ordinal < 0) {
    TF_ASSIGN_OR_RETURN(real_device_ordinal,
                        transfer_op_->GetDeviceOrdinal(ctx));
  }

  profiler::TraceMe activity(
      [real_device_ordinal] {
        return profiler::TraceMeEncode(
            "RunTransferWithOrdinal",
            {{"device_ordinal", real_device_ordinal}});
      },
      profiler::kInfo);
  return DoWork(ctx, real_device_ordinal);
}

TpuTransferAsyncOpKernel::TpuTransferAsyncOpKernel(
    OpKernelConstruction* ctx, const string& transfer_type,
    int number_of_threads, std::unique_ptr<TpuTransferOpInterface> transfer_op)
    : TpuTransferAsyncOpKernelBase(ctx, transfer_type, number_of_threads,
                                   std::move(transfer_op)) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("transfer_type: \"" + transfer_type + "\"");
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStransfer_opsDTcc mht_3(mht_3_v, 279, "", "./tensorflow/core/tpu/kernels/transfer_ops.cc", "TpuTransferAsyncOpKernel::TpuTransferAsyncOpKernel");

  OP_REQUIRES_OK(ctx, ctx->GetAttr("device_ordinal", &device_ordinal_));
  if (ctx->device_type() == DeviceType(DEVICE_CPU)) {
    OP_REQUIRES(
        ctx, device_ordinal_ >= 0,
        errors::InvalidArgument(transfer_type,
                                " ops must specify a device_ordinal when "
                                "placed on CPU."));
  }
}

Status TpuTransferAsyncOpKernel::RunTransfer(OpKernelContext* ctx) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStransfer_opsDTcc mht_4(mht_4_v, 293, "", "./tensorflow/core/tpu/kernels/transfer_ops.cc", "TpuTransferAsyncOpKernel::RunTransfer");

  return RunTransferWithOrdinal(ctx, device_ordinal_);
}

TpuTransferAsyncDynamicOrdinalOpKernel::TpuTransferAsyncDynamicOrdinalOpKernel(
    OpKernelConstruction* ctx, const string& transfer_type,
    int number_of_threads, std::unique_ptr<TpuTransferOpInterface> transfer_op)
    : TpuTransferAsyncOpKernelBase(ctx, transfer_type, number_of_threads,
                                   std::move(transfer_op)) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("transfer_type: \"" + transfer_type + "\"");
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStransfer_opsDTcc mht_5(mht_5_v, 305, "", "./tensorflow/core/tpu/kernels/transfer_ops.cc", "TpuTransferAsyncDynamicOrdinalOpKernel::TpuTransferAsyncDynamicOrdinalOpKernel");
}

Status TpuTransferAsyncDynamicOrdinalOpKernel::RunTransfer(
    OpKernelContext* ctx) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStransfer_opsDTcc mht_6(mht_6_v, 311, "", "./tensorflow/core/tpu/kernels/transfer_ops.cc", "TpuTransferAsyncDynamicOrdinalOpKernel::RunTransfer");

  const Tensor& device_ordinal_tensor = ctx->input(0);
  const int device_ordinal = device_ordinal_tensor.scalar<int32>()();
  XlaDevice* xla_device =
      dynamic_cast<XlaDevice*>(ctx->device()->UnderlyingDevice());
  if (((xla_device == nullptr) || (xla_device->device_type() == DEVICE_CPU)) &&
      (device_ordinal < 0)) {
    return errors::InvalidArgument(transfer_type_,
                                   " ops must specify a device_ordinal when "
                                   "placed on CPU.");
  }
  return RunTransferWithOrdinal(ctx, device_ordinal);
}

StreamExecutorTransferOpImpl::StreamExecutorTransferOpImpl()
    : transfer_manager_(
          xla::TpuTransferManagerInterface::GetRegisteredTpuTransferManager()),
      tpu_platform_(tpu::TpuPlatformInterface::GetRegisteredPlatform(
          /*initialize_platform=*/false)) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStransfer_opsDTcc mht_7(mht_7_v, 332, "", "./tensorflow/core/tpu/kernels/transfer_ops.cc", "StreamExecutorTransferOpImpl::StreamExecutorTransferOpImpl");
}

void StreamExecutorTransferOpImpl::Cancel() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStransfer_opsDTcc mht_8(mht_8_v, 337, "", "./tensorflow/core/tpu/kernels/transfer_ops.cc", "StreamExecutorTransferOpImpl::Cancel");

  TF_CHECK_OK(tpu::TpuNodeContext::CloseTpuHost());
}

StatusOr<int> StreamExecutorTransferOpImpl::GetDeviceOrdinal(
    OpKernelContext* ctx) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStransfer_opsDTcc mht_9(mht_9_v, 345, "", "./tensorflow/core/tpu/kernels/transfer_ops.cc", "StreamExecutorTransferOpImpl::GetDeviceOrdinal");

  const XlaDevice::Metadata* metadata;
  TF_RETURN_IF_ERROR(XlaDevice::GetMetadata(ctx, &metadata));
  return metadata->device_ordinal();
}

Status StreamExecutorTransferOpImpl::TransferBuffersToInfeed(
    int device_ordinal,
    const std::deque<tensorflow::tpu::NoncopyableBuffer>& buffers) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStransfer_opsDTcc mht_10(mht_10_v, 356, "", "./tensorflow/core/tpu/kernels/transfer_ops.cc", "StreamExecutorTransferOpImpl::TransferBuffersToInfeed");

  TF_ASSIGN_OR_RETURN(auto* executor, GetStreamExecutor(device_ordinal));
  return transfer_manager_->TransferBuffersToInfeed(executor, buffers);
}

Status StreamExecutorTransferOpImpl::TransferLiteralToInfeed(
    int device_ordinal, const xla::LiteralSlice& literal) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStransfer_opsDTcc mht_11(mht_11_v, 365, "", "./tensorflow/core/tpu/kernels/transfer_ops.cc", "StreamExecutorTransferOpImpl::TransferLiteralToInfeed");

  TF_ASSIGN_OR_RETURN(auto* executor, GetStreamExecutor(device_ordinal));
  return transfer_manager_->TransferLiteralToInfeed(executor, literal);
}

Status StreamExecutorTransferOpImpl::TransferLiteralFromOutfeed(
    int device_ordinal, xla::MutableBorrowingLiteral literal) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStransfer_opsDTcc mht_12(mht_12_v, 374, "", "./tensorflow/core/tpu/kernels/transfer_ops.cc", "StreamExecutorTransferOpImpl::TransferLiteralFromOutfeed");

  TF_ASSIGN_OR_RETURN(auto* executor, GetStreamExecutor(device_ordinal));
  return transfer_manager_->TransferLiteralFromOutfeed(executor, literal);
}

StatusOr<stream_executor::StreamExecutor*>
StreamExecutorTransferOpImpl::GetStreamExecutor(int device_ordinal) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStransfer_opsDTcc mht_13(mht_13_v, 383, "", "./tensorflow/core/tpu/kernels/transfer_ops.cc", "StreamExecutorTransferOpImpl::GetStreamExecutor");

  return tpu_platform_->ExecutorForDevice(device_ordinal);
}

}  // namespace tensorflow
