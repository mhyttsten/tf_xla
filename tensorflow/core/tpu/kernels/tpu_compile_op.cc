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
class MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_compile_opDTcc {
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
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_compile_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_compile_opDTcc() {
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
#include "tensorflow/core/tpu/kernels/tpu_compile_op.h"

#include <string>

#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/protobuf/tpu/compilation_result.pb.h"
#include "tensorflow/core/tpu/kernels/tpu_compile_op_options.h"
#include "tensorflow/stream_executor/tpu/tpu_node_context.h"

namespace tensorflow {
namespace tpu {
using ::stream_executor::port::StatusOr;

TpuCompileOp::TpuCompileOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_compile_opDTcc mht_0(mht_0_v, 197, "", "./tensorflow/core/tpu/kernels/tpu_compile_op.cc", "TpuCompileOp::TpuCompileOp");

  StatusOr<std::unique_ptr<TpuCompileOpKernelCommon>> compile_op_impl =
      CompileOpImplFactory::Get()->CreateNonMlirImpl(ctx);
  OP_REQUIRES_OK(ctx, compile_op_impl.status());
  impl_ = std::move(compile_op_impl.ValueOrDie());
}

void TpuCompileOp::Compute(OpKernelContext* ctx) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_compile_opDTcc mht_1(mht_1_v, 207, "", "./tensorflow/core/tpu/kernels/tpu_compile_op.cc", "TpuCompileOp::Compute");
 impl_->Compute(ctx); }

TpuCompileMlirOp::TpuCompileMlirOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_compile_opDTcc mht_2(mht_2_v, 212, "", "./tensorflow/core/tpu/kernels/tpu_compile_op.cc", "TpuCompileMlirOp::TpuCompileMlirOp");

  StatusOr<std::unique_ptr<TpuCompileOpKernelCommon>> compile_op_impl =
      CompileOpImplFactory::Get()->CreateMlirImpl(ctx);
  OP_REQUIRES_OK(ctx, compile_op_impl.status());
  impl_ = std::move(compile_op_impl.ValueOrDie());
}

void TpuCompileMlirOp::Compute(OpKernelContext* ctx) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_compile_opDTcc mht_3(mht_3_v, 222, "", "./tensorflow/core/tpu/kernels/tpu_compile_op.cc", "TpuCompileMlirOp::Compute");
 impl_->Compute(ctx); }

void TpuCompileSucceededAssertOp::Compute(OpKernelContext* ctx) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_compile_opDTcc mht_4(mht_4_v, 227, "", "./tensorflow/core/tpu/kernels/tpu_compile_op.cc", "TpuCompileSucceededAssertOp::Compute");

  const Tensor compilation_result = ctx->input(0);
  CompilationResultProto proto;
  Status status;
  if (!proto.ParseFromString(compilation_result.scalar<tstring>()())) {
    status =
        errors::InvalidArgument("Unable to parse compilation result proto");
  }
  if (!status.ok() || proto.status_code() != error::Code::OK) {
    status.Update(Status(proto.status_code(), proto.status_error_message()));
    LOG(WARNING) << "TPU compilation failed: " << status;
    errors::AppendToMessage(&status, "TPU compilation failed");
    if (tensorflow::internal::TpuCompilationFailureClosesChips()) {
      // At this point, if compilation fails we do not know if a task
      // is already running that expects results from this compiled
      // program to complete. So close the TPU driver to release all
      // awaiting interactions (all awaiting interaction will fail and
      // continue to fail until reinitialized).
      LOG(ERROR) << "Cloud TPU: Closing chips. TPU compilation is considered "
                    "as part of device state, and a failed compilation results "
                    "in a device reset.";

      Status close_status = TpuNodeContext::CloseTpuHost();

      if (!close_status.ok()) {
        errors::AppendToMessage(&status, close_status.error_message());
      }
    }
    ctx->CtxFailure(status);
  }
}

REGISTER_MODULE_INITIALIZER(register_tpu_compile_op_kernel, {
  VLOG(1) << "Register TpuCompileOp kernel.";
  REGISTER_KERNEL_BUILDER(Name("TPUCompile").Device(DEVICE_CPU), TpuCompileOp);
  REGISTER_KERNEL_BUILDER(Name("_TPUCompileMlir").Device(DEVICE_CPU),
                          TpuCompileMlirOp);
  REGISTER_KERNEL_BUILDER(Name("TPUCompileSucceededAssert").Device(DEVICE_CPU),
                          TpuCompileSucceededAssertOp);
});
}  // namespace tpu
}  // namespace tensorflow
