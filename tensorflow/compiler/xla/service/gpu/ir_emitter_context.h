/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_IR_EMITTER_CONTEXT_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_IR_EMITTER_CONTEXT_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emitter_contextDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emitter_contextDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emitter_contextDTh() {
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


#include "llvm/IR/Module.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/lhlo_gpu/IR/lhlo_gpu_ops.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_executable.h"
#include "tensorflow/compiler/xla/service/gpu/launch_dimensions.h"
#include "tensorflow/compiler/xla/service/hlo_execution_profile.h"
#include "tensorflow/compiler/xla/service/name_uniquer.h"

namespace xla {
namespace gpu {

// IrEmitterContext encapsulates common (mutable and immutable) data structures
// used by both IrEmitterNested and IrEmitterUnnested, such as the buffer
// assignment and the name uniquer.
class IrEmitterContext {
 public:
  IrEmitterContext(const HloModule* hlo_module,
                   const BufferAssignment* buffer_assignment,
                   std::string platform_name, GpuDeviceInfo gpu_device_info,
                   se::CudaComputeCapability cuda_compute_capability,
                   se::RocmComputeCapability rocm_compute_capability,
                   mlir::MLIRContext* mlir_context, llvm::Module* llvm_module)
      : hlo_module_(hlo_module),
        buffer_assignment_(buffer_assignment),
        platform_name_(std::move(platform_name)),
        gpu_device_info_(gpu_device_info),
        cuda_compute_capability_(cuda_compute_capability),
        rocm_compute_capability_(rocm_compute_capability),
        mlir_context_(mlir_context),
        llvm_module_(llvm_module) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("platform_name: \"" + platform_name + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emitter_contextDTh mht_0(mht_0_v, 222, "", "./tensorflow/compiler/xla/service/gpu/ir_emitter_context.h", "IrEmitterContext");
}
  // Disallow copy and assign.
  IrEmitterContext(const IrEmitterContext&) = delete;
  IrEmitterContext& operator=(const IrEmitterContext&) = delete;

  // Simple accessors.
  const HloModule& hlo_module() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emitter_contextDTh mht_1(mht_1_v, 231, "", "./tensorflow/compiler/xla/service/gpu/ir_emitter_context.h", "hlo_module");
 return *hlo_module_; }
  const BufferAssignment& buffer_assignment() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emitter_contextDTh mht_2(mht_2_v, 235, "", "./tensorflow/compiler/xla/service/gpu/ir_emitter_context.h", "buffer_assignment");

    return *buffer_assignment_;
  }
  absl::string_view platform_name() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emitter_contextDTh mht_3(mht_3_v, 241, "", "./tensorflow/compiler/xla/service/gpu/ir_emitter_context.h", "platform_name");
 return platform_name_; }
  GpuDeviceInfo gpu_device_info() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emitter_contextDTh mht_4(mht_4_v, 245, "", "./tensorflow/compiler/xla/service/gpu/ir_emitter_context.h", "gpu_device_info");
 return gpu_device_info_; }
  se::CudaComputeCapability cuda_compute_capability() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emitter_contextDTh mht_5(mht_5_v, 249, "", "./tensorflow/compiler/xla/service/gpu/ir_emitter_context.h", "cuda_compute_capability");

    return cuda_compute_capability_;
  }
  se::RocmComputeCapability rocm_compute_capability() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emitter_contextDTh mht_6(mht_6_v, 255, "", "./tensorflow/compiler/xla/service/gpu/ir_emitter_context.h", "rocm_compute_capability");

    return rocm_compute_capability_;
  }
  mlir::MLIRContext* mlir_context() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emitter_contextDTh mht_7(mht_7_v, 261, "", "./tensorflow/compiler/xla/service/gpu/ir_emitter_context.h", "mlir_context");
 return mlir_context_; }
  llvm::Module* llvm_module() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emitter_contextDTh mht_8(mht_8_v, 265, "", "./tensorflow/compiler/xla/service/gpu/ir_emitter_context.h", "llvm_module");
 return llvm_module_; }
  NameUniquer* name_uniquer() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emitter_contextDTh mht_9(mht_9_v, 269, "", "./tensorflow/compiler/xla/service/gpu/ir_emitter_context.h", "name_uniquer");
 return &name_uniquer_; }

  std::vector<GpuExecutable::ConstantInfo>& constants() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emitter_contextDTh mht_10(mht_10_v, 274, "", "./tensorflow/compiler/xla/service/gpu/ir_emitter_context.h", "constants");
 return constants_; }

  absl::Span<const BufferAllocation> allocations() const {
    if (buffer_assignment_) {
      return buffer_assignment_->Allocations();
    }
    return allocations_;
  }

  void set_allocations(absl::Span<const BufferAllocation> allocations) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emitter_contextDTh mht_11(mht_11_v, 286, "", "./tensorflow/compiler/xla/service/gpu/ir_emitter_context.h", "set_allocations");

    CHECK_EQ(nullptr, buffer_assignment_);
    allocations_ = allocations;
  }

 private:
  const HloModule* hlo_module_;
  const BufferAssignment* buffer_assignment_;
  absl::Span<const BufferAllocation> allocations_;
  std::string platform_name_;
  GpuDeviceInfo gpu_device_info_;
  se::CudaComputeCapability cuda_compute_capability_;
  se::RocmComputeCapability rocm_compute_capability_;
  mlir::MLIRContext* mlir_context_;
  llvm::Module* llvm_module_;
  NameUniquer name_uniquer_;
  std::vector<GpuExecutable::ConstantInfo> constants_;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_IR_EMITTER_CONTEXT_H_
