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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_COMPILER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_COMPILER_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_compilerDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_compilerDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_compilerDTh() {
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


#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "tensorflow/compiler/xla/service/executable.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_device_info.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_executable.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emitter_context.h"
#include "tensorflow/compiler/xla/service/hlo_dataflow_analysis.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/llvm_compiler.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/stream_executor/stream_executor_pimpl.h"

namespace xla {
namespace gpu {

class GpuAotCompilationResult : public AotCompilationResult {
 public:
  static StatusOr<std::unique_ptr<GpuAotCompilationResult>> FromString(
      const std::string& serialized) {
    GpuBefExecutableProto bef_executable;
    if (!bef_executable.ParseFromString(serialized)) {
      return InternalError("Failed to parse serialized GpuBefExecutableProto.");
    }
    return std::unique_ptr<GpuAotCompilationResult>(
        new GpuAotCompilationResult(std::move(bef_executable)));
  }

  GpuAotCompilationResult(HloModuleProto hlo_module_proto,
                          const std::string& bef,
                          EntryFunctionAttributes entry_func_attrs) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("bef: \"" + bef + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_compilerDTh mht_0(mht_0_v, 224, "", "./tensorflow/compiler/xla/service/gpu/gpu_compiler.h", "GpuAotCompilationResult");

    *bef_executable_.mutable_hlo_module_proto() = hlo_module_proto;
    bef_executable_.set_bef(bef);
    *bef_executable_.mutable_entry_func_attrs() = entry_func_attrs;
  }
  ~GpuAotCompilationResult() override = default;

  StatusOr<std::string> SerializeAsString() const override {
    return bef_executable_.SerializeAsString();
  }

  StatusOr<std::unique_ptr<Executable>> LoadExecutable(
      Compiler* compiler, se::StreamExecutor* executor) const override;

  const GpuBefExecutableProto& bef_executable() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_compilerDTh mht_1(mht_1_v, 241, "", "./tensorflow/compiler/xla/service/gpu/gpu_compiler.h", "bef_executable");

    return bef_executable_;
  }

 private:
  explicit GpuAotCompilationResult(GpuBefExecutableProto bef_executable)
      : bef_executable_(std::move(bef_executable)) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_compilerDTh mht_2(mht_2_v, 250, "", "./tensorflow/compiler/xla/service/gpu/gpu_compiler.h", "GpuAotCompilationResult");
}

  GpuBefExecutableProto bef_executable_;
};

// The GPU compiler generates efficient GPU executables.
class GpuCompiler : public LLVMCompiler {
 public:
  GpuCompiler(se::Platform::Id platform_id, const char* target_triple,
              const char* data_layout);
  ~GpuCompiler() override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_compilerDTh mht_3(mht_3_v, 263, "", "./tensorflow/compiler/xla/service/gpu/gpu_compiler.h", "~GpuCompiler");
}

  using LLVMCompiler::Compile;

  StatusOr<std::unique_ptr<HloModule>> RunHloPasses(
      std::unique_ptr<HloModule> module, se::StreamExecutor* stream_exec,
      const CompileOptions& options) override;

  StatusOr<std::unique_ptr<BufferAssignment>> AssignBuffers(
      const HloModule* hlo_module) override;

  virtual GpuVersion GetGpuVersion(se::StreamExecutor* stream_exec) = 0;

  StatusOr<std::unique_ptr<Executable>> RunBackend(
      std::unique_ptr<HloModule> module, se::StreamExecutor* stream_exec,
      const CompileOptions& options) override;

  StatusOr<std::unique_ptr<AotCompilationResult>> LoadAotCompilationResult(
      const std::string& serialized_aot_result) override;

  StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
  CompileAheadOfTime(std::unique_ptr<HloModuleGroup> module_group,
                     AotCompilationOptions const& options) override;

  StatusOr<std::pair<std::string, std::vector<uint8_t>>> CompileToTargetBinary(
      const HloModuleConfig& module_config,
      std::unique_ptr<llvm::Module> llvm_module,
      se::StreamExecutor* stream_exec, const CompileOptions& options,
      const HloModule* debug_module);

  se::Platform::Id PlatformId() const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_compilerDTh mht_4(mht_4_v, 296, "", "./tensorflow/compiler/xla/service/gpu/gpu_compiler.h", "PlatformId");
 return platform_id_; }

  HloCostAnalysis::ShapeSizeFunction ShapeSizeBytesFunction() const override;

 protected:
  virtual Status OptimizeHloPostLayoutAssignment(
      HloModule* hlo_module, se::StreamExecutor* stream_exec,
      se::DeviceMemoryAllocator* device_allocator);

 private:
  Status OptimizeHloModule(HloModule* hlo_module,
                           se::StreamExecutor* stream_exec,
                           se::DeviceMemoryAllocator* device_allocator);

  virtual Status OptimizeHloConvolutionCanonicalization(
      HloModule* hlo_module, se::StreamExecutor* stream_exec,
      se::DeviceMemoryAllocator* device_allocator) = 0;

  virtual HloDataflowAnalysis::CanShareBuffer GetCanShareBuffer() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_compilerDTh mht_5(mht_5_v, 317, "", "./tensorflow/compiler/xla/service/gpu/gpu_compiler.h", "GetCanShareBuffer");

    return
        [](const HloInstruction*, const HloInstruction*,
           const ShapeIndex&) -> absl::optional<bool> { return absl::nullopt; };
  }

  // TODO(timshen): Replace `debug_module` with some portable debug information
  // that accommodates both HLO and MLIR.
  virtual StatusOr<std::pair<std::string, std::vector<uint8_t>>>
  CompileTargetBinary(const HloModuleConfig& module_config,
                      llvm::Module* llvm_module, GpuVersion gpu_version,
                      se::StreamExecutor* stream_exec, bool relocatable,
                      const HloModule* debug_module) = 0;

  Status PrepareHloModuleForIrEmitting(HloModule* hlo_module);

  virtual StatusOr<std::vector<uint8_t>> LinkModules(
      se::StreamExecutor* stream_exec,
      std::vector<std::vector<uint8_t>> modules) {
    return Unimplemented("LinkModules is not implemented.");
  }

  // Optional HloProto, stashed for dumping snapshots.
  std::unique_ptr<HloProto> hlo_proto_;

  se::Platform::Id platform_id_;

  // The triple that represents our target.
  const char* target_triple_;

  // The data layout of the emitted module.
  const char* data_layout_;

  // The size in bytes of a pointer. Used by ShapeSizeBytesFunction.
  const int64_t pointer_size_;

  GpuCompiler(const GpuCompiler&) = delete;
  GpuCompiler& operator=(const GpuCompiler&) = delete;
};

GpuDeviceInfo GetGpuDeviceInfo(se::StreamExecutor* stream_exec);

// Compile `hlo_module` using XLA GPU and return the LLVM module thus generated.
// The GpuExecutable (and the Thunks that are part of it) are not returned.
StatusOr<std::unique_ptr<llvm::Module>> CompileModuleToLlvmIr(
    HloModule* hlo_module, llvm::LLVMContext* llvm_context,
    const std::string& target_triple, const std::string& data_layout,
    const std::string& platform_name, const se::Platform::Id platform_id,
    GpuDeviceInfo gpu_device_info,
    se::CudaComputeCapability cuda_compute_capability,
    se::RocmComputeCapability rocm_compute_capability, int pointer_size);

// Compiles the given LMHLO module to an executable.
// ir_emitter_context should be partially populated: buffer_assignment
// or buffer_allocations should not be populated, while other fields should be
// populated (or left empty if that field is optional).
//
// NOTE: buffer_assignment will be gone from ir_emitter_context once LMHLO
// transition is done.
StatusOr<std::unique_ptr<Executable>> CompileLmhloToExecutable(
    GpuCompiler* compiler, mlir::ModuleOp module, std::string module_name,
    const HloModuleConfig& module_config,
    const Compiler::CompileOptions& options,
    absl::string_view entry_function_name, se::StreamExecutor* stream_exec,
    std::unique_ptr<llvm::Module> llvm_module,
    IrEmitterContext* ir_emitter_context);

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_COMPILER_H_
