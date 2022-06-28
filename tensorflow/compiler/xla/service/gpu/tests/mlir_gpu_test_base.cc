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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPStestsPSmlir_gpu_test_baseDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPStestsPSmlir_gpu_test_baseDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPStestsPSmlir_gpu_test_baseDTcc() {
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

#include "tensorflow/compiler/xla/service/gpu/tests/mlir_gpu_test_base.h"

#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/InitAllDialects.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/register.h"
#include "tensorflow/compiler/mlir/xla/type_to_shape.h"
#include "tensorflow/compiler/xla/debug_options_flags.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_compiler.h"
#include "tensorflow/compiler/xla/service/gpu/target_constants.h"
#include "tensorflow/core/common_runtime/gpu/gpu_init.h"

namespace xla {
namespace gpu {

MlirGpuTestBase::MlirGpuTestBase() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPStestsPSmlir_gpu_test_baseDTcc mht_0(mht_0_v, 203, "", "./tensorflow/compiler/xla/service/gpu/tests/mlir_gpu_test_base.cc", "MlirGpuTestBase::MlirGpuTestBase");

  se::Platform* platform =
      se::MultiPlatformManager::PlatformWithName(tensorflow::GpuPlatformName())
          .ConsumeValueOrDie();
  BackendOptions options;
  options.set_platform(platform);
  backend_ = xla::Backend::CreateBackend(options).ConsumeValueOrDie();
}

StatusOr<std::unique_ptr<Executable>> MlirGpuTestBase::CompileMlirModule(
    mlir::ModuleOp module, se::Stream* stream) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPStestsPSmlir_gpu_test_baseDTcc mht_1(mht_1_v, 216, "", "./tensorflow/compiler/xla/service/gpu/tests/mlir_gpu_test_base.cc", "MlirGpuTestBase::CompileMlirModule");

  llvm::LLVMContext llvm_context;
  auto llvm_module = absl::make_unique<llvm::Module>("", llvm_context);
#if TENSORFLOW_USE_ROCM
  llvm_module->setTargetTriple(amdgpu::TargetTriple());
  llvm_module->setDataLayout(amdgpu::DataLayout());
#else
  llvm_module->setTargetTriple(nvptx::TargetTriple());
  llvm_module->setDataLayout(nvptx::DataLayout());
#endif

  se::StreamExecutor* stream_exec = stream->parent();
  GpuDeviceInfo gpu_device_info = GetGpuDeviceInfo(stream_exec);
  IrEmitterContext ir_emitter_context(
      /*hlo_module=*/nullptr, /*buffer_assignment=*/nullptr,
      backend_->platform()->Name(), gpu_device_info,
      stream_exec->GetDeviceDescription().cuda_compute_capability(),
      stream_exec->GetDeviceDescription().rocm_compute_capability(),
      /*mlir_context=*/nullptr, llvm_module.get());

  HloModuleConfig module_config;
  module_config.set_debug_options(GetDebugOptionsFromFlags());
  return CompileLmhloToExecutable(
      static_cast<GpuCompiler*>(backend_->compiler()), module, "TestModule",
      module_config, Compiler::CompileOptions(), "main", stream_exec,
      std::move(llvm_module), &ir_emitter_context);
}

StatusOr<ExecutionOutput> MlirGpuTestBase::RunMlirModule(
    mlir::ModuleOp module, se::Stream* stream,
    absl::Span<const se::DeviceMemoryBase> arguments) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPStestsPSmlir_gpu_test_baseDTcc mht_2(mht_2_v, 249, "", "./tensorflow/compiler/xla/service/gpu/tests/mlir_gpu_test_base.cc", "MlirGpuTestBase::RunMlirModule");

  TF_ASSIGN_OR_RETURN(auto executable, CompileMlirModule(module, stream));

  ExecutableRunOptions executable_run_options;
  executable_run_options.set_stream(stream);
  executable_run_options.set_allocator(backend_->memory_allocator());
  ServiceExecutableRunOptions run_options(executable_run_options,
                                          backend_->StreamBorrower());
  std::vector<ExecutionInput> execution_inputs;
  execution_inputs.reserve(arguments.size());

  for (auto arg : arguments) {
    Shape shape =
        ShapeUtil::MakeShape(xla::U8, {static_cast<int64_t>(arg.size())});
    execution_inputs.emplace_back(shape);
    execution_inputs.back().SetBuffer({}, MaybeOwningDeviceMemory(arg));
  }

  TF_ASSIGN_OR_RETURN(auto output,
                      executable->ExecuteAsyncOnStream(
                          &run_options, std::move(execution_inputs),
                          /*hlo_execution_profile=*/nullptr));

  TF_CHECK_OK(stream->BlockHostUntilDone());

  return std::move(output);
}

StatusOr<std::vector<std::vector<uint8_t>>>
MlirGpuTestBase::RunMlirModuleWithHostBuffers(
    mlir::ModuleOp module, std::vector<absl::Span<uint8_t>> arguments) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPStestsPSmlir_gpu_test_baseDTcc mht_3(mht_3_v, 282, "", "./tensorflow/compiler/xla/service/gpu/tests/mlir_gpu_test_base.cc", "MlirGpuTestBase::RunMlirModuleWithHostBuffers");

  auto* allocator = backend_->memory_allocator();
  std::vector<se::OwningDeviceMemory> owning_memory;
  owning_memory.reserve(arguments.size());
  for (auto host_buffer : arguments) {
    owning_memory.push_back(
        allocator
            ->Allocate(backend_->default_device_ordinal(), host_buffer.size())
            .ConsumeValueOrDie());
  }
  auto stream = backend_->BorrowStream(backend_->default_device_ordinal())
                    .ConsumeValueOrDie();
  std::vector<se::DeviceMemoryBase> args;
  for (int i = 0; i < owning_memory.size(); i++) {
    se::DeviceMemoryBase memory(*owning_memory[i]);
    stream->ThenMemcpy(&memory, static_cast<void*>(arguments[i].data()),
                       memory.size());
    args.push_back(memory);
  }
  TF_ASSIGN_OR_RETURN(ExecutionOutput output,
                      RunMlirModule(module, stream.get(), args));

  std::vector<std::vector<uint8_t>> host_outputs;
  for (const auto& result : output.Result().buffers().leaves()) {
    host_outputs.emplace_back();
    host_outputs.back().resize(result.second.size());
    stream->ThenMemcpy(static_cast<void*>(host_outputs.back().data()),
                       result.second, result.second.size());
  }
  TF_CHECK_OK(stream->BlockHostUntilDone());
  return host_outputs;
}

StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> MlirGpuTestBase::ParseMlirModule(
    absl::string_view module_text, mlir::MLIRContext& context) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("module_text: \"" + std::string(module_text.data(), module_text.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPStestsPSmlir_gpu_test_baseDTcc mht_4(mht_4_v, 320, "", "./tensorflow/compiler/xla/service/gpu/tests/mlir_gpu_test_base.cc", "MlirGpuTestBase::ParseMlirModule");

  context
      .loadDialect<mlir::arith::ArithmeticDialect, mlir::lmhlo::LmhloDialect,
                   mlir::mhlo::MhloDialect, mlir::func::FuncDialect,
                   mlir::gpu::GPUDialect, mlir::lmhlo_gpu::LmhloGpuDialect>();
  llvm::SourceMgr source_mgr;
  std::string diagnostic_str;
  llvm::raw_string_ostream os(diagnostic_str);
  mlir::SourceMgrDiagnosticHandler handler(source_mgr, &context, os);

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(
          llvm::StringRef(module_text.data(), module_text.size()), &context);
  if (!module) {
    return InvalidArgument("Failed to parse MLIR module: %s", diagnostic_str);
  }
  return module;
}

StatusOr<std::vector<std::vector<uint8_t>>>
MlirGpuTestBase::RunMlirTextWithHostBuffers(
    absl::string_view module_text, std::vector<absl::Span<uint8_t>> arguments) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("module_text: \"" + std::string(module_text.data(), module_text.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPStestsPSmlir_gpu_test_baseDTcc mht_5(mht_5_v, 345, "", "./tensorflow/compiler/xla/service/gpu/tests/mlir_gpu_test_base.cc", "MlirGpuTestBase::RunMlirTextWithHostBuffers");

  mlir::MLIRContext context;
  TF_ASSIGN_OR_RETURN(mlir::OwningOpRef<mlir::ModuleOp> module,
                      ParseMlirModule(module_text, context));
  return RunMlirModuleWithHostBuffers(*module, arguments);
}

StatusOr<std::unique_ptr<Executable>> MlirGpuTestBase::CompileMlirText(
    absl::string_view module_text) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("module_text: \"" + std::string(module_text.data(), module_text.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPStestsPSmlir_gpu_test_baseDTcc mht_6(mht_6_v, 357, "", "./tensorflow/compiler/xla/service/gpu/tests/mlir_gpu_test_base.cc", "MlirGpuTestBase::CompileMlirText");

  mlir::MLIRContext context;
  TF_ASSIGN_OR_RETURN(mlir::OwningOpRef<mlir::ModuleOp> module,
                      ParseMlirModule(module_text, context));
  auto stream = backend_->BorrowStream(backend_->default_device_ordinal())
                    .ConsumeValueOrDie();
  return CompileMlirModule(*module, stream.get());
}

}  // namespace gpu
}  // namespace xla
