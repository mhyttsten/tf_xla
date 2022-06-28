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
class MHTracer_DTPStensorflowPScorePStpuPStpu_on_demand_compilerDTcc {
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
   MHTracer_DTPStensorflowPScorePStpuPStpu_on_demand_compilerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePStpuPStpu_on_demand_compilerDTcc() {
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
#include <memory>

#include "absl/cleanup/cleanup.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/service/compiler.h"
#include "tensorflow/compiler/xla/service/executable.h"
#include "tensorflow/compiler/xla/service/hlo_cost_analysis.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_module_group.h"
#include "tensorflow/compiler/xla/service/shaped_buffer.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/stream_executor/device_memory_allocator.h"
#include "tensorflow/stream_executor/tpu/c_api_conversions.h"
#include "tensorflow/stream_executor/tpu/c_api_decl.h"
#include "tensorflow/stream_executor/tpu/proto_helper.h"
#include "tensorflow/stream_executor/tpu/status_helper.h"
#include "tensorflow/stream_executor/tpu/tpu_executable.h"
#include "tensorflow/stream_executor/tpu/tpu_executor.h"
#include "tensorflow/stream_executor/tpu/tpu_executor_c_api.h"
#include "tensorflow/stream_executor/tpu/tpu_platform.h"
#include "tensorflow/stream_executor/tpu/tpu_platform_id.h"

namespace xla {

namespace {

using ::tensorflow::tpu::ExecutorApiFn;

class TpuCompiler : public Compiler {
 public:
  TpuCompiler() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePStpuPStpu_on_demand_compilerDTcc mht_0(mht_0_v, 216, "", "./tensorflow/core/tpu/tpu_on_demand_compiler.cc", "TpuCompiler");
 compiler_ = ExecutorApiFn()->TpuCompiler_NewFn(); }
  ~TpuCompiler() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePStpuPStpu_on_demand_compilerDTcc mht_1(mht_1_v, 220, "", "./tensorflow/core/tpu/tpu_on_demand_compiler.cc", "~TpuCompiler");
 ExecutorApiFn()->TpuCompiler_FreeFn(compiler_); }

  stream_executor::Platform::Id PlatformId() const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePStpuPStpu_on_demand_compilerDTcc mht_2(mht_2_v, 225, "", "./tensorflow/core/tpu/tpu_on_demand_compiler.cc", "PlatformId");

    return tensorflow::tpu::GetTpuPlatformId();
  }

  StatusOr<std::unique_ptr<HloModule>> RunHloPasses(
      std::unique_ptr<HloModule> module,
      stream_executor::StreamExecutor* executor,
      const CompileOptions& options) override {
    XLA_HloModule hlo_module;
    auto cleanup = absl::MakeCleanup([&hlo_module]() {
      stream_executor::tpu::SerializedProto_Free(hlo_module.proto);
      ApiConverter::Destroy(&hlo_module.module_config);
    });
    hlo_module.module_config = ApiConverter::ToC(module->config());
    hlo_module.proto = stream_executor::tpu::SerializeProto(module->ToProto());
    auto allocator = ApiConverter::ToC(options.device_allocator);
    XLA_HloModule result;
    StatusHelper status;
    ExecutorApiFn()->TpuCompiler_RunHloPassesFn(
        compiler_, &hlo_module,
        static_cast<tensorflow::tpu::TpuExecutor*>(executor->implementation())
            ->se_executor(),
        &allocator, &result, status.c_status);
    if (!status.ok()) {
      return status.status();
    }
    HloModuleProto result_proto =
        stream_executor::tpu::DeserializeProto<HloModuleProto>(result.proto);
    stream_executor::tpu::SerializedProto_Free(result.proto);
    return HloModule::CreateFromProto(result_proto, module->config());
  }

  StatusOr<std::unique_ptr<Executable>> RunBackend(
      std::unique_ptr<HloModule> module,
      stream_executor::StreamExecutor* executor,
      const CompileOptions& options) override {
    XLA_HloModule hlo_module;
    auto cleanup = absl::MakeCleanup([&hlo_module]() {
      stream_executor::tpu::SerializedProto_Free(hlo_module.proto);
      ApiConverter::Destroy(&hlo_module.module_config);
    });
    SE_Executable* result;
    hlo_module.module_config = ApiConverter::ToC(module->config());
    hlo_module.proto = stream_executor::tpu::SerializeProto(module->ToProto());
    auto allocator = ApiConverter::ToC(options.device_allocator);

    StatusHelper status;
    ExecutorApiFn()->TpuCompiler_RunBackendFn(
        compiler_, &hlo_module,
        static_cast<tensorflow::tpu::TpuExecutor*>(executor->implementation())
            ->se_executor(),
        &allocator, &result, status.c_status);
    if (!status.ok()) {
      return status.status();
    }

    std::unique_ptr<Executable> exec =
        absl::make_unique<TpuExecutable>(result, std::move(module));
    return exec;
  }

  StatusOr<std::vector<std::unique_ptr<Executable>>> Compile(
      std::unique_ptr<HloModuleGroup> module_group,
      std::vector<std::vector<stream_executor::StreamExecutor*>> stream_exec,
      const CompileOptions& options) override {
    XLA_HloModuleGroup se_module_group;
    se_module_group.proto =
        stream_executor::tpu::SerializeProto(module_group->ToProto());
    se_module_group.module_config =
        new XLA_HloModuleConfig[module_group->size()];
    int module_group_size = module_group->size();
    auto cleanup_config =
        absl::MakeCleanup([&se_module_group, module_group_size]() {
          for (auto i = 0; i < module_group_size; ++i) {
            ApiConverter::Destroy(&se_module_group.module_config[i]);
          }
          delete[] se_module_group.module_config;
        });
    for (int i = 0; i < module_group->size(); ++i) {
      const auto& config = module_group->module(i).config();
      se_module_group.module_config[i] = ApiConverter::ToC(config);
    }
    std::vector<SE_StreamExecutorList> se_lists(stream_exec.size());
    std::vector<std::vector<SE_StreamExecutor*>> se_lists_storage;
    for (int i = 0; i < stream_exec.size(); ++i) {
      se_lists[i].count = stream_exec[i].size();
      se_lists_storage.emplace_back(stream_exec[i].size());
      se_lists[i].exec = se_lists_storage.back().data();
      for (int j = 0; j < stream_exec[i].size(); ++j) {
        se_lists[i].exec[j] = static_cast<tensorflow::tpu::TpuExecutor*>(
                                  stream_exec[i][j]->implementation())
                                  ->se_executor();
      }
    }

    SE_DeviceMemoryAllocator allocator =
        ApiConverter::ToC(options.device_allocator);

    SE_Executable** se_executables = new SE_Executable*[module_group->size()];

    StatusHelper status;

    ExecutorApiFn()->TpuCompiler_CompileFn(
        compiler_, &se_module_group, se_lists.data(), stream_exec.size(),
        &allocator, se_executables, status.c_status);

    if (!status.ok()) {
      return status.status();
    }

    std::vector<std::unique_ptr<Executable>> executables;
    for (int i = 0; i < module_group->size(); ++i) {
      // We get the HloModule from the compiled executable, rather than reusing
      // the input module from 'module_group', in case the module changed in
      // some way. For example, if the computation is automatically partitioned
      // via XLA, the executable's module may have different input/output shapes
      // than the input module.
      XLA_HloModule c_module =
          ExecutorApiFn()->TpuExecutable_HloModuleFn(se_executables[i]);
      auto cleanup_c_module = absl::MakeCleanup(
          [&c_module]() { ApiConverter::Destroy(&c_module); });
      TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                          ApiConverter::FromC(c_module));
      std::shared_ptr<HloModule> module_shared(module.release());
      executables.emplace_back(absl::make_unique<TpuExecutable>(
          se_executables[i], std::move(module_shared)));
    }

    stream_executor::tpu::SerializedProto_Free(se_module_group.proto);
    delete[] se_executables;

    return executables;
  }

  // Compiles the HLO module group for ahead-of-time execution.  This is
  // intended for use in static compilation.
  StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
  CompileAheadOfTime(std::unique_ptr<HloModuleGroup> module_group,
                     const AotCompilationOptions& options) override {
    return Unimplemented("This compiler does not support CompileAheadOfTime.");
  }

  // Returns a function that computes the size in bytes of the logical
  // buffer that contains a shape.
  HloCostAnalysis::ShapeSizeFunction ShapeSizeBytesFunction() const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePStpuPStpu_on_demand_compilerDTcc mht_3(mht_3_v, 372, "", "./tensorflow/core/tpu/tpu_on_demand_compiler.cc", "ShapeSizeBytesFunction");

    return [this](const xla::Shape& shape) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePStpuPStpu_on_demand_compilerDTcc mht_4(mht_4_v, 376, "", "./tensorflow/core/tpu/tpu_on_demand_compiler.cc", "lambda");

      XLA_Shape c_shape;
      ApiConverter::ToC(shape, &c_shape);
      int64_t bytes =
          ExecutorApiFn()->TpuCompiler_ShapeSizeFn(compiler_, &c_shape);
      ApiConverter::Destroy(&c_shape);
      return bytes;
    };
  }

 private:
  Tpu_Compiler* compiler_;
};

static bool InitModule() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePStpuPStpu_on_demand_compilerDTcc mht_5(mht_5_v, 393, "", "./tensorflow/core/tpu/tpu_on_demand_compiler.cc", "InitModule");

  xla::Compiler::RegisterCompilerFactory(
      tensorflow::tpu::GetTpuPlatformId(),
      []() { return absl::make_unique<TpuCompiler>(); });
  return true;
}

static bool module_initialized = InitModule();

}  // namespace
}  // namespace xla
