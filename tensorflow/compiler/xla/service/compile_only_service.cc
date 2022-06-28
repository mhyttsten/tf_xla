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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScompile_only_serviceDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScompile_only_serviceDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScompile_only_serviceDTcc() {
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

#include "tensorflow/compiler/xla/service/compile_only_service.h"

#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/debug_options_flags.h"
#include "tensorflow/compiler/xla/service/backend.h"
#include "tensorflow/compiler/xla/service/computation_layout.h"
#include "tensorflow/compiler/xla/service/dump.h"
#include "tensorflow/compiler/xla/service/platform_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/host_info.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace xla {

/* static */ StatusOr<std::unique_ptr<CompileOnlyService>>
CompileOnlyService::NewService(se::Platform* platform) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScompile_only_serviceDTcc mht_0(mht_0_v, 209, "", "./tensorflow/compiler/xla/service/compile_only_service.cc", "CompileOnlyService::NewService");

  ServiceOptions default_options;
  default_options.set_platform(platform);
  return NewService(default_options);
}

/* static */ StatusOr<std::unique_ptr<CompileOnlyService>>
CompileOnlyService::NewService(const ServiceOptions& options) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScompile_only_serviceDTcc mht_1(mht_1_v, 219, "", "./tensorflow/compiler/xla/service/compile_only_service.cc", "CompileOnlyService::NewService");

  se::Platform* platform = options.platform();
  if (platform == nullptr) {
    TF_ASSIGN_OR_RETURN(platform, PlatformUtil::GetDefaultPlatform());
  }

  TF_ASSIGN_OR_RETURN(auto compiler, Compiler::GetForPlatform(platform));

  std::unique_ptr<CompileOnlyService> service(
      new CompileOnlyService(options, compiler));
  return std::move(service);
}

CompileOnlyService::CompileOnlyService(const ServiceOptions& options,
                                       Compiler* compiler)
    : Service(options, /*execute_backend=*/nullptr), compiler_(compiler) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScompile_only_serviceDTcc mht_2(mht_2_v, 237, "", "./tensorflow/compiler/xla/service/compile_only_service.cc", "CompileOnlyService::CompileOnlyService");
}

StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
CompileOnlyService::CompileAheadOfTime(
    const absl::Span<const AotXlaComputationInstance> computations,
    const AotCompilationOptions& options,
    std::unique_ptr<AotCompilationMetadata>* metadata) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScompile_only_serviceDTcc mht_3(mht_3_v, 246, "", "./tensorflow/compiler/xla/service/compile_only_service.cc", "CompileOnlyService::CompileAheadOfTime");

  std::vector<std::unique_ptr<HloModule>> hlo_modules;

  const DebugOptions& debug_options = options.debug_options();
  ExecutionOptions execution_options;
  *execution_options.mutable_debug_options() = debug_options;
  // Capture replica_count, num_cores, and device_assignment in ExecutionOptions
  // to later save in a proto dump.
  if (options.replica_count() > 0) {
    execution_options.set_num_replicas(options.replica_count());
    if (options.has_static_device_assignment()) {
      CHECK_EQ(options.replica_count(),
               options.static_device_assignment().replica_count());
    }
  }
  if (options.num_cores() > 0) {
    execution_options.set_num_partitions(options.num_cores());
    if (options.has_static_device_assignment()) {
      CHECK_EQ(options.num_cores(),
               options.static_device_assignment().computation_count());
    }
  }
  if (options.has_static_device_assignment()) {
    TF_RETURN_IF_ERROR(options.static_device_assignment().Serialize(
        execution_options.mutable_device_assignment()));
  }
  execution_options.set_use_spmd_partitioning(options.use_spmd_partitioning());
  execution_options.set_use_auto_spmd_partitioning(
      options.use_auto_spmd_partitioning());
  execution_options.set_deduplicate_hlo(options.deduplicate_hlo());
  for (const AotXlaComputationInstance& instance : computations) {
    TF_RET_CHECK(instance.computation.has_host_program_shape());
    *execution_options.mutable_shape_with_output_layout() =
        instance.result_layout->ToProto();

    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<HloModuleConfig> module_config,
        CreateModuleConfig(
            ProgramShape(instance.computation.host_program_shape()),
            instance.argument_layouts, &execution_options, &options));

    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<HloModule> hlo_module,
        HloModule::CreateFromProto(instance.computation, *module_config));
    DumpHloModuleIfEnabled(*hlo_module, "before_optimizations");
    hlo_modules.push_back(std::move(hlo_module));
  }

  return compiler_->CompileAheadOfTime(
      absl::make_unique<HloModuleGroup>(hlo_modules[0]->name(),
                                        absl::MakeSpan(hlo_modules)),
      options, metadata);
}

}  // namespace xla
