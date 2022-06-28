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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSlocal_serviceDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSlocal_serviceDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSlocal_serviceDTcc() {
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

#include "tensorflow/compiler/xla/service/local_service.h"

#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "tensorflow/compiler/xla/client/executable_build_options.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/execution_options_util.h"
#include "tensorflow/compiler/xla/service/backend.h"
#include "tensorflow/compiler/xla/service/computation_layout.h"
#include "tensorflow/compiler/xla/service/executable.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_execution_profile.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/service/hlo_module_util.h"
#include "tensorflow/compiler/xla/service/platform_util.h"
#include "tensorflow/compiler/xla/shape_layout.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace xla {

/* static */ StatusOr<std::unique_ptr<LocalService>> LocalService::NewService(
    const ServiceOptions& options) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSlocal_serviceDTcc mht_0(mht_0_v, 218, "", "./tensorflow/compiler/xla/service/local_service.cc", "LocalService::NewService");

  se::Platform* platform = options.platform();
  if (platform == nullptr) {
    TF_ASSIGN_OR_RETURN(platform, PlatformUtil::GetDefaultPlatform());
  }

  BackendOptions backend_options;
  backend_options.set_platform(platform)
      .set_intra_op_parallelism_threads(options.intra_op_parallelism_threads())
      .set_allowed_devices(options.allowed_devices());

  TF_ASSIGN_OR_RETURN(std::unique_ptr<Backend> backend,
                      Backend::CreateBackend(backend_options));

  std::unique_ptr<LocalService> service(
      new LocalService(options, std::move(backend)));
  return std::move(service);
}

LocalService::LocalService(const ServiceOptions& options,
                           std::unique_ptr<Backend> execute_backend)
    : Service(options, std::move(execute_backend)) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSlocal_serviceDTcc mht_1(mht_1_v, 242, "", "./tensorflow/compiler/xla/service/local_service.cc", "LocalService::LocalService");
}

namespace {

// Retrieves the parameter metadata for the given computation and parameter
// number.
//
// If the parameter number is invalid for this computation, nullopt is
// returned. When the return value has_value(), nullptr will never be
// the held value.
absl::optional<const OpMetadata*> ParameterMetadata(
    const XlaComputation& computation, int parameter_number) {
  for (const HloComputationProto& comp : computation.proto().computations()) {
    if (comp.id() == computation.proto().entry_computation_id()) {
      for (const HloInstructionProto& instr : comp.instructions()) {
        if (instr.opcode() == HloOpcodeString(HloOpcode::kParameter) &&
            instr.parameter_number() == parameter_number) {
          if (!instr.has_metadata()) {
            return absl::nullopt;
          }
          return &instr.metadata();
        }
      }
    }
  }
  return absl::nullopt;
}

}  // namespace

StatusOr<std::unique_ptr<HloModuleConfig>> LocalService::GetHloModuleConfig(
    const XlaComputation& computation,
    const absl::Span<const Shape* const> argument_layouts,
    const ExecutableBuildOptions& build_options) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSlocal_serviceDTcc mht_2(mht_2_v, 278, "", "./tensorflow/compiler/xla/service/local_service.cc", "LocalService::GetHloModuleConfig");

  const HloModuleProto& proto = computation.proto();
  TF_RET_CHECK(proto.has_host_program_shape());
  ProgramShape program_shape(proto.host_program_shape());

  // Validate incoming layouts.
  if (argument_layouts.size() != program_shape.parameters_size()) {
    return InvalidArgument(
        "Invalid number of arguments for computation: expected %d, got %u.",
        program_shape.parameters_size(), argument_layouts.size());
  }

  for (int i = 0; i < argument_layouts.size(); ++i) {
    const Shape& argument_shape = *argument_layouts[i];
    TF_RETURN_IF_ERROR(
        ShapeUtil::ValidateShapeWithOptionalLayout(argument_shape));
    if (!ShapeUtil::Compatible(argument_shape, program_shape.parameters(i))) {
      absl::optional<const OpMetadata*> metadata =
          ParameterMetadata(computation, /*parameter_number=*/i);
      auto metadata_string = [&metadata]() -> std::string {
        if (!metadata.has_value()) {
          return "";
        }
        CHECK(metadata.value() != nullptr);
        const OpMetadata& m = *metadata.value();
        if (!m.source_file().empty()) {
          return absl::StrFormat(" (%s:%d)", m.source_file(), m.source_line());
        }
        return "";
      };
      return InvalidArgument(
          "Invalid argument shape for argument %d%s, expected %s, got %s.", i,
          metadata_string(),
          ShapeUtil::HumanString(program_shape.parameters(i)),
          ShapeUtil::HumanString(argument_shape));
    }
  }
  if (build_options.result_layout() != nullptr) {
    TF_RETURN_IF_ERROR(ValidateResultShape(*build_options.result_layout(),
                                           program_shape.result()));
  }

  ExecutionOptions execution_options =
      CreateExecutionOptions(build_options, &program_shape);

  return CreateModuleConfig(program_shape, argument_layouts,
                            &execution_options);
}

StatusOr<std::vector<std::unique_ptr<Executable>>>
LocalService::CompileExecutables(
    const XlaComputation& computation,
    const absl::Span<const Shape* const> argument_layouts,
    const ExecutableBuildOptions& build_options) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSlocal_serviceDTcc mht_3(mht_3_v, 334, "", "./tensorflow/compiler/xla/service/local_service.cc", "LocalService::CompileExecutables");

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<HloModuleConfig> module_config,
      GetHloModuleConfig(computation, argument_layouts, build_options));

  VLOG(3) << "Computation Layout: "
          << module_config->entry_computation_layout().ToString();

  TF_ASSIGN_OR_RETURN(
      se::StreamExecutor * executor,
      execute_backend_->stream_executor(build_options.device_ordinal()));

  // TODO(cjfj): Investigate why there are a couple of test failures when the
  // single partition computations are built using `BuildExecutables`, fix it,
  // and remove this special case (provided the performance if similar).
  if (build_options.num_partitions() == 1) {
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<Executable> executable,
        BuildExecutable(computation.proto(), std::move(module_config),
                        execute_backend_.get(), executor,
                        {build_options.device_allocator(),
                         build_options.compile_thread_pool()},
                        build_options.run_backend_only()));
    std::vector<std::unique_ptr<Executable>> executables;
    executables.push_back(std::move(executable));
    return executables;
  } else {
    std::vector<std::unique_ptr<HloModuleConfig>> module_configs;
    module_configs.push_back(std::move(module_config));
    // BuildExecutables uses the executors length to determine the number of
    // cores per module, but otherwise only uses the first executor.
    std::vector<se::StreamExecutor*> executors(build_options.num_partitions(),
                                               executor);

    return BuildExecutables(
        /*module_protos=*/{&computation.proto()}, std::move(module_configs),
        execute_backend_.get(), {executors},
        Compiler::CompileOptions{build_options.device_allocator(),
                                 build_options.compile_thread_pool()},
        build_options.run_backend_only());
  }
}

StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
LocalService::CompileAotResults(
    const XlaComputation& computation,
    const absl::Span<const Shape* const> argument_layouts,
    const ExecutableBuildOptions& build_options) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSlocal_serviceDTcc mht_4(mht_4_v, 384, "", "./tensorflow/compiler/xla/service/local_service.cc", "LocalService::CompileAotResults");

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<HloModuleConfig> module_config,
      GetHloModuleConfig(computation, argument_layouts, build_options));

  TF_ASSIGN_OR_RETURN(
      se::StreamExecutor * executor,
      execute_backend_->stream_executor(build_options.device_ordinal()));

  std::vector<std::unique_ptr<HloModuleConfig>> module_configs;
  module_configs.push_back(std::move(module_config));
  // BuildAotResults uses the executors length to determine the number of
  // cores per module, but otherwise only uses the first executor.
  std::vector<se::StreamExecutor*> executors(build_options.num_partitions(),
                                             executor);

  return BuildAotResults(
      /*module_protos=*/{&computation.proto()}, std::move(module_configs),
      execute_backend_.get(), {executors},
      Compiler::CompileOptions{build_options.device_allocator(),
                               build_options.compile_thread_pool()},
      build_options.run_backend_only());
}

StatusOr<int> LocalService::ReplicaNumberToDeviceOrdinal(int replica_number) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSlocal_serviceDTcc mht_5(mht_5_v, 411, "", "./tensorflow/compiler/xla/service/local_service.cc", "LocalService::ReplicaNumberToDeviceOrdinal");

  return backend().computation_placer()->DeviceId(
      replica_number, /*computation=*/0, options_.number_of_replicas(),
      /*computation_count=*/1);
}

StatusOr<const ShapedBuffer*> LocalService::GlobalDataToShapedBuffer(
    const GlobalDataHandle& data, int replica_number) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSlocal_serviceDTcc mht_6(mht_6_v, 421, "", "./tensorflow/compiler/xla/service/local_service.cc", "LocalService::GlobalDataToShapedBuffer");

  TF_ASSIGN_OR_RETURN(auto buffers, allocation_tracker_.Resolve(data));
  if (replica_number >= buffers.size()) {
    return InvalidArgument(
        "replica_number %d out of range; must be less than num_replicas = %u.",
        replica_number, buffers.size());
  }
  return buffers[replica_number];
}

StatusOr<GlobalDataHandle> LocalService::RegisterReplicatedBuffers(
    std::vector<ScopedShapedBuffer> replicated_buffers,
    const std::string& tag) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("tag: \"" + tag + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSlocal_serviceDTcc mht_7(mht_7_v, 437, "", "./tensorflow/compiler/xla/service/local_service.cc", "LocalService::RegisterReplicatedBuffers");

  return allocation_tracker_.RegisterReplicatedBuffers(
      std::move(replicated_buffers), tag);
}

}  // namespace xla
