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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_utilDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_utilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_utilDTcc() {
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

#include "tensorflow/compiler/xla/service/hlo_module_util.h"

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/service/compiler.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {

namespace {

Status ValidateResultShape(const Shape& client_shape,
                           const Shape& result_shape) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_module_utilDTcc mht_0(mht_0_v, 200, "", "./tensorflow/compiler/xla/service/hlo_module_util.cc", "ValidateResultShape");

  TF_RETURN_IF_ERROR(ShapeUtil::ValidateShapeWithOptionalLayout(client_shape));
  if (!ShapeUtil::Compatible(client_shape, result_shape)) {
    return InvalidArgument(
        "Shape used to set computation result layout %s is not compatible "
        "with result shape %s",
        ShapeUtil::HumanStringWithLayout(client_shape),
        ShapeUtil::HumanString(result_shape));
  }
  return Status::OK();
}
}  // namespace

StatusOr<std::unique_ptr<HloModuleConfig>> CreateModuleConfig(
    const ProgramShape& program_shape,
    absl::Span<const Shape* const> argument_shapes,
    const ExecutionOptions* execution_options, int default_num_replicas,
    absl::optional<int> num_threads, const AotCompilationOptions* aot_options) {
  auto config = absl::make_unique<HloModuleConfig>(program_shape);
  ComputationLayout* computation_layout =
      config->mutable_entry_computation_layout();
  const int64_t argument_shapes_size = argument_shapes.size();
  if (program_shape.parameters_size() != argument_shapes_size) {
    return InvalidArgument("computation takes %d parameters, but %u given",
                           program_shape.parameters_size(),
                           argument_shapes.size());
  }
  for (int i = 0, end = argument_shapes.size(); i < end; ++i) {
    // Verify that shape of arguments matches the shape of the arguments in the
    // ProgramShape.
    if (!ShapeUtil::Compatible(*argument_shapes[i],
                               program_shape.parameters(i))) {
      return InvalidArgument(
          "Argument does not match shape of computation parameter %d: want "
          "%s, got %s",
          i, ShapeUtil::HumanString(program_shape.parameters(i)),
          ShapeUtil::HumanString(*argument_shapes[i]));
    }
    TF_RETURN_IF_ERROR(
        computation_layout->mutable_parameter_layout(i)->CopyLayoutFromShape(
            *argument_shapes[i]));
  }
  if (execution_options != nullptr &&
      execution_options->has_shape_with_output_layout()) {
    const Shape shape_with_output_layout(
        execution_options->shape_with_output_layout());
    TF_RETURN_IF_ERROR(
        ValidateResultShape(shape_with_output_layout, program_shape.result()));
    TF_RETURN_IF_ERROR(
        computation_layout->mutable_result_layout()->CopyLayoutFromShape(
            shape_with_output_layout));
  } else {
    // If the result layout is not set, then choose the default.
    computation_layout->mutable_result_layout()->SetToDefaultLayout();
  }

  if (execution_options != nullptr) {
    if (execution_options->num_replicas() > 0) {
      config->set_replica_count(execution_options->num_replicas());
    } else {
      config->set_replica_count(default_num_replicas);
    }
    if (execution_options->num_partitions() > 0) {
      config->set_num_partitions(execution_options->num_partitions());
    }
    config->set_use_spmd_partitioning(
        execution_options->use_spmd_partitioning());
    config->set_use_auto_spmd_partitioning(
        execution_options->use_auto_spmd_partitioning());
    config->set_deduplicate_hlo(execution_options->deduplicate_hlo());
    config->set_seed(execution_options->seed());
    config->set_launch_id(execution_options->launch_id());
    config->set_debug_options(execution_options->debug_options());
  } else {
    config->set_replica_count(default_num_replicas);
    config->set_debug_options(GetDebugOptionsFromFlags());
  }

  if (num_threads.has_value()) {
    config->set_intra_op_parallelism_threads(*num_threads);
  }

  if (execution_options != nullptr &&
      execution_options->has_device_assignment()) {
    TF_ASSIGN_OR_RETURN(
        auto device_assignment,
        DeviceAssignment::Deserialize(execution_options->device_assignment()));
    config->set_static_device_assignment(*device_assignment);
  }
  config->set_alias_passthrough_params(
      execution_options->alias_passthrough_params());

  if (aot_options != nullptr &&
      aot_options->fusion_config_collection() != FusionConfigCollection::kOff) {
    config->set_fusion_config_collection(
        aot_options->fusion_config_collection());
    *config->mutable_fusion_config() = aot_options->fusion_config();
  }

  return std::move(config);
}

}  // namespace xla
