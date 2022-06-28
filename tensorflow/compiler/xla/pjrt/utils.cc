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
class MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSutilsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSutilsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSutilsDTcc() {
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

#include "tensorflow/compiler/xla/pjrt/utils.h"

#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/xla/client/executable_build_options.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_sharding.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {

namespace {
StatusOr<Shape> GetShardedShape(const Shape& shape,
                                const OpSharding& sharding) {
  if (sharding.type() == OpSharding::TUPLE) {
    if (!shape.IsTuple()) {
      return InvalidArgument(
          "Got tuple OpSharding (%s) for non-tuple shape (%s)",
          sharding.DebugString(), shape.ToString());
    }
    if (sharding.tuple_shardings_size() != shape.tuple_shapes_size()) {
      return InvalidArgument(
          "Got mismatched OpSharding tuple size (%d) and shape tuple size (%d)."
          " (OpSharding: %s, shape: %s)",
          sharding.tuple_shardings_size(), shape.tuple_shapes_size(),
          sharding.DebugString(), shape.ToString());
    }
    std::vector<Shape> sharded_subshapes;
    const int tuple_shapes_size = shape.tuple_shapes_size();
    sharded_subshapes.reserve(tuple_shapes_size);
    for (int i = 0; i < tuple_shapes_size; ++i) {
      TF_ASSIGN_OR_RETURN(
          Shape sharded_subshape,
          GetShardedShape(shape.tuple_shapes(i), sharding.tuple_shardings(i)));
      sharded_subshapes.emplace_back(std::move(sharded_subshape));
    }
    return ShapeUtil::MakeTupleShape(sharded_subshapes);
  }
  TF_ASSIGN_OR_RETURN(HloSharding hlo_sharding,
                      HloSharding::FromProto(sharding));
  return hlo_sharding.TileShape(shape);
}

StatusOr<Shape> GetShardedShape(const HloInstructionProto& instr) {
  const Shape unsharded_shape(instr.shape());
  Shape sharded_shape;
  if (instr.has_sharding()) {
    TF_ASSIGN_OR_RETURN(sharded_shape,
                        GetShardedShape(unsharded_shape, instr.sharding()));
  } else {
    sharded_shape = unsharded_shape;
  }
  LayoutUtil::ClearLayout(&sharded_shape);
  return sharded_shape;
}

// Returns sharded (argument shapes, result shape) without layouts.
StatusOr<std::pair<std::vector<Shape>, Shape>> GetShardedProgramShapes(
    const XlaComputation& computation, const ProgramShape& program_shape) {
  std::vector<Shape> arg_shapes;
  arg_shapes.resize(program_shape.parameters_size());
  Shape result_shape;
  for (const HloComputationProto& comp : computation.proto().computations()) {
    if (comp.id() != computation.proto().entry_computation_id()) {
      continue;
    }
    for (const HloInstructionProto& instr : comp.instructions()) {
      if (instr.opcode() == HloOpcodeString(HloOpcode::kParameter)) {
        if (instr.parameter_number() >= program_shape.parameters_size()) {
          return InvalidArgument(
              "Got invalid parameter number %d, expected %d parameters",
              instr.parameter_number(), program_shape.parameters_size());
        }
        TF_ASSIGN_OR_RETURN(arg_shapes[instr.parameter_number()],
                            GetShardedShape(instr));
      }
      if (instr.id() == comp.root_id()) {
        if (result_shape.element_type() != PRIMITIVE_TYPE_INVALID) {
          return InvalidArgument("Found multiple root instructions");
        }
        TF_ASSIGN_OR_RETURN(result_shape, GetShardedShape(instr));
      }
    }
  }
  for (int i = 0; i < arg_shapes.size(); ++i) {
    if (arg_shapes[i].element_type() == PRIMITIVE_TYPE_INVALID) {
      return InvalidArgument("Couldn't find parameter %d", i);
    }
  }
  if (result_shape.element_type() == PRIMITIVE_TYPE_INVALID) {
    return InvalidArgument("Couldn't find root instruction");
  }
  return std::make_pair(arg_shapes, result_shape);
}
}  // namespace

Status ParseDeviceAssignmentCompileOptions(
    bool compile_portable_executable, ExecutableBuildOptions* build_options,
    std::function<StatusOr<DeviceAssignment>(int, int)>
        GetDefaultDeviceAssignmentFunction,
    int* num_replicas, int* num_partitions,
    std::shared_ptr<DeviceAssignment>* device_assignment) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSutilsDTcc mht_0(mht_0_v, 289, "", "./tensorflow/compiler/xla/pjrt/utils.cc", "ParseDeviceAssignmentCompileOptions");

  if (compile_portable_executable) {
    if (build_options->has_device_assignment()) {
      return InvalidArgument(
          "CompileOptions requests portable executable but "
          "ExecutableBuildOptions includes a device assignment");
    }
    *num_replicas = 1;
    *num_partitions = 1;
  } else {
    if (!build_options->has_device_assignment()) {
      VLOG(2) << "Compile using default device_assignment.";
      TF_ASSIGN_OR_RETURN(
          DeviceAssignment device_assignment,
          GetDefaultDeviceAssignmentFunction(build_options->num_replicas(),
                                             build_options->num_partitions()));
      build_options->set_device_assignment(device_assignment);
    }
    VLOG(2) << "Compile device_assignment:\n"
            << build_options->device_assignment().ToString();
    *num_replicas = build_options->device_assignment().replica_count();
    *num_partitions = build_options->device_assignment().computation_count();
    *device_assignment =
        std::make_shared<DeviceAssignment>(build_options->device_assignment());
  }
  return Status::OK();
}

Status DetermineArgumentLayoutsFromCompileOptions(
    const XlaComputation& computation,
    std::function<StatusOr<Shape>(Shape)>
        choose_compact_layout_for_shape_function,
    absl::optional<std::vector<Shape>>& argument_layouts,
    ExecutableBuildOptions* build_options,
    std::vector<const Shape*>* argument_layout_pointers) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSutilsDTcc mht_1(mht_1_v, 326, "", "./tensorflow/compiler/xla/pjrt/utils.cc", "DetermineArgumentLayoutsFromCompileOptions");

  TF_ASSIGN_OR_RETURN(ProgramShape program_shape,
                      computation.GetProgramShape());
  if (!argument_layouts) {
    argument_layouts.emplace(program_shape.parameters());
    for (Shape& shape : *argument_layouts) {
      LayoutUtil::ClearLayout(&shape);
    }
  } else if (argument_layouts->size() != program_shape.parameters_size()) {
    return InvalidArgument(
        "CompileOptions specify %d argument layouts, but computation has %d "
        "arguments",
        argument_layouts->size(), program_shape.parameters_size());
  }
  argument_layout_pointers->reserve(argument_layouts->size());

  // Assign a default layout based on `sharded_shape` to any array subshapes in
  // `dst_shape` that are missing layouts.
  auto assign_layouts = [&choose_compact_layout_for_shape_function](
                            const Shape& sharded_shape, Shape* dst_shape) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSutilsDTcc mht_2(mht_2_v, 348, "", "./tensorflow/compiler/xla/pjrt/utils.cc", "lambda");

    return ShapeUtil::ForEachMutableSubshapeWithStatus(
        dst_shape, [&](Shape* subshape, const ShapeIndex& idx) {
          if (subshape->IsArray() && !subshape->has_layout()) {
            CHECK(ShapeUtil::IndexIsValid(sharded_shape, idx));
            const Shape& sharded_subshape =
                ShapeUtil::GetSubshape(sharded_shape, idx);
            LayoutUtil::SetToDefaultLayout(subshape);
            TF_ASSIGN_OR_RETURN(
                Shape layout,
                choose_compact_layout_for_shape_function(sharded_subshape));
            *subshape->mutable_layout() = layout.layout();
          }
          return Status::OK();
        });
  };
  TF_ASSIGN_OR_RETURN(auto sharded_shapes,
                      GetShardedProgramShapes(computation, program_shape));

  CHECK_EQ(sharded_shapes.first.size(), argument_layouts->size());
  for (int i = 0; i < argument_layouts->size(); ++i) {
    Shape* layout = &(*argument_layouts)[i];
    argument_layout_pointers->push_back(layout);
    TF_RETURN_IF_ERROR(assign_layouts(sharded_shapes.first[i], layout));
  }

  Shape result_layout;
  if (build_options->result_layout()) {
    result_layout = *build_options->result_layout();
  } else {
    result_layout = program_shape.result();
    LayoutUtil::ClearLayout(&result_layout);
  }
  TF_RETURN_IF_ERROR(assign_layouts(sharded_shapes.second, &result_layout));
  build_options->set_result_layout(result_layout);
  return Status::OK();
}

StatusOr<std::vector<int>> ComputeParametersThatMustBeDonated(
    const HloModule& module, bool tuple_inputs) {
  HloComputation* computation = module.entry_computation();
  int number_of_parameters = [&]() -> int {
    if (tuple_inputs) {
      CHECK_EQ(computation->num_parameters(), 1);
      const Shape& input_tuple_shape =
          computation->parameter_instruction(0)->shape();
      CHECK(input_tuple_shape.IsTuple());
      return input_tuple_shape.tuple_shapes_size();
    } else {
      return computation->num_parameters();
    }
  }();
  // If any buffer in a parameter is aliased we will donate the entire input
  // parameter.
  std::vector<int> parameters_to_donate;
  parameters_to_donate.reserve(computation->num_parameters());
  const HloInputOutputAliasConfig& config = module.input_output_alias_config();
  TF_RETURN_IF_ERROR(config.ForEachAliasWithStatus(
      [&](const ShapeIndex& output_index,
          const HloInputOutputAliasConfig::Alias& alias) {
        if (tuple_inputs) {
          if (alias.parameter_number != 0) {
            return InvalidArgument(
                "Unexpected parameter number %d in alias config with tupled "
                "inputs",
                alias.parameter_number);
          }
          const ShapeIndex& index = alias.parameter_index;
          if (!index.empty()) {
            int this_parameter = index.data()[0];
            if (this_parameter >= number_of_parameters) {
              return InvalidArgument(
                  "Unexpected parameter index %s in alias config with tupled "
                  "inputs and %d parameters",
                  index.ToString(), number_of_parameters);
            }
            parameters_to_donate.push_back(this_parameter);
          }
        } else {
          int this_parameter = alias.parameter_number;
          if (this_parameter >= number_of_parameters) {
            return InvalidArgument(
                "Unexpected parameter number %d in alias config without tupled "
                "inputs and %d parameters",
                this_parameter, number_of_parameters);
          }
          parameters_to_donate.push_back(this_parameter);
        }
        return Status::OK();
      }));
  absl::c_sort(parameters_to_donate);
  return parameters_to_donate;
}

int DefaultThreadPoolSize() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSutilsDTcc mht_3(mht_3_v, 445, "", "./tensorflow/compiler/xla/pjrt/utils.cc", "DefaultThreadPoolSize");

  // Google's CI system exposes an environment variable NPROC that describes
  // a CPU reservation for tests.
  // TODO(phawkins): expose a better thought-out set of knobs to control
  // parallelism.
  const char* nproc_str = std::getenv("NPROC");
  int nproc = 0;
  if (nproc_str && absl::SimpleAtoi(nproc_str, &nproc)) {
    return std::max(0, nproc);
  }
  return tensorflow::port::MaxParallelism();
}

bool HasMajorToMinorLayout(PrimitiveType type, absl::Span<int64_t const> dims,
                           absl::Span<int64_t const> byte_strides) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSutilsDTcc mht_4(mht_4_v, 462, "", "./tensorflow/compiler/xla/pjrt/utils.cc", "HasMajorToMinorLayout");

  CHECK_EQ(dims.size(), byte_strides.size());
  // If the array is size 0, the strides are irrelevant.
  if (absl::c_find(dims, 0) != dims.end()) {
    return true;
  }
  int64_t stride = primitive_util::ByteWidth(type);
  for (int i = static_cast<int>(dims.size()) - 1; i >= 0; --i) {
    // If a dimension is of size 1, its stride is irrelevant.
    if (dims[i] != 1) {
      if (byte_strides[i] != stride) {
        return false;
      }
      stride *= dims[i];
    }
  }
  return true;
}

}  // namespace xla
