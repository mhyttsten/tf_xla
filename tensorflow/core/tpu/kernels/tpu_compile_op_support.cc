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
class MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_compile_op_supportDTcc {
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
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_compile_op_supportDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_compile_op_supportDTcc() {
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
#include "tensorflow/core/tpu/kernels/tpu_compile_op_support.h"

#include <string>

#include "tensorflow/compiler/xla/debug_options_flags.h"
#include "tensorflow/compiler/xla/service/computation_layout.h"
#include "tensorflow/compiler/xla/service/computation_placer.h"
#include "tensorflow/compiler/xla/service/dump.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache_key.h"
#include "tensorflow/core/tpu/kernels/tpu_executable_info.pb.h"
#include "tensorflow/stream_executor/tpu/proto_helper.h"

namespace tensorflow {
namespace tpu {
using ::stream_executor::port::Status;
using ::stream_executor::port::StatusOr;
using ::xla::ComputationLayout;
using ::xla::DebugOptions;
using ::xla::DeviceAssignment;
using ::xla::HloModuleConfig;
using ::xla::HloSharding;
using ::xla::InvalidArgument;
using ::xla::ProgramShape;
using ::xla::Shape;
using ::xla::ShapeTree;
using ::xla::ShapeUtil;

Status ValidateResultShape(const Shape& client_shape,
                           const Shape& result_shape) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_compile_op_supportDTcc mht_0(mht_0_v, 214, "", "./tensorflow/core/tpu/kernels/tpu_compile_op_support.cc", "ValidateResultShape");

  TF_RETURN_IF_ERROR(
      xla::ShapeUtil::ValidateShapeWithOptionalLayout(client_shape));
  if (!xla::ShapeUtil::Compatible(client_shape, result_shape)) {
    return InvalidArgument(
        "Shape used to set computation result layout %s is not compatible "
        "with result shape %s",
        xla::ShapeUtil::HumanStringWithLayout(client_shape),
        xla::ShapeUtil::HumanString(result_shape));
  }
  return Status::OK();
}

StatusOr<std::unique_ptr<HloModuleConfig>> CreateModuleConfig(
    const ProgramShape& program_shape, absl::Span<const Shape> argument_shapes,
    absl::optional<const Shape> result_layout,
    absl::optional<const DeviceAssignment> device_assignment, int replica_count,
    int num_partitions, const DebugOptions* debug_options, const int* seed,
    const int* launch_id, const bool* alias_passthrough_params,
    const xla::FusionConfigCollection* fusion_config_collection,
    const std::vector<std::vector<bool>>* fusion_config) {
  auto config = absl::make_unique<HloModuleConfig>(program_shape);
  ComputationLayout* computation_layout =
      config->mutable_entry_computation_layout();
  if (program_shape.parameters_size() != argument_shapes.size()) {
    return InvalidArgument("computation takes %d parameters, but %u given",
                           program_shape.parameters_size(),
                           argument_shapes.size());
  }
  for (int i = 0; i < argument_shapes.size(); ++i) {
    // Verify that shape of arguments matches the shape of the arguments in the
    // ProgramShape.
    if (!ShapeUtil::Compatible(argument_shapes[i],
                               program_shape.parameters(i))) {
      return InvalidArgument(
          "Argument does not match shape of computation parameter %d: want "
          "%s, got %s",
          i, ShapeUtil::HumanString(program_shape.parameters(i)),
          ShapeUtil::HumanString(argument_shapes[i]));
    }
    TF_RETURN_IF_ERROR(
        computation_layout->mutable_parameter_layout(i)->CopyLayoutFromShape(
            argument_shapes[i]));
  }

  if (result_layout.has_value()) {
    TF_RETURN_IF_ERROR(
        ValidateResultShape(result_layout.value(), program_shape.result()));
    TF_RETURN_IF_ERROR(
        computation_layout->mutable_result_layout()->CopyLayoutFromShape(
            result_layout.value()));
  } else {
    // If the result layout is not set, then choose the default.
    computation_layout->mutable_result_layout()->SetToDefaultLayout();
  }

  config->set_replica_count(replica_count);
  config->set_num_partitions(num_partitions);
  if (seed != nullptr) {
    config->set_seed(*seed);
  }
  if (launch_id != nullptr) {
    config->set_launch_id(*launch_id);
  }
  if (debug_options != nullptr) {
    config->set_debug_options(*debug_options);
  } else {
    config->set_debug_options(xla::GetDebugOptionsFromFlags());
  }

  // TODO(henrytan): set intra_op_parallelism_threads.
  // Reference:
  // tensorflow/compiler/xla/service/service.cc?l=324.

  if (device_assignment.has_value()) {
    config->set_static_device_assignment(device_assignment.value());
  }

  if (alias_passthrough_params != nullptr) {
    config->set_alias_passthrough_params(*alias_passthrough_params);
  }

  if (fusion_config_collection != nullptr && fusion_config != nullptr &&
      *fusion_config_collection != xla::FusionConfigCollection::kOff) {
    config->set_fusion_config_collection(*fusion_config_collection);
    *config->mutable_fusion_config() = *fusion_config;
  }

  return std::move(config);
}

StatusOr<std::unique_ptr<xla::HloModuleConfig>> CreateModuleConfig(
    const xla::ProgramShape& program_shape,
    absl::Span<const Shape> argument_shapes,
    absl::optional<const Shape> result_layout,
    absl::optional<const DeviceAssignment> device_assignment, int replica_count,
    int num_partitions, const DebugOptions* debug_options) {
  return CreateModuleConfig(program_shape, argument_shapes, result_layout,
                            device_assignment, replica_count, num_partitions,
                            debug_options, /*seed=*/nullptr,
                            /*launch_id=*/nullptr,
                            /*alias_passthrough_params=*/nullptr,
                            /*fusion_config_collection=*/nullptr,
                            /*fusion_config=*/nullptr);
}

ShapeTree<HloSharding> GetSubtree(
    const ShapeTree<HloSharding>& tuple_shape_tree, int element_index) {
  ShapeTree<HloSharding> element_shape_tree(
      xla::ShapeUtil::GetTupleElementShape(tuple_shape_tree.shape(),
                                           element_index),
      HloSharding::Replicate());

  xla::ShapeIndex src_index;
  src_index.push_back(element_index);
  element_shape_tree.CopySubtreeFrom(tuple_shape_tree, src_index, {});
  return element_shape_tree;
}

Shape GetPerDeviceShape(const Shape& shape, const HloSharding& sharding,
                        int64_t device) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_compile_op_supportDTcc mht_1(mht_1_v, 337, "", "./tensorflow/core/tpu/kernels/tpu_compile_op_support.cc", "GetPerDeviceShape");

  if (shape.IsTuple()) {
    ShapeTree<HloSharding> tuple_shape_tree = sharding.GetAsShapeTree(shape);
    std::vector<Shape> arg_shapes;
    for (int64_t i = 0; i < xla::ShapeUtil::TupleElementCount(shape); ++i) {
      Shape element_shape = xla::ShapeUtil::GetTupleElementShape(shape, i);
      HloSharding element_sharding = tuple_shape_tree.element({i});
      if (element_shape.IsTuple()) {
        element_sharding = HloSharding::Tuple(GetSubtree(tuple_shape_tree, i));
      }
      if (element_sharding.UsesDevice(device)) {
        arg_shapes.push_back(
            GetPerDeviceShape(element_shape, element_sharding, device));
      }
    }
    return xla::ShapeUtil::MakeTupleShape(arg_shapes);
  }

  if (sharding.IsTileMaximal()) {
    return shape;
  }

  std::vector<int64_t> dimensions;
  std::vector<int64_t> offset = sharding.TileOffsetForDevice(shape, device);
  std::vector<int64_t> limit = sharding.TileLimitForDevice(shape, device);
  dimensions.resize(limit.size());
  for (int64_t i = 0; i < limit.size(); ++i) {
    dimensions[i] = limit[i] - offset[i];
  }
  if (shape.has_layout()) {
    return xla::ShapeUtil::MakeShapeWithLayout(shape.element_type(), dimensions,
                                               shape.layout().minor_to_major());
  }
  return xla::ShapeUtil::MakeShape(shape.element_type(), dimensions);
}

Status AddVariableUpdatesToCores(
    const TPUCompileMetadataProto& metadata,
    const XlaCompiler::CompilationResult& compilation_result,
    const std::vector<ShardingAndIndex>& arg_core_mapping,
    std::vector<bool>* may_modify_variables,
    std::vector<std::vector<xla::Shape>>* per_core_output_shapes,
    std::vector<std::vector<std::pair<int, bool>>>* per_core_variable_indices) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_compile_op_supportDTcc mht_2(mht_2_v, 382, "", "./tensorflow/core/tpu/kernels/tpu_compile_op_support.cc", "AddVariableUpdatesToCores");

  // Add all variables to the corresponding core.
  may_modify_variables->resize(metadata.num_cores_per_replica(), false);
  int resource_update_pos = 0;
  for (int i = 0; i < metadata.args_size(); ++i) {
    const tpu::TPUCompileMetadataProto::Arg& proto_arg = metadata.args(i);
    if (proto_arg.kind() == tpu::TPUCompileMetadataProto::Arg::VARIABLE) {
      const auto& sharding = proto_arg.sharding();
      bool updated = false;
      if (resource_update_pos < compilation_result.resource_updates.size()) {
        const XlaCompiler::ResourceUpdate& update =
            compilation_result.resource_updates[resource_update_pos];
        if (update.input_index == i) {
          updated = true;
          int pos = compilation_result.outputs.size() + resource_update_pos;
          xla::Shape shape = xla::ShapeUtil::GetTupleElementShape(
              compilation_result.xla_output_shape, pos);
          auto add_to_core = [&](int64_t core,
                                 const xla::Shape& per_core_shape) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_compile_op_supportDTcc mht_3(mht_3_v, 403, "", "./tensorflow/core/tpu/kernels/tpu_compile_op_support.cc", "lambda");

            (*per_core_output_shapes)[core].push_back(per_core_shape);
            (*may_modify_variables)[core] =
                (*may_modify_variables)[core] || update.modified;
          };
          if (sharding.type() == xla::OpSharding::MAXIMAL) {
            add_to_core(sharding.tile_assignment_devices(0), shape);
          } else if (sharding.type() == xla::OpSharding::OTHER) {
            auto sharding_or =
                xla::HloSharding::FromProto(proto_arg.sharding());
            TF_RET_CHECK(sharding_or.ok());
            for (int64_t core :
                 proto_arg.sharding().tile_assignment_devices()) {
              xla::Shape per_core_shape =
                  GetPerDeviceShape(shape, sharding_or.ValueOrDie(), core);
              add_to_core(core, per_core_shape);
            }
          } else {
            TF_RET_CHECK(sharding.type() == xla::OpSharding::REPLICATED);
            for (int64_t core = 0; core < metadata.num_cores_per_replica();
                 ++core) {
              add_to_core(core, shape);
            }
          }
          ++resource_update_pos;
        }
      }
      if (sharding.type() == xla::OpSharding::MAXIMAL) {
        (*per_core_variable_indices)[sharding.tile_assignment_devices(0)]
            .push_back(
                std::pair<int, bool>(arg_core_mapping[i].indices[0], updated));
      } else if (sharding.type() == xla::OpSharding::OTHER) {
        for (int core : sharding.tile_assignment_devices()) {
          (*per_core_variable_indices)[core].push_back(
              std::pair<int, bool>(arg_core_mapping[i].indices[core], updated));
        }
      } else {
        TF_RET_CHECK(sharding.type() == xla::OpSharding::REPLICATED);
        for (int64_t core = 0; core < metadata.num_cores_per_replica();
             ++core) {
          (*per_core_variable_indices)[core].push_back(
              std::pair<int, bool>(arg_core_mapping[i].indices[core], updated));
        }
      }
    }
  }
  return Status::OK();
}

Status ComputeOutputShapesForEachCore(
    const tpu::TPUCompileMetadataProto& metadata,
    const XlaCompiler::CompilationResult& compilation_result,
    std::vector<std::vector<xla::Shape>>* per_core_output_shapes) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_compile_op_supportDTcc mht_4(mht_4_v, 458, "", "./tensorflow/core/tpu/kernels/tpu_compile_op_support.cc", "ComputeOutputShapesForEachCore");

  for (int i = 0; i < metadata.retvals_size(); ++i) {
    const tpu::TPUCompileMetadataProto::Retval& retval = metadata.retvals(i);
    TF_RET_CHECK(!compilation_result.outputs[i].is_constant)
        << "TPU compilation output " << i
        << " has a compile-time constant value. "
           "This should never happen.";

    xla::Shape shape = xla::ShapeUtil::GetTupleElementShape(
        compilation_result.xla_output_shape, i);
    auto add_shape_to_core = [&](int core, xla::Shape per_core_shape) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_compile_op_supportDTcc mht_5(mht_5_v, 471, "", "./tensorflow/core/tpu/kernels/tpu_compile_op_support.cc", "lambda");

      (*per_core_output_shapes)[core].push_back(std::move(per_core_shape));
    };
    if (retval.sharding().type() == xla::OpSharding::MAXIMAL) {
      add_shape_to_core(retval.sharding().tile_assignment_devices(0),
                        std::move(shape));
    } else if (retval.sharding().type() == xla::OpSharding::OTHER) {
      auto sharding_or = xla::HloSharding::FromProto(retval.sharding());
      TF_RET_CHECK(sharding_or.ok());
      for (int64_t core : retval.sharding().tile_assignment_devices()) {
        xla::Shape per_core_shape =
            GetPerDeviceShape(shape, sharding_or.ValueOrDie(), core);
        add_shape_to_core(core, std::move(per_core_shape));
      }
    } else {
      TF_RET_CHECK(retval.sharding().type() == xla::OpSharding::REPLICATED)
          << "Not all of the constant tensors were consumed.";
      for (int core = 0; core < per_core_output_shapes->size(); ++core) {
        add_shape_to_core(core, shape);
      }
    }
  }
  return Status::OK();
}

Status CreateHloModules(
    const TPUCompileMetadataProto& metadata,
    const tensorflow::XlaCompiler::CompilationResult& compilation_result,
    const absl::optional<xla::DeviceAssignment>& device_assignment,
    std::vector<std::unique_ptr<xla::HloModule>>* hlo_modules) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_compile_op_supportDTcc mht_6(mht_6_v, 503, "", "./tensorflow/core/tpu/kernels/tpu_compile_op_support.cc", "CreateHloModules");

  TF_RET_CHECK(
      compilation_result.computation->proto().has_host_program_shape());

  auto debug_options = xla::DebugOptions();
  debug_options.set_xla_step_marker_location(metadata.step_marker_location());
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<xla::HloModuleConfig> module_config,
      CreateModuleConfig(
          xla::ProgramShape(
              compilation_result.computation->proto().host_program_shape()),
          compilation_result.xla_input_shapes,
          compilation_result.xla_output_shape, device_assignment,
          metadata.num_replicas(), metadata.num_cores_per_replica(),
          &debug_options));

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<xla::HloModule> hlo_module,
      xla::HloModule::CreateFromProto(compilation_result.computation->proto(),
                                      *module_config));
  DumpHloModuleIfEnabled(*hlo_module, "before_optimizations");
  hlo_modules->push_back(std::move(hlo_module));

  return Status::OK();
}

StatusOr<TpuCompilationRequestProto> CreateTpuCompilationRequest(
    const absl::variant<MlirToHloArgs, FunctionToHloArgs>& computation,
    const TPUCompileMetadataProto& metadata,
    const std::vector<TensorShape>& arg_shapes) {
  VLOG(1) << "CreateTpuCompilationRequest.";
  TpuCompilationRequestProto compilation_request;
  bool use_mlir = computation.index() == 0;
  compilation_request.set_use_mlir(use_mlir);
  if (use_mlir) {
    VLOG(1) << "Serializing MlirModule";
    const MlirToHloArgs& mlir_computation = absl::get<0>(computation);
    *compilation_request.mutable_mlir_module() =
        string(mlir_computation.mlir_module);
  } else {
    VLOG(1) << "Serializing FunctionDefinitionLibrary";
    const FunctionToHloArgs& function_computation = absl::get<1>(computation);
    *compilation_request.mutable_fdef_lib() =
        function_computation.flib_def->ToProto();
    compilation_request.set_graph_def_version(
        function_computation.graph_def_version);
    *compilation_request.mutable_function() = *function_computation.function;
    // TODO(b/160937500): serializing and copying large guaranteed_constants can
    // be a perf hit. There is a future work to refactor the compilation layer
    // to avoid passing guaranteed_constants over C_API.
    if (function_computation.guaranteed_constants.index() == 0) {
      absl::Span<const TensorProto* const> guaranteed_constants =
          absl::get<0>(function_computation.guaranteed_constants);
      for (const TensorProto* constant : guaranteed_constants) {
        *compilation_request.add_guaranteed_constants() = *constant;
      }
    } else {
      CHECK_EQ(function_computation.guaranteed_constants.index(), 1);
      const OpInputList& guaranteed_constants =
          *absl::get<1>(function_computation.guaranteed_constants);
      for (const Tensor& constant : guaranteed_constants) {
        constant.AsProtoTensorContent(
            compilation_request.add_guaranteed_constants());
      }
    }
  }

  for (const TensorShape& shape : arg_shapes) {
    shape.AsProto(compilation_request.add_arg_shapes());
  }

  *(compilation_request.mutable_metadata()) = metadata;

  VLOG(1) << "TpuCompilationRequest:\n" << compilation_request.DebugString();
  return compilation_request;
}

Status CompileOpMetadataFromContext(OpKernelConstruction* ctx,
                                    TPUCompileMetadataProto* metadata,
                                    NameAttrList* function_name,
                                    std::string* mlir_module) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_compile_op_supportDTcc mht_7(mht_7_v, 586, "", "./tensorflow/core/tpu/kernels/tpu_compile_op_support.cc", "CompileOpMetadataFromContext");

  CHECK_NE(metadata, nullptr);

  int num_computations;
  TF_RETURN_IF_ERROR(ctx->GetAttr("num_computations", &num_computations));

  std::string metadata_string;
  TF_RETURN_IF_ERROR(ctx->GetAttr("metadata", &metadata_string));
  if (!metadata->ParsePartialFromString(metadata_string)) {
    return errors::InvalidArgument("Unable to parse TPUCompileMetadataProto");
  }

  if (function_name != nullptr) {
    TF_RETURN_IF_ERROR(ctx->GetAttr("function", function_name));
  }

  if (mlir_module != nullptr) {
    TF_RETURN_IF_ERROR(ctx->GetAttr("mlir_module", mlir_module));
  }

  if (num_computations != metadata->num_cores_per_replica()) {
    return errors::InvalidArgument(
        "num_computations must be equal to "
        "num_cores_per_replica in the 'metadata' "
        "attribute (",
        num_computations, " vs ", metadata->num_cores_per_replica(), ")");
  }

  if (metadata->has_device_assignment()) {
    StatusOr<std::unique_ptr<DeviceAssignment>> device_assignment_or_error =
        DeviceAssignment::Deserialize(metadata->device_assignment());
    TF_RETURN_IF_ERROR(device_assignment_or_error.status());
    const DeviceAssignment& device_assignment =
        *device_assignment_or_error.ValueOrDie();
    const int num_replicas = metadata->num_replicas();
    if (device_assignment.replica_count() != num_replicas) {
      return errors::InvalidArgument(
          "Device assignment replica_count != num_replicas; ",
          device_assignment.replica_count(), " vs ", num_replicas);
    }
    if (device_assignment.computation_count() !=
        metadata->num_cores_per_replica()) {
      return errors::InvalidArgument(
          "Device assignment computation_count != num_cores_per_replica; ",
          device_assignment.computation_count(), " vs ",
          metadata->num_cores_per_replica());
    }
  }
  return Status::OK();
}

Status ComputeArgumentShapes(const tpu::TPUCompileMetadataProto& metadata,
                             const std::vector<TensorShape>& dynamic_shapes,
                             std::vector<TensorShape>* arg_shapes) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_compile_op_supportDTcc mht_8(mht_8_v, 642, "", "./tensorflow/core/tpu/kernels/tpu_compile_op_support.cc", "ComputeArgumentShapes");

  arg_shapes->resize(metadata.args_size());
  int dynamic_shape_pos = 0;
  for (int i = 0; i < metadata.args_size(); ++i) {
    const tpu::TPUCompileMetadataProto::Arg& arg = metadata.args(i);
    // The XLA compiler determines the shape of each constant by inspecting the
    // value of its corresponding host-memory tensor. As a result, we don't need
    // to give the compiler graph-inferred shapes for constant arguments.
    if (arg.kind() == tpu::TPUCompileMetadataProto::Arg::GUARANTEED_CONSTANT) {
      continue;
    }
    TF_RETURN_IF_ERROR(PartialTensorShape::IsValidShape(arg.shape()));
    PartialTensorShape static_shape(arg.shape());

    TensorShape& shape = (*arg_shapes)[i];
    if (static_shape.IsFullyDefined()) {
      TF_RET_CHECK(static_shape.AsTensorShape(&shape));
    } else {
      TF_RET_CHECK(dynamic_shape_pos < dynamic_shapes.size())
          << "Too few dynamic shapes";
      shape = dynamic_shapes[dynamic_shape_pos++];
      if (!static_shape.IsCompatibleWith(shape)) {
        return errors::InvalidArgument(
            "Mismatch between static and dynamic shape for argument. Static "
            "shape: ",
            static_shape.DebugString(),
            "; dynamic shape: ", shape.DebugString());
      }
    }
  }
  // Checks we consumed all of the dynamic shapes.
  TF_RET_CHECK(dynamic_shape_pos == dynamic_shapes.size())
      << "Too many dynamic shapes";
  return Status::OK();
}
}  // namespace tpu
}  // namespace tensorflow
