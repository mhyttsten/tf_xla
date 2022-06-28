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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTcc() {
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

#include "tensorflow/compiler/xla/service/hlo_module.h"

#include <algorithm>
#include <iterator>
#include <set>
#include <sstream>
#include <string>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_schedule.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/fingerprint.h"
#include "tensorflow/core/platform/stacktrace.h"

namespace xla {

HloModule::HloModule(const std::string& name, HloModuleConfig config)
    : name_(NameUniquer::GetSanitizedName(name)),
      config_(std::move(config)),
      unique_id_(next_unique_module_id_++),
      metadata_(tensorflow::Env::Default()) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTcc mht_0(mht_0_v, 219, "", "./tensorflow/compiler/xla/service/hlo_module.cc", "HloModule::HloModule");

  metadata_.set_canonical_module_id(unique_id_);
}

Status HloModule::set_schedule(HloSchedule schedule) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTcc mht_1(mht_1_v, 226, "", "./tensorflow/compiler/xla/service/hlo_module.cc", "HloModule::set_schedule");

  TF_RET_CHECK(schedule.module() == this);
  TF_RETURN_IF_ERROR(schedule.Verify());
  schedule_ = std::move(schedule);
  return Status::OK();
}

void HloModule::ReplaceEntryComputation(HloComputation* entry_computation) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTcc mht_2(mht_2_v, 236, "", "./tensorflow/compiler/xla/service/hlo_module.cc", "HloModule::ReplaceEntryComputation");

  entry_computation_ = entry_computation;
  config_.SetDefaultComputationLayout(
      entry_computation_->ComputeProgramShape());
  input_output_alias_config_ = HloInputOutputAliasConfig(
      entry_computation_->root_instruction()->shape());
}

HloComputation* HloModule::AddComputationInternal(
    std::unique_ptr<HloComputation> computation, bool is_entry,
    bool uniquify_identifiers, bool preserve_entry_layouts) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTcc mht_3(mht_3_v, 249, "", "./tensorflow/compiler/xla/service/hlo_module.cc", "HloModule::AddComputationInternal");

  if (is_entry) {
    CHECK_EQ(nullptr, entry_computation_);
    entry_computation_ = computation.get();

    if (preserve_entry_layouts) {
      config_.SetComputationLayoutIfExists(
          entry_computation_->ComputeProgramShape());
    } else if (!config_.has_entry_computation_layout()) {
      // If the module configuration has no entry layout computation set, create
      // a default one based on the program shape.
      config_.SetDefaultComputationLayout(
          entry_computation_->ComputeProgramShape());
    }
    input_output_alias_config_ = HloInputOutputAliasConfig(
        entry_computation_->root_instruction()->shape());
  }

  if (uniquify_identifiers) {
    computation->UniquifyName(&computation_name_uniquer_);
    for (auto* instruction : computation->instructions()) {
      instruction->UniquifyName(&instruction_name_uniquer_);
    }

    // Pick unique IDs for each instruction.
    for (auto* instruction : computation->instructions()) {
      instruction->SetUniqueId(NewUniqueInstructionId());
    }
    // Set unique id to this computation.
    CHECK_NE(computation->root_instruction()->unique_id(), -1)
        << "Root has no valid id: " << computation->ToString();
    computation->SetUniqueId(computation->root_instruction()->unique_id());
  } else {
    // Don't uniquify the names of the computation or instruction, but we must
    // run the names through the uniquifiers to prevent future name collisions
    // for computations and instructions created later. Also, set the
    // next_unique_id_ to the one greater than the max unique id of any
    // instruction (or the computation) to avoid ID collisions.
    computation_name_uniquer_.GetUniqueName(computation->name());
    for (auto* instruction : computation->instructions()) {
      instruction_name_uniquer_.GetUniqueName(instruction->name());
      next_unique_id_ = std::max(next_unique_id_, instruction->unique_id() + 1);
    }
    if (next_unique_id_ < computation->unique_id() + 1) {
      next_unique_id_ = computation->unique_id() + 1;
    }
  }

  computation->set_parent(this);
  computations_.push_back(std::move(computation));
  return computations_.back().get();
}

HloComputation* HloModule::AddEntryComputation(
    std::unique_ptr<HloComputation> computation) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTcc mht_4(mht_4_v, 306, "", "./tensorflow/compiler/xla/service/hlo_module.cc", "HloModule::AddEntryComputation");

  return AddComputationInternal(std::move(computation), /*is_entry=*/true,
                                /*uniquify_identifiers=*/true,
                                /*preserve_entry_layouts=*/false);
}

HloComputation* HloModule::AddEntryComputationWithLayouts(
    std::unique_ptr<HloComputation> computation) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTcc mht_5(mht_5_v, 316, "", "./tensorflow/compiler/xla/service/hlo_module.cc", "HloModule::AddEntryComputationWithLayouts");

  return AddComputationInternal(std::move(computation), /*is_entry=*/true,
                                /*uniquify_identifiers=*/true,
                                /*preserve_entry_layouts=*/true);
}

Status HloModule::RemoveEmbeddedComputation(HloComputation* to_remove) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTcc mht_6(mht_6_v, 325, "", "./tensorflow/compiler/xla/service/hlo_module.cc", "HloModule::RemoveEmbeddedComputation");

  if (has_schedule() && !to_remove->IsCalledComputation()) {
    schedule_->remove_computation(to_remove);
  }

  auto it = absl::c_find_if(
      computations_, [&to_remove](const std::unique_ptr<HloComputation>& comp) {
        return comp.get() == to_remove;
      });
  TF_RET_CHECK(it != computations_.end());
  TF_RET_CHECK(it->get() == to_remove);
  computations_.erase(it);
  return Status::OK();
}

HloComputation* HloModule::AddEmbeddedComputation(
    std::unique_ptr<HloComputation> computation) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTcc mht_7(mht_7_v, 344, "", "./tensorflow/compiler/xla/service/hlo_module.cc", "HloModule::AddEmbeddedComputation");

  return AddComputationInternal(std::move(computation), /*is_entry=*/false,
                                /*uniquify_identifiers=*/true,
                                /*preserve_entry_layouts=*/false);
}

void HloModule::ReplaceComputations(
    const absl::flat_hash_map<HloComputation*, HloComputation*>& replacements) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTcc mht_8(mht_8_v, 354, "", "./tensorflow/compiler/xla/service/hlo_module.cc", "HloModule::ReplaceComputations");

  // Replace all uses of non-canonical computations with their
  // representatives.
  std::vector<std::unique_ptr<HloComputation>> new_computations;
  new_computations.reserve(computations_.size());

  for (std::unique_ptr<HloComputation>& computation : computations_) {
    for (auto* instruction : computation->instructions()) {
      switch (instruction->opcode()) {
        case HloOpcode::kAllReduce:
        case HloOpcode::kCall:
        case HloOpcode::kMap:
        case HloOpcode::kReduce:
        case HloOpcode::kReduceScatter:
        case HloOpcode::kReduceWindow:
        case HloOpcode::kScatter:
        case HloOpcode::kSort: {
          HloComputation* new_arg = tensorflow::gtl::FindWithDefault(
              replacements, instruction->to_apply(), nullptr);
          if (new_arg != nullptr) {
            instruction->set_to_apply(new_arg);
          }
          break;
        }
        case HloOpcode::kWhile: {
          HloComputation* new_condition = tensorflow::gtl::FindWithDefault(
              replacements, instruction->while_condition(), nullptr);
          if (new_condition != nullptr) {
            instruction->set_while_condition(new_condition);
          }
          HloComputation* new_body = tensorflow::gtl::FindWithDefault(
              replacements, instruction->while_body(), nullptr);
          if (new_body != nullptr) {
            instruction->set_while_body(new_body);
          }
          break;
        }
        case HloOpcode::kConditional: {
          for (int b = 0; b < instruction->branch_count(); ++b) {
            HloComputation* new_computation = tensorflow::gtl::FindWithDefault(
                replacements, instruction->branch_computation(b), nullptr);
            if (new_computation != nullptr) {
              instruction->set_branch_computation(b, new_computation);
            }
          }
          break;
        }
        case HloOpcode::kSelectAndScatter: {
          HloComputation* new_select = tensorflow::gtl::FindWithDefault(
              replacements, instruction->select(), nullptr);
          if (new_select != nullptr) {
            instruction->set_select(new_select);
          }
          HloComputation* new_scatter = tensorflow::gtl::FindWithDefault(
              replacements, instruction->scatter(), nullptr);
          if (new_scatter != nullptr) {
            instruction->set_scatter(new_scatter);
          }
          break;
        }
        default:
          break;
      }
    }

    if (replacements.find(computation.get()) == replacements.end()) {
      new_computations.push_back(std::move(computation));
    }
  }

  // Replace entry_computation if necessary.
  entry_computation_ = tensorflow::gtl::FindWithDefault(
      replacements, entry_computation_, entry_computation_);

  computations_ = std::move(new_computations);
}

std::string HloModule::ToString(const HloPrintOptions& options) const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTcc mht_9(mht_9_v, 434, "", "./tensorflow/compiler/xla/service/hlo_module.cc", "HloModule::ToString");

  return std::string(ToCord(options));
}

absl::Cord HloModule::ToCord(const HloPrintOptions& options) const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTcc mht_10(mht_10_v, 441, "", "./tensorflow/compiler/xla/service/hlo_module.cc", "HloModule::ToCord");

  absl::Cord result;
  result.Append("HloModule ");
  if (options.print_ids()) {
    // When print_ids() is false, exclude module's name because it includes and
    // leads to non-deterministic fingerprint.
    result.Append(name());
  }
  if (has_schedule()) {
    TF_CHECK_OK(schedule().Verify());
    result.Append(", is_scheduled=true");
  }
  std::string serialized_aliasing = input_output_alias_config().ToShortString();
  if (!serialized_aliasing.empty()) {
    result.Append(", input_output_alias={ ");
    result.Append(std::move(serialized_aliasing));
    result.Append(" }");
  }
  if (config_.alias_passthrough_params()) {
    result.Append(", alias_passthrough_params=true");
  }
  if (config_.allow_spmd_sharding_propagation_to_output()) {
    result.Append(", allow_spmd_sharding_propagation_to_output=true");
  }
  result.Append("\n\n");
  const auto& computations = options.canonicalize_computations()
                                 ? MakeComputationSorted()
                                 : MakeComputationPostOrder();
  for (const HloComputation* computation : computations) {
    // Don't print async computations when the sytax sugar is enabled since that
    // is redundant information.
    if (options.syntax_sugar_async_ops() && computation->IsAsyncComputation()) {
      continue;
    }
    if (computation == entry_computation()) {
      result.Append("ENTRY ");
    }
    if (has_schedule() && schedule().is_computation_scheduled(computation)) {
      result.Append(computation->ToCord(
          options, schedule().sequence(computation).instructions()));
    } else {
      result.Append(computation->ToCord(options));
    }
    result.Append("\n\n");
  }
  return result;
}

HloModuleProto HloModule::ToProto() const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTcc mht_11(mht_11_v, 492, "", "./tensorflow/compiler/xla/service/hlo_module.cc", "HloModule::ToProto");

  HloModuleProto proto;
  proto.set_id(unique_id_);
  proto.set_name(name_);
  proto.set_entry_computation_name(entry_computation_->name());
  proto.set_entry_computation_id(entry_computation_->unique_id());
  for (const HloComputation* computation : MakeComputationPostOrder()) {
    HloComputationProto computation_proto = computation->ToProto();
    proto.add_computations()->Swap(&computation_proto);
  }
  if (has_schedule()) {
    *proto.mutable_schedule() = schedule().ToProto().ValueOrDie();
  }
  *proto.mutable_host_program_shape() =
      entry_computation_layout().ComputeProgramShape().ToProto();
  *proto.mutable_input_output_alias() = input_output_alias_config().ToProto();
  *proto.mutable_dynamic_parameter_binding() =
      dynamic_parameter_binding().ToProto();
  for (const auto& parameter_indices : CrossProgramPrefetches()) {
    const auto& parameter = parameter_indices.first;
    const auto& indices = parameter_indices.second;
    auto* prefetch = proto.mutable_cross_program_prefetches()->Add();
    prefetch->set_parameter(parameter);
    for (auto index : indices) {
      prefetch->add_index(index);
    }
  }
  proto.set_is_dynamic(is_dynamic_);
  if (has_spmd_output_sharding()) {
    *proto.mutable_spmd_output_sharding() = spmd_output_sharding().ToProto();
  }

  if (has_spmd_parameters_shardings()) {
    for (const auto& parameter_sharding : spmd_parameters_shardings()) {
      *proto.add_spmd_parameters_shardings() = parameter_sharding.ToProto();
    }
  }

  for (const HloModuleProto::ProfileInfo& profile_info : profile_info_list_) {
    HloModuleProto::ProfileInfo& profile_info_proto =
        *proto.mutable_profile_info()->Add();
    profile_info_proto.set_profile_type(profile_info.profile_type());
    profile_info_proto.set_relative_speedup(profile_info.relative_speedup());
    profile_info_proto.set_profile_source(profile_info.profile_source());
    profile_info_proto.set_compilation_event(profile_info.compilation_event());
  }
  return proto;
}

Status HloModule::CheckUniqueNamesAndIdsForComputationsAndInstructions() const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTcc mht_12(mht_12_v, 544, "", "./tensorflow/compiler/xla/service/hlo_module.cc", "HloModule::CheckUniqueNamesAndIdsForComputationsAndInstructions");

  absl::flat_hash_set<std::string> computation_names;
  absl::flat_hash_set<int> computation_ids;
  absl::flat_hash_set<std::string> instruction_names;
  absl::flat_hash_set<int> instruction_ids;

  for (const HloComputation* computation : computations()) {
    TF_RET_CHECK(!ContainsKey(computation_names, computation->name()))
        << "Computation name is not unique: " << computation->name();
    computation_names.insert(computation->name());

    TF_RET_CHECK(!ContainsKey(computation_ids, computation->unique_id()))
        << "Computation id is not unique: " << computation->unique_id();
    computation_ids.insert(computation->unique_id());

    for (const HloInstruction* instruction : computation->instructions()) {
      TF_RET_CHECK(!ContainsKey(instruction_names, instruction->name()))
          << "Instruction name is not unique: " << instruction->name();
      instruction_names.insert(instruction->name());

      TF_RET_CHECK(!ContainsKey(instruction_ids, instruction->unique_id()))
          << "Instruction id is not unique: " << instruction->unique_id();
      instruction_ids.insert(instruction->unique_id());
    }
  }
  return Status::OK();
}

/* static */
StatusOr<std::unique_ptr<HloModule>> HloModule::CreateFromProto(
    const HloModuleProto& proto, const HloModuleConfig& module_config,
    bool prohibit_empty_literal) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTcc mht_13(mht_13_v, 578, "", "./tensorflow/compiler/xla/service/hlo_module.cc", "HloModule::CreateFromProto");

  VLOG(2) << "CreateFromProto()";
  XLA_VLOG_LINES(3, proto.DebugString());

  // The ProgramShape in the passed in module config must match the shapes of
  // the entry parameters and root.
  TF_RET_CHECK(proto.has_host_program_shape())
      << "No program shape found in the proto";
  ProgramShape expected_program_shape(proto.host_program_shape());
  TF_RET_CHECK(expected_program_shape.parameters_size() ==
               module_config.entry_computation_layout().parameter_count());
  for (int i = 0; i < expected_program_shape.parameters_size(); ++i) {
    const Shape& parameter_shape =
        module_config.entry_computation_layout().parameter_layout(i).shape();
    TF_RET_CHECK(ShapeUtil::Compatible(expected_program_shape.parameters(i),
                                       parameter_shape))
        << "HloModuleConfig has different shape for parameter " << i
        << " than the HLO module. Expected: "
        << ShapeUtil::HumanStringWithLayout(
               expected_program_shape.parameters(i))
        << ", actual: " << ShapeUtil::HumanStringWithLayout(parameter_shape);
  }
  const Shape& result_shape =
      module_config.entry_computation_layout().result_layout().shape();
  TF_RET_CHECK(
      ShapeUtil::Compatible(expected_program_shape.result(), result_shape))
      << "HloModuleConfig has different result shape than the HLO module. "
         "Expected: "
      << ShapeUtil::HumanStringWithLayout(expected_program_shape.result())
      << ", actual: " << ShapeUtil::HumanStringWithLayout(result_shape);

  absl::flat_hash_map<int64_t, HloComputation*> computation_map;
  absl::flat_hash_map<HloComputation*, int64_t> to_proto_id;
  std::vector<std::unique_ptr<HloComputation>> computations;
  HloComputation* entry = nullptr;
  for (const HloComputationProto& computation_proto : proto.computations()) {
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<HloComputation> computation,
        HloComputation::CreateFromProto(computation_proto, computation_map,
                                        prohibit_empty_literal));
    CHECK_NE(computation.get(), nullptr);
    int64_t computation_id = computation_proto.id();
    TF_RET_CHECK(computation_id != -1);
    TF_RET_CHECK(!ContainsKey(computation_map, computation_id));
    computation_map[computation_id] = computation.get();
    to_proto_id[computation.get()] = computation_id;
    if (computation_id == proto.entry_computation_id()) {
      entry = computation.get();
    }
    computations.push_back(std::move(computation));
  }
  TF_RET_CHECK(entry != nullptr);

  auto module = absl::make_unique<HloModule>(proto.name(), module_config);

  // Sort the computations in the proto id's order.
  absl::c_sort(computations, [&](const std::unique_ptr<HloComputation>& a,
                                 const std::unique_ptr<HloComputation>& b) {
    return to_proto_id[a.get()] < to_proto_id[b.get()];
  });

  // Add sorted computations to the module.
  for (auto& computation : computations) {
    bool is_entry = computation.get() == entry;
    // Don't uniquify names because we want names to be stable across
    // serialization and deserialization.
    module->AddComputationInternal(std::move(computation), is_entry,
                                   /*uniquify_identifiers=*/false,
                                   /*preserve_entry_layouts=*/false);
  }
  TF_RET_CHECK(module->entry_computation_ != nullptr);
  if (proto.has_schedule()) {
    TF_RETURN_IF_ERROR(module->RemoveUnusedComputations());
  }
  TF_ASSIGN_OR_RETURN(
      module->input_output_alias_config_,
      HloInputOutputAliasConfig::CreateFromProto(
          entry->ComputeProgramShape().result(), proto.input_output_alias()));

  // Because we didn't uniquify the names or the ids, double-check that the
  // instruction and computation names and ids are unique from the proto.
  TF_ASSIGN_OR_RETURN(module->dynamic_parameter_binding_,
                      DynamicParameterBinding::CreateFromProto(
                          proto.dynamic_parameter_binding()));

  TF_RETURN_IF_ERROR(
      module->CheckUniqueNamesAndIdsForComputationsAndInstructions());

  if (proto.has_schedule()) {
    TF_ASSIGN_OR_RETURN(
        HloSchedule schedule,
        HloSchedule::CreateFromProto(module.get(), proto.schedule()));
    TF_RETURN_IF_ERROR(module->set_schedule(std::move(schedule)));
  }

  for (const auto& prefetch : proto.cross_program_prefetches()) {
    module->AddCrossProgramPrefetch(
        prefetch.parameter(),
        ShapeIndex(prefetch.index().begin(), prefetch.index().end()));
  }

  module->set_is_dynamic(proto.is_dynamic());

  if (proto.has_spmd_output_sharding()) {
    TF_ASSIGN_OR_RETURN(HloSharding hlo_sharding,
                        HloSharding::FromProto(proto.spmd_output_sharding()));
    module->set_spmd_output_sharding(hlo_sharding);
  }

  std::vector<HloSharding> param_shardings;
  for (const auto& sharding_proto : proto.spmd_parameters_shardings()) {
    TF_ASSIGN_OR_RETURN(HloSharding sharding,
                        HloSharding::FromProto(sharding_proto));
    param_shardings.push_back(sharding);
  }
  if (!param_shardings.empty()) {
    module->set_spmd_parameters_shardings(param_shardings);
  }

  for (const auto& profile_info : proto.profile_info()) {
    module->add_profile_info(profile_info);
  }
  return std::move(module);
}

/* static */
StatusOr<HloModuleConfig> HloModule::CreateModuleConfigFromShape(
    const ProgramShape& program_shape, const DebugOptions& debug_options,
    const ExecutionOptions* execution_options) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTcc mht_14(mht_14_v, 709, "", "./tensorflow/compiler/xla/service/hlo_module.cc", "HloModule::CreateModuleConfigFromShape");

  HloModuleConfig module_config(ProgramShape{program_shape});
  module_config.set_debug_options(debug_options);
  if (execution_options) {
    if (execution_options->num_replicas() > 0) {
      module_config.set_replica_count(execution_options->num_replicas());
    }
    if (execution_options->num_partitions() > 0) {
      module_config.set_num_partitions(execution_options->num_partitions());
    }
    module_config.set_use_spmd_partitioning(
        execution_options->use_spmd_partitioning());
    module_config.set_use_auto_spmd_partitioning(
        execution_options->use_auto_spmd_partitioning());
    module_config.set_deduplicate_hlo(execution_options->deduplicate_hlo());
    module_config.set_allow_spmd_sharding_propagation_to_output(
        execution_options->allow_spmd_sharding_propagation_to_output());
    if (execution_options->has_device_assignment()) {
      TF_ASSIGN_OR_RETURN(std::unique_ptr<DeviceAssignment> device_assignment,
                          DeviceAssignment::Deserialize(
                              execution_options->device_assignment()));
      module_config.set_static_device_assignment(*device_assignment);
      if (execution_options->num_replicas() > 0) {
        CHECK_EQ(module_config.static_device_assignment().replica_count(),
                 module_config.replica_count());
      }
      if (execution_options->num_partitions() > 0) {
        CHECK_EQ(module_config.static_device_assignment().computation_count(),
                 module_config.num_partitions());
      }
    }
  }

  // The module config is constructed with default layouts regardless of what is
  // passed in via the ProgramShape. Set the layouts to the appropriate values.
  ComputationLayout* entry_layout =
      module_config.mutable_entry_computation_layout();
  for (int64_t i = 0; i < entry_layout->parameter_count(); ++i) {
    TF_RETURN_IF_ERROR(
        entry_layout->mutable_parameter_layout(i)->CopyLayoutFromShape(
            program_shape.parameters(i)));
  }
  TF_RETURN_IF_ERROR(entry_layout->mutable_result_layout()->CopyLayoutFromShape(
      program_shape.result()));
  return module_config;
}

/* static */
StatusOr<HloModuleConfig> HloModule::CreateModuleConfigFromProto(
    const HloModuleProto& module, const DebugOptions& debug_options,
    const ExecutionOptions* execution_options) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTcc mht_15(mht_15_v, 762, "", "./tensorflow/compiler/xla/service/hlo_module.cc", "HloModule::CreateModuleConfigFromProto");

  TF_RET_CHECK(module.has_host_program_shape())
      << "No program shape found in the proto";
  ProgramShape program_shape(module.host_program_shape());
  return CreateModuleConfigFromShape(program_shape, debug_options,
                                     execution_options);
}

namespace {
// Returns whether `hlo` is used outside the given subcomputation.
// `instructions_in_subcomputation` is the instruction set of the given
// subcomputation.
bool IsUsedOutsideSubcomputation(const HloInstruction& hlo,
                                 const absl::flat_hash_set<HloInstruction*>&
                                     instructions_in_subcomputation) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTcc mht_16(mht_16_v, 779, "", "./tensorflow/compiler/xla/service/hlo_module.cc", "IsUsedOutsideSubcomputation");

  return absl::c_any_of(hlo.users(), [&](HloInstruction* user) {
    return !instructions_in_subcomputation.contains(user);
  });
}
}  // anonymous namespace

HloInstruction* HloModule::OutlineExpressionFromComputation(
    absl::Span<HloInstruction* const> instructions_to_outline,
    const std::string& outlined_computation_name, HloComputation* computation) {
   std::vector<std::string> mht_17_v;
   mht_17_v.push_back("outlined_computation_name: \"" + outlined_computation_name + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTcc mht_17(mht_17_v, 792, "", "./tensorflow/compiler/xla/service/hlo_module.cc", "HloModule::OutlineExpressionFromComputation");

  auto builder = HloComputation::Builder(outlined_computation_name);

  // A map from original instructions to their counterparts in the new outlined
  // function.
  absl::flat_hash_map<HloInstruction*, HloInstruction*> outlined_instructions;
  // A set that contains all instructions to be outlined.
  absl::flat_hash_set<HloInstruction*> instruction_set_to_outline(
      instructions_to_outline.begin(), instructions_to_outline.end());
  std::vector<HloInstruction*> arguments;
  std::vector<HloInstruction*> outputs;
  int64_t parameter_count = 0;
  for (HloInstruction* instruction_to_outline : instructions_to_outline) {
    // Clone the original instruction.
    HloInstruction* outlined_instruction =
        builder.AddInstruction(instruction_to_outline->Clone());

    // Replace its operands to their counterparts in the new function.
    for (int64_t operand_num = 0;
         operand_num < outlined_instruction->operand_count(); ++operand_num) {
      HloInstruction* old_operand =
          outlined_instruction->mutable_operand(operand_num);

      HloInstruction** operand_slot = &(outlined_instructions[old_operand]);
      if (*operand_slot == nullptr) {
        // Because instructions_to_outline is in topological order, if
        // old_operand is not in outlined_instructions, old_operand must be an
        // input of the outlined subcomputation and thus should be represented
        // as a parameter in the new function.
        arguments.push_back(old_operand);
        *operand_slot = builder.AddInstruction(HloInstruction::CreateParameter(
            parameter_count, old_operand->shape(), "p"));
        ++parameter_count;
      }
      TF_CHECK_OK(
          outlined_instruction->ReplaceOperandWith(operand_num, *operand_slot));
    }

    // Insert the new instruction into the outlined_instructions map.
    InsertOrDie(&outlined_instructions, instruction_to_outline,
                outlined_instruction);

    // Mark instruction_to_outline an output if it is used outside the
    // subcomputation or is the output of the original computation (i.e. used
    // externally).
    if (instruction_to_outline->user_count() == 0 ||
        IsUsedOutsideSubcomputation(*instruction_to_outline,
                                    instruction_set_to_outline)) {
      outputs.push_back(instruction_to_outline);
    }
  }

  if (outputs.size() != 1) {
    std::string error_message =
        "The subcomputation to outline has multiple outputs:\n";
    for (HloInstruction* output : outputs) {
      absl::StrAppend(&error_message, output->ToString(), "\n");
    }
    LOG(FATAL) << error_message;
  }
  HloInstruction* output = outputs[0];

  // Creates a call to the nested computation.
  HloComputation* nested_computation = AddEmbeddedComputation(
      builder.Build(FindOrDie(outlined_instructions, output)));
  HloInstruction* call = computation->AddInstruction(HloInstruction::CreateCall(
      output->shape(), arguments, nested_computation));

  VLOG(2) << "Outlining the following instructions";
  for (auto* instruction_to_outline : instructions_to_outline) {
    VLOG(2) << "  " << instruction_to_outline->ToString();
  }
  VLOG(2) << "as a call " << call->ToString();
  VLOG(2) << "to " << nested_computation->ToString();

  TF_CHECK_OK(output->ReplaceAllUsesWith(call));
  for (auto i = instructions_to_outline.rbegin();
       i != instructions_to_outline.rend(); ++i) {
    TF_CHECK_OK(computation->RemoveInstruction(*i));
  }

  return call;
}

int64_t HloModule::instruction_count() const {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTcc mht_18(mht_18_v, 879, "", "./tensorflow/compiler/xla/service/hlo_module.cc", "HloModule::instruction_count");

  int64_t n = 0;
  for (const auto& computation : computations_) {
    n += computation->instruction_count();
  }
  return n;
}

std::vector<HloComputation*> HloModule::MakeComputationPostOrder(
    const absl::flat_hash_set<HloComputation*>& allow_list) const {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTcc mht_19(mht_19_v, 891, "", "./tensorflow/compiler/xla/service/hlo_module.cc", "HloModule::MakeComputationPostOrder");

  std::vector<HloComputation*> filtered_post_order(allow_list.size());
  auto post_order = this->MakeComputationPostOrder();

  int filtered_idx = 0;
  for (auto& computation : post_order) {
    if (allow_list.contains(computation)) {
      filtered_post_order[filtered_idx] = computation;
      filtered_idx += 1;
    }
  }

  return filtered_post_order;
}

std::vector<HloComputation*> HloModule::MakeComputationPostOrder() const {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTcc mht_20(mht_20_v, 909, "", "./tensorflow/compiler/xla/service/hlo_module.cc", "HloModule::MakeComputationPostOrder");

  if (computations_.empty()) {
    return {};
  }
  // First determine all root computations by building a set of nonroot
  // computations (computations which are called by an instruction in the
  // module).
  absl::flat_hash_set<HloComputation*> nonroot_computations;
  nonroot_computations.reserve(computations_.size() - 1);
  for (auto& computation : computations_) {
    for (auto* instruction : computation->instructions()) {
      for (HloComputation* called_computation :
           instruction->called_computations()) {
        nonroot_computations.insert(called_computation);
      }
    }
  }

  // Keep track of computations which have already been added to the post
  // order. This prevents duplication as an embedded computation may be called
  // from two different root computations.
  absl::flat_hash_set<HloComputation*> added_computations;
  std::vector<HloComputation*> post_order;
  added_computations.reserve(computations_.size());
  post_order.reserve(computations_.size());
  for (auto& computation : computations_) {
    if (nonroot_computations.contains(computation.get())) {
      continue;
    }
    for (HloComputation* embedded_computation :
         computation->MakeEmbeddedComputationsList()) {
      if (!added_computations.contains(embedded_computation)) {
        post_order.push_back(embedded_computation);
        added_computations.insert(embedded_computation);
      }
    }
    // Root computations should only be encountered once.
    CHECK(!added_computations.contains(computation.get()));
    post_order.push_back(computation.get());
    added_computations.insert(computation.get());
  }
  if (post_order.size() != computations_.size()) {
    for (HloComputation* computation : post_order) {
      LOG(ERROR) << "Post Order: " << computation->name() << " ("
                 << computation->parent()->name() << ")";
    }
    for (auto& computation : computations_) {
      LOG(ERROR) << "Computations: " << computation->name() << " ("
                 << computation->parent()->name() << ")";
    }
    LOG(FATAL) << "Mismatch computation count: post_order=" << post_order.size()
               << " computation_count=" << computations_.size();
  }
  return post_order;
}

namespace {
bool CompareComputationsByContent(const HloComputation* a,
                                  const HloComputation* b) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTcc mht_21(mht_21_v, 970, "", "./tensorflow/compiler/xla/service/hlo_module.cc", "CompareComputationsByContent");

  if (a->instruction_count() != b->instruction_count()) {
    return a->instruction_count() < b->instruction_count();
  }
  return a->ToString(HloPrintOptions::ModuleFingerprint()) <
         b->ToString(HloPrintOptions::ModuleFingerprint());
}

uint64_t GetFingerprint(
    absl::flat_hash_map<const HloComputation*, uint64_t>& fingerprint_map,
    const HloComputation* computation) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTcc mht_22(mht_22_v, 983, "", "./tensorflow/compiler/xla/service/hlo_module.cc", "GetFingerprint");

  auto it = fingerprint_map.find(computation);
  if (it != fingerprint_map.end()) {
    return it->second;
  } else {
    const uint64_t fingerprint = tensorflow::Fingerprint64(
        computation->ToString(HloPrintOptions::ModuleFingerprint()));
    fingerprint_map[computation] = fingerprint;
    return fingerprint;
  }
}

void SortComputationsByContent(std::vector<HloComputation*>* computations) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTcc mht_23(mht_23_v, 998, "", "./tensorflow/compiler/xla/service/hlo_module.cc", "SortComputationsByContent");

  absl::flat_hash_map<const HloComputation*, uint64_t> fingerprint_map;
  auto cmp = [&fingerprint_map](const HloComputation* a,
                                const HloComputation* b) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTcc mht_24(mht_24_v, 1004, "", "./tensorflow/compiler/xla/service/hlo_module.cc", "lambda");

    if (a->instruction_count() != b->instruction_count()) {
      return a->instruction_count() < b->instruction_count();
    }
    return GetFingerprint(fingerprint_map, a) <
           GetFingerprint(fingerprint_map, b);
  };
  absl::c_sort(*computations, cmp);
}

}  // anonymous namespace

std::vector<HloComputation*> HloModule::MakeComputationSorted() const {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTcc mht_25(mht_25_v, 1019, "", "./tensorflow/compiler/xla/service/hlo_module.cc", "HloModule::MakeComputationSorted");

  std::vector<HloComputation*> result = MakeComputationPostOrder();
  if (config().content_aware_computation_sorting()) {
    SortComputationsByContent(&result);
  }
  return result;
}

std::vector<HloComputation*> HloModule::MakeNonfusionComputations() const {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTcc mht_26(mht_26_v, 1030, "", "./tensorflow/compiler/xla/service/hlo_module.cc", "HloModule::MakeNonfusionComputations");

  std::vector<HloComputation*> result = MakeComputationPostOrder();
  result.erase(std::remove_if(
                   result.begin(), result.end(),
                   [](HloComputation* c) { return c->IsFusionComputation(); }),
               result.end());
  return result;
}

std::vector<HloComputation*> HloModule::MakeNonfusionComputationsSorted()
    const {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTcc mht_27(mht_27_v, 1043, "", "./tensorflow/compiler/xla/service/hlo_module.cc", "HloModule::MakeNonfusionComputationsSorted");

  auto result = MakeNonfusionComputations();
  if (config().content_aware_computation_sorting()) {
    SortComputationsByContent(&result);
  }
  return result;
}

std::unique_ptr<HloModule> HloModule::Clone(const std::string& suffix) const {
   std::vector<std::string> mht_28_v;
   mht_28_v.push_back("suffix: \"" + suffix + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTcc mht_28(mht_28_v, 1055, "", "./tensorflow/compiler/xla/service/hlo_module.cc", "HloModule::Clone");

  return Clone(config(), suffix);
}

std::unique_ptr<HloModule> HloModule::Clone(const HloModuleConfig& config,
                                            const std::string& suffix) const {
   std::vector<std::string> mht_29_v;
   mht_29_v.push_back("suffix: \"" + suffix + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTcc mht_29(mht_29_v, 1064, "", "./tensorflow/compiler/xla/service/hlo_module.cc", "HloModule::Clone");

  VLOG(1) << "Cloning module :" << name_ << " --> " << suffix << "\n";
  auto module = absl::make_unique<HloModule>(
      absl::StrCat(name_, suffix.empty() ? "" : "-", suffix), config);

  HloCloneContext context(module.get(), suffix);
  auto cloned_computation = entry_computation_->Clone(suffix, &context);
  module->AddEntryComputation(std::move(cloned_computation));
  module->input_output_alias_config() = input_output_alias_config();
  module->set_is_dynamic(is_dynamic());
  if (has_schedule() && schedule().Verify().ok()) {
    HloSchedule clone_schedule(module.get());
    for (HloComputation* computation : computations()) {
      if (schedule().is_computation_scheduled(computation)) {
        HloComputation* new_computation = context.FindComputation(computation);
        // The module being cloned may have computations that are dead, i.e.,
        // unreachable from the entry computation. In that case, new_computation
        // is nullptr.
        if (new_computation != nullptr) {
          HloInstructionSequence& clone_sequence =
              clone_schedule.GetOrCreateSequence(new_computation);
          for (const HloInstruction* instruction :
               schedule().sequence(computation).instructions()) {
            clone_sequence.push_back(context.GetInstruction(instruction));
          }
        }
      }
    }
    TF_CHECK_OK(module->set_schedule(std::move(clone_schedule)));
  }
  for (const auto& parameter_indices : CrossProgramPrefetches()) {
    const auto& parameter = parameter_indices.first;
    const auto& indices = parameter_indices.second;
    module->AddCrossProgramPrefetch(parameter, indices);
  }
  return module;
}

Status HloModule::RemoveUnusedComputations() {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTcc mht_30(mht_30_v, 1105, "", "./tensorflow/compiler/xla/service/hlo_module.cc", "HloModule::RemoveUnusedComputations");

  std::string suffix = "tmp";
  auto module = absl::make_unique<HloModule>(
      absl::StrCat(name_, suffix.empty() ? "" : "-", suffix), config());
  HloCloneContext context(module.get(), suffix);
  entry_computation_->Clone(suffix, &context);
  std::vector<HloComputation*> to_remove;
  for (auto computation : computations()) {
    auto found_computation = context.FindComputation(computation);
    if (found_computation == nullptr) {
      to_remove.push_back(computation);
    }
  }
  for (auto computation : to_remove) {
    TF_RETURN_IF_ERROR(RemoveEmbeddedComputation(computation));
  }
  return Status::OK();
}

HloComputation* HloModule::DeepCloneComputation(HloComputation* computation,
                                                HloCloneContext* context) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTcc mht_31(mht_31_v, 1128, "", "./tensorflow/compiler/xla/service/hlo_module.cc", "HloModule::DeepCloneComputation");

  HloComputation* new_computation;
  if (context != nullptr) {
    if ((new_computation = context->FindComputation(computation)) != nullptr) {
      return new_computation;
    }
    new_computation =
        AddEmbeddedComputation(computation->Clone(context->suffix(), context));
  } else {
    new_computation = AddEmbeddedComputation(computation->Clone(""));
  }
  return new_computation;
}

uint64_t HloModule::RandomNew64() const {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTcc mht_32(mht_32_v, 1145, "", "./tensorflow/compiler/xla/service/hlo_module.cc", "HloModule::RandomNew64");

  absl::MutexLock l(&rng_mutex_);
  return rng_();
}

HloComputation* HloModule::GetComputationWithName(absl::string_view name) {
   std::vector<std::string> mht_33_v;
   mht_33_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTcc mht_33(mht_33_v, 1154, "", "./tensorflow/compiler/xla/service/hlo_module.cc", "HloModule::GetComputationWithName");

  auto computations_in_module = computations();
  auto it = absl::c_find_if(
      computations_in_module,
      [&](HloComputation* computation) { return computation->name() == name; });
  return it == computations_in_module.end() ? nullptr : *it;
}

/* static */ std::atomic<int> HloModule::next_unique_module_id_(0);

}  // namespace xla
