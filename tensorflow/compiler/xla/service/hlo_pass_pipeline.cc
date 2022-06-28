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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_pass_pipelineDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_pass_pipelineDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_pass_pipelineDTcc() {
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

#include "tensorflow/compiler/xla/service/hlo_pass_pipeline.h"

#include <functional>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/service/dump.h"
#include "tensorflow/compiler/xla/service/hlo_graph_dumper.h"
#include "tensorflow/compiler/xla/service/hlo_proto_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

namespace {

void RecordPassStartMetadata(HloModule& module, const std::string& pass_name,
                             const std::string& pipeline_name) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("pass_name: \"" + pass_name + "\"");
   mht_0_v.push_back("pipeline_name: \"" + pipeline_name + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_pass_pipelineDTcc mht_0(mht_0_v, 210, "", "./tensorflow/compiler/xla/service/hlo_pass_pipeline.cc", "RecordPassStartMetadata");

  module.metadata()->RecordPassStart();
  // An HloPassMetadata was just created so Status should always be OK.
  TF_CHECK_OK(module.metadata()->set_current_pass_name(pass_name));
  TF_CHECK_OK(module.metadata()->set_current_pass_pipeline_name(pipeline_name));
}

void RecordPassStartMetadata(HloModuleGroup& module_group,
                             const std::string& pass_name,
                             const std::string& pipeline_name) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("pass_name: \"" + pass_name + "\"");
   mht_1_v.push_back("pipeline_name: \"" + pipeline_name + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_pass_pipelineDTcc mht_1(mht_1_v, 224, "", "./tensorflow/compiler/xla/service/hlo_pass_pipeline.cc", "RecordPassStartMetadata");

  for (HloModule* module : module_group.modules()) {
    RecordPassStartMetadata(*module, pass_name, pipeline_name);
  }
}

Status AttemptRecordPassEndMetadata(HloModule& module,
                                    const std::string& pass_name,
                                    bool module_changed) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("pass_name: \"" + pass_name + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_pass_pipelineDTcc mht_2(mht_2_v, 236, "", "./tensorflow/compiler/xla/service/hlo_pass_pipeline.cc", "AttemptRecordPassEndMetadata");

  // Module id is set here instead of RecordPassStartMetadata because it may
  // change in the middle of the pass, and we want the final id.
  TF_RETURN_IF_ERROR(
      module.metadata()->set_current_pass_module_id(module.unique_id()));
  TF_RETURN_IF_ERROR(
      module.metadata()->set_current_pass_module_changed(module_changed));
  TF_RETURN_IF_ERROR(module.metadata()->RecordPassEnd());
  return Status::OK();
}

void RecordPassEndMetadata(HloModule& module, const std::string& pass_name,
                           bool module_changed) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("pass_name: \"" + pass_name + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_pass_pipelineDTcc mht_3(mht_3_v, 252, "", "./tensorflow/compiler/xla/service/hlo_pass_pipeline.cc", "RecordPassEndMetadata");

  Status status =
      AttemptRecordPassEndMetadata(module, pass_name, module_changed);
  if (!status.ok()) {
    LOG(FATAL) << status;
  }
}

Status AttemptRecordPassEndMetadata(HloModuleGroup& module_group,
                                    const std::string& pass_name,
                                    bool module_changed) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("pass_name: \"" + pass_name + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_pass_pipelineDTcc mht_4(mht_4_v, 266, "", "./tensorflow/compiler/xla/service/hlo_pass_pipeline.cc", "AttemptRecordPassEndMetadata");

  for (HloModule* module : module_group.modules()) {
    for (HloModule* other_module : module_group.modules()) {
      TF_RETURN_IF_ERROR(
          module->metadata()->add_current_pass_module_group_module_id(
              other_module->unique_id()));
    }
    TF_RETURN_IF_ERROR(
        AttemptRecordPassEndMetadata(*module, pass_name, module_changed));
  }
  return Status::OK();
}

void RecordPassEndMetadata(HloModuleGroup& module_group,
                           const std::string& pass_name, bool module_changed) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("pass_name: \"" + pass_name + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_pass_pipelineDTcc mht_5(mht_5_v, 284, "", "./tensorflow/compiler/xla/service/hlo_pass_pipeline.cc", "RecordPassEndMetadata");

  Status status =
      AttemptRecordPassEndMetadata(module_group, pass_name, module_changed);
  if (!status.ok()) {
    LOG(FATAL) << status;
  }
}

void SetInstructionMetadata(HloModule& module) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_pass_pipelineDTcc mht_6(mht_6_v, 295, "", "./tensorflow/compiler/xla/service/hlo_pass_pipeline.cc", "SetInstructionMetadata");

  StatusOr<int64_t> pass_id = module.metadata()->current_pass_id();
  if (!pass_id.ok()) {
    LOG(FATAL) << pass_id.status();
  }
  for (xla::HloComputation* computation : module.computations()) {
    for (xla::HloInstruction* instruction : computation->instructions()) {
      if (instruction->metadata().creation_pass_id() == 0) {
        instruction->set_creation_pass_id(*pass_id);
      }
      if (instruction->metadata().logical_creation_pass_id() == 0) {
        instruction->set_logical_creation_pass_id(*pass_id);
      }
    }
  }
}

void SetInstructionMetadata(HloModuleGroup& module_group) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_pass_pipelineDTcc mht_7(mht_7_v, 315, "", "./tensorflow/compiler/xla/service/hlo_pass_pipeline.cc", "SetInstructionMetadata");

  for (HloModule* module : module_group.modules()) {
    SetInstructionMetadata(*module);
  }
}

}  // namespace

template <typename HloT>
Status HloPassPipeline::RunInvariantCheckers(
    HloT* hlo, absl::string_view after_pass_name) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("after_pass_name: \"" + std::string(after_pass_name.data(), after_pass_name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_pass_pipelineDTcc mht_8(mht_8_v, 329, "", "./tensorflow/compiler/xla/service/hlo_pass_pipeline.cc", "HloPassPipeline::RunInvariantCheckers");

  for (auto& invariant_checker : invariant_checkers_) {
    VLOG(1) << "    Invariant checker " << invariant_checker->name();
    StatusOr<bool> changed_status = RunHelper(invariant_checker.get(), hlo);
    VLOG(1) << "    Invariant checker done " << invariant_checker->name();
    if (!changed_status.ok()) {
      VLOG(2) << "Failed invariant check:";
      XLA_VLOG_LINES(2, hlo->ToString());
      return tensorflow::errors::CreateWithUpdatedMessage(
          changed_status.status(),
          absl::StrCat(changed_status.status().error_message(),
                       "\n\nFailed after ", after_pass_name));
    }
    TF_RET_CHECK(!changed_status.ValueOrDie())
        << "invariant checkers must not change the graph";
  }
  return Status::OK();
}

template <typename HloT>
StatusOr<bool> HloPassPipeline::RunPassesInternal(
    HloT* hlo, const DebugOptions& debug_options) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_pass_pipelineDTcc mht_9(mht_9_v, 353, "", "./tensorflow/compiler/xla/service/hlo_pass_pipeline.cc", "HloPassPipeline::RunPassesInternal");

  auto passes = GetEnabledPasses(debug_options);
  // Copy string by value since debug options could get clobbered in an hlo
  // module group pass.
  std::string dump_regex = debug_options.xla_dump_hlo_pass_re();
  static constexpr absl::string_view kPipelineStart = "pipeline-start";
  static constexpr absl::string_view kPipelineEnd = "pipeline-end";
  std::string pipeline_name = std::string(name());

  TF_RETURN_IF_ERROR(RunInvariantCheckers(hlo, kPipelineStart));

  RecordPassStartMetadata(*hlo, std::string(kPipelineStart), pipeline_name);
  SetInstructionMetadata(*hlo);
  MaybeDumpHloAndSaveFilenames(*hlo,
                               /*after_pass_name=*/kPipelineStart,
                               /*before_pass_name=*/passes.empty()
                                   ? kPipelineEnd
                                   : passes.front()->name());
  RecordPassEndMetadata(*hlo, std::string(kPipelineStart),
                        /*module_changed=*/false);

  bool changed = false;
  for (int i = 0; i < passes.size(); i++) {
    HloPassInterface* pass = passes[i];
    XLA_SCOPED_LOGGING_TIMER(absl::StrCat("HLO pass: ", pass->name()));
    std::string pass_name = std::string(pass->name());
    VLOG(1) << "  HLO pass " << pass_name;
    VLOG(2) << "  Module hash " << absl::HashOf(*hlo);
    if (!pass->IsPassPipeline()) {
      compilation_stats_->StartPass(pass_name);
    }
    RecordPassStartMetadata(*hlo, pass_name, pipeline_name);
    TF_ASSIGN_OR_RETURN(bool pass_changed, RunHelper(pass, hlo));
    SetInstructionMetadata(*hlo);
    if (!dump_regex.empty() && (pass_changed || dump_regex != ".*")) {
      MaybeDumpHloAndSaveFilenames(*hlo,
                                   /*after_pass_name=*/pass_name,
                                   /*before_pass_name=*/i + 1 >= passes.size()
                                       ? kPipelineEnd
                                       : passes[i + 1]->name());
    }
    RecordPassEndMetadata(*hlo, pass_name, pass_changed);
    changed |= pass_changed;
    if (pass_changed) {
      VLOG(3) << "  Pass caused changes " << pass->name();
    }
    TF_RETURN_IF_ERROR(RunInvariantCheckers(hlo, pass_name));
    if (!pass->IsPassPipeline()) {
      compilation_stats_->EndPass(pass_name);
    }
  }
  return changed;
}

std::vector<HloPassInterface*> HloPassPipeline::GetEnabledPasses(
    const DebugOptions& debug_options) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_pass_pipelineDTcc mht_10(mht_10_v, 411, "", "./tensorflow/compiler/xla/service/hlo_pass_pipeline.cc", "HloPassPipeline::GetEnabledPasses");

  if (debug_options.xla_disable_all_hlo_passes()) {
    VLOG(1) << "*All* passes disabled by --xla_disable_all_hlo_passes.";
    return {};
  }

  absl::flat_hash_set<std::string> disabled_pass_names(
      debug_options.xla_disable_hlo_passes().begin(),
      debug_options.xla_disable_hlo_passes().end());

  absl::flat_hash_set<std::string> enabled_pass_names(
      debug_options.xla_enable_hlo_passes_only().begin(),
      debug_options.xla_enable_hlo_passes_only().end());

  if (!disabled_pass_names.empty()) {
    VLOG(1) << "Passes disabled by --xla_disable_hlo_passes: "
            << absl::StrJoin(disabled_pass_names, ", ");
  }

  if (!enabled_pass_names.empty()) {
    VLOG(1) << "Passes enabled by --xla_enable_hlo_passes_only: "
            << absl::StrJoin(enabled_pass_names, ", ");
  }

  CHECK(disabled_pass_names.empty() || enabled_pass_names.empty());

  std::vector<HloPassInterface*> enabled_passes;
  if (!enabled_pass_names.empty()) {
    for (auto& pass : passes_) {
      if (enabled_pass_names.contains(pass->name())) {
        enabled_passes.push_back(pass.get());
      }
    }
  } else {
    for (auto& pass : passes_) {
      if (!disabled_pass_names.contains(pass->name())) {
        enabled_passes.push_back(pass.get());
      }
    }
  }
  return enabled_passes;
}

void HloPassPipeline::MaybeDumpHloAndSaveFilenames(
    HloModule& module, absl::string_view after_pass_name,
    absl::string_view before_pass_name) {
   std::vector<std::string> mht_11_v;
   mht_11_v.push_back("after_pass_name: \"" + std::string(after_pass_name.data(), after_pass_name.size()) + "\"");
   mht_11_v.push_back("before_pass_name: \"" + std::string(before_pass_name.data(), before_pass_name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_pass_pipelineDTcc mht_11(mht_11_v, 461, "", "./tensorflow/compiler/xla/service/hlo_pass_pipeline.cc", "HloPassPipeline::MaybeDumpHloAndSaveFilenames");

  for (const std::string& filename : DumpHloModuleBetweenPassesIfEnabled(
           name(), before_pass_name, after_pass_name, module)) {
    Status status = module.metadata()->add_current_pass_dump_filename(filename);
    if (!status.ok()) {
      LOG(FATAL) << status;
    }
  }
}

void HloPassPipeline::MaybeDumpHloAndSaveFilenames(
    HloModuleGroup& module_group, absl::string_view after_pass_name,
    absl::string_view before_pass_name) {
   std::vector<std::string> mht_12_v;
   mht_12_v.push_back("after_pass_name: \"" + std::string(after_pass_name.data(), after_pass_name.size()) + "\"");
   mht_12_v.push_back("before_pass_name: \"" + std::string(before_pass_name.data(), before_pass_name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_pass_pipelineDTcc mht_12(mht_12_v, 478, "", "./tensorflow/compiler/xla/service/hlo_pass_pipeline.cc", "HloPassPipeline::MaybeDumpHloAndSaveFilenames");

  for (HloModule* module : module_group.modules()) {
    MaybeDumpHloAndSaveFilenames(*module, after_pass_name, before_pass_name);
  }
}

StatusOr<bool> HloPassPipeline::Run(HloModule* module) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_pass_pipelineDTcc mht_13(mht_13_v, 487, "", "./tensorflow/compiler/xla/service/hlo_pass_pipeline.cc", "HloPassPipeline::Run");

  run_called_ = true;

  VLOG(1) << "Running HLO pass pipeline on module " << module->name() << ": "
          << name();

  return RunPassesInternal(module, module->config().debug_options());
}

StatusOr<bool> HloPassPipeline::RunOnModuleGroup(HloModuleGroup* module_group) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_pass_pipelineDTcc mht_14(mht_14_v, 499, "", "./tensorflow/compiler/xla/service/hlo_pass_pipeline.cc", "HloPassPipeline::RunOnModuleGroup");

  run_called_ = true;

  VLOG(1) << "Running HLO pass pipeline on module group "
          << module_group->name() << ": " << name();

  if (module_group->modules().empty()) {
    VLOG(1) << "Module group is empty. Nothing to do.";
    return false;
  }

  return RunPassesInternal(module_group,
                           module_group->module(0).config().debug_options());
}

}  // namespace xla
