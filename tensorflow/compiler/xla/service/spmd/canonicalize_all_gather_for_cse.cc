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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSspmdPScanonicalize_all_gather_for_cseDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSspmdPScanonicalize_all_gather_for_cseDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSspmdPScanonicalize_all_gather_for_cseDTcc() {
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

#include "tensorflow/compiler/xla/service/spmd/canonicalize_all_gather_for_cse.h"

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_query.h"

namespace xla {

StatusOr<bool> CanonicalizeAllGatherForCSE::RunOnComputation(
    HloComputation* comp) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSspmdPScanonicalize_all_gather_for_cseDTcc mht_0(mht_0_v, 197, "", "./tensorflow/compiler/xla/service/spmd/canonicalize_all_gather_for_cse.cc", "CanonicalizeAllGatherForCSE::RunOnComputation");

  bool changed = false;
  // Helper to find the respective shape input dimension of an shape output
  // dimension of a reshape.
  std::vector<HloInstruction*> ordered_hlos = comp->MakeInstructionPostOrder();
  for (HloInstruction* hlo : ordered_hlos) {
    HloAllGatherInstruction* ag = DynCast<HloAllGatherInstruction>(hlo);

    // TODO(cjfj): Support all-gathers with more than one operand.
    if (!ag || ag->operand_count() > 1) {
      continue;
    }

    // Also only do this for degenerate dimension sizes as the additional
    // reshaping may not be worth the potential for CSE.
    HloInstruction* real_data = ag->mutable_operand(0);
    while (std::get<0>(
        real_data->ReshapeMerelyInsertsOrDeletes1SizedDimensions())) {
      real_data = real_data->mutable_operand(0);
    }

    if (real_data == ag->operand(0)) {
      continue;
    }

    const int64_t ag_dim = ag->all_gather_dimension();
    int64_t new_ag_dim;
    if (auto dims = ShapeUtil::ReshapeLeavesDimensionsUnmodified(
            ag->operand(0)->shape(), real_data->shape(), {ag_dim})) {
      new_ag_dim = dims->at(0);
    } else {
      int64_t major_elements =
          Product(absl::MakeConstSpan(ag->operand(0)->shape().dimensions())
                      .subspan(0, ag_dim));
      new_ag_dim = 0;
      while (major_elements > 1) {
        major_elements /= real_data->shape().dimensions(new_ag_dim++);
      }
    }
    if (new_ag_dim == real_data->shape().rank()) {
      continue;
    }

    const int64_t all_gather_participants =
        ShapeUtil::ElementsIn(ag->shape()) /
        ShapeUtil::ElementsIn(ag->operand(0)->shape());
    Shape new_ag_shape = real_data->shape();
    new_ag_shape.set_dimensions(
        new_ag_dim,
        all_gather_participants * new_ag_shape.dimensions(new_ag_dim));
    absl::optional<int64_t> new_channel_id =
        ag->channel_id() ? absl::make_optional(this->NextChannelId())
                         : absl::nullopt;
    HloInstruction* new_ag =
        comp->AddInstruction(HloInstruction::CreateAllGather(
            new_ag_shape, {real_data}, /*all_gather_dimension=*/new_ag_dim,
            ag->replica_groups(), ag->constrain_layout(), new_channel_id,
            ag->use_global_device_ids()));
    ag->SetupDerivedInstruction(new_ag);
    HloInstruction* new_formatting = comp->AddInstruction(
        HloInstruction::CreateReshape(ag->shape(), new_ag));
    TF_RETURN_IF_ERROR(comp->ReplaceInstruction(ag, new_formatting));
    changed = true;
  }
  return changed;
}

StatusOr<bool> CanonicalizeAllGatherForCSE::Run(HloModule* module) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSspmdPScanonicalize_all_gather_for_cseDTcc mht_1(mht_1_v, 267, "", "./tensorflow/compiler/xla/service/spmd/canonicalize_all_gather_for_cse.cc", "CanonicalizeAllGatherForCSE::Run");

  bool changed = false;
  next_channel_id_ = hlo_query::NextChannelId(*module);
  for (HloComputation* comp : module->computations()) {
    TF_ASSIGN_OR_RETURN(bool comp_changed, RunOnComputation(comp));
    changed |= comp_changed;
  }
  return changed;
}

}  // namespace xla
