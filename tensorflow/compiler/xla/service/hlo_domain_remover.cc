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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_domain_removerDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_domain_removerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_domain_removerDTcc() {
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

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/hlo_domain_remover.h"

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_domain_map.h"
#include "tensorflow/compiler/xla/service/hlo_domain_verifier.h"
#include "tensorflow/compiler/xla/service/hlo_graph_dumper.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {

class HloDomainRemover::RunContext {
 public:
  RunContext(HloModule* module, HloDomainRemover* remover)
      : module_(module), remover_(remover) {}

  StatusOr<bool> Run();

 private:
  // Verifies the consistency of the domain, and normalizes the instructions
  // within it.
  Status VerifyAndNormalizeDomain(const DomainMetadata::Domain& domain);

  HloModule* module_;
  HloDomainRemover* remover_;
};

Status HloDomainRemover::RunContext::VerifyAndNormalizeDomain(
    const DomainMetadata::Domain& domain) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_domain_removerDTcc mht_0(mht_0_v, 214, "", "./tensorflow/compiler/xla/service/hlo_domain_remover.cc", "HloDomainRemover::RunContext::VerifyAndNormalizeDomain");

  TF_ASSIGN_OR_RETURN(const DomainMetadata* ref_metadata,
                      HloDomainVerifier::VerifyDomain(domain));
  if (ref_metadata != nullptr) {
    VLOG(4) << "Applying domain normalization: " << ref_metadata->ToString();
    TF_RETURN_IF_ERROR(remover_->normalizer_(domain, ref_metadata));
  } else {
    // No kDomain instruction was present within this domain, so call the
    // generic normalization functions and have them apply their heuristic.
    VLOG(2) << "Applying domain-less normalization";
    TF_RETURN_IF_ERROR(remover_->normalizer_(domain, nullptr));
  }
  return Status::OK();
}

StatusOr<bool> HloDomainRemover::RunContext::Run() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_domain_removerDTcc mht_1(mht_1_v, 232, "", "./tensorflow/compiler/xla/service/hlo_domain_remover.cc", "HloDomainRemover::RunContext::Run");

  VLOG(4) << "Processing metadata domain: '" << remover_->kind_ << "'";
  int64_t removed_domains = 0;
  for (HloComputation* computation : module_->computations()) {
    // First create the domain instruction sets. A domain instruction set is
    // the set of instructions whose edges never cross a kDomain instruction.
    TF_ASSIGN_OR_RETURN(std::unique_ptr<HloDomainMap> domain_map,
                        HloDomainMap::Create(computation, remover_->kind_));
    // Verify and normalize every domain populated within the map.
    for (auto& domain : domain_map->GetDomains()) {
      TF_RETURN_IF_ERROR(VerifyAndNormalizeDomain(*domain));
    }

    // Now remove all the kDomain instructions of the kind specified by the
    // remover, that are within the currently processed computation from the
    // graph.
    for (HloInstruction* instruction :
         computation->MakeInstructionPostOrder()) {
      for (HloInstruction* operand : instruction->unique_operands()) {
        if (domain_map->IsDomainInstruction(operand)) {
          VLOG(5) << "Removing " << operand->name();
          TF_RETURN_IF_ERROR(
              operand->ReplaceAllUsesWith(operand->mutable_operand(0)));
          TF_RETURN_IF_ERROR(computation->RemoveInstruction(operand));
          ++removed_domains;
        }
      }
    }
    HloInstruction* root = computation->root_instruction();
    if (root != nullptr && domain_map->IsDomainInstruction(root)) {
      VLOG(5) << "Removing " << root->name();
      computation->set_root_instruction(root->mutable_operand(0));
      TF_RETURN_IF_ERROR(computation->RemoveInstruction(root));
      ++removed_domains;
    }
  }
  VLOG(3) << "Removed " << removed_domains << " kDomain instructions of '"
          << remover_->kind_ << "' kind";
  return removed_domains > 0;
}

StatusOr<int64_t> HloDomainRemover::RemoveExitDomains(
    HloInstruction* instruction, absl::string_view domain_kind) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("domain_kind: \"" + std::string(domain_kind.data(), domain_kind.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_domain_removerDTcc mht_2(mht_2_v, 278, "", "./tensorflow/compiler/xla/service/hlo_domain_remover.cc", "HloDomainRemover::RemoveExitDomains");

  int64_t removed_domains = 0;
  HloComputation* computation = instruction->parent();
  // Make a const copy of instruction's users to loop through later, as the
  // users vector could be changed during the loop(e.g. ReplaceAllUsesWith).
  const std::vector<HloInstruction*> users(instruction->users());
  for (HloInstruction* user : users) {
    if (user->opcode() == HloOpcode::kDomain &&
        user->user_side_metadata().Kind() == domain_kind &&
        user->operand_side_metadata().Kind() == domain_kind) {
      VLOG(5) << "Removing exit domain " << user->name();
      TF_RETURN_IF_ERROR(user->ReplaceAllUsesWith(instruction));
      TF_RETURN_IF_ERROR(computation->RemoveInstruction(user));
      ++removed_domains;
    }
  }
  return removed_domains;
}

StatusOr<bool> HloDomainRemover::Run(HloModule* module) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_domain_removerDTcc mht_3(mht_3_v, 300, "", "./tensorflow/compiler/xla/service/hlo_domain_remover.cc", "HloDomainRemover::Run");

  RunContext run_context(module, this);
  return run_context.Run();
}

}  // namespace xla
