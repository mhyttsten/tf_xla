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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_domain_isolatorDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_domain_isolatorDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_domain_isolatorDTcc() {
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

#include "tensorflow/compiler/xla/service/hlo_domain_isolator.h"

#include <cstdint>

#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_domain_remover.h"
#include "tensorflow/compiler/xla/service/hlo_graph_dumper.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_sharding_metadata.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/platform/statusor.h"

namespace xla {

namespace {

// Add domains which are used as users of a specific instruction.
StatusOr<int64_t> AddExitDomains(HloInstruction* instruction,
                                 HloDomainIsolator::DomainCreator* creator) {
  int64_t added_domains = 0;
  if (instruction->opcode() == HloOpcode::kDomain) {
    return added_domains;
  }
  // Make a const copy of instruction's users to loop through later, as the
  // users vector could be changed during the loop
  // (e.g. ReplaceUseWithDifferentShape).
  const std::vector<HloInstruction*> users(instruction->users());
  for (HloInstruction* user : users) {
    // Check whether a kDomain is necessary between user and instruction.
    HloInstruction* domain = (*creator)(user, instruction, instruction);
    if (domain != nullptr) {
      VLOG(4) << "New domain: " << domain->ToString();
      // Call ReplaceUseWithDifferentShape even though the shapes are
      // expected to match to avoid an expensive shape check between the
      // original and the new instruction.
      TF_RETURN_IF_ERROR(
          instruction->ReplaceUseWithDifferentShape(user, domain));
      ++added_domains;
    }
  }
  return added_domains;
}

StatusOr<bool> RunInternal(HloModule* module,
                           HloDomainIsolator::DomainCreator* creator) {
  int64_t added_domains = 0;
  for (HloComputation* computation : module->computations()) {
    // Walk in post order and place all the required kDomain instructions.
    for (HloInstruction* instruction :
         computation->MakeInstructionPostOrder()) {
      if (instruction->opcode() == HloOpcode::kDomain) {
        continue;
      }
      for (HloInstruction* operand : instruction->unique_operands()) {
        // When applying multiple domains, we could end up stacking more than
        // one in one edge, so here we want to build the effective
        // (kDomain-less) instruction->operand edge.
        HloInstruction* root = operand;
        while (root->opcode() == HloOpcode::kDomain) {
          root = root->mutable_operand(0);
        }
        // Check whether a kDomain is necessary between instruction and operand.
        HloInstruction* domain = (*creator)(instruction, root, operand);
        if (domain != nullptr) {
          VLOG(4) << "New domain: " << domain->ToString();
          // Call ReplaceUseWithDifferentShape even though the shapes are
          // expected to match to avoid an expensive shape check between the
          // original and the new instruction.
          TF_RETURN_IF_ERROR(
              operand->ReplaceUseWithDifferentShape(instruction, domain));
          ++added_domains;
        }
      }
    }
  }
  VLOG(3) << "Added " << added_domains << " kDomain instructions";
  return added_domains > 0;
}

}  // namespace

HloDomainIsolator::HloDomainIsolator(DomainCreatorFactory creator_factory)
    : creator_factory_(std::move(creator_factory)) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_domain_isolatorDTcc mht_0(mht_0_v, 268, "", "./tensorflow/compiler/xla/service/hlo_domain_isolator.cc", "HloDomainIsolator::HloDomainIsolator");
}

StatusOr<bool> HloDomainIsolator::UpdateDomains(HloInstruction* instruction) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_domain_isolatorDTcc mht_1(mht_1_v, 273, "", "./tensorflow/compiler/xla/service/hlo_domain_isolator.cc", "HloDomainIsolator::UpdateDomains");

  TF_ASSIGN_OR_RETURN(const int64_t removed_domains,
                      HloDomainRemover::RemoveExitDomains(
                          instruction, ShardingMetadata::KindName()));
  DomainCreator creator = creator_factory_();
  TF_ASSIGN_OR_RETURN(const int64_t added_domains,
                      AddExitDomains(instruction, &creator));
  return removed_domains > 0 || added_domains > 0;
}

StatusOr<bool> HloDomainIsolator::Run(HloModule* module) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_domain_isolatorDTcc mht_2(mht_2_v, 286, "", "./tensorflow/compiler/xla/service/hlo_domain_isolator.cc", "HloDomainIsolator::Run");

  DomainCreator creator = creator_factory_();
  return RunInternal(module, &creator);
}

}  // namespace xla
