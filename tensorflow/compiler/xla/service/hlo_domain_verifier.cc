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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_domain_verifierDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_domain_verifierDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_domain_verifierDTcc() {
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

#include "tensorflow/compiler/xla/service/hlo_domain_verifier.h"

#include <set>

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_domain_map.h"
#include "tensorflow/compiler/xla/service/hlo_graph_dumper.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {

class HloDomainVerifier::RunContext {
 public:
  RunContext(HloModule* module, HloDomainVerifier* verifier)
      : module_(module), verifier_(verifier) {}

  Status Run();

 private:
  // If the verifier caller passed an empty vector for kinds, we collect all the
  // available domain types.
  Status PopulateDomainKinds();

  HloModule* module_;
  HloDomainVerifier* verifier_;
};

Status HloDomainVerifier::RunContext::PopulateDomainKinds() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_domain_verifierDTcc mht_0(mht_0_v, 214, "", "./tensorflow/compiler/xla/service/hlo_domain_verifier.cc", "HloDomainVerifier::RunContext::PopulateDomainKinds");

  if (verifier_->kinds_.empty()) {
    // The caller specified no domain kinds, collect all the ones available.
    std::set<std::string> kinds;
    for (HloComputation* computation : module_->computations()) {
      for (HloInstruction* instruction : computation->instructions()) {
        if (instruction->opcode() == HloOpcode::kDomain) {
          TF_RET_CHECK(instruction->user_side_metadata().Kind() ==
                       instruction->operand_side_metadata().Kind())
              << instruction->ToString();
          kinds.insert(std::string(instruction->user_side_metadata().Kind()));
        }
      }
    }
    verifier_->kinds_.insert(verifier_->kinds_.end(), kinds.begin(),
                             kinds.end());
  }
  return Status::OK();
}

Status HloDomainVerifier::RunContext::Run() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_domain_verifierDTcc mht_1(mht_1_v, 237, "", "./tensorflow/compiler/xla/service/hlo_domain_verifier.cc", "HloDomainVerifier::RunContext::Run");

  VLOG(4) << "Running HLO Domain Verifier";
  TF_RETURN_IF_ERROR(PopulateDomainKinds());
  for (HloComputation* computation : module_->computations()) {
    for (auto& kind : verifier_->kinds_) {
      // First create the domain instruction sets. A domain instruction set is
      // the set of instructions whose edges never cross a kDomain instruction.
      TF_ASSIGN_OR_RETURN(std::unique_ptr<HloDomainMap> domain_map,
                          HloDomainMap::Create(computation, kind));
      // Verify every domain populated within the map.
      for (auto& domain : domain_map->GetDomains()) {
        TF_RETURN_IF_ERROR(VerifyDomain(*domain).status());
      }
    }
  }
  return Status::OK();
}

StatusOr<bool> HloDomainVerifier::Run(HloModule* module) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_domain_verifierDTcc mht_2(mht_2_v, 258, "", "./tensorflow/compiler/xla/service/hlo_domain_verifier.cc", "HloDomainVerifier::Run");

  RunContext run_context(module, this);
  TF_RETURN_IF_ERROR(run_context.Run());
  return false;
}

StatusOr<const DomainMetadata*> HloDomainVerifier::VerifyDomain(
    const DomainMetadata::Domain& domain) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_domain_verifierDTcc mht_3(mht_3_v, 268, "", "./tensorflow/compiler/xla/service/hlo_domain_verifier.cc", "HloDomainVerifier::VerifyDomain");

  const DomainMetadata* ref_metadata = nullptr;
  VLOG(4) << "Reach set:";
  for (HloInstruction* instruction : domain.instructions) {
    VLOG(4) << "  " << instruction->name();
  }
  VLOG(4) << "  Domains:";
  for (HloInstruction* instruction : domain.enter_domains) {
    const DomainMetadata& meta = instruction->user_side_metadata();
    VLOG(4) << "    User side: " << instruction->name();
    VLOG(4) << "      " << meta.ToString();
    if (ref_metadata == nullptr) {
      ref_metadata = &meta;
    } else {
      TF_RET_CHECK(meta.Matches(*ref_metadata))
          << "Metadata mismatch at instruction " << instruction->name() << " : "
          << meta.ToString() << " vs " << ref_metadata->ToString();
    }
  }
  for (HloInstruction* instruction : domain.exit_domains) {
    const DomainMetadata& meta = instruction->operand_side_metadata();
    VLOG(4) << "    Operand side: " << instruction->name();
    VLOG(4) << "      " << meta.ToString();
    if (ref_metadata == nullptr) {
      ref_metadata = &meta;
    } else {
      TF_RET_CHECK(meta.Matches(*ref_metadata))
          << "Metadata mismatch at instruction " << instruction->name() << " : "
          << meta.ToString() << " vs " << ref_metadata->ToString();
    }
  }
  return ref_metadata;
}

}  // namespace xla
