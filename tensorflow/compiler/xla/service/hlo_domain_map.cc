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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_domain_mapDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_domain_mapDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_domain_mapDTcc() {
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

#include "tensorflow/compiler/xla/service/hlo_domain_map.h"

#include <algorithm>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {

/* static */ StatusOr<std::unique_ptr<HloDomainMap>> HloDomainMap::Create(
    HloComputation* computation, std::string domain_kind) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("domain_kind: \"" + domain_kind + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_domain_mapDTcc mht_0(mht_0_v, 202, "", "./tensorflow/compiler/xla/service/hlo_domain_map.cc", "HloDomainMap::Create");

  auto domain_map = absl::WrapUnique(new HloDomainMap(std::move(domain_kind)));
  TF_RETURN_IF_ERROR(domain_map->Populate(computation));
  return std::move(domain_map);
}

/* static */ StatusOr<std::unique_ptr<HloDomainMap>> HloDomainMap::Create(
    HloModule* module, std::string domain_kind) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("domain_kind: \"" + domain_kind + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_domain_mapDTcc mht_1(mht_1_v, 213, "", "./tensorflow/compiler/xla/service/hlo_domain_map.cc", "HloDomainMap::Create");

  auto domain_map = absl::WrapUnique(new HloDomainMap(std::move(domain_kind)));
  for (HloComputation* computation : module->computations()) {
    TF_RETURN_IF_ERROR(domain_map->Populate(computation));
  }
  return std::move(domain_map);
}

bool HloDomainMap::InSameDomain(const HloInstruction* instruction1,
                                const HloInstruction* instruction2) const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_domain_mapDTcc mht_2(mht_2_v, 225, "", "./tensorflow/compiler/xla/service/hlo_domain_map.cc", "HloDomainMap::InSameDomain");

  int64_t domain_id1 = GetDomainId(instruction1);
  int64_t domain_id2 = GetDomainId(instruction2);
  return domain_id1 >= 0 && domain_id1 == domain_id2;
}

int64_t HloDomainMap::GetDomainId(const HloInstruction* instruction) const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_domain_mapDTcc mht_3(mht_3_v, 234, "", "./tensorflow/compiler/xla/service/hlo_domain_map.cc", "HloDomainMap::GetDomainId");

  return FindOrDefault(instruction_to_domain_, instruction, -1);
}

int64_t HloDomainMap::GetDomainMetadataId(
    const HloInstruction* instruction) const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_domain_mapDTcc mht_4(mht_4_v, 242, "", "./tensorflow/compiler/xla/service/hlo_domain_map.cc", "HloDomainMap::GetDomainMetadataId");

  return FindOrDie(domain_metadata_id_, instruction);
}

Status HloDomainMap::TryProcessEmptyDomain(HloInstruction* instruction) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_domain_mapDTcc mht_5(mht_5_v, 249, "", "./tensorflow/compiler/xla/service/hlo_domain_map.cc", "HloDomainMap::TryProcessEmptyDomain");

  TF_RET_CHECK(instruction->opcode() == HloOpcode::kDomain);
  // We only check operands, so we are sure to not process the empty domain from
  // both sides.
  for (HloInstruction* operand : instruction->unique_operands()) {
    if (IsDomainInstruction(operand)) {
      auto domain = absl::make_unique<DomainMetadata::Domain>();
      domain->enter_domains.insert(operand);
      domain->exit_domains.insert(instruction);
      TF_RETURN_IF_ERROR(InsertDomain(std::move(domain)));
    }
  }
  if (instruction == instruction->parent()->root_instruction()) {
    auto domain = absl::make_unique<DomainMetadata::Domain>();
    domain->enter_domains.insert(instruction);
    TF_RETURN_IF_ERROR(InsertDomain(std::move(domain)));
  }
  return Status::OK();
}

Status HloDomainMap::Populate(HloComputation* computation) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_domain_mapDTcc mht_6(mht_6_v, 272, "", "./tensorflow/compiler/xla/service/hlo_domain_map.cc", "HloDomainMap::Populate");

  InstructionOrderMap instructions_post_order;
  int64_t count = 0;
  for (HloInstruction* instruction : computation->MakeInstructionPostOrder()) {
    instructions_post_order.insert(std::make_pair(instruction, count++));
  }
  for (HloInstruction* instruction : computation->instructions()) {
    if (IsDomainInstruction(instruction)) {
      // If this is a kDomain of the kind we are currently processing, check
      // whether this is an "empty domain".
      TF_RETURN_IF_ERROR(TryProcessEmptyDomain(instruction));
      continue;
    }
    int64_t domain_id = FindOrDefault(instruction_to_domain_, instruction, -1);
    if (domain_id >= 0) {
      // We have already processed this instruction.
      continue;
    }
    TF_ASSIGN_OR_RETURN(std::unique_ptr<DomainMetadata::Domain> domain,
                        CreateDomain(instruction, instructions_post_order));
    TF_RETURN_IF_ERROR(InsertDomain(std::move(domain)));
  }
  TF_RETURN_IF_ERROR(PopulateDomainMetadataMap());
  return Status::OK();
}

Status HloDomainMap::PopulateDomainMetadataMap() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_domain_mapDTcc mht_7(mht_7_v, 301, "", "./tensorflow/compiler/xla/service/hlo_domain_map.cc", "HloDomainMap::PopulateDomainMetadataMap");

  auto hash = [](const DomainMetadata* m) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_domain_mapDTcc mht_8(mht_8_v, 305, "", "./tensorflow/compiler/xla/service/hlo_domain_map.cc", "lambda");
 return m->Hash(); };
  auto equal = [](const DomainMetadata* a, const DomainMetadata* b) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_domain_mapDTcc mht_9(mht_9_v, 309, "", "./tensorflow/compiler/xla/service/hlo_domain_map.cc", "lambda");

    return a->Matches(*b);
  };
  absl::flat_hash_map<const DomainMetadata*, int64_t, decltype(hash),
                      decltype(equal)>
      domain_metadata(1024, hash, equal);

  for (auto& domain : instruction_domains_) {
    int64_t domain_metadata_id = -1;
    if (!domain->enter_domains.empty()) {
      const HloInstruction* domain_instruction = *domain->enter_domains.begin();
      domain_metadata_id =
          domain_metadata
              .insert({&domain_instruction->user_side_metadata(),
                       domain_metadata.size() + 1})
              .first->second;
    } else if (!domain->exit_domains.empty()) {
      const HloInstruction* domain_instruction = *domain->exit_domains.begin();
      domain_metadata_id =
          domain_metadata
              .insert({&domain_instruction->operand_side_metadata(),
                       domain_metadata.size() + 1})
              .first->second;
    } else {
      domain_metadata_id = 0;
    }
    TF_RET_CHECK(domain_metadata_id >= 0);
    for (HloInstruction* instruction : domain->instructions) {
      domain_metadata_id_[instruction] = domain_metadata_id;
    }
  }
  return Status::OK();
}

Status HloDomainMap::InsertDomain(
    std::unique_ptr<DomainMetadata::Domain> domain) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_domain_mapDTcc mht_10(mht_10_v, 347, "", "./tensorflow/compiler/xla/service/hlo_domain_map.cc", "HloDomainMap::InsertDomain");

  int64_t domain_id = instruction_domains_.size();
  instruction_domains_.push_back(std::move(domain));
  for (HloInstruction* instruction : instruction_domains_.back()->reach_set) {
    instruction_to_domain_[instruction] = domain_id;
  }
  return Status::OK();
}

Status HloDomainMap::ExpandDomain(HloInstruction* instruction,
                                  DomainMetadata::Domain* domain) const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_domain_mapDTcc mht_11(mht_11_v, 360, "", "./tensorflow/compiler/xla/service/hlo_domain_map.cc", "HloDomainMap::ExpandDomain");

  std::vector<HloInstruction*> in_queue;
  in_queue.push_back(instruction);
  while (!in_queue.empty()) {
    HloInstruction* current_instruction = in_queue.back();
    in_queue.pop_back();
    if (domain->reach_set.insert(current_instruction).second) {
      // We should not be finding instructions with assigned domain here.
      // If we assigned a domain to the instruction, it means that all the
      // instructions reached by it, should have a domain as well.
      int64_t domain_id =
          FindOrDefault(instruction_to_domain_, current_instruction, -1);
      TF_RET_CHECK(domain_id < 0)
          << "Instruction " << current_instruction->ToString()
          << " already has domain " << domain_id;
      for (HloInstruction* operand : current_instruction->operands()) {
        if (IsDomainInstruction(operand)) {
          // The reach set instruction is a user of the domain instruction
          // (the instruction sees the kDomain as operand).
          // IOW the dataflow enters the domain through the kDomain instruction.
          domain->enter_domains.insert(operand);
        } else {
          in_queue.push_back(operand);
        }
      }
      for (HloInstruction* user : current_instruction->users()) {
        if (IsDomainInstruction(user)) {
          // The reach set instruction is an operand of the domain instruction
          // (the instruction sees the kDomain as user).
          // IOW the dataflow exits the domain through the kDomain instruction.
          domain->exit_domains.insert(user);
        } else {
          in_queue.push_back(user);
        }
      }
    }
  }
  return Status::OK();
}

StatusOr<std::unique_ptr<DomainMetadata::Domain>> HloDomainMap::CreateDomain(
    HloInstruction* instruction,
    const InstructionOrderMap& instructions_order) const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_domain_mapDTcc mht_12(mht_12_v, 405, "", "./tensorflow/compiler/xla/service/hlo_domain_map.cc", "HloDomainMap::CreateDomain");

  auto domain = absl::make_unique<DomainMetadata::Domain>();
  TF_RETURN_IF_ERROR(ExpandDomain(instruction, domain.get()));
  domain->instructions =
      MakeNonDomainInstructions(domain->reach_set, instructions_order);
  return std::move(domain);
}

bool HloDomainMap::IsDomainInstruction(
    const HloInstruction* instruction) const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_domain_mapDTcc mht_13(mht_13_v, 417, "", "./tensorflow/compiler/xla/service/hlo_domain_map.cc", "HloDomainMap::IsDomainInstruction");

  if (instruction->opcode() != HloOpcode::kDomain) {
    return false;
  }
  if (!domain_kind_.empty()) {
    if (instruction->user_side_metadata().Kind() != domain_kind_) {
      return false;
    }
    // Both user and operand side of the metadata must be of the same kind.
    CHECK(instruction->operand_side_metadata().Kind() == domain_kind_)
        << "Instruction " << instruction->ToString()
        << " has mismatching metadata kinds";
  }
  return true;
}

/* static */ std::vector<HloInstruction*>
HloDomainMap::MakeNonDomainInstructions(
    const absl::flat_hash_set<HloInstruction*>& instruction_set,
    const InstructionOrderMap& instructions_order) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_domain_mapDTcc mht_14(mht_14_v, 439, "", "./tensorflow/compiler/xla/service/hlo_domain_map.cc", "HloDomainMap::MakeNonDomainInstructions");

  std::vector<HloInstruction*> instructions;
  instructions.reserve(instruction_set.size());
  for (HloInstruction* instruction : instruction_set) {
    if (instruction->opcode() != HloOpcode::kDomain) {
      instructions.push_back(instruction);
    }
  }
  // sort instructions according to instructions_order
  absl::c_sort(instructions,
               [&instructions_order](HloInstruction* a, HloInstruction* b) {
                 return instructions_order.at(a) < instructions_order.at(b);
               });
  return instructions;
}

}  // namespace xla
