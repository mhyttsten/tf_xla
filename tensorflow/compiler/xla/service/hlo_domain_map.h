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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_DOMAIN_MAP_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_DOMAIN_MAP_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_domain_mapDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_domain_mapDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_domain_mapDTh() {
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


#include <memory>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_domain_metadata.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {

// The HloDomainMap splits a set of instructions within a module or computation,
// into different domains, separated by kDomain instructions.
// A domain is composed by a set of instructions which can reach each other via
// operand/user edges, without crossing a kDomain insutrction of a given kind.
// A domain never crosses computation boundaries.
class HloDomainMap {
 public:
  // Creates a new HloDomainMap, creating all the domains within the input
  // computation, of the given kind. If domain_kind is not empty, only the
  // kDomain instructions of domain_kind will be considered as separators.
  // Otherwise every kDomain instruction will be splitting domains.
  static StatusOr<std::unique_ptr<HloDomainMap>> Create(
      HloComputation* computation, std::string domain_kind);

  // Creates a new HloDomainMap, creating all the domains within the input
  // module, of the given kind. If domain_kind is not empty, only the
  // kDomain instructions of domain_kind will be considered as separators.
  // Otherwise every kDomain instruction will be splitting domains.
  static StatusOr<std::unique_ptr<HloDomainMap>> Create(
      HloModule* module, std::string domain_kind);

  // Retrieves all the domains the input module or computation are composed by.
  const std::vector<std::unique_ptr<DomainMetadata::Domain>>& GetDomains()
      const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_domain_mapDTh mht_0(mht_0_v, 225, "", "./tensorflow/compiler/xla/service/hlo_domain_map.h", "GetDomains");

    return instruction_domains_;
  }

  // Checks whether two instructions are within the same domain.
  bool InSameDomain(const HloInstruction* instruction1,
                    const HloInstruction* instruction2) const;

  // Checks whether instruction is a kDomain instruction of the kind we are
  // currently processing.
  bool IsDomainInstruction(const HloInstruction* instruction) const;

  // Retrieves the domain identifier of the instruction, or -1 in case
  // instruction is not found within any domain.
  int64_t GetDomainId(const HloInstruction* instruction) const;

  // Returns the unique id of the domain metadata for the domain the given
  // instruction belongs to. The given instruction must not be a kDomain
  // instruction since each domain instruction is associated with 2 domains.
  int64_t GetDomainMetadataId(const HloInstruction* instruction) const;

 private:
  // Map used for representing instruction ordering, i.e.
  // order_map[a] < order_map[b] means a must be ordered before b.
  using InstructionOrderMap =
      absl::flat_hash_map<const HloInstruction*, int64_t>;

  HloDomainMap(std::string domain_kind)
      : domain_kind_(std::move(domain_kind)) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("domain_kind: \"" + domain_kind + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_domain_mapDTh mht_1(mht_1_v, 257, "", "./tensorflow/compiler/xla/service/hlo_domain_map.h", "HloDomainMap");
}

  // Check if the kDomain instruction is facing (via its operand link) another
  // kDomain instruction of the same kind, hence defining an empty domain.
  // If that is the case, create the empty domain and call the proper
  // normalizer.
  Status TryProcessEmptyDomain(HloInstruction* instruction);

  Status Populate(HloComputation* computation);

  // Inserts the provided domain into the ones tracked by this object,
  // creating a new domain ID.
  Status InsertDomain(std::unique_ptr<DomainMetadata::Domain> domain);

  // From the given instruction, expands operand and user wise, the set of
  // instructions which can be reached without crossing a kDomain instruction
  // of the kind specified by domain_kind_.
  // The domain data structure will be populated with all the reached
  // instructions, and the boundaries of the domain, with the kDomain
  // instructions encountered while expanding the reach.
  Status ExpandDomain(HloInstruction* instruction,
                      DomainMetadata::Domain* domain) const;

  // Creates a domain data structure using the ExpandDomain() API.
  StatusOr<std::unique_ptr<DomainMetadata::Domain>> CreateDomain(
      HloInstruction* instruction,
      const InstructionOrderMap& instructions_order) const;

  // Out of an instruction set, returns a vector of all the ones which are not
  // a kDomain kind.
  static std::vector<HloInstruction*> MakeNonDomainInstructions(
      const absl::flat_hash_set<HloInstruction*>& instruction_set,
      const InstructionOrderMap& instructions_order);

  // Populates domain_metadata_id_ that maps each HloInstruction to the unique
  // ID of its associated domain metatadata.
  Status PopulateDomainMetadataMap();

  std::string domain_kind_;
  std::vector<std::unique_ptr<DomainMetadata::Domain>> instruction_domains_;
  absl::flat_hash_map<const HloInstruction*, int64_t> instruction_to_domain_;
  absl::flat_hash_map<const HloInstruction*, int64_t> domain_metadata_id_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_DOMAIN_MAP_H_
