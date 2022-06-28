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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_SHARDING_METADATA_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_SHARDING_METADATA_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_sharding_metadataDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_sharding_metadataDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_sharding_metadataDTh() {
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


#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/service/hlo_domain_metadata.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_sharding.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {

// A DomainMetadata implementation that internally wraps a sharding attribute.
class ShardingMetadata : public DomainMetadata {
 public:
  explicit ShardingMetadata(std::shared_ptr<const HloSharding> sharding)
      : sharding_(std::move(sharding)) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_sharding_metadataDTh mht_0(mht_0_v, 204, "", "./tensorflow/compiler/xla/service/hlo_sharding_metadata.h", "ShardingMetadata");
}

  std::unique_ptr<DomainMetadata> Clone() const override;

  absl::string_view Kind() const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_sharding_metadataDTh mht_1(mht_1_v, 211, "", "./tensorflow/compiler/xla/service/hlo_sharding_metadata.h", "Kind");
 return KindName(); }

  bool Matches(const DomainMetadata& other) const override;

  template <typename H>
  friend H AbslHashValue(H h, const ShardingMetadata& sharding_metadata) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_sharding_metadataDTh mht_2(mht_2_v, 219, "", "./tensorflow/compiler/xla/service/hlo_sharding_metadata.h", "AbslHashValue");

    const bool has_sharding = sharding_metadata.sharding_ != nullptr;
    if (has_sharding) {
      h = H::combine(std::move(h), *sharding_metadata.sharding_);
    }
    return H::combine(std::move(h), has_sharding);
  }

  size_t Hash() const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_sharding_metadataDTh mht_3(mht_3_v, 230, "", "./tensorflow/compiler/xla/service/hlo_sharding_metadata.h", "Hash");
 return absl::HashOf(*this); }

  std::string ToString() const override;

  const HloSharding* sharding() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_sharding_metadataDTh mht_4(mht_4_v, 237, "", "./tensorflow/compiler/xla/service/hlo_sharding_metadata.h", "sharding");
 return sharding_.get(); }

  static absl::string_view KindName() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_sharding_metadataDTh mht_5(mht_5_v, 242, "", "./tensorflow/compiler/xla/service/hlo_sharding_metadata.h", "KindName");
 return "sharding"; }

  static StatusOr<const ShardingMetadata*> ToShardingMetadata(
      const DomainMetadata* metadata);

  // Apply the specified domain metadata onto the specified domain. If no
  // metadata is specified then apply sharding heuristics and normalize the
  // instructions whose sharding deviates from the one which is inferred as to
  // be the original one. Policy wise, HLO passes are allowed to create new
  // unassigned instructions, but if they do create assigned ones, they have to
  // conform to the ones around.
  static Status NormalizeShardingDomain(const DomainMetadata::Domain& domain,
                                        const DomainMetadata* metadata);

 private:
  std::shared_ptr<const HloSharding> sharding_;
};

// If the sharding between root and instruction changes then returns a
// ShardingMetadata based kDomain instruction what can be used to separate
// operand and instruction.
// Returns nullptr if there is no need for a domain separation.
class ShardingDomainCreator {
 public:
  HloInstruction* operator()(HloInstruction* instruction, HloInstruction* root,
                             HloInstruction* operand);

 private:
  // Map from instruction and user sharding to domain users to CSE identical
  // domains.
  struct DomainCseMapKey {
    const HloInstruction* instruction;
    std::shared_ptr<const HloSharding> sharding;

    bool operator==(const DomainCseMapKey& other) const;

    template <typename H>
    friend H AbslHashValue(H h, const DomainCseMapKey& key) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_sharding_metadataDTh mht_6(mht_6_v, 282, "", "./tensorflow/compiler/xla/service/hlo_sharding_metadata.h", "AbslHashValue");

      h = H::combine(std::move(h), key.instruction);
      const bool has_sharding = key.sharding != nullptr;
      if (has_sharding) {
        h = H::combine(std::move(h), *key.sharding);
      }
      return H::combine(std::move(h), has_sharding);
    }
  };
  absl::flat_hash_map<DomainCseMapKey, HloInstruction*> domain_cse_map_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_SHARDING_METADATA_H_
