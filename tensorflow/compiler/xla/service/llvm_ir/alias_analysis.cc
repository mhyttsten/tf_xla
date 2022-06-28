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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSalias_analysisDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSalias_analysisDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSalias_analysisDTcc() {
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

#include "tensorflow/compiler/xla/service/llvm_ir/alias_analysis.h"

#include <map>

#include "absl/container/flat_hash_set.h"
#include "llvm/IR/MDBuilder.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_type_conversion_util.h"
#include "tensorflow/compiler/xla/service/logical_buffer.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {
namespace llvm_ir {

// Sentry allocation used to represent parameters of the entry computation in
// alias_scope_metadata_ and noalias_metadata_.
static const BufferAllocation* kParameterAllocation = new BufferAllocation(
    /*index=*/-1, /*size=*/0, LogicalBuffer::Color(0));

void AliasAnalysis::AddAliasingInformationToIrArray(const HloInstruction& hlo,
                                                    llvm_ir::IrArray* array,
                                                    const ShapeIndex& index) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSalias_analysisDTcc mht_0(mht_0_v, 205, "", "./tensorflow/compiler/xla/service/llvm_ir/alias_analysis.cc", "AliasAnalysis::AddAliasingInformationToIrArray");

  BufferAllocation::Slice buffer_slice;
  if (hlo.opcode() == HloOpcode::kParameter &&
      hlo.parent() == module_.entry_computation()) {
    // Entry computation parameters may alias with each other but may not alias
    // with our temporary buffers.
    buffer_slice = BufferAllocation::Slice(kParameterAllocation, 0, 0);
  } else {
    auto unique_slice = assignment_.GetUniqueSlice(&hlo, index);
    if (!unique_slice.ok()) {
      // Skip HLOs which don't have a buffer assigned or for which the
      // buffer can't be determined statically. We cannot determine their
      // aliasing properties in these cases.
      return;
    }
    buffer_slice = unique_slice.ValueOrDie();
  }

  if (module_.config().debug_options().xla_llvm_enable_alias_scope_metadata()) {
    llvm::MDNode*& alias_scope_md = alias_scope_metadata_[buffer_slice];
    if (alias_scope_md == nullptr) {
      alias_scope_md =
          GetAliasScopeMetadataForBuffer(buffer_slice, GetAliasDomain());
    }
    if (alias_scope_md != nullptr) {
      array->AddAliasScopeMetadata(alias_scope_md);
    }
  }

  if (module_.config().debug_options().xla_llvm_enable_noalias_metadata()) {
    llvm::MDNode*& noalias_md = noalias_metadata_[{buffer_slice, &hlo}];
    if (noalias_md == nullptr) {
      noalias_md = GetNoaliasMetadataForBuffer(buffer_slice, GetAliasDomain(),
                                               assignment_, hlo);
    }
    if (noalias_md != nullptr) {
      array->AddNoaliasMetadata(noalias_md);
    }
  }

  if (module_.config()
          .debug_options()
          .xla_llvm_enable_invariant_load_metadata()) {
    // Parameters of the entry computation are never stored to, loading from a
    // parameter pointer should always return the same result within a loop.
    if (hlo.opcode() == HloOpcode::kParameter &&
        hlo.parent() == module_.entry_computation()) {
      array->MarkInvariantOverWholeProgram(context_);
    }
  }
}

llvm::MDNode* AliasAnalysis::GetAliasDomain() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSalias_analysisDTcc mht_1(mht_1_v, 260, "", "./tensorflow/compiler/xla/service/llvm_ir/alias_analysis.cc", "AliasAnalysis::GetAliasDomain");

  llvm::MDBuilder metadata_builder(*context_);
  if (alias_domain_ == nullptr) {
    // We use createAliasScopeDomain rather than createAnonymousAliasScopeDomain
    // so that when functions get inlined, we continue using the one domain,
    // rather than duplicating it (and thus having two AA domains in one
    // function).
    //
    // A side-effect of this is that if you ever compile two HLO modules in the
    // same LLVM module, they'll have the same alias scope domain.  This isn't a
    // problem because the two HLO modules will never interact with one another.
    alias_domain_ =
        metadata_builder.createAliasScopeDomain("XLA global AA domain");
  }
  return alias_domain_;
}

llvm::MDNode* AliasAnalysis::GetAliasScopeMetadataForBuffer(
    const BufferAllocation::Slice& buffer_slice, llvm::MDNode* domain) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSalias_analysisDTcc mht_2(mht_2_v, 281, "", "./tensorflow/compiler/xla/service/llvm_ir/alias_analysis.cc", "AliasAnalysis::GetAliasScopeMetadataForBuffer");

  // While we could synthesize an alias.scope, doing so is not more profitable
  // than LLVM's default behavior.
  if (buffer_slice.allocation() == kParameterAllocation) {
    return nullptr;
  }

  llvm::MDBuilder metadata_builder(domain->getContext());
  llvm::MDNode* scope = metadata_builder.createAliasScope(
      "buffer: " + buffer_slice.ToString(), domain);
  llvm::MDNode* scope_list = llvm::MDNode::get(domain->getContext(), scope);
  return scope_list;
}

llvm::MDNode* AliasAnalysis::GetNoaliasMetadataForBuffer(
    const BufferAllocation::Slice& buffer_slice, llvm::MDNode* domain,
    const BufferAssignment& assignment, const HloInstruction& hlo) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSalias_analysisDTcc mht_3(mht_3_v, 300, "", "./tensorflow/compiler/xla/service/llvm_ir/alias_analysis.cc", "AliasAnalysis::GetNoaliasMetadataForBuffer");

  // We want to construct a list of buffers which:
  //
  // 1. Do not alias the given buffer.
  // 2. Will plausibly be used in the vicinity of the given buffer.
  //
  // Making the noalias set overly large will result in either a massive
  // slowdown in LLVM or LLVM will just ignore the noalias set.
  //
  // A plausible list of instructions are:
  // 1. Users of the given hlo.
  // 2. Operands of users of the given hlo.
  // 3. Operands of the given hlo.
  //
  // This set can be increased as we need.
  std::vector<const HloValue*> worklist;
  absl::flat_hash_set<const HloInstruction*> added_to_worklist;
  auto add_buffers_to_worklist =
      [&](const HloInstruction* instruction) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSalias_analysisDTcc mht_4(mht_4_v, 321, "", "./tensorflow/compiler/xla/service/llvm_ir/alias_analysis.cc", "lambda");

        // Buffers of parameters cannot be added to the noalias set.
        if (instruction->opcode() == HloOpcode::kParameter) {
          return;
        }
        if (added_to_worklist.contains(instruction)) {
          return;
        }
        added_to_worklist.insert(instruction);
        ShapeUtil::ForEachSubshape(
            instruction->shape(),
            [&](const Shape& /*shape*/, const ShapeIndex& index) {
              for (const HloValue* buffer :
                   assignment.GetSourceBuffers(instruction, index)) {
                if (assignment.HasAllocation(*buffer)) {
                  worklist.push_back(buffer);
                }
              }
            });
      };

  for (HloInstruction* user : hlo.users()) {
    add_buffers_to_worklist(user);
    for (HloInstruction* operand : user->operands()) {
      add_buffers_to_worklist(operand);
    }
  }

  add_buffers_to_worklist(&hlo);
  for (HloInstruction* operand : hlo.operands()) {
    add_buffers_to_worklist(operand);
  }

  std::set<BufferAllocation::Slice> buffers;
  for (const HloValue* buffer : worklist) {
    const BufferAllocation::Slice noalias_slice =
        assignment.GetAssignedAllocation(*buffer).GetSlice(*buffer);
    // Our buffer must not overlap with the noalias slice.
    if (!buffer_slice.OverlapsWith(noalias_slice)) {
      buffers.insert(noalias_slice);
      // Some instructions have too many operands, causing the noalias set to be
      // too large. To reduce compilation time (b/31901575), truncate noalias
      // sets to at most 500 elements.
      //
      // Future work: improvements to LLVM's scoped AA that avoid creating a
      // MDNode set for every alias query can help to reduce the compilation
      // time as well.
      constexpr int kMaxNoAliasSetSize = 500;
      if (buffers.size() >= kMaxNoAliasSetSize) {
        break;
      }
    }
  }

  // Don't bother constructing a noalias metadata node if it would be empty.
  if (buffers.empty()) {
    return nullptr;
  }

  llvm::MDBuilder metadata_builder(domain->getContext());
  std::vector<llvm::Metadata*> scopes;
  for (const BufferAllocation::Slice noalias_slice : buffers) {
    llvm::MDNode* scope = metadata_builder.createAliasScope(
        "buffer: " + noalias_slice.ToString(), domain);
    scopes.push_back(scope);
  }
  llvm::MDNode* noalias_list =
      llvm::MDNode::get(domain->getContext(), AsArrayRef(scopes));
  return noalias_list;
}

}  // namespace llvm_ir
}  // namespace xla
