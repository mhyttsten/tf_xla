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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScall_inlinerDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScall_inlinerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScall_inlinerDTcc() {
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

#include "tensorflow/compiler/xla/service/call_inliner.h"

#include <deque>

#include "tensorflow/compiler/xla/service/call_graph.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_domain_isolator.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_sharding_metadata.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {
namespace {

// Traverses the callee computation, inlining cloned nodes into the caller
// computation and connecting them to producers/consumers appropriately.
// When the traversal has completed, the provided call instruction is entirely
// replaced in the caller's graph.
class SubcomputationInsertionVisitor : public DfsHloVisitorWithDefault {
 public:
  // call is the call operation -- it will be replaced with the body of the
  // called computation.
  explicit SubcomputationInsertionVisitor(HloInstruction* call)
      : call_(call), outer_(call->parent()) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScall_inlinerDTcc mht_0(mht_0_v, 209, "", "./tensorflow/compiler/xla/service/call_inliner.cc", "SubcomputationInsertionVisitor");

    CHECK_EQ(HloOpcode::kCall, call_->opcode());
  }

  // Resolves the operands to the HLO instruction in the inlined (caller) graph,
  // and clones the HLO instruction into that graph with the new operands.
  Status DefaultAction(HloInstruction* hlo) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScall_inlinerDTcc mht_1(mht_1_v, 218, "", "./tensorflow/compiler/xla/service/call_inliner.cc", "DefaultAction");

    std::vector<HloInstruction*> new_operands;
    for (HloInstruction* operand : hlo->operands()) {
      TF_ASSIGN_OR_RETURN(HloInstruction * new_operand, Resolve(operand));
      new_operands.push_back(new_operand);
    }
    VLOG(1) << "Cloning HLO and adding to caller: " << hlo->ToString();
    auto new_hlo = hlo->CloneWithNewOperands(hlo->shape(), new_operands);
    HloInstruction* new_hlo_pointer =
        outer_->AddInstruction(std::move(new_hlo));
    TF_RETURN_IF_ERROR(NoteMapping(hlo, new_hlo_pointer));

    // Account for control edges.
    for (HloInstruction* control_predecessor : hlo->control_predecessors()) {
      TF_ASSIGN_OR_RETURN(HloInstruction * new_control_predecessor,
                          Resolve(control_predecessor));
      TF_RETURN_IF_ERROR(
          new_control_predecessor->AddControlDependencyTo(new_hlo_pointer));
    }

    return Status::OK();
  }

  // Does not create new nodes for the parameter; rather, notes the mapping from
  // the subcomputation parameter node to the call operands in the caller
  // computation.
  Status HandleParameter(HloInstruction* parameter) override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScall_inlinerDTcc mht_2(mht_2_v, 247, "", "./tensorflow/compiler/xla/service/call_inliner.cc", "HandleParameter");

    TF_RETURN_IF_ERROR(NoteMapping(
        parameter, call_->mutable_operand(parameter->parameter_number())));
    return Status::OK();
  }

  // Wires the consumers of the call to instead point at the newly created root,
  // replacing the call operation in the caller computation.
  Status FinishVisit(HloInstruction* root) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScall_inlinerDTcc mht_3(mht_3_v, 258, "", "./tensorflow/compiler/xla/service/call_inliner.cc", "FinishVisit");

    TF_ASSIGN_OR_RETURN(HloInstruction * new_root, Resolve(root));
    VLOG(1) << "Replacing all uses of " << call_->ToString()
            << " with new root " << new_root->ToString();
    return outer_->ReplaceInstruction(call_, new_root);
  }

  CallInliner::InlinedInstructionMap ConsumeInstructionMap() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScall_inlinerDTcc mht_4(mht_4_v, 268, "", "./tensorflow/compiler/xla/service/call_inliner.cc", "ConsumeInstructionMap");

    return std::move(subcomputation_hlo_to_new_hlo_);
  }

 private:
  // Resolves the callee subcomputation_hlo to the new (inline) HLO in the
  // caller computation, or returns a NotFound error if that subcomputation HLO
  // has not been mapped.
  StatusOr<HloInstruction*> Resolve(HloInstruction* subcomputation_hlo) {
    auto it = subcomputation_hlo_to_new_hlo_.find(subcomputation_hlo);
    if (it == subcomputation_hlo_to_new_hlo_.end()) {
      return NotFound(
          "Could not find mapping from subcomputation HLO %s to a cloned HLO.",
          subcomputation_hlo->ToString());
    }
    return it->second;
  }

  // Notes that the given subcomputation_hlo in the callee has been mapped to
  // the (inline) new_hlo in the caller computation.
  //
  // Returns an error status if the subcomputation_hlo is mapped more than
  // once.
  Status NoteMapping(HloInstruction* subcomputation_hlo,
                     HloInstruction* new_hlo) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScall_inlinerDTcc mht_5(mht_5_v, 295, "", "./tensorflow/compiler/xla/service/call_inliner.cc", "NoteMapping");

    auto result = subcomputation_hlo_to_new_hlo_.insert(
        std::make_pair(subcomputation_hlo, new_hlo));
    TF_RET_CHECK(result.second)
        << "A mapping for the subcomputation HLO is already present.";
    return Status::OK();
  }

  HloInstruction* call_;
  HloComputation* outer_;
  CallInliner::InlinedInstructionMap subcomputation_hlo_to_new_hlo_;
};

}  // namespace

/* static */ StatusOr<CallInliner::InlinedInstructionMap> CallInliner::Inline(
    HloInstruction* call) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScall_inlinerDTcc mht_6(mht_6_v, 314, "", "./tensorflow/compiler/xla/service/call_inliner.cc", "CallInliner::Inline");

  TF_RET_CHECK(call->opcode() == HloOpcode::kCall)
      << "Instruction was not a call op: " << call->opcode();
  const auto& callees = call->called_computations();
  TF_RET_CHECK(callees.size() == 1);
  HloComputation* callee = callees[0];
  // We visit the callee, cloning its body into its caller.
  SubcomputationInsertionVisitor visitor(call);
  TF_RETURN_IF_ERROR(callee->Accept(&visitor));
  return visitor.ConsumeInstructionMap();
}

StatusOr<bool> CallInliner::Run(HloModule* module) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScall_inlinerDTcc mht_7(mht_7_v, 329, "", "./tensorflow/compiler/xla/service/call_inliner.cc", "CallInliner::Run");

  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(module);
  // Because call graph nodes are visited in post-order (callees before callers)
  // we'll always inline kCalls into their callers in the appropriate order.
  bool did_mutate = false;
  TF_RETURN_IF_ERROR(
      call_graph->VisitNodes([&](const CallGraphNode& node) -> Status {
        VLOG(1) << "Visiting node: " << node.ToString();
        for (HloInstruction* instruction :
             node.computation()->MakeInstructionPostOrder()) {
          if (instruction->opcode() == HloOpcode::kCall) {
            const auto& callees = instruction->called_computations();
            TF_RET_CHECK(callees.size() == 1);
            HloInstruction* call_root = callees[0]->root_instruction();
            if (!single_call_site_ ||
                call_graph->GetNode(instruction->to_apply())
                        .caller_callsites()
                        .size() == 1) {
              TF_ASSIGN_OR_RETURN(CallInliner::InlinedInstructionMap inline_map,
                                  Inline(instruction));
              HloInstruction* inlined_root = inline_map[call_root];
              if (update_domain_) {
                HloDomainIsolator isolator(
                    []() { return ShardingDomainCreator{}; });
                TF_RETURN_IF_ERROR(
                    isolator.UpdateDomains(inlined_root).status());
              }
              did_mutate = true;
            }
          }
        }
        return Status::OK();
      }));
  if (did_mutate) {
    // Run DCE to remove called computations which are now becoming unused.
    // This can result then in problems if within the called computation, there
    // were send/recv instructions, which the module group verifier will flag as
    // error findingthe same channel ID used for multiple send/recv
    // instructions.
    TF_RETURN_IF_ERROR(HloDCE().Run(module).status());
  }
  return did_mutate;
}

}  // namespace xla
