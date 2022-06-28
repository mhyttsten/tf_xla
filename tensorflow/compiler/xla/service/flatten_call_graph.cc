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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSflatten_call_graphDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSflatten_call_graphDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSflatten_call_graphDTcc() {
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

#include "tensorflow/compiler/xla/service/flatten_call_graph.h"

#include "tensorflow/compiler/xla/service/call_graph.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {

namespace {

// Helper to replace the called computation at a while-, call-, case-, or
// conditional-instruction. This function replaces exactly one instance of
// 'computation' with 'new_computation' even if 'instruction' calls
// 'computation' more than once.
void ReplaceCalledComputation(HloInstruction* instruction,
                              HloComputation* computation,
                              HloComputation* new_computation) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSflatten_call_graphDTcc mht_0(mht_0_v, 204, "", "./tensorflow/compiler/xla/service/flatten_call_graph.cc", "ReplaceCalledComputation");

  switch (instruction->opcode()) {
    case HloOpcode::kWhile: {
      if (computation == instruction->while_condition()) {
        instruction->set_while_condition(new_computation);
      } else {
        CHECK_EQ(computation, instruction->while_body());
        instruction->set_while_body(new_computation);
      }
      break;
    }
    case HloOpcode::kCall: {
      CHECK_EQ(instruction->to_apply(), computation);
      instruction->set_to_apply(new_computation);
      break;
    }
    case HloOpcode::kConditional: {
      for (int b = 0; b < instruction->branch_count(); ++b) {
        if (b == instruction->branch_count() - 1) {
          CHECK_EQ(computation, instruction->branch_computation(b));
        }
        if (computation == instruction->branch_computation(b)) {
          instruction->set_branch_computation(b, new_computation);
          break;
        }
      }
      break;
    }
    default:
      LOG(FATAL) << "unexpected opcode: "
                 << HloOpcodeString(instruction->opcode());
  }
}

// Flatten a single call graph node. Expects to visit nodes in postorder.
Status FlattenNode(const CallGraphNode& node) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSflatten_call_graphDTcc mht_1(mht_1_v, 242, "", "./tensorflow/compiler/xla/service/flatten_call_graph.cc", "FlattenNode");

  HloComputation* computation = node.computation();
  HloModule* module = computation->parent();
  // Clone callee for all call-sites except the first one.
  for (int i = 0; i < node.caller_callsites().size(); ++i) {
    CallSite call_site = node.caller_callsites()[i];
    // Only consider sequential call contexts.
    if (call_site.context() == CallContext::kEmbedded) {
      continue;
    }
    CHECK_EQ(call_site.context(), CallContext::kControlFlow);

    // Skip first element if this computation is only called from a sequential
    // context.
    if (node.context() != CallContext::kBoth && i == 0) {
      continue;
    }

    // Clone computation for the remaining sequential context call sites.
    HloComputation* clone =
        module->AddEmbeddedComputation(computation->Clone());
    ReplaceCalledComputation(call_site.instruction(), computation, clone);
    // Clone the sub-tree of all computations called from this node.
    std::vector<HloComputation*> worklist;
    worklist.push_back(clone);
    while (!worklist.empty()) {
      auto current = worklist.back();
      worklist.pop_back();
      for (auto* instruction : current->instructions()) {
        if (GetInstructionCallContext(instruction->opcode()) !=
            CallContext::kControlFlow) {
          continue;
        }
        for (auto callee : instruction->called_computations()) {
          HloComputation* callee_clone =
              module->AddEmbeddedComputation(callee->Clone());
          ReplaceCalledComputation(instruction, callee, callee_clone);
          worklist.push_back(callee_clone);
        }
      }
    }
  }
  return Status::OK();
}

}  // namespace

StatusOr<bool> FlattenCallGraph::Run(HloModule* module) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSflatten_call_graphDTcc mht_2(mht_2_v, 292, "", "./tensorflow/compiler/xla/service/flatten_call_graph.cc", "FlattenCallGraph::Run");

  XLA_VLOG_LINES(3, "Before flatten call graph:\n" + module->ToString());

  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(module);
  TF_RETURN_IF_ERROR(call_graph->VisitNodes(FlattenNode));

  XLA_VLOG_LINES(3, "After flatten call graph:\n" + module->ToString());
  return true;
}

}  // namespace xla
