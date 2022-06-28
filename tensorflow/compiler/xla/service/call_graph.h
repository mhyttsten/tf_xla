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

// Call graph for an HLO module.

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_CALL_GRAPH_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_CALL_GRAPH_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScall_graphDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScall_graphDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScall_graphDTh() {
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


#include <ostream>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"

namespace xla {

// The context in which a computation is called by another computation.
enum class CallContext {
  // In an embedded call context, the body of the function cannot allocate
  // buffers.
  kEmbedded,

  // A control flow call context can allocate buffers.
  kControlFlow,

  // A computation is called from both an embedded and control flow context.
  kBoth,

  // During call graph construction kNone is used to indicate that the context
  // has not been determined. This is the top value for the context
  // lattice. After construction, no call sites or call graph nodes should have
  // this value.
  kNone
};

std::string CallContextToString(CallContext context);
std::ostream& operator<<(std::ostream& out, const CallContext& context);

CallContext GetInstructionCallContext(HloOpcode opcode);

// Represents an HLO instruction which calls one or more computations.
class CallSite {
 public:
  CallSite(HloInstruction* instruction,
           absl::Span<HloComputation* const> called_computations,
           CallContext context)
      : instruction_(CHECK_NOTNULL(instruction)),
        called_computations_(called_computations.begin(),
                             called_computations.end()),
        context_(context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScall_graphDTh mht_0(mht_0_v, 233, "", "./tensorflow/compiler/xla/service/call_graph.h", "CallSite");
}

  // Returns the instruction associated with this call site.
  HloInstruction* instruction() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScall_graphDTh mht_1(mht_1_v, 239, "", "./tensorflow/compiler/xla/service/call_graph.h", "instruction");
 return instruction_; }

  // Returns the computations called at this call site.
  absl::Span<HloComputation* const> called_computations() const {
    return called_computations_;
  }

  // Returns the context in which computations are called at this call site.
  CallContext context() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScall_graphDTh mht_2(mht_2_v, 250, "", "./tensorflow/compiler/xla/service/call_graph.h", "context");
 return context_; }

  std::string ToString() const;

 private:
  // The calling instruction.
  HloInstruction* instruction_;

  // The computations called by this callsite.
  const absl::InlinedVector<HloComputation*, 2> called_computations_;

  // The context in which the computations are called.
  const CallContext context_;
};

// A node in the call graph representing an HLO computation.
class CallGraphNode {
 public:
  CallGraphNode(HloComputation* computation);

  // Returns the computation represented by this call graph node.
  HloComputation* computation() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScall_graphDTh mht_3(mht_3_v, 274, "", "./tensorflow/compiler/xla/service/call_graph.h", "computation");
 return computation_; }

  // Returns the call sites in this computation. These are the instructions in
  // this computation which call other computations.
  absl::Span<const CallSite> callsites() const { return callsites_; }

  // Returns the callsite associated with the given instruction. If this
  // instruction calls no computations nullptr is returned.
  // Prerequisite: instruction is in the computation associated with this call
  // graph node.
  const CallSite* GetCallSite(const HloInstruction* instruction) const;

  // Returns the computations called by this computation.
  absl::Span<HloComputation* const> callees() const { return callees_; }

  // Returns the call sites in other computations which call this computation.
  absl::Span<const CallSite> caller_callsites() const {
    return caller_callsites_;
  }

  // Returns the computations which call this computation.
  absl::Span<HloComputation* const> callers() const { return callers_; }

  // Returns the context in which this computation is called.
  CallContext context() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScall_graphDTh mht_4(mht_4_v, 301, "", "./tensorflow/compiler/xla/service/call_graph.h", "context");
 return context_; }

  // Returns the depth of this node in the call graph. The depth is defined as
  // the length of the longest call chain from a computation with no callers
  // (usually the entry computation node) to this node.
  int depth() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScall_graphDTh mht_5(mht_5_v, 309, "", "./tensorflow/compiler/xla/service/call_graph.h", "depth");
 return depth_; }

  std::string ToString() const;

  CallGraphNode(const CallGraphNode&) = delete;
  CallGraphNode& operator=(const CallGraphNode&) = delete;
  CallGraphNode(CallGraphNode&&) = default;
  CallGraphNode& operator=(CallGraphNode&&) = default;

 private:
  // Only CallGraph can modify CallGraphNode.
  friend class CallGraph;

  // Sets the context in which this computation is called.
  void set_context(CallContext value) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScall_graphDTh mht_6(mht_6_v, 326, "", "./tensorflow/compiler/xla/service/call_graph.h", "set_context");
 context_ = value; }

  // Sets the depth of this node in the graph.
  void set_depth(int value) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScall_graphDTh mht_7(mht_7_v, 332, "", "./tensorflow/compiler/xla/service/call_graph.h", "set_depth");
 depth_ = value; }

  // Adds a callsite which calls this computation. Updates callers to include
  // the calling computation.
  void AddCallerCallSite(const CallSite& caller_callsite);

  // If instruction calls any computations adds a call site for this instruction
  // to the call graph node. If the instruction calls no computations then no
  // call site is added.
  void AddCallSiteForInstruction(HloInstruction* instruction);

  // Computation represented by this call graph node.
  HloComputation* computation_;

  // The computations called by this computation. The vector is used for a
  // stable ordering and the set enables fast membership testing.
  absl::InlinedVector<HloComputation*, 1> callees_;
  absl::flat_hash_set<HloComputation*> callee_set_;

  // The computations which call this computation. The vector is used for a
  // stable ordering and the set enables fast membership testing.
  absl::InlinedVector<HloComputation*, 1> callers_;
  absl::flat_hash_set<HloComputation*> caller_set_;

  // The call sites in this computation
  absl::InlinedVector<CallSite, 1> callsites_;

  // The map from instruction to index in callsites_ for looking up the callsite
  // (if any) associated with a particular instruction in this computation.
  absl::flat_hash_map<const HloInstruction*, int64_t> callsite_instructions_;

  // The call sites in other computations which call this computation.
  absl::InlinedVector<CallSite, 1> caller_callsites_;

  // The context in which this computation is called.
  CallContext context_ = CallContext::kNone;

  // The depth of this node in the call graph.
  int depth_ = 0;
};

// The call graph for an HLO module. The graph includes a node for each
// computation in the module.
class CallGraph {
 public:
  using VisitorFunction = std::function<Status(const CallGraphNode&)>;

  // Builds and returns a call graph for the given HLO module.
  static std::unique_ptr<CallGraph> Build(const HloModule* module);

  // Returns the node associated with the given computation.
  const CallGraphNode& GetNode(const HloComputation* computation) const;
  CallGraphNode& GetNode(const HloComputation* computation);

  // Returns the vector of all nodes in the call graph.
  const std::vector<CallGraphNode>& nodes() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScall_graphDTh mht_8(mht_8_v, 390, "", "./tensorflow/compiler/xla/service/call_graph.h", "nodes");
 return nodes_; }

  // Calls the given function on each node in the call graph. Nodes are visited
  // in post order (callees before callers). If visit_unreachable_nodes is true
  // then all nodes in the call graph are visited. Otherwise only those nodes
  // reachable from the entry computation are visited.
  Status VisitNodes(const VisitorFunction& visitor_func,
                    bool visit_unreachable_nodes = true) const;

  // Returns true if 'a' dominates 'b' in the call graph. Computation 'a'
  // dominates computation 'b' iff all callgraph paths in the caller-to-callee
  // direction from a root computation to 'b' pass through computation
  // 'a'. Trivially, a computation dominates itself.
  bool Dominates(const HloComputation* a, const HloComputation* b) const;

  // Returns whether 'instruction' is contained in 'computation' either directly
  // ('instruction->parent' is 'computation') or indirectly ('computation'
  // dominates 'instruction->parent' in the call graph).
  bool InstructionIsNestedIn(const HloInstruction* instruction,
                             const HloComputation* computation) const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScall_graphDTh mht_9(mht_9_v, 412, "", "./tensorflow/compiler/xla/service/call_graph.h", "InstructionIsNestedIn");

    return Dominates(computation, instruction->parent());
  }

  // Returns the nearest call graph ancestors of instructions 'a' and 'b' for
  // which the ancestors are in the same computation. An instruction is an call
  // graph ancestor of 'a' if the instruction calls the computation containing
  // 'a' either directly or transitively. Degeneratively an instruction is an
  // ancestor of itself. nullptr is returned if there is no common ancestor or
  // if the caller chain of 'a' or 'b' diverges (has multiple callers) before
  // the nearest common ancestor.
  //
  // Example:
  //
  // Entry computation:
  //   %x = Call(A, {Constant(42.0)})
  //   %y = Call(B, {%x})
  //
  // Computation A:
  //   %a = Negate(Param())
  //
  // Computation B:
  //   %b = Exp(Param());
  //
  // If called with %a and %b, this function would return (%x, %y). %x is an
  // ancestor of %a, and %y is an ancestor of %b, and %x and %y are in the same
  // computation.
  std::pair<HloInstruction*, HloInstruction*> NearestAncestorsInSameComputation(
      HloInstruction* a, HloInstruction* b) const;

  // Returns whether the call graph is flattened. A call graph is flattened if
  // every computation called in a sequential context (eg, kWhile or kCall) has
  // zero or one callsite, and no computation is called from both a parallel and
  // sequential context. The call graph of a module can be flattened with
  // FlattenCallGraph.
  bool IsFlattened() const;

  // Returns a vector of instructions calling the passed computation.
  // (Often a vector of size 1.)
  std::vector<HloInstruction*> GetComputationCallers(HloComputation* c);

  std::string ToString() const;

 private:
  CallGraph(const HloModule* module);

  // Not copyable.
  CallGraph(const CallGraph&) = delete;
  CallGraph& operator=(const CallGraph&) = delete;

  // Sets the call contexts for every node in the graph.
  void SetCallContexts();

  // Sets the call node depths for every node in the graph.
  void SetNodeDepths();

  // Helper method for VisitNodes(). Traverses the call graph from 'node' in DFS
  // post order (callee before caller) calling visitor_func on each node. Adds
  // nodes to 'visited' as each node is visited. Skips nodes already in
  // 'visited'.
  Status VisitNodesInternal(
      const VisitorFunction& visitor_func, const CallGraphNode& node,
      absl::flat_hash_set<const CallGraphNode*>* visited) const;

  // Recursive helper for computing whether 'a' dominates 'b' in the call
  // graph. 'b_ancestor' is the currently visited node (which starts at 'b'),
  // and 'visited' is the set of computations which have been visited.
  bool DominatesHelper(
      const HloComputation* a, const HloComputation* b,
      absl::flat_hash_set<const HloComputation*>* visited) const;

  // The HLO module represented by this call graph.
  const HloModule* module_ = nullptr;

  // Vector of all nodes in the call graph.
  std::vector<CallGraphNode> nodes_;

  // Map from HLO computation to the index of the corresponding call graph node
  // in nodes_.
  absl::flat_hash_map<const HloComputation*, int64_t> node_indices_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_CALL_GRAPH_H_
