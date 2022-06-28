/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_ORDERING_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_ORDERING_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_orderingDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_orderingDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_orderingDTh() {
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
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/xla/service/call_graph.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/service/hlo_dataflow_analysis.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_reachability.h"
#include "tensorflow/compiler/xla/service/hlo_schedule.h"
#include "tensorflow/compiler/xla/service/hlo_value.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {

// Base class for describing a partial ordering of HLO instructions. Used to
// determine live range overlap of HLO instruction output buffers.
class HloOrdering {
 public:
  explicit HloOrdering(const HloModule* module)
      : module_(module), call_graph_(CallGraph::Build(module)) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_orderingDTh mht_0(mht_0_v, 210, "", "./tensorflow/compiler/xla/service/hlo_ordering.h", "HloOrdering");
}
  virtual ~HloOrdering() = default;

  // Specify the ordering constraints between a pair of instructions a and b.
  enum class ExecutionConstraint {
    // Indicate a and b are the same instruction;
    kIsSame,
    // Indicate a runs before b starts;
    kRunBeforeStart,
    // Indicate a runs before b ends but after b starts, e.g., when b is a
    // conditional or while loop;
    kRunBeforeEnd,
    // Only one of a or b runs each time their common ancestor is evaluated,
    // and a is in an earlier branch than b.
    kRunExclusiveBefore,
    // Only one of a or b runs each time, and a is in a later branch than b.
    kRunExclusiveAfter,
    // Indicate a runs after b ends.
    kRunAfter,
    // An order cannot be detrermined as a and b do not have a common ancestor.
    kUnordered,
  };
  // Return the execution constraint between a and b.
  HloOrdering::ExecutionConstraint GetExecutionConstraint(
      const HloInstruction* a, const HloInstruction* b) const;

  // Returns true if instruction 'a' executes before instruction 'b'. This is
  // not reflexive, that is, an instruction does not execute before itself.
  bool ExecutesBefore(const HloInstruction* a, const HloInstruction* b) const;

  // Returns whether the value 'a' is defined before the value 'b' under the
  // given ordering.
  bool IsDefinedBefore(const HloValue& a, const HloValue& b) const;

  // Returns whether the given use is before the given value definition under
  // the given ordering. Set use_is_always_before_def_in_same_instr to false if
  // you want the analysis to always consider a use at an instruction's operand
  // to be strictly before that instructions definition. The configuration needs
  // to be false when result will be used to remove unnecessary copy
  // instructions, due to additional buffer sharing constraints.
  bool UsesBeforeValueDefinition(
      absl::Span<const HloUse* const> uses, const HloValue& value,
      const HloDataflowAnalysis& dataflow,
      bool use_is_always_before_def_in_same_instr = false) const;
  // Returns whether the given values interfere. Two values interfere if they
  // may both be simultaneously live.
  bool MayInterfere(const HloValue& a, const HloValue& b,
                    const HloDataflowAnalysis& dataflow) const;

  // Returns true if the live range of the given value 'a' is strictly before
  // the live range of value 'b' using the given HLO ordering.
  bool LiveRangeStrictlyBefore(
      const HloValue& a, const HloValue& b, const HloDataflowAnalysis& dataflow,
      bool use_is_always_before_def_in_same_instr = false) const;

  // Returns the sequential instruction order for the given computation, or
  // nullptr if the computation does not have a sequential ordering.
  virtual const HloInstructionSequence* SequentialOrder(
      const HloComputation& computation) const = 0;

  // Return the call graph of the module used to compute ordering.
  const CallGraph& call_graph() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_orderingDTh mht_1(mht_1_v, 274, "", "./tensorflow/compiler/xla/service/hlo_ordering.h", "call_graph");
 return *call_graph_; }

  virtual std::string ToString() const = 0;

 protected:
  // Returns true if instruction 'a' executes before instruction 'b'.
  // Precondition: 'a' and 'b' are in the same computation.
  //
  // Derived classes should implement this method for determining order of
  // instructions in the same computation. ExecutesBefore() analyzes the
  // callgraph and uses this method to determine ordering of instructions in
  // different computations.
  virtual bool ExecutesBeforeInSameComputation(
      const HloInstruction* a, const HloInstruction* b) const = 0;

  const HloModule* module_;

  std::unique_ptr<CallGraph> call_graph_;
};

// Base class for partial orderings implemented by a map of predecessors for
// each instruction. Subclasses should fill in predecessors_.
class PredecessorHloOrdering : public HloOrdering {
 public:
  ~PredecessorHloOrdering() override = default;

  // Returns nullptr indicating the computation does not have a sequential
  // ordering.
  const HloInstructionSequence* SequentialOrder(
      const HloComputation& computation) const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_orderingDTh mht_2(mht_2_v, 306, "", "./tensorflow/compiler/xla/service/hlo_ordering.h", "SequentialOrder");

    return nullptr;
  }

  HloReachabilityMap& reachability_map(const HloComputation* computation) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_orderingDTh mht_3(mht_3_v, 313, "", "./tensorflow/compiler/xla/service/hlo_ordering.h", "reachability_map");

    return *predecessors_.at(computation);
  }
  const HloReachabilityMap& reachability_map(
      const HloComputation* computation) const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_orderingDTh mht_4(mht_4_v, 320, "", "./tensorflow/compiler/xla/service/hlo_ordering.h", "reachability_map");

    return *predecessors_.at(computation);
  }

 protected:
  explicit PredecessorHloOrdering(const HloModule* module);
  std::string ToStringHelper(const std::string& name) const;

  bool ExecutesBeforeInSameComputation(const HloInstruction* a,
                                       const HloInstruction* b) const override;

  // For each computation in the module, this is the set of the instruction's
  // predecessors. An instruction is an element of its own predecessor set.
  //
  // Subclasses should fill this in to define the desired ordering.
  absl::flat_hash_map<const HloComputation*,
                      std::unique_ptr<HloReachabilityMap>>
      predecessors_;
};

// An HLO ordering based on data dependencies in the HLO graph. In this partial
// order, instruction A executes before instruction B only if there is a path
// from A to B in the HLO graph. For example, given the following graph:
/*
          param
         /     \
      negate   exp
          \    /
           add
*/
// DependencyHloOrdering gives the following executes-before relations:
//   param executes before negate, exp, and add
//   negate executes before add
//   exp executes before add
//   add executes before nothing
// negate and exp are not ordered because the dependencies allow either to
// execute before the other (or in parallel). DependencyHloOrdering ordering
// allows maximum parallelism and enables any execution order which satisfies
// data dependencies. This requires pessimistic assumptions about buffer live
// ranges and can result in more memory used than more constrained orderings.
class DependencyHloOrdering : public PredecessorHloOrdering {
 public:
  explicit DependencyHloOrdering(const HloModule* module);
  ~DependencyHloOrdering() override = default;

  std::string ToString() const override;
};

// An HLO ordering based on a total order of instructions in each computation.
// The computation total order is a sequencing of all of its instructions in
// the computation (eg, {inst0, inst1, inst2,...}) as in single-threaded
// execution. For example, given the following HLO graph:
/*
          param
         /     \
      negate   exp
          \    /
           add
*/
// and the following sequence:
//
//  {param, negate, exp, add}
//
// SequentialHloOrdering gives the following executes-before relations:
//   param executes before negate, exp, and add
//   negate executes before exp and add
//   exp executes before add
//   add executes before nothing
// This is more constrained than DependencyHloOrdering in this example because
// negate and exp are ordered (negate before exp). This enables param to share
// the same buffer as exp (param buffer is dead after exp). Generally, this
// ordering enables more buffer sharing (reduced memory usage) because buffer
// interference is reduced relative to DependencyHloOrdering.
class SequentialHloOrdering : public HloOrdering {
 public:
  explicit SequentialHloOrdering(const HloSchedule& schedule);
  explicit SequentialHloOrdering(HloSchedule&& schedule);
  ~SequentialHloOrdering() override = default;

  // Returns the sequential instruction order for the given computation.
  const HloInstructionSequence* SequentialOrder(
      const HloComputation& computation) const override;

  std::string ToString() const override;

 protected:
  void Initialize();

  bool ExecutesBeforeInSameComputation(const HloInstruction* a,
                                       const HloInstruction* b) const override;

  const HloSchedule schedule_;

  // The position of every instruction in the HLO module in its respective
  // computation sequence (a value of zero indicates the instruction is first in
  // the sequence, etc). Instructions from all computations are contained in
  // this map so more than one instruction may have the same position
  // value. This is not a problem because ExecutesBefore also verifies
  // instructions are in the same computation.
  absl::flat_hash_map<const HloInstruction*, int> order_position_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_ORDERING_H_
