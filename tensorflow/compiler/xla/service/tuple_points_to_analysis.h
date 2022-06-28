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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_TUPLE_POINTS_TO_ANALYSIS_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_TUPLE_POINTS_TO_ANALYSIS_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStuple_points_to_analysisDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStuple_points_to_analysisDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStuple_points_to_analysisDTh() {
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


#include <stddef.h>

#include <iosfwd>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/logical_buffer.h"
#include "tensorflow/compiler/xla/service/logical_buffer_analysis.h"
#include "tensorflow/compiler/xla/shape_tree.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/compactptrset.h"

namespace xla {

// A class describing the source(s) of the Buffer(s) contained in the output of
// a particular HLO instruction. The structure of PointsToSet mirrors the
// structure of the instruction's shape, which may be an arbitrary tree (eg, a
// nested tuple). Each node in this tree corresponds to a single buffer in the
// instruction's output and contains the set of Buffers which might define
// the corresponding buffer.
class PointsToSet {
 public:
  // Construct our ShapeTree with a pointer rather than a reference to a Shape
  // because this is very hot code, and copying (and then destroying) all these
  // Shapes is slow.
  explicit PointsToSet(const Shape* shape) : tree_(shape) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStuple_points_to_analysisDTh mht_0(mht_0_v, 224, "", "./tensorflow/compiler/xla/service/tuple_points_to_analysis.h", "PointsToSet");
}

  // Returns true if any points-to sets for any subshape element is not a
  // singleton.
  bool IsAmbiguous() const;

  // Returns true if no LogicalBuffer appears in more than one points-to set of
  // the shape nodes.
  bool IsDistinct() const;

  // Returns the total number of different LogicalBuffers contained in this
  // object. This is equal to CreateFlattenedSet().size().
  size_t size() const;

  // Creates a set containing the union of all LogicalBuffers contained in the
  // PointsToSet.
  using BufferSet = tensorflow::gtl::CompactPointerSet<const LogicalBuffer*>;
  BufferSet CreateFlattenedSet() const;

  // Returns true if the given buffer is in the points-to set at the given
  // index.
  bool ContainsBufferAtIndex(const LogicalBuffer& buffer,
                             const ShapeIndex& index) const;

  // Returns true if the given buffer is in the points-to set at any index.
  bool ContainsBuffer(const LogicalBuffer& buffer) const;

  // Adds the given buffer to the points-to set at the given index. This is a
  // nop if the buffer already is in the set at that index.
  void AddPointedToBuffer(const LogicalBuffer& buffer, const ShapeIndex& index);

  // For the subshape at the given index (where index is defined as in
  // ShapeUtil::GetSubshape) this method returns the set of HLO instructions
  // which may produce the tuple subshape at that index. For example, given:
  //
  // %tuple1 = tuple(...)
  // %tuple2 = tuple(...)
  // %select = select(%tuple1, %tuple2)
  // %nested_tuple = tuple(%select, %tuple1)
  //
  // These are the values for tuple_sources() for the PointsToSet of
  // %nested_tuple:
  //
  // tuple_sources({}) = {%nested_tuple}
  // tuple_sources({0}) = {%tuple1, %tuple2}
  // tuple_sources({1}) = {%tuple1}
  //
  // tuple_sources() at the index of an array shape (not a tuple) returns the
  // empty set. The instructions in the set returned by tuple_sources
  // necessarily are either Tuple instructions, constants, or parameters.
  using SourceSet = tensorflow::gtl::CompactPointerSet<HloInstruction*>;
  const SourceSet& tuple_sources(const ShapeIndex& index) const;

  // Add a tuple source instruction for the given index.
  void add_tuple_source(const ShapeIndex& index, HloInstruction* tuple);

  using BufferList = absl::InlinedVector<const LogicalBuffer*, 1>;

  // Return the list of logical buffers for the subshape at index.
  const BufferList& element(const ShapeIndex& index) const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStuple_points_to_analysisDTh mht_1(mht_1_v, 286, "", "./tensorflow/compiler/xla/service/tuple_points_to_analysis.h", "element");

    return tree_.element(index).buffers;
  }
  BufferList* mutable_element(const ShapeIndex& index) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStuple_points_to_analysisDTh mht_2(mht_2_v, 292, "", "./tensorflow/compiler/xla/service/tuple_points_to_analysis.h", "mutable_element");

    return &tree_.mutable_element(index)->buffers;
  }

  // Call fn(index, buflist) for every subshape index.
  template <typename Fn>
  void ForEachElement(const Fn& fn) const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStuple_points_to_analysisDTh mht_3(mht_3_v, 301, "", "./tensorflow/compiler/xla/service/tuple_points_to_analysis.h", "ForEachElement");

    tree_.ForEachElement([&fn](const ShapeIndex& index, const Elem& elem) {
      fn(index, elem.buffers);
    });
  }
  template <typename Fn>
  void ForEachMutableElement(const Fn& fn) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStuple_points_to_analysisDTh mht_4(mht_4_v, 310, "", "./tensorflow/compiler/xla/service/tuple_points_to_analysis.h", "ForEachMutableElement");

    tree_.ForEachMutableElement([&fn](const ShapeIndex& index, Elem* elem) {
      fn(index, &elem->buffers);
    });
  }
  template <typename Fn>
  Status ForEachElementWithStatus(const Fn& fn) const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStuple_points_to_analysisDTh mht_5(mht_5_v, 319, "", "./tensorflow/compiler/xla/service/tuple_points_to_analysis.h", "ForEachElementWithStatus");

    return tree_.ForEachElementWithStatus(
        [&fn](const ShapeIndex& index, const Elem& elem) {
          return fn(index, elem.buffers);
        });
  }

 private:
  struct Elem {
    BufferList buffers;
    SourceSet tuple_sources;
  };
  ShapeTree<Elem> tree_;

  // PointsToSet contains references (const LogicalBuffer*) to elements within
  // TuplePointsToAnalysis, so disable copying.
  PointsToSet(const PointsToSet&) = delete;
  PointsToSet& operator=(const PointsToSet&) = delete;
};

// This class describes a particular subshape in a computation (instruction and
// shape index) and the logical buffer which may be a source of the subshape
// value.
class BufferAlias {
 public:
  BufferAlias(HloInstruction* instruction, const ShapeIndex& index)
      : instruction_(instruction), index_(index) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStuple_points_to_analysisDTh mht_6(mht_6_v, 348, "", "./tensorflow/compiler/xla/service/tuple_points_to_analysis.h", "BufferAlias");
}

  // Return the instruction/index of the subshape.
  HloInstruction* instruction() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStuple_points_to_analysisDTh mht_7(mht_7_v, 354, "", "./tensorflow/compiler/xla/service/tuple_points_to_analysis.h", "instruction");
 return instruction_; }
  const ShapeIndex& index() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStuple_points_to_analysisDTh mht_8(mht_8_v, 358, "", "./tensorflow/compiler/xla/service/tuple_points_to_analysis.h", "index");
 return index_; }

  bool operator==(const BufferAlias& other) const {
    return instruction_ == other.instruction_ && index_ == other.index_;
  }
  bool operator!=(const BufferAlias& other) const { return !(*this == other); }

  std::string ToString() const;

 private:
  HloInstruction* instruction_;
  ShapeIndex index_;
};

std::ostream& operator<<(std::ostream& out, const BufferAlias& buffer_alias);

// DFS visitor that performs tuple points-to analysis. This analysis determines
// the potential sources of each buffer in each instruction's output.
class TuplePointsToAnalysis : public DfsHloVisitorWithDefault {
 public:
  // Runs points-to analysis on 'module'.
  static StatusOr<std::unique_ptr<TuplePointsToAnalysis>> Run(
      const HloModule* module);

  // Return the points-to set of an instruction. This describes the potential
  // sources of each buffer in the instruction's output.
  const PointsToSet& GetPointsToSet(
      const HloInstruction* hlo_instruction) const;

  // Returns the logical buffer with the given ID.
  const LogicalBuffer& GetBuffer(LogicalBuffer::Id id) const;

  // Returns the buffer defined at the given instruction and index. An error is
  // returned if no buffer is defined at that point.
  StatusOr<const LogicalBuffer*> GetBufferDefinedAt(
      const HloInstruction* instruction, const ShapeIndex& index) const;

  // Return a (possibly empty) vector containing all BufferAliases of the given
  // logical buffer The buffer alias set is the inverse of the points-to set.
  // That is, LogicalBuffer B is in the points-to set of instruction I at index
  // N iff instruction I, index N is a BufferAlias of B.
  using BufferAliasVector = absl::InlinedVector<BufferAlias, 1>;
  const BufferAliasVector& GetBufferAliases(const LogicalBuffer& buffer) const;

  // Returns the number of logical buffers in the module
  LogicalBuffer::Id num_logical_buffers() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStuple_points_to_analysisDTh mht_9(mht_9_v, 406, "", "./tensorflow/compiler/xla/service/tuple_points_to_analysis.h", "num_logical_buffers");

    return logical_buffer_analysis_->num_logical_buffers();
  }

  // Return a the logical buffer with id "id" in the module. Iteration
  // over all logical buffers is usually done with something like:
  //
  // for (LogicalBuffer:Id id = 0; id < points_to.num_logical_buffers(); id++){
  //   const auto& buffer = points_to.logical_buffer(id);
  //   ... do something with buffer ...
  // }
  LogicalBuffer& logical_buffer(LogicalBuffer::Id id) const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStuple_points_to_analysisDTh mht_10(mht_10_v, 420, "", "./tensorflow/compiler/xla/service/tuple_points_to_analysis.h", "logical_buffer");

    return logical_buffer_analysis_->GetBuffer(id);
  }

  // Returns a vector of buffers that the instruction produces. Most
  // instructions produce a single buffer (the top-level buffer), some produce
  // no buffers (eg bitcast), and some produce more than one buffer (eg,
  // tuple-shaped parameters).
  using BufferDefinitionVector = absl::InlinedVector<const LogicalBuffer*, 1>;
  const BufferDefinitionVector& GetBuffersDefinedByInstruction(
      const HloInstruction* instruction) const;

  // Returns true if the given instruction defines a buffer at the given index.
  bool InstructionDefinesBufferAtIndex(const HloInstruction* instruction,
                                       const ShapeIndex& index) const;

  // Returns an OK status if the given buffer is defined by instruction
  // 'buffer.instruction()' at index 'buffer.index()' and if the given buffer
  // matches the TuplePointsToAnalysis' LogicalBuffer with 'buffer.id'. Returns
  // an FailedPrecondition error status otherwise. An example of a LogicalBuffer
  // which is not defined is a tuple element in a Tuple instruction. In this
  // case, the Tuple instruction does not define the LogicalBuffer, rather that
  // index aliases one of its operands.
  Status VerifyBuffer(const LogicalBuffer& buffer) const;

  Status DefaultAction(HloInstruction* hlo_instruction) override;
  Status HandleTuple(HloInstruction* tuple) override;
  Status HandleGetTupleElement(HloInstruction* get_tuple_element) override;
  Status HandleAsyncStart(HloInstruction* async_start) override;
  Status HandleAsyncUpdate(HloInstruction* async_update) override;
  Status HandleAsyncDone(HloInstruction* async_done) override;
  Status HandleBitcast(HloInstruction* bitcast) override;
  Status HandleDomain(HloInstruction* domain) override;
  Status HandleCopy(HloInstruction* copy) override;
  Status HandleCopyStart(HloInstruction* copy_start) override;
  Status HandleCopyDone(HloInstruction* copy_done) override;
  Status HandleRecvDone(HloInstruction* recv_done) override;
  Status HandleSend(HloInstruction* send) override;
  Status HandleTupleSelect(HloInstruction* tuple_select) override;
  Status HandleAddDependency(HloInstruction* add_dependency) override;
  Status HandleCustomCall(HloInstruction* custom_call) override;
  Status HandleOptimizationBarrier(HloInstruction* barrier) override;

  std::string ToString() const;

  // Returns true if 'user' cannot possibly use the buffer at 'index' in
  // 'operand'. Returns false otherwise.
  //
  // REQUIRES: 'operand' is an operand of 'user'.
  bool DoesNotUseOperandBuffer(const HloInstruction* operand,
                               const ShapeIndex& index,
                               const HloInstruction* user) const;

 private:
  explicit TuplePointsToAnalysis(
      const HloModule* module,
      std::unique_ptr<LogicalBufferAnalysis> logical_buffer_analysis)
      : module_(module),
        logical_buffer_analysis_(std::move(logical_buffer_analysis)) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStuple_points_to_analysisDTh mht_11(mht_11_v, 481, "", "./tensorflow/compiler/xla/service/tuple_points_to_analysis.h", "TuplePointsToAnalysis");
}

  // Perform the analysis. Should be called immediately after constructing the
  // object and before calling GetPointsToSet.
  Status Analyze();

  // Populates instruction-defined buffers and aliases for each instruction
  // in 'instructions'.
  Status PopulateDefinedBuffersAndAliases(
      const decltype(std::declval<HloComputation>()
                         .instructions())& instructions);

  // Creates an empty PointsToSet in the points_to_ map for the given
  // instruction.
  PointsToSet& CreateEmptyPointsToSet(const HloInstruction* instruction);

  // Creates a PointsToSet in the points_to_ map for 'instruction' which is a
  // copy of the existing PointsToSet for 'src'.
  PointsToSet& CreateCopiedPointsToSet(const HloInstruction* instruction,
                                       const HloInstruction* src);

  // Adds the buffers defined by the given instruction to the given vector.
  Status GatherBuffersDefinedByInstruction(const HloInstruction* instruction,
                                           BufferDefinitionVector* buffers);

  // Print points-to set for 'instruction' to 'output'.
  void InstructionToString(const HloInstruction* instruction,
                           std::string* output) const;

  // Information kept per instruction
  struct PerInstruction {
    std::unique_ptr<PointsToSet> points_to_set;
    // Empirically, ~92% of instructions have 1
    // instruction_defined_buffer, and 99% have 0 or 1
    BufferDefinitionVector instruction_defined_buffers;
  };

  const PerInstruction* PerInst(const HloInstruction* inst) const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStuple_points_to_analysisDTh mht_12(mht_12_v, 521, "", "./tensorflow/compiler/xla/service/tuple_points_to_analysis.h", "PerInst");

    int id = inst->unique_id();
    DCHECK_GE(id, 0);
    auto iter = per_instruction_.find(id);
    if (iter == per_instruction_.end()) {
      LOG(FATAL) << "Expected per-instruction information to already exist";
    } else {
      return iter->second.get();
    }
  }
  PerInstruction* PerInst(const HloInstruction* inst) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStuple_points_to_analysisDTh mht_13(mht_13_v, 534, "", "./tensorflow/compiler/xla/service/tuple_points_to_analysis.h", "PerInst");

    int id = inst->unique_id();
    DCHECK_GE(id, 0);
    auto iter = per_instruction_.find(id);
    if (iter == per_instruction_.end()) {
      return per_instruction_.emplace(id, absl::make_unique<PerInstruction>())
          .first->second.get();
    } else {
      return iter->second.get();
    }
  }

  std::vector<std::pair<HloInstruction*, int64_t>>
  GetAllUsesOfInstructionAtIndex(HloInstruction* instruction,
                                 const ShapeIndex& index) const;
  bool HasUniqueFusedUseOfOperandAt(HloInstruction* operand,
                                    const ShapeIndex& operand_index,
                                    HloInstruction* fusion,
                                    const int64_t use_operand_index) const;

  // The module this analysis is performed on.
  const HloModule* module_;

  // The logical buffers for this module.
  const std::unique_ptr<LogicalBufferAnalysis> logical_buffer_analysis_;

  // A map from instruction->unique_id() to
  absl::flat_hash_map<int, std::unique_ptr<PerInstruction>> per_instruction_;

  // A map from LogicalBuffer->id() to alias information about that logical
  // buffer
  std::vector<BufferAliasVector> logical_buffer_aliases_;

  TuplePointsToAnalysis(const TuplePointsToAnalysis&) = delete;
  TuplePointsToAnalysis& operator=(const TuplePointsToAnalysis&) = delete;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_TUPLE_POINTS_TO_ANALYSIS_H_
