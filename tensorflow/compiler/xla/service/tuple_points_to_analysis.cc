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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStuple_points_to_analysisDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStuple_points_to_analysisDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStuple_points_to_analysisDTcc() {
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

#include "tensorflow/compiler/xla/service/tuple_points_to_analysis.h"

#include <ostream>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_dataflow_analysis.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

std::string BufferAlias::ToString() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStuple_points_to_analysisDTcc mht_0(mht_0_v, 210, "", "./tensorflow/compiler/xla/service/tuple_points_to_analysis.cc", "BufferAlias::ToString");

  return absl::StrCat("BufferAlias(", instruction_->name(), "[",
                      absl::StrJoin(index_, ","), "])");
}

std::ostream& operator<<(std::ostream& out, const BufferAlias& buffer_alias) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStuple_points_to_analysisDTcc mht_1(mht_1_v, 218, "", "./tensorflow/compiler/xla/service/tuple_points_to_analysis.cc", "operator<<");

  out << buffer_alias.ToString();
  return out;
}

bool PointsToSet::IsAmbiguous() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStuple_points_to_analysisDTcc mht_2(mht_2_v, 226, "", "./tensorflow/compiler/xla/service/tuple_points_to_analysis.cc", "PointsToSet::IsAmbiguous");

  bool ambiguous = false;
  ForEachElement(
      [&ambiguous](const ShapeIndex& /*index*/, const BufferList& points_to) {
        ambiguous |= points_to.size() > 1;
      });
  return ambiguous;
}

bool PointsToSet::IsDistinct() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStuple_points_to_analysisDTcc mht_3(mht_3_v, 238, "", "./tensorflow/compiler/xla/service/tuple_points_to_analysis.cc", "PointsToSet::IsDistinct");

  bool distinct = true;
  absl::flat_hash_set<const LogicalBuffer*> all_points_to;
  ForEachElement([&](const ShapeIndex& /*index*/, const BufferList& points_to) {
    for (auto& buffer : points_to) {
      if (all_points_to.contains(buffer)) {
        distinct = false;
      }
      all_points_to.insert(buffer);
    }
  });
  return distinct;
}

size_t PointsToSet::size() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStuple_points_to_analysisDTcc mht_4(mht_4_v, 255, "", "./tensorflow/compiler/xla/service/tuple_points_to_analysis.cc", "PointsToSet::size");

  // Because pointed-to elements may be duplicated we have to create a flattened
  // set and return the size.
  return CreateFlattenedSet().size();
}

PointsToSet::BufferSet PointsToSet::CreateFlattenedSet() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStuple_points_to_analysisDTcc mht_5(mht_5_v, 264, "", "./tensorflow/compiler/xla/service/tuple_points_to_analysis.cc", "PointsToSet::CreateFlattenedSet");

  BufferSet flat_set;
  ForEachElement(
      [&flat_set](const ShapeIndex& /*index*/, const BufferList& buffers) {
        flat_set.insert(buffers.begin(), buffers.end());
      });
  return flat_set;
}

bool PointsToSet::ContainsBuffer(const LogicalBuffer& buffer) const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStuple_points_to_analysisDTcc mht_6(mht_6_v, 276, "", "./tensorflow/compiler/xla/service/tuple_points_to_analysis.cc", "PointsToSet::ContainsBuffer");

  bool found = false;
  ForEachElement([&found, &buffer](const ShapeIndex& /*index*/,
                                   const BufferList& pointed_to_buffers) {
    if (!found && absl::c_linear_search(pointed_to_buffers, &buffer)) {
      found = true;
    }
  });
  return found;
}

bool PointsToSet::ContainsBufferAtIndex(const LogicalBuffer& buffer,
                                        const ShapeIndex& index) const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStuple_points_to_analysisDTcc mht_7(mht_7_v, 291, "", "./tensorflow/compiler/xla/service/tuple_points_to_analysis.cc", "PointsToSet::ContainsBufferAtIndex");

  const auto& pointed_to_buffers = element(index);
  return absl::c_linear_search(pointed_to_buffers, &buffer);
}

void PointsToSet::AddPointedToBuffer(const LogicalBuffer& buffer,
                                     const ShapeIndex& index) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStuple_points_to_analysisDTcc mht_8(mht_8_v, 300, "", "./tensorflow/compiler/xla/service/tuple_points_to_analysis.cc", "PointsToSet::AddPointedToBuffer");

  if (ContainsBufferAtIndex(buffer, index)) {
    return;
  }
  mutable_element(index)->push_back(&buffer);
}

const PointsToSet::SourceSet& PointsToSet::tuple_sources(
    const ShapeIndex& index) const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStuple_points_to_analysisDTcc mht_9(mht_9_v, 311, "", "./tensorflow/compiler/xla/service/tuple_points_to_analysis.cc", "PointsToSet::tuple_sources");

  return tree_.element(index).tuple_sources;
}

void PointsToSet::add_tuple_source(const ShapeIndex& index,
                                   HloInstruction* tuple) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStuple_points_to_analysisDTcc mht_10(mht_10_v, 319, "", "./tensorflow/compiler/xla/service/tuple_points_to_analysis.cc", "PointsToSet::add_tuple_source");

  tree_.mutable_element(index)->tuple_sources.insert(tuple);
}

namespace {
// Gather fusion instructions from 'instruction' into 'fusion_instructions'.
void GatherFusionInstructions(
    HloInstruction* instruction,
    std::vector<HloInstruction*>* fusion_instructions) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStuple_points_to_analysisDTcc mht_11(mht_11_v, 330, "", "./tensorflow/compiler/xla/service/tuple_points_to_analysis.cc", "GatherFusionInstructions");

  CHECK_EQ(HloOpcode::kFusion, instruction->opcode());
  for (auto* fused : instruction->fused_instructions()) {
    if (fused->opcode() == HloOpcode::kFusion) {
      GatherFusionInstructions(fused, fusion_instructions);
    }
  }
  fusion_instructions->push_back(instruction);
}

}  // namespace

/* static */ StatusOr<std::unique_ptr<TuplePointsToAnalysis>>
TuplePointsToAnalysis::Run(const HloModule* module) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStuple_points_to_analysisDTcc mht_12(mht_12_v, 346, "", "./tensorflow/compiler/xla/service/tuple_points_to_analysis.cc", "TuplePointsToAnalysis::Run");

  auto logical_buffer_analysis = LogicalBufferAnalysis::Run(module);
  std::unique_ptr<TuplePointsToAnalysis> analysis(new TuplePointsToAnalysis(
      module, logical_buffer_analysis.ConsumeValueOrDie()));
  TF_RETURN_IF_ERROR(analysis->Analyze());
  return std::move(analysis);
}

Status TuplePointsToAnalysis::Analyze() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStuple_points_to_analysisDTcc mht_13(mht_13_v, 357, "", "./tensorflow/compiler/xla/service/tuple_points_to_analysis.cc", "TuplePointsToAnalysis::Analyze");

  per_instruction_.clear();
  per_instruction_.reserve(module_->instruction_count());

  logical_buffer_aliases_.clear();
  logical_buffer_aliases_.resize(
      logical_buffer_analysis_->num_logical_buffers());

  std::vector<HloInstruction*> fusion_instructions;
  for (auto* computation : module_->MakeNonfusionComputations()) {
    TF_RETURN_IF_ERROR(computation->Accept(this));
    TF_RETURN_IF_ERROR(
        PopulateDefinedBuffersAndAliases(computation->instructions()));
    for (auto* instruction : computation->instructions()) {
      if (instruction->opcode() == HloOpcode::kFusion) {
        GatherFusionInstructions(instruction, &fusion_instructions);
      }
    }
  }
  // Run points-to analysis on fusion instructions in 'computation'.
  for (auto* instruction : fusion_instructions) {
    TF_RETURN_IF_ERROR(instruction->fused_expression_root()->Accept(this));
    TF_RETURN_IF_ERROR(
        PopulateDefinedBuffersAndAliases(instruction->fused_instructions()));
  }

  XLA_VLOG_LINES(3, ToString());

  return Status::OK();
}

Status TuplePointsToAnalysis::PopulateDefinedBuffersAndAliases(
    const decltype(std::declval<HloComputation>()
                       .instructions())& instructions) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStuple_points_to_analysisDTcc mht_14(mht_14_v, 393, "", "./tensorflow/compiler/xla/service/tuple_points_to_analysis.cc", "TuplePointsToAnalysis::PopulateDefinedBuffersAndAliases");

  for (auto* instruction : instructions) {
    PerInstruction* pi = PerInst(instruction);
    TF_RETURN_IF_ERROR(GatherBuffersDefinedByInstruction(
        instruction, &pi->instruction_defined_buffers));

    const PointsToSet& points_to_set = GetPointsToSet(instruction);
    points_to_set.ForEachElement(
        [this, &instruction](
            const ShapeIndex& index,
            const PointsToSet::BufferList& pointed_to_buffers) {
          for (const LogicalBuffer* buffer : pointed_to_buffers) {
            logical_buffer_aliases_[buffer->id()].emplace_back(instruction,
                                                               index);
          }
        });
  }
  return Status::OK();
}

Status TuplePointsToAnalysis::DefaultAction(HloInstruction* hlo_instruction) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStuple_points_to_analysisDTcc mht_15(mht_15_v, 416, "", "./tensorflow/compiler/xla/service/tuple_points_to_analysis.cc", "TuplePointsToAnalysis::DefaultAction");

  // Create trivial points-to set for instruction. Each points-to set at index i
  // contains a single element LogicalBuffer(hlo_instruction, i). This indicates
  // that this instruction is the source of all buffers in its own output.
  PointsToSet& points_to_set = CreateEmptyPointsToSet(hlo_instruction);
  points_to_set.ForEachMutableElement(
      [this, hlo_instruction](const ShapeIndex& index,
                              PointsToSet::BufferList* buffers) {
        buffers->push_back(
            &logical_buffer_analysis_->GetBuffer(hlo_instruction, index));
      });

  if (hlo_instruction->shape().IsTuple()) {
    // If the hlo instruction is a tuple-shaped, then trivially the instruction
    // itself is the source of the tuple.
    points_to_set.add_tuple_source({}, hlo_instruction);
  }

  return Status::OK();
}

Status TuplePointsToAnalysis::HandleGetTupleElement(
    HloInstruction* get_tuple_element) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStuple_points_to_analysisDTcc mht_16(mht_16_v, 441, "", "./tensorflow/compiler/xla/service/tuple_points_to_analysis.cc", "TuplePointsToAnalysis::HandleGetTupleElement");

  // GetTupleElement forwards a pointer to a particular element of the tuple
  // operand.
  int64_t element_index = get_tuple_element->tuple_index();

  PointsToSet& points_to_set = CreateEmptyPointsToSet(get_tuple_element);
  const PointsToSet& operand_points_to_set =
      *PerInst(get_tuple_element->operand(0))->points_to_set;

  // Copy the points-to set (and tuple sources) at index {element_index} of the
  // operand to the points-to set for this GetTupleElement instruction.
  points_to_set.ForEachMutableElement(
      [&](const ShapeIndex& target_index, PointsToSet::BufferList* points_to) {
        // Construct an index into the operand by prepending element_index to
        // the index for the GetTupleElement instruction's points-to set.
        ShapeIndex src_index;
        src_index.push_back(element_index);
        for (auto element : target_index) {
          src_index.push_back(element);
        }

        *points_to = operand_points_to_set.element(src_index);
        for (HloInstruction* tuple :
             operand_points_to_set.tuple_sources(src_index)) {
          points_to_set.add_tuple_source(target_index, tuple);
        }
      });

  return Status::OK();
}

Status TuplePointsToAnalysis::HandleCopy(HloInstruction* copy) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStuple_points_to_analysisDTcc mht_17(mht_17_v, 475, "", "./tensorflow/compiler/xla/service/tuple_points_to_analysis.cc", "TuplePointsToAnalysis::HandleCopy");

  // A kCopy instruction performs a shallow copy of the operand. The top-level
  // buffer (index={}) is newly created, but all other buffers (in the case of a
  // tuple shape) come from the operand
  PointsToSet& points_to_set = CreateCopiedPointsToSet(copy, copy->operand(0));
  points_to_set.mutable_element(/*index=*/{})->clear();
  points_to_set.AddPointedToBuffer(
      logical_buffer_analysis_->GetBuffer(copy, /*index=*/{}),
      /*index=*/{});

  return Status::OK();
}

Status TuplePointsToAnalysis::HandleBitcast(HloInstruction* bitcast) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStuple_points_to_analysisDTcc mht_18(mht_18_v, 491, "", "./tensorflow/compiler/xla/service/tuple_points_to_analysis.cc", "TuplePointsToAnalysis::HandleBitcast");

  // A kBitcast instruction aliases its operand. That is, the buffer of its
  // result *is* the buffer of its operand, so just copy the operands points-to
  // set.
  CreateCopiedPointsToSet(bitcast, bitcast->operand(0));
  return Status::OK();
}

Status TuplePointsToAnalysis::HandleDomain(HloInstruction* domain) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStuple_points_to_analysisDTcc mht_19(mht_19_v, 502, "", "./tensorflow/compiler/xla/service/tuple_points_to_analysis.cc", "TuplePointsToAnalysis::HandleDomain");

  // A kDomain instruction aliases its operand. That is, the buffer of its
  // result *is* the buffer of its operand, so just copy the operands points-to
  // set.
  CreateCopiedPointsToSet(domain, domain->operand(0));
  return Status::OK();
}

Status TuplePointsToAnalysis::HandleAddDependency(
    HloInstruction* add_dependency) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStuple_points_to_analysisDTcc mht_20(mht_20_v, 514, "", "./tensorflow/compiler/xla/service/tuple_points_to_analysis.cc", "TuplePointsToAnalysis::HandleAddDependency");

  // AddDependency just forwards the value of its zero-th operand.
  CreateCopiedPointsToSet(add_dependency, add_dependency->operand(0));
  return Status::OK();
}

Status TuplePointsToAnalysis::HandleRecvDone(HloInstruction* recv_done) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStuple_points_to_analysisDTcc mht_21(mht_21_v, 523, "", "./tensorflow/compiler/xla/service/tuple_points_to_analysis.cc", "TuplePointsToAnalysis::HandleRecvDone");

  // RecvDone aliases its input (Recv) tuple element {0} to element {0} of its
  // output. The other indices ({} and {1}) define their own buffers.
  PointsToSet& points_to_set = CreateEmptyPointsToSet(recv_done);
  points_to_set.AddPointedToBuffer(
      logical_buffer_analysis_->GetBuffer(recv_done, /*index=*/{}),
      /*index=*/{});
  points_to_set.AddPointedToBuffer(
      logical_buffer_analysis_->GetBuffer(recv_done, /*index=*/{1}),
      /*index=*/{1});

  const PointsToSet& operand_points_to_set =
      GetPointsToSet(recv_done->operand(0));

  // Recursively copy the points to set of the operand tuple {0} to the output
  // element {0}.
  points_to_set.ForEachMutableElement(
      [&points_to_set, &operand_points_to_set](
          const ShapeIndex& index, PointsToSet::BufferList* buffers) {
        if (index.empty() || index[0] != 0) {
          return;
        }
        *buffers = operand_points_to_set.element(index);
        for (auto& tuple_source : operand_points_to_set.tuple_sources(index)) {
          points_to_set.add_tuple_source(index, tuple_source);
        }
      });
  return Status::OK();
}

Status TuplePointsToAnalysis::HandleAsyncStart(HloInstruction* async_start) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStuple_points_to_analysisDTcc mht_22(mht_22_v, 556, "", "./tensorflow/compiler/xla/service/tuple_points_to_analysis.cc", "TuplePointsToAnalysis::HandleAsyncStart");

  // AsyncStart forwards its aliased operands to {0}.
  PointsToSet& points_to_set = CreateEmptyPointsToSet(async_start);

  points_to_set.ForEachMutableElement(
      [&](const ShapeIndex& target_index, PointsToSet::BufferList* buffers) {
        if (target_index.size() >= 2 && target_index.front() == 0) {
          const PointsToSet& operand_points_to_set =
              GetPointsToSet(async_start->operand(target_index.at(1)));
          ShapeIndex source_index(target_index.begin() + 2, target_index.end());
          *buffers = operand_points_to_set.element(source_index);
          for (HloInstruction* tuple :
               operand_points_to_set.tuple_sources(source_index)) {
            points_to_set.add_tuple_source(target_index, tuple);
          }
        } else {
          buffers->push_back(
              &logical_buffer_analysis_->GetBuffer(async_start, target_index));
        }
      });

  return Status::OK();
}

Status TuplePointsToAnalysis::HandleAsyncUpdate(HloInstruction* async_update) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStuple_points_to_analysisDTcc mht_23(mht_23_v, 583, "", "./tensorflow/compiler/xla/service/tuple_points_to_analysis.cc", "TuplePointsToAnalysis::HandleAsyncUpdate");

  // AsyncUpdate forwards its aliased operand to {}.
  PointsToSet& points_to_set = CreateEmptyPointsToSet(async_update);
  const PointsToSet& operand_points_to_set =
      GetPointsToSet(async_update->operand(0));
  CHECK_EQ(async_update->shape(), async_update->operand(0)->shape());

  points_to_set.ForEachMutableElement([&](const ShapeIndex& index,
                                          PointsToSet::BufferList* buffers) {
    *buffers = operand_points_to_set.element(index);
    for (HloInstruction* tuple : operand_points_to_set.tuple_sources(index)) {
      points_to_set.add_tuple_source(index, tuple);
    }
  });

  return Status::OK();
}

Status TuplePointsToAnalysis::HandleAsyncDone(HloInstruction* async_done) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStuple_points_to_analysisDTcc mht_24(mht_24_v, 604, "", "./tensorflow/compiler/xla/service/tuple_points_to_analysis.cc", "TuplePointsToAnalysis::HandleAsyncDone");

  // AsyncDone forwards its aliased operand.
  PointsToSet& points_to_set = CreateEmptyPointsToSet(async_done);
  const PointsToSet& operand_points_to_set =
      GetPointsToSet(async_done->operand(0));
  operand_points_to_set.ForEachElement(
      [&points_to_set, &operand_points_to_set](
          const ShapeIndex& src_index,
          const PointsToSet::BufferList& points_to) {
        if (!src_index.empty() && src_index.front() == 1) {
          const ShapeIndex target_index(src_index.begin() + 1, src_index.end());
          *points_to_set.mutable_element(target_index) = points_to;

          for (HloInstruction* tuple :
               operand_points_to_set.tuple_sources(src_index)) {
            points_to_set.add_tuple_source(target_index, tuple);
          }
        }
      });

  return Status::OK();
}

Status TuplePointsToAnalysis::HandleCopyStart(HloInstruction* copy_start) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStuple_points_to_analysisDTcc mht_25(mht_25_v, 630, "", "./tensorflow/compiler/xla/service/tuple_points_to_analysis.cc", "TuplePointsToAnalysis::HandleCopyStart");

  // CopyStart forwards its aliased operand to {1}.
  PointsToSet& points_to_set = CreateEmptyPointsToSet(copy_start);
  const PointsToSet& operand_points_to_set =
      GetPointsToSet(copy_start->operand(0));

  points_to_set.ForEachMutableElement(
      [&](const ShapeIndex& target_index, PointsToSet::BufferList* buffers) {
        if (target_index == ShapeIndex({1})) {
          *buffers = operand_points_to_set.element(/*index=*/{});
        } else {
          buffers->push_back(
              &logical_buffer_analysis_->GetBuffer(copy_start, target_index));
        }
      });

  for (HloInstruction* tuple :
       operand_points_to_set.tuple_sources(/*index=*/{})) {
    points_to_set.add_tuple_source(/*index=*/{1}, tuple);
  }

  return Status::OK();
}

Status TuplePointsToAnalysis::HandleCopyDone(HloInstruction* copy_done) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStuple_points_to_analysisDTcc mht_26(mht_26_v, 657, "", "./tensorflow/compiler/xla/service/tuple_points_to_analysis.cc", "TuplePointsToAnalysis::HandleCopyDone");

  // CopyDone forwards its aliased operand.
  PointsToSet& points_to_set = CreateEmptyPointsToSet(copy_done);
  const PointsToSet& operand_points_to_set =
      GetPointsToSet(copy_done->operand(0));
  operand_points_to_set.ForEachElement(
      [&points_to_set, &operand_points_to_set](
          const ShapeIndex& src_index,
          const PointsToSet::BufferList& points_to) {
        if (src_index == ShapeIndex({0})) {
          const ShapeIndex target_index = {};
          *points_to_set.mutable_element(target_index) = points_to;

          for (HloInstruction* tuple :
               operand_points_to_set.tuple_sources(src_index)) {
            points_to_set.add_tuple_source(target_index, tuple);
          }
        }
      });

  return Status::OK();
}

Status TuplePointsToAnalysis::HandleSend(HloInstruction* send) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStuple_points_to_analysisDTcc mht_27(mht_27_v, 683, "", "./tensorflow/compiler/xla/service/tuple_points_to_analysis.cc", "TuplePointsToAnalysis::HandleSend");

  // Send creates a tuple of {aliased operand, U32 context, token}.
  PointsToSet& points_to_set = CreateEmptyPointsToSet(send);

  // Creates the points to set for the tuple and its element at {1}.
  auto top_buffer = points_to_set.mutable_element(ShapeIndex({}));
  top_buffer->push_back(
      &logical_buffer_analysis_->GetBuffer(send, ShapeIndex({})));
  points_to_set.add_tuple_source({}, send);

  auto context_buffer = points_to_set.mutable_element(ShapeIndex({1}));
  context_buffer->push_back(
      &logical_buffer_analysis_->GetBuffer(send, ShapeIndex({1})));

  auto token_buffer = points_to_set.mutable_element(ShapeIndex({2}));
  token_buffer->push_back(
      &logical_buffer_analysis_->GetBuffer(send, ShapeIndex({2})));

  // Recursively copy the points to set of the operand to output tuple {0}.
  const PointsToSet& operand_points_to_set = GetPointsToSet(send->operand(0));
  operand_points_to_set.ForEachElement(
      [&points_to_set, &operand_points_to_set](
          const ShapeIndex& src_index,
          const PointsToSet::BufferList& points_to) {
        ShapeIndex target_index({0});
        for (auto element : src_index) {
          target_index.push_back(element);
        }
        *points_to_set.mutable_element(target_index) = points_to;

        for (HloInstruction* tuple :
             operand_points_to_set.tuple_sources(src_index)) {
          points_to_set.add_tuple_source(target_index, tuple);
        }
      });

  return Status::OK();
}

Status TuplePointsToAnalysis::HandleTuple(HloInstruction* tuple) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStuple_points_to_analysisDTcc mht_28(mht_28_v, 725, "", "./tensorflow/compiler/xla/service/tuple_points_to_analysis.cc", "TuplePointsToAnalysis::HandleTuple");

  absl::Span<HloInstruction* const> operands(tuple->operands());
  PointsToSet& points_to_set = CreateEmptyPointsToSet(tuple);
  points_to_set.AddPointedToBuffer(
      logical_buffer_analysis_->GetBuffer(tuple, /*index=*/{}),
      /*index=*/{});

  // A tuple contains references to all input operands and transitively any
  // references in those operands.
  for (int64_t i = 0; i < operands.size(); ++i) {
    const PointsToSet& operand_points_to_set =
        *PerInst(operands[i])->points_to_set;

    // Copy the points-to set (and tuple sources) of the operand into the
    // respective subtree of the tuple instructions points-to set.
    operand_points_to_set.ForEachElement(
        [&points_to_set, &operand_points_to_set, i](
            const ShapeIndex& src_index,
            const PointsToSet::BufferList& points_to) {
          ShapeIndex target_index;
          target_index.push_back(i);
          for (auto element : src_index) {
            target_index.push_back(element);
          }

          *points_to_set.mutable_element(target_index) = points_to;

          for (HloInstruction* tuple :
               operand_points_to_set.tuple_sources(src_index)) {
            points_to_set.add_tuple_source(target_index, tuple);
          }
        });
  }

  points_to_set.add_tuple_source({}, tuple);

  return Status::OK();
}

Status TuplePointsToAnalysis::HandleTupleSelect(HloInstruction* tuple_select) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStuple_points_to_analysisDTcc mht_29(mht_29_v, 767, "", "./tensorflow/compiler/xla/service/tuple_points_to_analysis.cc", "TuplePointsToAnalysis::HandleTupleSelect");

  // Select allocates a new buffer and then shallow copies the on_true or
  // on_false buffer into this new buffer. Which side is chosen cannot be
  // determined statically so conservatively set the points-to set to the union
  // of these on_true and on_false operands.
  //
  // First create a copy of the on_true points-to set (and tuple sources), then
  // add in elements of the on_false points-to set (tuple sources).
  auto on_true = tuple_select->operand(1);
  auto on_false = tuple_select->operand(2);
  PointsToSet& points_to_set = CreateCopiedPointsToSet(tuple_select, on_true);
  const PointsToSet& false_points_to_set = *PerInst(on_false)->points_to_set;
  points_to_set.ForEachMutableElement(
      [&](const ShapeIndex& index, PointsToSet::BufferList* buffers) {
        for (const LogicalBuffer* false_buffer :
             false_points_to_set.element(index)) {
          points_to_set.AddPointedToBuffer(*false_buffer, index);
        }

        for (HloInstruction* tuple : false_points_to_set.tuple_sources(index)) {
          points_to_set.add_tuple_source(index, tuple);
        }
      });

  // Select creates a new (top-level) buffer to store its result, so its
  // respective element in the points-to set should contain only itself.
  points_to_set.mutable_element({})->clear();
  points_to_set.AddPointedToBuffer(
      logical_buffer_analysis_->GetBuffer(tuple_select, /*index=*/{}),
      /*index=*/{});
  return Status::OK();
}

Status TuplePointsToAnalysis::HandleCustomCall(HloInstruction* custom_call) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStuple_points_to_analysisDTcc mht_30(mht_30_v, 803, "", "./tensorflow/compiler/xla/service/tuple_points_to_analysis.cc", "TuplePointsToAnalysis::HandleCustomCall");

  auto ccall = Cast<HloCustomCallInstruction>(custom_call);
  PointsToSet& points_to_set = CreateEmptyPointsToSet(custom_call);
  absl::flat_hash_map<ShapeIndex, std::pair<int64_t, ShapeIndex>>
      aliased_outputs;
  for (const auto& pair : ccall->output_to_operand_aliasing()) {
    aliased_outputs.emplace(pair.first, pair.second);
  }
  points_to_set.ForEachMutableElement([&](const ShapeIndex& index,
                                          PointsToSet::BufferList* buffers) {
    auto it = aliased_outputs.find(index);
    if (it == aliased_outputs.end()) {
      points_to_set.AddPointedToBuffer(
          logical_buffer_analysis_->GetBuffer(custom_call, index), index);
    } else {
      const PointsToSet& input_set =
          *PerInst(ccall->operand(it->second.first))->points_to_set;
      for (const LogicalBuffer* input_buffer :
           input_set.element(it->second.second)) {
        points_to_set.AddPointedToBuffer(*input_buffer, index);
      }

      for (HloInstruction* tuple : input_set.tuple_sources(it->second.second)) {
        points_to_set.add_tuple_source(index, tuple);
      }
    }
  });
  points_to_set.add_tuple_source({}, custom_call);
  return Status::OK();
}

Status TuplePointsToAnalysis::HandleOptimizationBarrier(
    HloInstruction* barrier) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStuple_points_to_analysisDTcc mht_31(mht_31_v, 838, "", "./tensorflow/compiler/xla/service/tuple_points_to_analysis.cc", "TuplePointsToAnalysis::HandleOptimizationBarrier");

  // A kOptimizationBarrier instruction is a no-op.
  CreateCopiedPointsToSet(barrier, barrier->operand(0));
  return Status::OK();
}

const PointsToSet& TuplePointsToAnalysis::GetPointsToSet(
    const HloInstruction* hlo_instruction) const {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStuple_points_to_analysisDTcc mht_32(mht_32_v, 848, "", "./tensorflow/compiler/xla/service/tuple_points_to_analysis.cc", "TuplePointsToAnalysis::GetPointsToSet");

  return *PerInst(hlo_instruction)->points_to_set;
}

PointsToSet& TuplePointsToAnalysis::CreateEmptyPointsToSet(
    const HloInstruction* instruction) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStuple_points_to_analysisDTcc mht_33(mht_33_v, 856, "", "./tensorflow/compiler/xla/service/tuple_points_to_analysis.cc", "TuplePointsToAnalysis::CreateEmptyPointsToSet");

  PerInstruction* pi = PerInst(instruction);
  CHECK(pi->points_to_set == nullptr)
      << "instruction should not have been present in the map.";
  auto set = absl::make_unique<PointsToSet>(&instruction->shape());
  pi->points_to_set = std::move(set);
  // Return *set using the iterator returned by emplace.
  return *pi->points_to_set;
}

bool TuplePointsToAnalysis::InstructionDefinesBufferAtIndex(
    const HloInstruction* instruction, const ShapeIndex& index) const {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStuple_points_to_analysisDTcc mht_34(mht_34_v, 870, "", "./tensorflow/compiler/xla/service/tuple_points_to_analysis.cc", "TuplePointsToAnalysis::InstructionDefinesBufferAtIndex");

  const auto& buffers = GetPointsToSet(instruction).element(index);
  return (buffers.size() == 1 && buffers[0]->instruction() == instruction);
}

Status TuplePointsToAnalysis::VerifyBuffer(const LogicalBuffer& buffer) const {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStuple_points_to_analysisDTcc mht_35(mht_35_v, 878, "", "./tensorflow/compiler/xla/service/tuple_points_to_analysis.cc", "TuplePointsToAnalysis::VerifyBuffer");

  if (!InstructionDefinesBufferAtIndex(buffer.instruction(), buffer.index())) {
    return FailedPrecondition(
        "LogicalBuffer %s is ill-defined: instruction %s does not define a "
        "buffer at that index",
        buffer.ToString(), buffer.instruction()->name());
  }

  if (buffer.id() < 0 ||
      buffer.id() >= logical_buffer_analysis_->num_logical_buffers()) {
    return FailedPrecondition("LogicalBuffer %s is ill-defined: invalid id %d",
                              buffer.ToString(), buffer.id());
  }
  if (GetBuffer(buffer.id()).instruction() != buffer.instruction() ||
      GetBuffer(buffer.id()).index() != buffer.index()) {
    return FailedPrecondition(
        "LogicalBuffer %s is ill-defined: buffer with same id differs: %s",
        buffer.ToString(), GetBuffer(buffer.id()).ToString());
  }

  return Status::OK();
}

const LogicalBuffer& TuplePointsToAnalysis::GetBuffer(
    LogicalBuffer::Id id) const {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStuple_points_to_analysisDTcc mht_36(mht_36_v, 905, "", "./tensorflow/compiler/xla/service/tuple_points_to_analysis.cc", "TuplePointsToAnalysis::GetBuffer");

  CHECK_GE(id, 0);
  CHECK_LT(id, logical_buffer_analysis_->num_logical_buffers());
  return logical_buffer_analysis_->GetBuffer(id);
}

StatusOr<const LogicalBuffer*> TuplePointsToAnalysis::GetBufferDefinedAt(
    const HloInstruction* instruction, const ShapeIndex& index) const {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStuple_points_to_analysisDTcc mht_37(mht_37_v, 915, "", "./tensorflow/compiler/xla/service/tuple_points_to_analysis.cc", "TuplePointsToAnalysis::GetBufferDefinedAt");

  const auto& buffers = GetPointsToSet(instruction).element(index);
  if (buffers.size() != 1 || buffers[0]->instruction() != instruction) {
    return FailedPrecondition(
        "instruction %s does not define buffer at index {%s}",
        instruction->name(), absl::StrJoin(index, ","));
  }
  return buffers[0];
}

const TuplePointsToAnalysis::BufferAliasVector&
TuplePointsToAnalysis::GetBufferAliases(const LogicalBuffer& buffer) const {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStuple_points_to_analysisDTcc mht_38(mht_38_v, 929, "", "./tensorflow/compiler/xla/service/tuple_points_to_analysis.cc", "TuplePointsToAnalysis::GetBufferAliases");

  return logical_buffer_aliases_.at(buffer.id());
}

const TuplePointsToAnalysis::BufferDefinitionVector&
TuplePointsToAnalysis::GetBuffersDefinedByInstruction(
    const HloInstruction* instruction) const {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStuple_points_to_analysisDTcc mht_39(mht_39_v, 938, "", "./tensorflow/compiler/xla/service/tuple_points_to_analysis.cc", "TuplePointsToAnalysis::GetBuffersDefinedByInstruction");

  return PerInst(instruction)->instruction_defined_buffers;
}

Status TuplePointsToAnalysis::GatherBuffersDefinedByInstruction(
    const HloInstruction* instruction,
    TuplePointsToAnalysis::BufferDefinitionVector* buffers) {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStuple_points_to_analysisDTcc mht_40(mht_40_v, 947, "", "./tensorflow/compiler/xla/service/tuple_points_to_analysis.cc", "TuplePointsToAnalysis::GatherBuffersDefinedByInstruction");

  GetPointsToSet(instruction)
      .ForEachElement([buffers, instruction](
                          const ShapeIndex& index,
                          const PointsToSet::BufferList& source_buffers) {
        // Add buffers which 'instruction' is the source of.
        CHECK(!source_buffers.empty());
        if (source_buffers.size() == 1 &&
            source_buffers[0]->instruction() == instruction) {
          // If this instruction is the source of this buffer the
          // indices must match.
          DCHECK(source_buffers[0]->index() == index);
          buffers->push_back(source_buffers[0]);
        } else {
          // If the points-to set includes more than one buffer then
          // necessarily this instruction did not produce the
          // buffer.
          for (const LogicalBuffer* source_buffer : source_buffers) {
            DCHECK(source_buffer->instruction() != instruction);
          }
        }
      });
  return Status::OK();
}

PointsToSet& TuplePointsToAnalysis::CreateCopiedPointsToSet(
    const HloInstruction* instruction, const HloInstruction* src) {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStuple_points_to_analysisDTcc mht_41(mht_41_v, 976, "", "./tensorflow/compiler/xla/service/tuple_points_to_analysis.cc", "TuplePointsToAnalysis::CreateCopiedPointsToSet");

  // PointsToSet doesn't have a copy constructor so copy over element-by-element
  // from src PointsToSet.
  PointsToSet& dst_points_to_set = CreateEmptyPointsToSet(instruction);
  const PointsToSet& src_points_to_set = GetPointsToSet(src);
  dst_points_to_set.ForEachMutableElement(
      [&dst_points_to_set, &src_points_to_set](
          const ShapeIndex& index, PointsToSet::BufferList* buffers) {
        *buffers = src_points_to_set.element(index);
        for (auto& tuple_source : src_points_to_set.tuple_sources(index)) {
          dst_points_to_set.add_tuple_source(index, tuple_source);
        }
      });
  return *PerInst(instruction)->points_to_set;
}

std::string TuplePointsToAnalysis::ToString() const {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStuple_points_to_analysisDTcc mht_42(mht_42_v, 995, "", "./tensorflow/compiler/xla/service/tuple_points_to_analysis.cc", "TuplePointsToAnalysis::ToString");

  std::string output =
      absl::StrFormat("TuplePointsToSet for module %s:\n", module_->name());
  for (const auto* computation : module_->MakeNonfusionComputations()) {
    const char* entry =
        computation == module_->entry_computation() ? "entry " : "";
    absl::StrAppend(&output, entry, "computation ", computation->name(), ":\n");
    for (const HloInstruction* instruction :
         computation->MakeInstructionPostOrder()) {
      InstructionToString(instruction, &output);
      if (instruction->opcode() == HloOpcode::kFusion) {
        for (auto* fused : instruction->fused_instructions()) {
          InstructionToString(fused, &output);
        }
      }
    }
  }

  absl::StrAppend(&output, "LogicalBuffers:\n");
  for (const auto& b : logical_buffer_analysis_->logical_buffers()) {
    absl::StrAppend(&output, "  buffer ", b->ToString(), ":\n");
    for (const BufferAlias& alias : logical_buffer_aliases_.at(b->id())) {
      absl::StrAppend(&output, "    alias ", alias.ToString(), "\n");
    }
  }
  return output;
}

void TuplePointsToAnalysis::InstructionToString(
    const HloInstruction* instruction, std::string* output) const {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStuple_points_to_analysisDTcc mht_43(mht_43_v, 1027, "", "./tensorflow/compiler/xla/service/tuple_points_to_analysis.cc", "TuplePointsToAnalysis::InstructionToString");

  const std::string prefix = instruction->IsFused() ? "    " : "";
  absl::StrAppend(output, prefix, "  instruction ",
                  instruction->ToShortString(), ":\n");
  const PointsToSet& points_to_set = GetPointsToSet(instruction);
  points_to_set.ForEachElement(
      [&prefix, &output](const ShapeIndex& index,
                         const PointsToSet::BufferList& points_to) {
        absl::StrAppend(
            output, prefix, "    {", absl::StrJoin(index, ","), "}: ",
            absl::StrJoin(points_to, ", ",
                          [](std::string* out, const LogicalBuffer* source) {
                            out->append(source->ToString());
                          }),
            "\n");
      });
}

bool TuplePointsToAnalysis::DoesNotUseOperandBuffer(
    const HloInstruction* operand, const ShapeIndex& index,
    const HloInstruction* user) const {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStuple_points_to_analysisDTcc mht_44(mht_44_v, 1050, "", "./tensorflow/compiler/xla/service/tuple_points_to_analysis.cc", "TuplePointsToAnalysis::DoesNotUseOperandBuffer");

  CHECK(user->IsUserOf(operand))
      << "user: " << user->ToString() << " operand: " << operand->ToString();
  if (user->opcode() == HloOpcode::kGetTupleElement && !index.empty()) {
    // GetTupleElement instructions only access the top-level buffer of their
    // operand.
    return true;
  } else if (user->IsLoopFusion()) {
    // Find fusion parameter associated with 'operand'.
    auto it = absl::c_find_if(
        user->fused_parameters(), [&](HloInstruction* fused_param) {
          return user->operand(fused_param->parameter_number()) == operand;
        });
    CHECK(it != user->fused_parameters().end());
    // Iterate through all users of all buffer aliases of the buffer in the
    // points-to set of fusion parameter at 'index'.
    // Return false if any uses are detected at 'index', returns true otherwise.
    const LogicalBuffer* buffer = GetBufferDefinedAt(*it, index).ValueOrDie();
    for (const BufferAlias& alias : GetBufferAliases(*buffer)) {
      for (HloInstruction* alias_user : alias.instruction()->users()) {
        if (DoesNotUseOperandBuffer(alias.instruction(), alias.index(),
                                    alias_user)) {
          continue;
        }
        // Return false: use detected at 'buffer' -> 'alias' -> 'alias_user'.
        return false;
      }
    }
    // Return true: found no uses of 'operand' at 'index' in 'user'.
    return true;
  }
  return false;
}

// Returns all uses of all aliases of 'instruction' at 'index' in 'uses'.
// Each use in 'uses' is a pair (HloInstruction* user, int64_t operand_index)
// where 'user' is a user of an alias of 'instruction' at 'index', and
// 'operand_index' is the operand index at which the alias appears in the
// operand list of 'user'.
std::vector<std::pair<HloInstruction*, int64_t>>
TuplePointsToAnalysis::GetAllUsesOfInstructionAtIndex(
    HloInstruction* instruction, const ShapeIndex& index) const {
  std::vector<std::pair<HloInstruction*, int64_t>> uses;
  const PointsToSet::BufferList& points_to =
      GetPointsToSet(instruction).element(index);
  for (const LogicalBuffer* buffer : points_to) {
    for (const BufferAlias& alias : GetBufferAliases(*buffer)) {
      for (HloInstruction* alias_user : alias.instruction()->users()) {
        if (DoesNotUseOperandBuffer(alias.instruction(), alias.index(),
                                    alias_user)) {
          continue;
        }
        for (int64_t op_idx : alias_user->OperandIndices(alias.instruction())) {
          uses.emplace_back(alias_user, op_idx);
        }
      }
    }
  }
  return uses;
}

// Returns true if there is exactly one use of 'operand' at 'operand_index'
// in 'fusion.fused_instructions', where the singleton use is the fused
// root at operand index 'use_operand_index'. Returns false otherwise.
//
// REQUIRES: 'fusion' opcode is a kFusion instruction.
bool TuplePointsToAnalysis::HasUniqueFusedUseOfOperandAt(
    HloInstruction* operand, const ShapeIndex& operand_index,
    HloInstruction* fusion, const int64_t use_operand_index) const {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStuple_points_to_analysisDTcc mht_45(mht_45_v, 1121, "", "./tensorflow/compiler/xla/service/tuple_points_to_analysis.cc", "TuplePointsToAnalysis::HasUniqueFusedUseOfOperandAt");

  CHECK_EQ(HloOpcode::kFusion, fusion->opcode());
  // Check that 'operand' is unique in the operand list of 'fusion'.
  if (fusion->OperandIndices(operand).size() > 1) {
    return false;
  }
  // Find fusion parameter associated with 'operand'.
  const auto& fused_params = fusion->fused_parameters();
  auto fused_param_it =
      absl::c_find_if(fused_params, [&](HloInstruction* fused_param) {
        return fusion->operand(fused_param->parameter_number()) == operand;
      });
  if (fused_param_it == fused_params.end()) {
    return false;
  }
  auto* fused_param = *fused_param_it;
  // Get all uses of 'operand' at 'index' from 'fusion.fused_instructions'.
  auto fused_param_uses =
      GetAllUsesOfInstructionAtIndex(fused_param, operand_index);
  // Return true iff there is exactly one use of 'operand' at 'index', and
  // this singleton use is the fused root (at index in 'use_operand_indices').
  return fused_param_uses.size() == 1 &&
         fused_param_uses[0].first == fusion->fused_expression_root() &&
         fused_param_uses[0].second == use_operand_index;
}
}  // namespace xla
