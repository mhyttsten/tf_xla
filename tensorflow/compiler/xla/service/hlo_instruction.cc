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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc() {
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

#include "tensorflow/compiler/xla/service/hlo_instruction.h"

#include <algorithm>
#include <functional>
#include <iostream>
#include <ostream>
#include <set>
#include <string>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/memory/memory.h"
#include "absl/strings/ascii.h"
#include "absl/strings/escaping.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/protobuf_util.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_op_metadata.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_sharding_metadata.h"
#include "tensorflow/compiler/xla/service/name_uniquer.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/human_readable_json.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

using absl::CEscape;
using absl::StrAppend;
using absl::StrCat;
using absl::StrJoin;

HloInstruction* HloInstruction::AddInstruction(
    std::unique_ptr<HloInstruction> derived_instruction) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_0(mht_0_v, 237, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::AddInstruction");

  HloInstruction* derived =
      parent()->AddInstruction(std::move(derived_instruction));
  const bool has_prior_sharding = derived->has_sharding();
  SetupDerivedInstruction(derived);
  if (!has_prior_sharding && (derived->opcode() == HloOpcode::kReshape ||
                              derived->opcode() == HloOpcode::kTranspose)) {
    derived->clear_sharding();
  }
  return derived;
}

/* static */
StatusOr<std::unique_ptr<HloInstruction>> HloInstruction::CreateFromProto(
    const HloInstructionProto& proto,
    const absl::flat_hash_map<int64_t, HloInstruction*>& instruction_map,
    const absl::flat_hash_map<int64_t, HloComputation*>& computation_map,
    bool prohibit_empty_literal) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_1(mht_1_v, 257, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CreateFromProto");

  TF_RET_CHECK(!proto.opcode().empty());
  HloOpcode opcode;
  auto opcode_or = StringToHloOpcode(proto.opcode());
  absl::optional<ComparisonDirection> comparison_direction;
  if (opcode_or.ok()) {
    opcode = opcode_or.ConsumeValueOrDie();
  } else {
    // Unknown opcode. Try auto-upgrading deprecated "less-than",
    // "greater-than", etc opcodes, which are now rolled into the kCompare
    // opcode.
    if (proto.opcode() == "equal-to") {
      comparison_direction = ComparisonDirection::kEq;
    } else if (proto.opcode() == "not-equal-to") {
      comparison_direction = ComparisonDirection::kNe;
    } else if (proto.opcode() == "greater-than-or-equal-to") {
      comparison_direction = ComparisonDirection::kGe;
    } else if (proto.opcode() == "greater-than") {
      comparison_direction = ComparisonDirection::kGt;
    } else if (proto.opcode() == "less-than-or-equal-to") {
      comparison_direction = ComparisonDirection::kLe;
    } else if (proto.opcode() == "less-than") {
      comparison_direction = ComparisonDirection::kLt;
    }
    if (comparison_direction) {
      opcode = HloOpcode::kCompare;
    } else {
      return InvalidArgument("Unknown opcode: %s", proto.opcode());
    }
  }

  TF_RET_CHECK(proto.has_shape());

  std::unique_ptr<HloInstruction> instruction;
  const auto operands = [&instruction_map, &proto](int index) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_2(mht_2_v, 294, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "lambda");

    return instruction_map.at(proto.operand_ids(index));
  };
  const auto all_operands = [&instruction_map, &proto]() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_3(mht_3_v, 300, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "lambda");

    std::vector<HloInstruction*> result(proto.operand_ids_size());
    std::transform(proto.operand_ids().begin(), proto.operand_ids().end(),
                   result.begin(), [&instruction_map](int64_t operand_id) {
                     return instruction_map.at(operand_id);
                   });
    return result;
  };
  const auto computations = [&computation_map, &proto](int index) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_4(mht_4_v, 311, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "lambda");

    return computation_map.at(proto.called_computation_ids(index));
  };
  const auto all_computations = [&computation_map, &proto]() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_5(mht_5_v, 317, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "lambda");

    std::vector<HloComputation*> result(proto.called_computation_ids_size());
    std::transform(proto.called_computation_ids().begin(),
                   proto.called_computation_ids().end(), result.begin(),
                   [&computation_map](int64_t computation_id) {
                     return computation_map.at(computation_id);
                   });
    return result;
  };

  TF_RET_CHECK(
      absl::c_all_of(proto.operand_ids(),
                     [&](int64_t id) { return instruction_map.contains(id); }))
      << proto.name() << " instruction contains invalid operand id(s)";

  TF_RET_CHECK(
      absl::c_all_of(proto.called_computation_ids(),
                     [&](int64_t id) { return computation_map.contains(id); }))
      << proto.name() << " instruction references invalid computation id(s)";

  Shape shape(proto.shape());
  TF_RETURN_IF_ERROR(ShapeUtil::ValidateShapeWithOptionalLayout(shape));

  absl::optional<int> arity = HloOpcodeArity(opcode);
  if (arity) {
    TF_RET_CHECK(proto.operand_ids_size() == *arity)
        << proto.opcode() << " instruction should have " << *arity
        << " operands but sees " << proto.operand_ids_size();
  }

  switch (opcode) {
    // Ops migrated to subclasses.
    case HloOpcode::kBatchNormTraining:
      instruction =
          CreateBatchNormTraining(shape, operands(0), operands(1), operands(2),
                                  proto.epsilon(), proto.feature_index());
      break;
    case HloOpcode::kBatchNormInference:
      instruction = CreateBatchNormInference(
          shape, operands(0), operands(1), operands(2), operands(3),
          operands(4), proto.epsilon(), proto.feature_index());
      break;
    case HloOpcode::kBatchNormGrad:
      instruction = CreateBatchNormGrad(shape, operands(0), operands(1),
                                        operands(2), operands(3), operands(4),
                                        proto.epsilon(), proto.feature_index());
      break;
    case HloOpcode::kFft: {
      std::vector<int64_t> fft_length(proto.fft_length().begin(),
                                      proto.fft_length().end());
      instruction = CreateFft(shape, operands(0), proto.fft_type(),
                              absl::Span<const int64_t>(fft_length));
      break;
    }
    case HloOpcode::kAsyncStart: {
      TF_RET_CHECK(proto.called_computation_ids_size() == 1)
          << "Async start instruction should have 1 called computation but "
             "sees "
          << proto.called_computation_ids_size();
      instruction = CreateAsyncStart(shape, all_operands(), computations(0));
      break;
    }
    case HloOpcode::kAsyncUpdate: {
      TF_RET_CHECK(proto.called_computation_ids_size() == 1)
          << "Async update instruction should have 1 called computation but "
             "sees "
          << proto.called_computation_ids_size();
      instruction = CreateAsyncUpdate(shape, operands(0), computations(0));
      break;
    }
    case HloOpcode::kAsyncDone: {
      TF_RET_CHECK(proto.called_computation_ids_size() == 1)
          << "Async done instruction should have 1 called computation but sees "
          << proto.called_computation_ids_size();
      instruction = CreateAsyncDone(shape, operands(0), computations(0));
      break;
    }
    case HloOpcode::kCopyStart: {
      instruction = CreateCopyStart(shape, operands(0),
                                    proto.is_cross_program_prefetch());
      break;
    }
    case HloOpcode::kCompare: {
      // Auto-upgraded from deprecated opcode skips the following.
      if (!comparison_direction) {
        TF_ASSIGN_OR_RETURN(
            comparison_direction,
            StringToComparisonDirection(proto.comparison_direction()));
      }
      auto comparison_type_str = proto.comparison_type();
      if (!comparison_type_str.empty()) {
        // If a comparison type is specified, it *must* be valid.
        TF_ASSIGN_OR_RETURN(auto comparison_type,
                            StringToComparisonType(comparison_type_str));
        instruction = CreateCompare(shape, operands(0), operands(1),
                                    *comparison_direction, comparison_type);
      } else {
        // Allow the specify of comparison type to be optional.
        // The comparison type will be determined by the types of the operands.
        instruction = CreateCompare(shape, operands(0), operands(1),
                                    *comparison_direction);
      }
      break;
    }
    case HloOpcode::kTriangularSolve: {
      instruction = CreateTriangularSolve(shape, operands(0), operands(1),
                                          proto.triangular_solve_options());
      break;
    }
    case HloOpcode::kCholesky: {
      instruction =
          CreateCholesky(shape, operands(0), proto.cholesky_options());
      break;
    }
    case HloOpcode::kSend:
      instruction = CreateSend(operands(0), operands(1), proto.channel_id(),
                               proto.is_host_transfer());
      break;
    case HloOpcode::kSendDone:
      instruction = CreateSendDone(operands(0), proto.is_host_transfer());
      break;
    case HloOpcode::kRecv:
      instruction = CreateRecv(shape.tuple_shapes(0), operands(0),
                               proto.channel_id(), proto.is_host_transfer());
      break;
    case HloOpcode::kRecvDone:
      instruction = CreateRecvDone(operands(0), proto.is_host_transfer());
      break;
    case HloOpcode::kReverse:
      instruction =
          CreateReverse(shape, operands(0),
                        std::vector<int64_t>(proto.dimensions().begin(),
                                             proto.dimensions().end()));
      break;
    case HloOpcode::kConcatenate:
      TF_RET_CHECK(proto.dimensions_size() == 1)
          << "Concatenate instruction should have 1 dimension but sees "
          << proto.dimensions_size();
      instruction =
          CreateConcatenate(shape, all_operands(), proto.dimensions(0));
      break;
    case HloOpcode::kConditional: {
      TF_RET_CHECK(proto.called_computation_ids_size() > 0)
          << "conditional should have at least 1 called computation";
      if (operands(0)->shape().element_type() == PRED) {
        TF_RET_CHECK(proto.called_computation_ids_size() == 2)
            << "conditional should have exactly 2 called computations but got "
            << proto.called_computation_ids_size();
      }
      TF_RET_CHECK(proto.operand_ids_size() ==
                   proto.called_computation_ids_size() + 1)
          << "conditional should have one branch_index operand plus one "
             "operand per called computation but got "
          << proto.operand_ids_size() << " operands for "
          << proto.called_computation_ids_size() << " branch computations";
      auto cond_operands = all_operands();
      instruction =
          CreateConditional(shape, cond_operands[0], all_computations(),
                            absl::MakeSpan(cond_operands).subspan(1));
      break;
    }
    case HloOpcode::kReduce:
      TF_RET_CHECK(proto.operand_ids_size() % 2 == 0)
          << "Reduce instruction should have an even number of operands but "
             "sees "
          << proto.operand_ids_size();
      TF_RET_CHECK(proto.called_computation_ids_size() == 1)
          << "Reduce instruction should have 1 called computation but sees "
          << proto.called_computation_ids_size();
      {
        const auto reduce_operands = all_operands();
        auto inputs = absl::MakeSpan(reduce_operands)
                          .subspan(0, reduce_operands.size() / 2);
        auto init_values =
            absl::MakeSpan(reduce_operands)
                .subspan(reduce_operands.size() / 2, reduce_operands.size());
        instruction =
            CreateReduce(shape, inputs, init_values,
                         std::vector<int64_t>(proto.dimensions().begin(),
                                              proto.dimensions().end()),
                         computations(0));
      }
      break;
    case HloOpcode::kSort: {
      TF_RET_CHECK(proto.operand_ids_size() >= 1)
          << "Sort instruction should have at least 1 operand but has "
          << proto.operand_ids_size();
      TF_RET_CHECK(proto.dimensions().size() == 1)
          << "Sort instruction should have 1 dimension";
      TF_RET_CHECK(proto.called_computation_ids_size() == 1)
          << "Sort instruction should one called computation but sees "
          << proto.called_computation_ids_size();
      auto sort_operands = all_operands();
      instruction = CreateSort(shape, proto.dimensions(0), all_operands(),
                               computations(0), proto.is_stable());
      break;
    }
    case HloOpcode::kTranspose:
      instruction =
          CreateTranspose(shape, operands(0),
                          std::vector<int64_t>(proto.dimensions().begin(),
                                               proto.dimensions().end()));
      break;
    case HloOpcode::kBroadcast:
      instruction =
          CreateBroadcast(shape, operands(0),
                          std::vector<int64_t>(proto.dimensions().begin(),
                                               proto.dimensions().end()));
      break;
    case HloOpcode::kMap:
      TF_RET_CHECK(proto.called_computation_ids_size() == 1)
          << "Map instruction should have 1 called computation but sees "
          << proto.called_computation_ids_size();
      instruction = CreateMap(shape, all_operands(), computations(0));
      break;
    case HloOpcode::kSlice: {
      std::vector<int64_t> slice_starts, slice_limits, slice_strides;
      for (const HloInstructionProto::SliceDimensions& slice_dimensions :
           proto.slice_dimensions()) {
        slice_starts.push_back(slice_dimensions.start());
        slice_limits.push_back(slice_dimensions.limit());
        slice_strides.push_back(slice_dimensions.stride());
      }
      instruction = CreateSlice(shape, operands(0), slice_starts, slice_limits,
                                slice_strides);
      break;
    }
    case HloOpcode::kConstant: {
      // TODO(b/110214922): Revert this to CHECK(proto.has_literal()).
      if (proto.has_literal()) {
        TF_ASSIGN_OR_RETURN(
            auto literal,
            Literal::CreateFromProto(proto.literal(), prohibit_empty_literal));
        instruction = CreateConstant(std::move(literal));
        // Literal's shape may have no/different tiling info.
        TF_RET_CHECK(Shape::Equal().MinorToMajorOnlyInLayout()(
            instruction->shape(), shape))
            << instruction->shape().ToString(true) << " vs "
            << shape.ToString(true);
        *instruction->mutable_shape() = shape;
      } else {
        instruction = absl::make_unique<HloConstantInstruction>(shape);
      }
      break;
    }
    case HloOpcode::kTrace: {
      TF_RET_CHECK(proto.has_literal());
      TF_ASSIGN_OR_RETURN(
          auto literal,
          Literal::CreateFromProto(proto.literal(), prohibit_empty_literal));
      instruction = CreateTrace(literal.GetR1U8AsString(), operands(0));
      break;
    }
    case HloOpcode::kFusion: {
      // In the proto, fused computations are held exclusively within the
      // HloInstructionProto and do not appear as an HloComputationProto within
      // the HloModuleProto.
      TF_RET_CHECK(!proto.fusion_kind().empty());
      TF_ASSIGN_OR_RETURN(FusionKind fusion_kind,
                          StringToFusionKind(proto.fusion_kind()));

      // Find the fused computation and set its fusion instruction.
      TF_RET_CHECK(proto.called_computation_ids_size() == 1)
          << "Expect 1 called computation for fusion instruction but sees "
          << proto.called_computation_ids_size();
      const int64_t fusion_id = proto.called_computation_ids(0);
      auto* fused_computation =
          tensorflow::gtl::FindPtrOrNull(computation_map, fusion_id);
      TF_RET_CHECK(fused_computation != nullptr)
          << "No fusion computation with id " << fusion_id;
      instruction =
          CreateFusion(shape, fusion_kind, all_operands(), fused_computation);
      break;
    }
    case HloOpcode::kRng:
      instruction = CreateRng(shape, proto.distribution(), all_operands());
      break;
    case HloOpcode::kRngBitGenerator:
      instruction =
          CreateRngBitGenerator(shape, operands(0), proto.rng_algorithm());
      break;
    case HloOpcode::kRngGetAndUpdateState:
      instruction = CreateRngGetAndUpdateState(shape, proto.delta());
      break;
    case HloOpcode::kParameter:
      instruction =
          CreateParameter(proto.parameter_number(), shape, proto.name());
      if (!proto.parameter_replication().replicated_at_leaf_buffers().empty()) {
        instruction->set_parameter_replicated_at_leaf_buffers(
            proto.parameter_replication().replicated_at_leaf_buffers());
      }
      break;
    case HloOpcode::kGetTupleElement:
      instruction =
          CreateGetTupleElement(shape, operands(0), proto.tuple_index());
      break;
    case HloOpcode::kReducePrecision:
      instruction = CreateReducePrecision(
          shape, operands(0), proto.exponent_bits(), proto.mantissa_bits());
      break;
    case HloOpcode::kInfeed: {
      TF_RET_CHECK(shape.IsTuple() &&
                   (ShapeUtil::TupleElementCount(shape) == 2))
          << "Infeed should have a tuple shape with 2 operands, but has: "
          << shape;
      const Shape& data_shape = ShapeUtil::GetTupleElementShape(shape, 0);
      instruction =
          CreateInfeed(data_shape, operands(0), proto.infeed_config());
    } break;
    case HloOpcode::kOutfeed: {
      Shape outfeed_shape(proto.outfeed_shape());
      TF_RETURN_IF_ERROR(
          ShapeUtil::ValidateShapeWithOptionalLayout(outfeed_shape));
      instruction = CreateOutfeed(outfeed_shape, operands(0), operands(1),
                                  proto.outfeed_config());
      break;
    }
    case HloOpcode::kAllGather:
    case HloOpcode::kAllGatherStart: {
      absl::optional<int64_t> channel_id;
      if (proto.channel_id() > 0) {
        channel_id = proto.channel_id();
      }

      TF_RET_CHECK(proto.dimensions_size() == 1)
          << "AllGather cannot have more than 1 all-gather dimensions";
      int64_t all_gather_dimension = proto.dimensions(0);
      if (opcode == HloOpcode::kAllGather) {
        instruction = CreateAllGather(
            shape, all_operands(), all_gather_dimension,
            std::vector<ReplicaGroup>(proto.replica_groups().begin(),
                                      proto.replica_groups().end()),
            proto.constrain_layout(), channel_id,
            proto.use_global_device_ids());
      } else {
        instruction = CreateAllGatherStart(
            shape, all_operands(), all_gather_dimension,
            std::vector<ReplicaGroup>(proto.replica_groups().begin(),
                                      proto.replica_groups().end()),
            proto.constrain_layout(), channel_id,
            proto.use_global_device_ids());
      }
      break;
    }
    case HloOpcode::kAllReduce:
    case HloOpcode::kAllReduceStart:
    case HloOpcode::kReduceScatter: {
      TF_RET_CHECK(proto.called_computation_ids_size() == 1)
          << "AllReduce should have 1 called computation but sees "
          << proto.called_computation_ids_size();
      TF_RET_CHECK(proto.channel_id() <= 0 || proto.all_reduce_id() <= 0)
          << "AllReduce cannot have both channel_id() and all_reduce_id()";
      absl::optional<int64_t> channel_id;
      if (proto.channel_id() > 0) {
        channel_id = proto.channel_id();
      }
      if (proto.all_reduce_id() > 0) {
        channel_id = proto.all_reduce_id();
      }
      std::vector<ReplicaGroup> replica_groups(proto.replica_groups().begin(),
                                               proto.replica_groups().end());
      if (opcode == HloOpcode::kAllReduce) {
        instruction =
            CreateAllReduce(shape, all_operands(), computations(0),
                            replica_groups, proto.constrain_layout(),
                            channel_id, proto.use_global_device_ids());
      } else if (opcode == HloOpcode::kReduceScatter) {
        TF_RET_CHECK(proto.dimensions_size() == 1)
            << "ReduceScatter cannot have more than 1 scatter dimensions";
        int64_t scatter_dimension = proto.dimensions(0);
        instruction = CreateReduceScatter(
            shape, all_operands(), computations(0), replica_groups,
            proto.constrain_layout(), channel_id, proto.use_global_device_ids(),
            scatter_dimension);
      } else {
        instruction =
            CreateAllReduceStart(shape, all_operands(), computations(0),
                                 replica_groups, proto.constrain_layout(),
                                 channel_id, proto.use_global_device_ids());
      }
      break;
    }
    case HloOpcode::kAllToAll: {
      absl::optional<int64_t> channel_id;
      if (proto.channel_id() > 0) {
        channel_id = proto.channel_id();
      }
      absl::optional<int64_t> split_dimension;
      if (proto.dimensions_size() > 0) {
        TF_RET_CHECK(proto.dimensions_size() == 1)
            << "AllToAll cannot have more than 1 dimension (split dimension)";
        TF_RET_CHECK(all_operands().size() == 1)
            << "AllToAll must have a single operand when the split dimension "
               "is specified";
        split_dimension = proto.dimensions(0);
      }
      instruction = CreateAllToAll(
          shape, all_operands(),
          /*replica_groups=*/
          std::vector<ReplicaGroup>(proto.replica_groups().begin(),
                                    proto.replica_groups().end()),
          /*constrain_layout=*/proto.constrain_layout(),
          /*channel_id=*/channel_id, split_dimension);
      break;
    }
    case HloOpcode::kCollectivePermute:
    case HloOpcode::kCollectivePermuteStart: {
      TF_RET_CHECK(proto.operand_ids().size() == 1 ||
                   proto.operand_ids().size() == 4);
      std::vector<std::pair<int64_t, int64_t>> source_target_pairs(
          proto.source_target_pairs_size());
      absl::optional<int64_t> channel_id;
      if (proto.channel_id() > 0) {
        channel_id = proto.channel_id();
      }
      for (int i = 0; i < source_target_pairs.size(); ++i) {
        source_target_pairs[i].first = proto.source_target_pairs(i).source();
        source_target_pairs[i].second = proto.source_target_pairs(i).target();
      }
      if (proto.dynamic_slice_sizes_size() == 0) {
        if (opcode == HloOpcode::kCollectivePermute) {
          instruction = CreateCollectivePermute(
              shape, operands(0), source_target_pairs, channel_id);
        } else if (opcode == HloOpcode::kCollectivePermuteStart) {
          instruction = CreateCollectivePermuteStart(
              shape, operands(0), source_target_pairs, channel_id);
        } else {
          LOG(FATAL) << "Expect CollectivePermute or CollectivePermuteStart, "
                     << "but got " << HloOpcodeString(opcode);
        }
      } else {
        std::vector<std::vector<int64_t>> slice_sizes;
        HloInstruction* input = operands(0);
        HloInstruction* input_start_indices = operands(2);
        if (input->shape().IsTuple() &&
            input->shape().tuple_shapes_size() > 1) {
          slice_sizes.resize(input->shape().tuple_shapes_size());
        } else {
          slice_sizes.resize(1);
        }
        int proto_index = 0;
        if (input->shape().IsTuple()) {
          if (input_start_indices->shape()
                  .tuple_shapes(0)
                  .tuple_shapes(0)
                  .IsArray()) {
            slice_sizes.resize(input->shape().tuple_shapes_size());
            for (int i = 0; i < input->shape().tuple_shapes_size(); ++i) {
              slice_sizes[i].resize(
                  input->shape().tuple_shapes(i).dimensions_size());
              for (int j = 0;
                   j < input->shape().tuple_shapes(i).dimensions_size(); ++j) {
                CHECK_GE(proto.dynamic_slice_sizes_size(), proto_index);
                slice_sizes[i][j] = proto.dynamic_slice_sizes(proto_index);
                proto_index += 1;
              }
            }
          } else {
            slice_sizes.resize(
                input->shape().tuple_shapes_size() *
                ShapeUtil::TupleElementCount(
                    input_start_indices->shape().tuple_shapes(0)));
            int slice_sizes_count = 0;
            for (int i = 0; i < input->shape().tuple_shapes_size(); ++i) {
              for (int j = 0;
                   j < ShapeUtil::TupleElementCount(
                           input_start_indices->shape().tuple_shapes(i));
                   ++j) {
                slice_sizes[slice_sizes_count].resize(
                    input->shape().tuple_shapes(i).rank());
                for (int k = 0; k < input->shape().tuple_shapes(i).rank();
                     ++k) {
                  CHECK_GE(proto.dynamic_slice_sizes_size(), proto_index);
                  slice_sizes[slice_sizes_count][k] =
                      proto.dynamic_slice_sizes(proto_index);
                  proto_index += 1;
                }
                slice_sizes_count += 1;
              }
            }
          }
        } else {
          slice_sizes.resize(
              ShapeUtil::TupleElementCount(input_start_indices->shape()));
          if (input_start_indices->shape().tuple_shapes(0).IsTuple()) {
            for (int i = 0;
                 i < ShapeUtil::TupleElementCount(input_start_indices->shape());
                 ++i) {
              slice_sizes[i].resize(input->shape().dimensions_size());
              for (int j = 0; j < input->shape().dimensions_size(); ++j) {
                slice_sizes[i][j] = proto.dynamic_slice_sizes(proto_index);
                proto_index += 1;
              }
            }
          } else {
            slice_sizes.resize(1);
            slice_sizes[0].resize(input->shape().dimensions_size());
            for (int j = 0; j < input->shape().dimensions_size(); ++j) {
              slice_sizes[0][j] = proto.dynamic_slice_sizes(proto_index);
              proto_index += 1;
            }
          }
        }
        if (opcode == HloOpcode::kCollectivePermute) {
          instruction = CreateCollectivePermute(
              shape, operands(0), operands(1), operands(2), operands(3),
              source_target_pairs, slice_sizes, channel_id);
        } else if (opcode == HloOpcode::kCollectivePermuteStart) {
          instruction = CreateCollectivePermuteStart(
              shape, operands(0), operands(1), operands(2), operands(3),
              source_target_pairs, slice_sizes, channel_id);
        } else {
          LOG(FATAL) << "Expect CollectivePermute or CollectivePermuteStart, "
                     << "but got " << HloOpcodeString(opcode);
        }
      }
      break;
    }
    case HloOpcode::kReplicaId: {
      instruction = CreateReplicaId(shape);
      break;
    }
    case HloOpcode::kPartitionId: {
      instruction = CreatePartitionId(shape);
      break;
    }
    case HloOpcode::kConvolution: {
      TF_RET_CHECK(proto.has_window());
      TF_RET_CHECK(proto.has_convolution_dimension_numbers());
      PrecisionConfig precision_config = proto.precision_config();
      precision_config.mutable_operand_precision()->Resize(
          proto.operand_ids_size(), PrecisionConfig::DEFAULT);
      instruction = CreateConvolve(
          shape, operands(0), operands(1),
          std::max<int64_t>(proto.feature_group_count(), 1),
          std::max<int64_t>(proto.batch_group_count(), 1), proto.window(),
          proto.convolution_dimension_numbers(), precision_config);
      break;
    }
    case HloOpcode::kReduceWindow:
      TF_RET_CHECK(proto.operand_ids_size() % 2 == 0)
          << "Reduce window should have an even number of operands but "
             "sees "
          << proto.operand_ids_size();
      TF_RET_CHECK(proto.called_computation_ids_size() == 1)
          << "ReduceWindow should have 1 called computation but sees "
          << proto.called_computation_ids_size();
      {
        const auto reduce_operands = all_operands();
        auto inputs = absl::MakeSpan(reduce_operands)
                          .subspan(0, reduce_operands.size() / 2);
        auto init_values =
            absl::MakeSpan(reduce_operands)
                .subspan(reduce_operands.size() / 2, reduce_operands.size());
        instruction = CreateReduceWindow(shape, inputs, init_values,
                                         proto.window(), computations(0));
      }
      break;
    case HloOpcode::kSelectAndScatter:
      TF_RET_CHECK(proto.called_computation_ids_size() == 2)
          << "SelectAndScatter should have 2 called computations but sees "
          << proto.called_computation_ids_size();
      instruction = CreateSelectAndScatter(shape, operands(0), computations(0),
                                           proto.window(), operands(1),
                                           operands(2), computations(1));
      break;
    case HloOpcode::kCustomCall: {
      if (proto.constrain_layout()) {
        // A proto RepeatedPtrField cannot be converted to a Span (it is a
        // vector of pointers essentially) so create a vector of shapes to pass
        // in.
        std::vector<Shape> operand_shapes;
        const auto& operand_shapes_with_layout =
            proto.operand_shapes_with_layout();
        operand_shapes.reserve(operand_shapes_with_layout.size());
        for (const ShapeProto& shape_proto : operand_shapes_with_layout) {
          operand_shapes.emplace_back(shape_proto);
        }
        instruction =
            CreateCustomCall(shape, all_operands(), proto.custom_call_target(),
                             operand_shapes, proto.backend_config());
      } else {
        if (proto.called_computation_ids_size() == 1) {
          instruction = CreateCustomCall(shape, all_operands(), computations(0),
                                         proto.custom_call_target(),
                                         proto.backend_config());
        } else if (proto.called_computation_ids_size() > 1) {
          instruction = CreateCustomCall(
              shape, all_operands(), all_computations(),
              proto.custom_call_target(), proto.backend_config());

        } else {
          instruction = CreateCustomCall(shape, all_operands(),
                                         proto.custom_call_target(),
                                         proto.backend_config());
        }
      }
      auto custom_call_instr =
          Cast<HloCustomCallInstruction>(instruction.get());
      if (proto.has_window()) {
        custom_call_instr->set_window(proto.window());
      }
      if (proto.has_literal()) {
        TF_ASSIGN_OR_RETURN(
            auto literal,
            Literal::CreateFromProto(proto.literal(), prohibit_empty_literal));
        custom_call_instr->set_literal(std::move(literal));
      }
      if (proto.has_convolution_dimension_numbers()) {
        custom_call_instr->set_convolution_dimension_numbers(
            proto.convolution_dimension_numbers());
      }
      custom_call_instr->set_feature_group_count(std::max(
          static_cast<int64_t>(proto.feature_group_count()), int64_t{1}));
      custom_call_instr->set_batch_group_count(std::max(
          static_cast<int64_t>(proto.batch_group_count()), int64_t{1}));
      custom_call_instr->set_custom_call_has_side_effect(
          proto.custom_call_has_side_effect());
      custom_call_instr->set_padding_type(proto.padding_type());

      PrecisionConfig precision_config = proto.precision_config();
      precision_config.mutable_operand_precision()->Resize(
          proto.operand_ids_size(), PrecisionConfig::DEFAULT);
      *custom_call_instr->mutable_precision_config() = precision_config;
      std::vector<std::pair<ShapeIndex, std::pair<int64_t, ShapeIndex>>>
          output_to_operand_aliasing;
      for (const auto& aliasing : proto.custom_call_output_operand_aliasing()) {
        output_to_operand_aliasing.emplace_back(
            ShapeIndex(aliasing.output_shape_index().begin(),
                       aliasing.output_shape_index().end()),
            std::pair<int64_t, ShapeIndex>{
                aliasing.operand_index(),
                ShapeIndex(aliasing.operand_shape_index().begin(),
                           aliasing.operand_shape_index().end())});
      }
      custom_call_instr->set_output_to_operand_aliasing(
          std::move(output_to_operand_aliasing));
      custom_call_instr->set_custom_call_schedule(proto.custom_call_schedule());
      custom_call_instr->set_api_version(proto.custom_call_api_version());
      break;
    }
    case HloOpcode::kPad:
      TF_RET_CHECK(proto.has_padding_config());
      instruction =
          CreatePad(shape, operands(0), operands(1), proto.padding_config());
      break;
    case HloOpcode::kDynamicSlice: {
      std::vector<int64_t> slice_sizes(proto.dynamic_slice_sizes_size());
      absl::c_copy(proto.dynamic_slice_sizes(), slice_sizes.begin());
      TF_RET_CHECK(proto.operand_ids_size() >= 1)
          << "DynamicSlice instruction should have at least 1 operands but "
             "sees "
          << proto.operand_ids_size();
      // TODO(b/118437727): Old form, make the check unconditional.
      if (proto.operand_ids_size() != 2 || operands(1)->shape().rank() != 1) {
        auto expected_operands = 1 + operands(0)->shape().rank();
        TF_RET_CHECK(proto.operand_ids_size() == expected_operands)
            << "DynamicSlice instruction should have " << expected_operands
            << " operands, but has " << proto.operand_ids_size();
      }
      const auto& operand_vector = all_operands();
      instruction = CreateDynamicSlice(
          shape, operands(0), absl::MakeSpan(operand_vector).subspan(1),
          slice_sizes);
      break;
    }
    case HloOpcode::kDynamicUpdateSlice: {
      TF_RET_CHECK(proto.operand_ids_size() >= 2)
          << "DynamicUpdateSlice instruction should have at least 2 operands "
             "but sees "
          << proto.operand_ids_size();
      // TODO(b/118437727): Old form, make the check unconditional.
      if (proto.operand_ids_size() != 3 || operands(2)->shape().rank() != 1) {
        auto expected_operands = 2 + operands(0)->shape().rank();
        TF_RET_CHECK(proto.operand_ids_size() == expected_operands)
            << "DynamicUpdateSlice instruction should have "
            << expected_operands << " operands, but has "
            << proto.operand_ids_size();
      }
      const auto& operand_vector = all_operands();
      instruction =
          CreateDynamicUpdateSlice(shape, operands(0), operands(1),
                                   absl::MakeSpan(operand_vector).subspan(2));

      break;
    }
    case HloOpcode::kGather: {
      TF_RET_CHECK(proto.has_gather_dimension_numbers())
          << "Gather instruction should have GatherDimensionNumbers set.";
      auto gather_dimension_numbers = absl::make_unique<GatherDimensionNumbers>(
          proto.gather_dimension_numbers());
      std::vector<int64_t> gather_slice_sizes;
      const auto& slice_sizes = proto.gather_slice_sizes();
      gather_slice_sizes.reserve(slice_sizes.size());
      for (int64_t bound : slice_sizes) {
        gather_slice_sizes.push_back(bound);
      }
      instruction = CreateGather(shape, operands(0), operands(1),
                                 *gather_dimension_numbers, gather_slice_sizes,
                                 proto.indices_are_sorted());
      break;
    }
    case HloOpcode::kScatter: {
      TF_RET_CHECK(proto.has_scatter_dimension_numbers())
          << "Scatter instruction should have ScatterDimensionNumbers set.";
      TF_RET_CHECK(proto.called_computation_ids_size() == 1)
          << "Scatter instruction should have 1 called computation but sees "
          << proto.called_computation_ids_size();
      auto scatter_dimension_numbers =
          absl::make_unique<ScatterDimensionNumbers>(
              proto.scatter_dimension_numbers());
      instruction =
          CreateScatter(shape, operands(0), operands(1), operands(2),
                        computations(0), *scatter_dimension_numbers,
                        proto.indices_are_sorted(), proto.unique_indices());
      break;
    }
    case HloOpcode::kIota:
      TF_RET_CHECK(proto.dimensions_size() == 1)
          << "Iota instruction should have 1 dimension but sees "
          << proto.dimensions_size();
      instruction = CreateIota(shape, proto.dimensions(0));
      break;
    case HloOpcode::kDot: {
      TF_RET_CHECK(proto.has_dot_dimension_numbers())
          << "Dot instruction should have dot_dimension_numbers.";
      PrecisionConfig precision_config = proto.precision_config();
      precision_config.mutable_operand_precision()->Resize(
          proto.operand_ids_size(), PrecisionConfig::DEFAULT);
      instruction = absl::make_unique<HloDotInstruction>(
          shape, operands(0), operands(1), proto.dot_dimension_numbers(),
          precision_config);
      break;
    }
    case HloOpcode::kDomain: {
      std::shared_ptr<const HloSharding> entry_hlo_sharding;
      std::shared_ptr<const HloSharding> exit_hlo_sharding;
      if (proto.has_domain_entry_sharding()) {
        TF_ASSIGN_OR_RETURN(
            HloSharding sharding,
            HloSharding::FromProto(proto.domain_entry_sharding()));
        entry_hlo_sharding = std::make_shared<const HloSharding>(sharding);
      }
      if (proto.has_domain_exit_sharding()) {
        TF_ASSIGN_OR_RETURN(
            HloSharding sharding,
            HloSharding::FromProto(proto.domain_exit_sharding()));
        exit_hlo_sharding = std::make_shared<const HloSharding>(sharding);
      }
      instruction = absl::make_unique<HloDomainInstruction>(
          shape, operands(0),
          absl::make_unique<ShardingMetadata>(entry_hlo_sharding),
          absl::make_unique<ShardingMetadata>(exit_hlo_sharding));
      break;
    }
    case HloOpcode::kGetDimensionSize:
      TF_RET_CHECK(proto.dimensions_size() == 1);
      instruction =
          CreateGetDimensionSize(shape, operands(0), proto.dimensions(0));
      break;
    case HloOpcode::kSetDimensionSize:
      TF_RET_CHECK(proto.dimensions_size() == 1);
      instruction = CreateSetDimensionSize(shape, operands(0), operands(1),
                                           proto.dimensions(0));
      break;
    case HloOpcode::kReshape: {
      int64_t inferred_dimension = -1;
      if (!proto.dimensions().empty()) {
        inferred_dimension = proto.dimensions()[0];
      }
      TF_RET_CHECK(shape.IsArray() && operands(0)->shape().IsArray() &&
                   ShapeUtil::ElementsIn(shape) ==
                       ShapeUtil::ElementsIn(operands(0)->shape()))
          << "shape: " << ShapeUtil::HumanString(shape)
          << " operand: " << ShapeUtil::HumanString(operands(0)->shape());
      instruction = CreateReshape(shape, operands(0), inferred_dimension);
      break;
    }
    case HloOpcode::kDynamicReshape: {
      TF_RET_CHECK(shape.IsArray() && operands(0)->shape().IsArray() &&
                   ShapeUtil::ElementsIn(shape) ==
                       ShapeUtil::ElementsIn(operands(0)->shape()))
          << "shape: " << ShapeUtil::HumanString(shape)
          << " operand: " << ShapeUtil::HumanString(operands(0)->shape());
      const auto& operand_vector = all_operands();
      instruction = CreateDynamicReshape(
          shape, operands(0), absl::MakeSpan(operand_vector).subspan(1));
      break;
    }
    default: {
      instruction = absl::WrapUnique(new HloInstruction(opcode, shape));
      for (const int64_t operand_id : proto.operand_ids()) {
        instruction->AppendOperand(instruction_map.at(operand_id));
      }
      if (instruction->opcode() != HloOpcode::kFusion) {
        if (instruction->opcode() == HloOpcode::kCall) {
          TF_RET_CHECK(proto.called_computation_ids_size() == 1)
              << "Call should have 1 called computation but has "
              << proto.called_computation_ids_size();
        }
        for (const int64_t computation_id : proto.called_computation_ids()) {
          instruction->called_computations_.push_back(
              computation_map.at(computation_id));
        }
      }
      TF_RET_CHECK(!proto.has_precision_config())
          << instruction->opcode() << proto.DebugString();
      TF_RET_CHECK(!proto.has_dot_dimension_numbers()) << instruction->opcode();
      break;
    }
  }

  for (const int64_t predecessor_id : proto.control_predecessor_ids()) {
    TF_RET_CHECK(ContainsKey(instruction_map, predecessor_id))
        << "No instruction with id " << predecessor_id;
    TF_RETURN_IF_ERROR(instruction_map.at(predecessor_id)
                           ->AddControlDependencyTo(instruction.get()));
  }

  TF_RET_CHECK(!proto.name().empty());
  instruction->SetAndSanitizeName(proto.name());
  instruction->metadata_ = proto.metadata();
  instruction->backend_config_ = proto.backend_config();
  instruction->outer_dimension_partitions_.assign(
      proto.outer_dimension_partitions().begin(),
      proto.outer_dimension_partitions().end());

  TF_RET_CHECK(proto.id() >= 0)
      << "Instruction with negative id: " << proto.id();
  TF_RET_CHECK(proto.id() <= INT_MAX)
      << "Instruction with id > INT_MAX: " << proto.id();
  instruction->unique_id_ = proto.id();

  if (proto.has_sharding()) {
    TF_ASSIGN_OR_RETURN(const auto& sharding,
                        HloSharding::FromProto(proto.sharding()));
    instruction->set_sharding(sharding);
  }

  if (proto.has_frontend_attributes()) {
    instruction->set_frontend_attributes(proto.frontend_attributes());
  }

  return std::move(instruction);
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateParameter(
    int64_t parameter_number, const Shape& shape, const std::string& name) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_6(mht_6_v, 1168, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CreateParameter");

  return absl::make_unique<HloParameterInstruction>(parameter_number, shape,
                                                    name);
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateTrace(
    const std::string& tag, HloInstruction* operand) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("tag: \"" + tag + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_7(mht_7_v, 1178, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CreateTrace");

  return absl::make_unique<HloTraceInstruction>(tag, operand);
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateConstant(
    Literal literal) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_8(mht_8_v, 1186, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CreateConstant");

  return absl::make_unique<HloConstantInstruction>(std::move(literal));
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateIota(
    const Shape& shape, int64_t iota_dimension) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_9(mht_9_v, 1194, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CreateIota");

  return absl::make_unique<HloIotaInstruction>(shape, iota_dimension);
}

/* static */ std::unique_ptr<HloInstruction>
HloInstruction::CreateGetTupleElement(const Shape& shape,
                                      HloInstruction* operand, int64_t index) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_10(mht_10_v, 1203, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CreateGetTupleElement");

  return absl::make_unique<HloGetTupleElementInstruction>(shape, operand,
                                                          index);
}

/* static */ std::unique_ptr<HloInstruction>
HloInstruction::CreateGetTupleElement(HloInstruction* operand, int64_t index) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_11(mht_11_v, 1212, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CreateGetTupleElement");

  return absl::make_unique<HloGetTupleElementInstruction>(
      operand->shape().tuple_shapes(index), operand, index);
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateRng(
    const Shape& shape, RandomDistribution distribution,
    absl::Span<HloInstruction* const> parameters) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_12(mht_12_v, 1222, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CreateRng");

  return absl::make_unique<HloRngInstruction>(shape, distribution, parameters);
}

/* static */ std::unique_ptr<HloInstruction>
HloInstruction::CreateRngGetAndUpdateState(const Shape& shape, int64_t delta) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_13(mht_13_v, 1230, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CreateRngGetAndUpdateState");

  return absl::make_unique<HloRngGetAndUpdateStateInstruction>(shape, delta);
}

/* static */ std::unique_ptr<HloInstruction>
HloInstruction::CreateRngBitGenerator(const Shape& shape, HloInstruction* state,
                                      RandomAlgorithm algorithm) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_14(mht_14_v, 1239, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CreateRngBitGenerator");

  return absl::make_unique<HloRngBitGeneratorInstruction>(shape, state,
                                                          algorithm);
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateNary(
    const Shape& shape, HloOpcode opcode,
    absl::Span<HloInstruction* const> operands) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_15(mht_15_v, 1249, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CreateNary");

  if (opcode == HloOpcode::kCopy) {
    // It is impossible to copy an opaque shape, we don't know how big it is.
    CHECK(!shape.IsOpaque());
  }
  auto instruction = absl::WrapUnique(new HloInstruction(opcode, shape));
  for (auto operand : operands) {
    instruction->AppendOperand(operand);
  }
  return instruction;
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateUnary(
    const Shape& shape, HloOpcode opcode, HloInstruction* operand) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_16(mht_16_v, 1265, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CreateUnary");

  // Only certain opcodes are supported with CreateUnary: opcodes of unary
  // instructions with no auxiliary fields.
  switch (opcode) {
    case HloOpcode::kAbs:
    case HloOpcode::kAllGatherDone:
    case HloOpcode::kAllReduceDone:
    case HloOpcode::kRoundNearestAfz:
    case HloOpcode::kBitcast:
    case HloOpcode::kCeil:
    case HloOpcode::kCollectivePermuteDone:
    case HloOpcode::kCopy:
    case HloOpcode::kCopyDone:
    case HloOpcode::kCos:
    case HloOpcode::kOptimizationBarrier:
    case HloOpcode::kClz:
    case HloOpcode::kExp:
    case HloOpcode::kExpm1:
    case HloOpcode::kFloor:
    case HloOpcode::kImag:
    case HloOpcode::kIsFinite:
    case HloOpcode::kLog:
    case HloOpcode::kLog1p:
    case HloOpcode::kNot:
    case HloOpcode::kNegate:
    case HloOpcode::kPopulationCount:
    case HloOpcode::kReal:
    case HloOpcode::kRsqrt:
    case HloOpcode::kLogistic:
    case HloOpcode::kSign:
    case HloOpcode::kSin:
    case HloOpcode::kSqrt:
    case HloOpcode::kCbrt:
    case HloOpcode::kTanh:
      break;
    default:
      LOG(FATAL) << "Invalid unary instruction opcode "
                 << HloOpcodeString(opcode);
  }
  return CreateNary(shape, opcode, {operand});
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateBinary(
    const Shape& shape, HloOpcode opcode, HloInstruction* lhs,
    HloInstruction* rhs) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_17(mht_17_v, 1312, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CreateBinary");

  // Only certain opcodes are supported with CreateBinary: opcodes of binary
  // instructions with no auxiliary fields.
  switch (opcode) {
    case HloOpcode::kAdd:
    case HloOpcode::kAtan2:
    case HloOpcode::kDivide:
    case HloOpcode::kComplex:
    case HloOpcode::kMaximum:
    case HloOpcode::kMinimum:
    case HloOpcode::kMultiply:
    case HloOpcode::kPower:
    case HloOpcode::kRemainder:
    case HloOpcode::kSubtract:
    case HloOpcode::kAnd:
    case HloOpcode::kOr:
    case HloOpcode::kXor:
    case HloOpcode::kShiftLeft:
    case HloOpcode::kShiftRightArithmetic:
    case HloOpcode::kShiftRightLogical:
      break;
    default:
      LOG(FATAL) << "Invalid binary instruction opcode "
                 << HloOpcodeString(opcode);
  }
  return CreateNary(shape, opcode, {lhs, rhs});
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateTernary(
    const Shape& shape, HloOpcode opcode, HloInstruction* lhs,
    HloInstruction* rhs, HloInstruction* ehs) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_18(mht_18_v, 1345, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CreateTernary");

  // Only certain opcodes are supported with CreateTernary: opcodes of ternary
  // instructions with no auxiliary fields.
  switch (opcode) {
    case HloOpcode::kClamp:
    case HloOpcode::kSelect:
    case HloOpcode::kTupleSelect:
      break;
    default:
      LOG(FATAL) << "Invalid ternary instruction opcode "
                 << HloOpcodeString(opcode);
  }
  return CreateNary(shape, opcode, {lhs, rhs, ehs});
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateVariadic(
    const Shape& shape, HloOpcode opcode,
    absl::Span<HloInstruction* const> operands) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_19(mht_19_v, 1365, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CreateVariadic");

  CHECK_EQ(HloOpcode::kTuple, opcode);
  return CreateNary(shape, opcode, operands);
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateMap(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloComputation* map_computation) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_20(mht_20_v, 1375, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CreateMap");

  return absl::make_unique<HloMapInstruction>(shape, operands, map_computation);
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateConvolve(
    const Shape& shape, HloInstruction* lhs, HloInstruction* rhs,
    int64_t feature_group_count, int64_t batch_group_count,
    const Window& window, const ConvolutionDimensionNumbers& dimension_numbers,
    const PrecisionConfig& precision_config) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_21(mht_21_v, 1386, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CreateConvolve");

  return absl::make_unique<HloConvolutionInstruction>(
      shape, lhs, rhs, feature_group_count, batch_group_count, window,
      dimension_numbers, precision_config);
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateFft(
    const Shape& shape, HloInstruction* operand, FftType fft_type,
    absl::Span<const int64_t> fft_length) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_22(mht_22_v, 1397, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CreateFft");

  return absl::make_unique<HloFftInstruction>(shape, operand, fft_type,
                                              fft_length);
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateAsyncStart(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloComputation* async_computation) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_23(mht_23_v, 1407, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CreateAsyncStart");

  return absl::make_unique<HloAsyncInstruction>(HloOpcode::kAsyncStart, shape,
                                                operands, async_computation);
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateAsyncUpdate(
    const Shape& shape, HloInstruction* operand,
    HloComputation* async_computation) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_24(mht_24_v, 1417, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CreateAsyncUpdate");

  return absl::make_unique<HloAsyncInstruction>(HloOpcode::kAsyncUpdate, shape,
                                                operand, async_computation);
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateAsyncDone(
    const Shape& shape, HloInstruction* operand,
    HloComputation* async_computation) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_25(mht_25_v, 1427, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CreateAsyncDone");

  return absl::make_unique<HloAsyncInstruction>(HloOpcode::kAsyncDone, shape,
                                                operand, async_computation);
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateCopyStart(
    const Shape& shape, HloInstruction* operand,
    bool is_cross_program_prefetch) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_26(mht_26_v, 1437, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CreateCopyStart");

  return absl::make_unique<HloCopyStartInstruction>(shape, operand,
                                                    is_cross_program_prefetch);
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateCompare(
    const Shape& shape, HloInstruction* lhs, HloInstruction* rhs,
    ComparisonDirection direction, absl::optional<Comparison::Type> type) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_27(mht_27_v, 1447, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CreateCompare");

  return absl::make_unique<HloCompareInstruction>(shape, lhs, rhs, direction,
                                                  type);
}

/* static */ std::unique_ptr<HloInstruction>
HloInstruction::CreateTriangularSolve(const Shape& shape, HloInstruction* a,
                                      HloInstruction* b,
                                      const TriangularSolveOptions& options) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_28(mht_28_v, 1458, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CreateTriangularSolve");

  return absl::make_unique<HloTriangularSolveInstruction>(shape, a, b, options);
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateCholesky(
    const Shape& shape, HloInstruction* a, const CholeskyOptions& options) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_29(mht_29_v, 1466, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CreateCholesky");

  return absl::make_unique<HloCholeskyInstruction>(shape, a, options);
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateDot(
    const Shape& shape, HloInstruction* lhs, HloInstruction* rhs,
    const DotDimensionNumbers& dimension_numbers,
    const PrecisionConfig& precision_config) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_30(mht_30_v, 1476, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CreateDot");

  return absl::make_unique<HloDotInstruction>(
      shape, lhs, rhs, dimension_numbers, precision_config);
}

/* static */ std::unique_ptr<HloInstruction>
HloInstruction::CreateReducePrecision(const Shape& shape,
                                      HloInstruction* operand,
                                      const int exponent_bits,
                                      const int mantissa_bits) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_31(mht_31_v, 1488, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CreateReducePrecision");

  return absl::make_unique<HloReducePrecisionInstruction>(
      shape, operand, exponent_bits, mantissa_bits);
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateAllGather(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    int64_t all_gather_dimension, absl::Span<const ReplicaGroup> replica_groups,
    bool constrain_layout, const absl::optional<int64_t>& channel_id,
    bool use_global_device_ids) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_32(mht_32_v, 1500, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CreateAllGather");

  return absl::make_unique<HloAllGatherInstruction>(
      HloOpcode::kAllGather, shape, operands, all_gather_dimension,
      replica_groups, constrain_layout, channel_id, use_global_device_ids);
}

/* static */ std::unique_ptr<HloInstruction>
HloInstruction::CreateAllGatherStart(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    int64_t all_gather_dimension, absl::Span<const ReplicaGroup> replica_groups,
    bool constrain_layout, const absl::optional<int64_t>& channel_id,
    bool use_global_device_ids) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_33(mht_33_v, 1514, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CreateAllGatherStart");

  return absl::make_unique<HloAllGatherInstruction>(
      HloOpcode::kAllGatherStart, shape, operands, all_gather_dimension,
      replica_groups, constrain_layout, channel_id, use_global_device_ids);
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateAllReduce(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloComputation* reduce_computation,
    absl::Span<const ReplicaGroup> replica_groups, bool constrain_layout,
    const absl::optional<int64_t>& channel_id, bool use_global_device_ids) {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_34(mht_34_v, 1527, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CreateAllReduce");

  return absl::make_unique<HloAllReduceInstruction>(
      HloOpcode::kAllReduce, shape, operands, reduce_computation,
      replica_groups, constrain_layout, channel_id, use_global_device_ids);
}

/* static */ std::unique_ptr<HloInstruction>
HloInstruction::CreateReduceScatter(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloComputation* reduce_computation,
    absl::Span<const ReplicaGroup> replica_groups, bool constrain_layout,
    const absl::optional<int64_t>& channel_id, bool use_global_device_ids,
    int64_t scatter_dimension) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_35(mht_35_v, 1542, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CreateReduceScatter");

  return absl::make_unique<HloReduceScatterInstruction>(
      shape, operands, reduce_computation, replica_groups, constrain_layout,
      channel_id, use_global_device_ids, scatter_dimension);
}

/* static */ std::unique_ptr<HloInstruction>
HloInstruction::CreateAllReduceStart(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloComputation* reduce_computation,
    absl::Span<const ReplicaGroup> replica_groups, bool constrain_layout,
    const absl::optional<int64_t>& channel_id, bool use_global_device_ids) {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_36(mht_36_v, 1556, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CreateAllReduceStart");

  return absl::make_unique<HloAllReduceInstruction>(
      HloOpcode::kAllReduceStart, shape, operands, reduce_computation,
      replica_groups, constrain_layout, channel_id, use_global_device_ids);
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateAllToAll(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    absl::Span<const ReplicaGroup> replica_groups, bool constrain_layout,
    const absl::optional<int64_t>& channel_id,
    const absl::optional<int64_t>& split_dimension) {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_37(mht_37_v, 1569, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CreateAllToAll");

  return absl::make_unique<HloAllToAllInstruction>(
      shape, operands, replica_groups, constrain_layout, channel_id,
      split_dimension);
}

/* static */ std::unique_ptr<HloInstruction>
HloInstruction::CreateCollectivePermute(
    const Shape& shape, HloInstruction* operand,
    const std::vector<std::pair<int64_t, int64_t>>& source_target_pairs,
    const absl::optional<int64_t>& channel_id) {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_38(mht_38_v, 1582, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CreateCollectivePermute");

  return absl::make_unique<HloCollectivePermuteInstruction>(
      HloOpcode::kCollectivePermute, shape, operand, source_target_pairs,
      channel_id);
}

/* static */ std::unique_ptr<HloInstruction>
HloInstruction::CreateCollectivePermute(
    const Shape& shape, HloInstruction* input, HloInstruction* output,
    HloInstruction* input_start_indices, HloInstruction* output_start_indices,
    absl::Span<const std::pair<int64_t, int64_t>> source_target_pairs,
    absl::Span<const std::vector<int64_t>> slice_sizes,
    const absl::optional<int64_t>& channel_id) {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_39(mht_39_v, 1597, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CreateCollectivePermute");

  return absl::make_unique<HloCollectivePermuteInstruction>(
      HloOpcode::kCollectivePermute, shape, input, output, input_start_indices,
      output_start_indices, source_target_pairs, slice_sizes, channel_id);
}

/* static */ std::unique_ptr<HloInstruction>
HloInstruction::CreateCollectivePermuteStart(
    const Shape& shape, HloInstruction* operand,
    const std::vector<std::pair<int64_t, int64_t>>& source_target_pairs,
    const absl::optional<int64_t>& channel_id) {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_40(mht_40_v, 1610, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CreateCollectivePermuteStart");

  return absl::make_unique<HloCollectivePermuteInstruction>(
      HloOpcode::kCollectivePermuteStart, shape, operand, source_target_pairs,
      channel_id);
}

/* static */ std::unique_ptr<HloInstruction>
HloInstruction::CreateCollectivePermuteStart(
    const Shape& shape, HloInstruction* input, HloInstruction* output,
    HloInstruction* input_start_indices, HloInstruction* output_start_indices,
    absl::Span<const std::pair<int64_t, int64_t>> source_target_pairs,
    absl::Span<const std::vector<int64_t>> slice_sizes,
    const absl::optional<int64_t>& channel_id) {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_41(mht_41_v, 1625, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CreateCollectivePermuteStart");

  return absl::make_unique<HloCollectivePermuteInstruction>(
      HloOpcode::kCollectivePermuteStart, shape, input, output,
      input_start_indices, output_start_indices, source_target_pairs,
      slice_sizes, channel_id);
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateReplicaId(
    const Shape& shape) {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_42(mht_42_v, 1636, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CreateReplicaId");

  CHECK(Shape::Equal().IgnoreLayout()(shape, ShapeUtil::MakeShape(U32, {})))
      << "HloInstruction replica-id must have a shape of u32[], but "
      << shape.ToString() << " is specified";
  return absl::WrapUnique(new HloInstruction(HloOpcode::kReplicaId, shape));
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreatePartitionId(
    const Shape& shape) {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_43(mht_43_v, 1647, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CreatePartitionId");

  CHECK(Shape::Equal().IgnoreLayout()(shape, ShapeUtil::MakeShape(U32, {})))
      << "HloInstruction partition-id must have a shape of u32[], but "
      << shape.ToString() << " is specified";
  return absl::WrapUnique(new HloInstruction(HloOpcode::kPartitionId, shape));
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateInfeed(
    const Shape& infeed_shape, HloInstruction* token_operand,
    const std::string& config) {
   std::vector<std::string> mht_44_v;
   mht_44_v.push_back("config: \"" + config + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_44(mht_44_v, 1660, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CreateInfeed");

  return absl::make_unique<HloInfeedInstruction>(infeed_shape, token_operand,
                                                 config);
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateOutfeed(
    const Shape& outfeed_shape, HloInstruction* operand,
    HloInstruction* token_operand, absl::string_view outfeed_config) {
   std::vector<std::string> mht_45_v;
   mht_45_v.push_back("outfeed_config: \"" + std::string(outfeed_config.data(), outfeed_config.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_45(mht_45_v, 1671, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CreateOutfeed");

  return absl::make_unique<HloOutfeedInstruction>(
      outfeed_shape, operand, token_operand, outfeed_config);
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateSend(
    HloInstruction* operand, HloInstruction* token, int64_t channel_id,
    bool is_host_transfer) {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_46(mht_46_v, 1681, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CreateSend");

  return absl::make_unique<HloSendInstruction>(operand, token, channel_id,
                                               is_host_transfer);
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateSendDone(
    HloInstruction* operand, bool is_host_transfer) {
   std::vector<std::string> mht_47_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_47(mht_47_v, 1690, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CreateSendDone");

  auto send_operand = DynCast<HloSendInstruction>(operand);
  CHECK(send_operand != nullptr)
      << "SendDone must take the context operand from Send";
  return absl::make_unique<HloSendDoneInstruction>(send_operand,
                                                   is_host_transfer);
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateRecv(
    const Shape& shape, HloInstruction* token, int64_t channel_id,
    bool is_host_transfer) {
   std::vector<std::string> mht_48_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_48(mht_48_v, 1703, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CreateRecv");

  return absl::make_unique<HloRecvInstruction>(shape, token, channel_id,
                                               is_host_transfer);
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateRecvDone(
    HloInstruction* operand, bool is_host_transfer) {
   std::vector<std::string> mht_49_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_49(mht_49_v, 1712, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CreateRecvDone");

  auto recv_operand = DynCast<HloRecvInstruction>(operand);
  CHECK(recv_operand != nullptr)
      << "RecvDone must take the context operand from Recv";
  return absl::make_unique<HloRecvDoneInstruction>(recv_operand,
                                                   is_host_transfer);
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateReverse(
    const Shape& shape, HloInstruction* operand,
    absl::Span<const int64_t> dimensions) {
   std::vector<std::string> mht_50_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_50(mht_50_v, 1725, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CreateReverse");

  return absl::make_unique<HloReverseInstruction>(shape, operand, dimensions);
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateAfterAll(
    absl::Span<HloInstruction* const> operands) {
   std::vector<std::string> mht_51_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_51(mht_51_v, 1733, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CreateAfterAll");

  CHECK(!operands.empty());
  auto instruction = absl::WrapUnique(
      new HloInstruction(HloOpcode::kAfterAll, ShapeUtil::MakeTokenShape()));
  for (auto operand : operands) {
    instruction->AppendOperand(operand);
  }
  return instruction;
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateToken() {
   std::vector<std::string> mht_52_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_52(mht_52_v, 1746, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CreateToken");

  return absl::WrapUnique(
      new HloInstruction(HloOpcode::kAfterAll, ShapeUtil::MakeTokenShape()));
}

/* static */ std::unique_ptr<HloInstruction>
HloInstruction::CreateAddDependency(HloInstruction* data_operand,
                                    HloInstruction* token_operand) {
   std::vector<std::string> mht_53_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_53(mht_53_v, 1756, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CreateAddDependency");

  auto instruction = absl::WrapUnique(
      new HloInstruction(HloOpcode::kAddDependency, data_operand->shape()));
  instruction->AppendOperand(data_operand);
  instruction->AppendOperand(token_operand);
  return instruction;
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateWhile(
    const Shape& shape, HloComputation* condition, HloComputation* body,
    HloInstruction* init) {
   std::vector<std::string> mht_54_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_54(mht_54_v, 1769, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CreateWhile");

  auto instruction =
      absl::WrapUnique(new HloInstruction(HloOpcode::kWhile, shape));
  instruction->AppendOperand(init);
  // Body comes before condition computation in the vector.
  instruction->called_computations_.push_back(body);
  instruction->called_computations_.push_back(condition);
  return instruction;
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateConditional(
    const Shape& shape, HloInstruction* pred,
    HloInstruction* true_computation_arg, HloComputation* true_computation,
    HloInstruction* false_computation_arg, HloComputation* false_computation) {
   std::vector<std::string> mht_55_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_55(mht_55_v, 1785, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CreateConditional");

  auto instruction =
      absl::WrapUnique(new HloInstruction(HloOpcode::kConditional, shape));
  instruction->AppendOperand(pred);
  instruction->AppendOperand(true_computation_arg);
  instruction->AppendOperand(false_computation_arg);
  // In called_computations_, the index of true_computation must be 0 and that
  // of false computation must be 1, as defined by kTrueComputationIndex and
  // kFalseComputationIndex.
  instruction->called_computations_.push_back(true_computation);
  instruction->called_computations_.push_back(false_computation);
  return instruction;
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateConditional(
    const Shape& shape, HloInstruction* branch_index,
    absl::Span<HloComputation* const> branch_computations,
    absl::Span<HloInstruction* const> branch_computation_args) {
   std::vector<std::string> mht_56_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_56(mht_56_v, 1805, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CreateConditional");

  auto instruction =
      absl::WrapUnique(new HloInstruction(HloOpcode::kConditional, shape));
  instruction->AppendOperand(branch_index);
  CHECK_EQ(branch_computations.size(), branch_computation_args.size());
  for (int i = 0; i < branch_computations.size(); ++i) {
    instruction->called_computations_.push_back(branch_computations[i]);
    instruction->AppendOperand(branch_computation_args[i]);
  }
  return instruction;
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateSlice(
    const Shape& shape, HloInstruction* operand,
    absl::Span<const int64_t> start_indices,
    absl::Span<const int64_t> limit_indices,
    absl::Span<const int64_t> strides) {
   std::vector<std::string> mht_57_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_57(mht_57_v, 1824, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CreateSlice");

  return absl::make_unique<HloSliceInstruction>(shape, operand, start_indices,
                                                limit_indices, strides);
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateDynamicSlice(
    const Shape& shape, HloInstruction* operand,
    absl::Span<HloInstruction* const> start_indices,
    absl::Span<const int64_t> slice_sizes) {
   std::vector<std::string> mht_58_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_58(mht_58_v, 1835, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CreateDynamicSlice");

  return absl::make_unique<HloDynamicSliceInstruction>(
      shape, operand, start_indices, slice_sizes);
}

/* static */ std::unique_ptr<HloInstruction>
HloInstruction::CreateDynamicUpdateSlice(
    const Shape& shape, HloInstruction* operand, HloInstruction* update,
    absl::Span<HloInstruction* const> start_indices) {
   std::vector<std::string> mht_59_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_59(mht_59_v, 1846, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CreateDynamicUpdateSlice");

  return absl::make_unique<HloDynamicUpdateSliceInstruction>(
      shape, operand, update, start_indices);
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateConcatenate(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    int64_t dimension) {
   std::vector<std::string> mht_60_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_60(mht_60_v, 1856, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CreateConcatenate");

  return absl::make_unique<HloConcatenateInstruction>(shape, operands,
                                                      dimension);
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateConvert(
    const Shape& shape, HloInstruction* operand) {
   std::vector<std::string> mht_61_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_61(mht_61_v, 1865, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CreateConvert");

  auto instruction =
      absl::WrapUnique(new HloInstruction(HloOpcode::kConvert, shape));
  instruction->AppendOperand(operand);
  return instruction;
}

/* static */ std::unique_ptr<HloInstruction>
HloInstruction::CreateBitcastConvert(const Shape& shape,
                                     HloInstruction* operand) {
   std::vector<std::string> mht_62_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_62(mht_62_v, 1877, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CreateBitcastConvert");

  auto instruction =
      absl::WrapUnique(new HloInstruction(HloOpcode::kBitcastConvert, shape));
  instruction->AppendOperand(operand);
  return instruction;
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateBitcast(
    const Shape& shape, HloInstruction* operand) {
   std::vector<std::string> mht_63_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_63(mht_63_v, 1888, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CreateBitcast");

  auto instruction =
      absl::WrapUnique(new HloInstruction(HloOpcode::kBitcast, shape));
  instruction->AppendOperand(operand);
  return instruction;
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateReduce(
    const Shape& shape, HloInstruction* operand, HloInstruction* init_value,
    absl::Span<const int64_t> dimensions_to_reduce,
    HloComputation* reduce_computation) {
   std::vector<std::string> mht_64_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_64(mht_64_v, 1901, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CreateReduce");

  auto instruction = absl::WrapUnique(new HloReduceInstruction(
      shape, {operand, init_value}, dimensions_to_reduce, reduce_computation));
  return std::move(instruction);
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateReduce(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    absl::Span<HloInstruction* const> init_values,
    absl::Span<const int64_t> dimensions_to_reduce,
    HloComputation* reduce_computation) {
   std::vector<std::string> mht_65_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_65(mht_65_v, 1914, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CreateReduce");

  std::vector<HloInstruction*> all_args;
  all_args.reserve(operands.size() * 2);
  all_args.insert(all_args.end(), operands.begin(), operands.end());
  all_args.insert(all_args.end(), init_values.begin(), init_values.end());
  return absl::make_unique<HloReduceInstruction>(
      shape, all_args, dimensions_to_reduce, reduce_computation);
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateReduce(
    const Shape& shape, HloInstruction* tuple_of_instructions,
    absl::Span<HloInstruction* const> init_values,
    absl::Span<const int64_t> dimensions_to_reduce,
    HloComputation* reduce_computation) {
   std::vector<std::string> mht_66_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_66(mht_66_v, 1930, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CreateReduce");

  if (!tuple_of_instructions->shape().IsTuple()) {
    CHECK_EQ(init_values.size(), 1)
        << "The first input has to be a tuple, or the number of init values "
           "has to be one.";
    return CreateReduce(shape, tuple_of_instructions, init_values[0],
                        dimensions_to_reduce, reduce_computation);
  }
  absl::InlinedVector<HloInstruction*, 4> inputs;
  for (int idx = 0; idx < tuple_of_instructions->shape().tuple_shapes_size();
       idx++) {
    std::unique_ptr<HloInstruction> gte =
        HloInstruction::CreateGetTupleElement(tuple_of_instructions, idx);
    inputs.push_back(
        tuple_of_instructions->parent()->AddInstruction(std::move(gte)));
  }
  return CreateReduce(shape, inputs, init_values, dimensions_to_reduce,
                      reduce_computation);
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateReduceWindow(
    const Shape& shape, HloInstruction* operand, HloInstruction* init_value,
    const Window& window, HloComputation* reduce_computation) {
   std::vector<std::string> mht_67_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_67(mht_67_v, 1955, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CreateReduceWindow");

  return absl::make_unique<HloReduceWindowInstruction>(
      shape, operand, init_value, window, reduce_computation);
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateReduceWindow(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    absl::Span<HloInstruction* const> init_values, const Window& window,
    HloComputation* reduce_computation) {
   std::vector<std::string> mht_68_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_68(mht_68_v, 1966, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CreateReduceWindow");

  return absl::make_unique<HloReduceWindowInstruction>(
      shape, operands, init_values, window, reduce_computation);
}
/* static */ std::unique_ptr<HloInstruction>
HloInstruction::CreateBatchNormTraining(const Shape& shape,
                                        HloInstruction* operand,
                                        HloInstruction* scale,
                                        HloInstruction* offset, float epsilon,
                                        int64_t feature_index) {
   std::vector<std::string> mht_69_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_69(mht_69_v, 1978, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CreateBatchNormTraining");

  return absl::make_unique<HloBatchNormTrainingInstruction>(
      shape, operand, scale, offset, epsilon, feature_index);
}

/* static */ std::unique_ptr<HloInstruction>
HloInstruction::CreateBatchNormInference(
    const Shape& shape, HloInstruction* operand, HloInstruction* scale,
    HloInstruction* offset, HloInstruction* mean, HloInstruction* variance,
    float epsilon, int64_t feature_index) {
   std::vector<std::string> mht_70_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_70(mht_70_v, 1990, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CreateBatchNormInference");

  return absl::make_unique<HloBatchNormInferenceInstruction>(
      shape, operand, scale, offset, mean, variance, epsilon, feature_index);
}

/* static */ std::unique_ptr<HloInstruction>
HloInstruction::CreateBatchNormGrad(const Shape& shape, HloInstruction* operand,
                                    HloInstruction* scale, HloInstruction* mean,
                                    HloInstruction* variance,
                                    HloInstruction* grad_output, float epsilon,
                                    int64_t feature_index) {
   std::vector<std::string> mht_71_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_71(mht_71_v, 2003, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CreateBatchNormGrad");

  return absl::make_unique<HloBatchNormGradInstruction>(
      shape, operand, scale, mean, variance, grad_output, epsilon,
      feature_index);
}

/* static */ std::unique_ptr<HloInstruction>
HloInstruction::CreateSelectAndScatter(
    const Shape& shape, HloInstruction* operand, HloComputation* select,
    const Window& window, HloInstruction* source, HloInstruction* init_value,
    HloComputation* scatter) {
   std::vector<std::string> mht_72_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_72(mht_72_v, 2016, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CreateSelectAndScatter");

  return absl::make_unique<HloSelectAndScatterInstruction>(
      shape, operand, select, window, source, init_value, scatter);
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateBroadcast(
    const Shape& shape, HloInstruction* operand,
    absl::Span<const int64_t> broadcast_dimensions) {
   std::vector<std::string> mht_73_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_73(mht_73_v, 2026, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CreateBroadcast");

  return absl::make_unique<HloBroadcastInstruction>(shape, operand,
                                                    broadcast_dimensions);
}

/* static */ std::unique_ptr<HloInstruction>
HloInstruction::CreateGetDimensionSize(const Shape& shape,
                                       HloInstruction* operand,
                                       int64_t dimension) {
   std::vector<std::string> mht_74_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_74(mht_74_v, 2037, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CreateGetDimensionSize");

  return absl::make_unique<HloGetDimensionSizeInstruction>(shape, operand,
                                                           dimension);
}

/* static */ std::unique_ptr<HloInstruction>
HloInstruction::CreateSetDimensionSize(const Shape& shape,
                                       HloInstruction* operand,
                                       HloInstruction* val, int64_t dimension) {
   std::vector<std::string> mht_75_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_75(mht_75_v, 2048, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CreateSetDimensionSize");

  return absl::make_unique<HloSetDimensionSizeInstruction>(shape, operand, val,
                                                           dimension);
}

/* static */ std::unique_ptr<HloInstruction>
HloInstruction::CreateBroadcastSequence(
    const Shape& output_shape, HloInstruction* operand,
    const std::function<HloInstruction*(std::unique_ptr<HloInstruction>)>&
        adder) {
   std::vector<std::string> mht_76_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_76(mht_76_v, 2060, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CreateBroadcastSequence");

  CHECK(ShapeUtil::IsScalar(operand->shape()) ||
        operand->shape().rank() == output_shape.rank());
  Shape broadcast_shape = ShapeUtil::ChangeElementType(
      output_shape, operand->shape().element_type());
  // Do explicit broadcast for scalar.
  if (ShapeUtil::IsScalar(operand->shape())) {
    auto broadcast =
        HloInstruction::CreateBroadcast(broadcast_shape, operand, {});
    broadcast->set_metadata(operand->metadata());
    if (operand->has_sharding()) {
      broadcast->set_sharding(operand->sharding());
    }
    broadcast->set_frontend_attributes(operand->frontend_attributes());
    return broadcast;
  }
  // Do explicit broadcast for degenerate broadcast.
  std::vector<int64_t> broadcast_dimensions;
  std::vector<int64_t> reshaped_dimensions;
  for (int i = 0; i < operand->shape().rank(); i++) {
    if (operand->shape().dimensions(i) == output_shape.dimensions(i)) {
      broadcast_dimensions.push_back(i);
      reshaped_dimensions.push_back(operand->shape().dimensions(i));
    } else {
      CHECK_EQ(operand->shape().dimensions(i), 1)
          << "An explicit broadcast sequence requires the broadcasted "
             "dimensions to be trivial; operand: "
          << operand->ToString() << "; output_shape: " << output_shape;
    }
  }
  // Eliminate the size one dimensions.
  HloInstruction* reshaped_operand = adder(HloInstruction::CreateReshape(
      ShapeUtil::MakeShape(operand->shape().element_type(),
                           reshaped_dimensions),
      operand));
  reshaped_operand->set_metadata(operand->metadata());
  if (operand->has_sharding()) {
    reshaped_operand->set_sharding(operand->sharding());
  }
  reshaped_operand->set_frontend_attributes(operand->frontend_attributes());
  // Broadcast 'reshape' up to the larger size.
  auto broadcast = HloInstruction::CreateBroadcast(
      broadcast_shape, reshaped_operand, broadcast_dimensions);
  broadcast->set_metadata(operand->metadata());
  if (operand->has_sharding()) {
    broadcast->set_sharding(operand->sharding());
  }
  broadcast->set_frontend_attributes(operand->frontend_attributes());
  return broadcast;
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreatePad(
    const Shape& shape, HloInstruction* operand, HloInstruction* padding_value,
    const PaddingConfig& padding_config) {
   std::vector<std::string> mht_77_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_77(mht_77_v, 2116, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CreatePad");

  return absl::make_unique<HloPadInstruction>(shape, operand, padding_value,
                                              padding_config);
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateReshape(
    const Shape& shape, HloInstruction* operand, int64_t inferred_dimension) {
   std::vector<std::string> mht_78_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_78(mht_78_v, 2125, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CreateReshape");

  CHECK_EQ(ShapeUtil::ElementsIn(shape),
           ShapeUtil::ElementsIn(operand->shape()))
      << "shape: " << ShapeUtil::HumanString(shape)
      << " operand: " << ShapeUtil::HumanString(operand->shape());

  return absl::make_unique<HloReshapeInstruction>(shape, operand,
                                                  inferred_dimension);
}

/* static */ std::unique_ptr<HloInstruction>
HloInstruction::CreateDynamicReshape(
    const Shape& shape, HloInstruction* data_operand,
    absl::Span<HloInstruction* const> dim_sizes) {
   std::vector<std::string> mht_79_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_79(mht_79_v, 2141, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CreateDynamicReshape");

  CHECK_EQ(ShapeUtil::ElementsIn(shape),
           ShapeUtil::ElementsIn(data_operand[0].shape()))
      << "shape: " << ShapeUtil::HumanString(shape)
      << " operand: " << ShapeUtil::HumanString(data_operand[0].shape());
  CHECK_EQ(shape.rank(), dim_sizes.size());
  return absl::make_unique<HloDynamicReshapeInstruction>(shape, data_operand,
                                                         dim_sizes);
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateTranspose(
    const Shape& shape, HloInstruction* operand,
    absl::Span<const int64_t> dimensions) {
   std::vector<std::string> mht_80_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_80(mht_80_v, 2156, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CreateTranspose");

  return absl::make_unique<HloTransposeInstruction>(shape, operand, dimensions);
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateSort(
    const Shape& shape, int64_t dimension,
    absl::Span<HloInstruction* const> operands, HloComputation* compare,
    bool is_stable) {
   std::vector<std::string> mht_81_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_81(mht_81_v, 2166, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CreateSort");

  return absl::make_unique<HloSortInstruction>(shape, dimension, operands,
                                               compare, is_stable);
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateFusion(
    const Shape& shape, FusionKind fusion_kind, HloInstruction* fused_root) {
   std::vector<std::string> mht_82_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_82(mht_82_v, 2175, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CreateFusion");

  return absl::make_unique<HloFusionInstruction>(shape, fusion_kind,
                                                 fused_root);
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateFusion(
    const Shape& shape, FusionKind fusion_kind,
    absl::Span<HloInstruction* const> operands,
    HloComputation* fusion_computation) {
   std::vector<std::string> mht_83_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_83(mht_83_v, 2186, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CreateFusion");

  return absl::make_unique<HloFusionInstruction>(shape, fusion_kind, operands,
                                                 fusion_computation);
}

void HloInstruction::set_single_sharding(const HloSharding& sharding) {
   std::vector<std::string> mht_84_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_84(mht_84_v, 2194, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::set_single_sharding");

  CHECK(!sharding.IsTuple()) << sharding;
  if (shape().IsTuple()) {
    set_sharding(HloSharding::Tuple(sharding.GetAsShapeTree(shape())));
  } else {
    set_sharding(sharding);
  }
}

void HloInstruction::SetupDerivedInstruction(
    HloInstruction* derived_instruction) const {
   std::vector<std::string> mht_85_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_85(mht_85_v, 2207, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::SetupDerivedInstruction");

  if (sharding_ != nullptr &&
      ShapeUtil::CompatibleKind(shape_, derived_instruction->shape())) {
    // Only copy sharding if the tuple tree shape of the two instruction is
    // compatible because copying it between differently shaped instructions
    // can produce invalid shardings.
    derived_instruction->set_sharding(*sharding_);
  } else {
    derived_instruction->clear_sharding();
  }
  derived_instruction->set_metadata(metadata_);
  derived_instruction->set_frontend_attributes(frontend_attributes_);
}

bool HloInstruction::IsRoot() const {
   std::vector<std::string> mht_86_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_86(mht_86_v, 2224, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::IsRoot");

  return parent_ != nullptr && this == parent_->root_instruction();
}

bool HloInstruction::HasSideEffectNoRecurse() const {
   std::vector<std::string> mht_87_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_87(mht_87_v, 2231, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::HasSideEffectNoRecurse");

  switch (opcode_) {
    case HloOpcode::kSend:
    case HloOpcode::kSendDone:
    case HloOpcode::kRecv:
    case HloOpcode::kRecvDone:
    case HloOpcode::kRng:
    case HloOpcode::kRngGetAndUpdateState:
    case HloOpcode::kInfeed:
    case HloOpcode::kOutfeed:
    case HloOpcode::kTrace:
    case HloOpcode::kAllReduceStart:
    case HloOpcode::kAllReduceDone:
    case HloOpcode::kAllGatherStart:
    case HloOpcode::kAllGatherDone:
    case HloOpcode::kCollectivePermuteStart:
    case HloOpcode::kCollectivePermuteDone:
      return true;
    case HloOpcode::kAllReduce:
      return channel_id().has_value() ||
             Cast<HloAllReduceInstruction>(this)->constrain_layout();
    case HloOpcode::kAllToAll:
      return Cast<HloAllToAllInstruction>(this)->constrain_layout();
    case HloOpcode::kCustomCall:
      return Cast<HloCustomCallInstruction>(this)
          ->custom_call_has_side_effect();
    default:
      return false;
  }
}

bool HloInstruction::HasSideEffect() const {
   std::vector<std::string> mht_88_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_88(mht_88_v, 2265, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::HasSideEffect");

  if (HasSideEffectNoRecurse()) {
    return true;
  }
  // Check if any of the called computations has a side effect.
  for (const auto& computation : called_computations()) {
    if (computation->HasSideEffect()) {
      return true;
    }
  }
  return false;
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateCall(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloComputation* computation) {
   std::vector<std::string> mht_89_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_89(mht_89_v, 2283, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CreateCall");

  std::unique_ptr<HloInstruction> instruction =
      absl::WrapUnique(new HloInstruction(HloOpcode::kCall, shape));
  for (auto operand : operands) {
    instruction->AppendOperand(operand);
  }
  instruction->called_computations_.push_back(computation);
  return instruction;
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateCustomCall(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    absl::string_view custom_call_target, std::string opaque,
    CustomCallApiVersion api_version) {
   std::vector<std::string> mht_90_v;
   mht_90_v.push_back("custom_call_target: \"" + std::string(custom_call_target.data(), custom_call_target.size()) + "\"");
   mht_90_v.push_back("opaque: \"" + opaque + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_90(mht_90_v, 2301, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CreateCustomCall");

  return absl::make_unique<HloCustomCallInstruction>(
      shape, operands, custom_call_target, std::move(opaque), api_version);
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateCustomCall(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloComputation* to_apply, absl::string_view custom_call_target,
    std::string opaque, CustomCallApiVersion api_version) {
   std::vector<std::string> mht_91_v;
   mht_91_v.push_back("custom_call_target: \"" + std::string(custom_call_target.data(), custom_call_target.size()) + "\"");
   mht_91_v.push_back("opaque: \"" + opaque + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_91(mht_91_v, 2314, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CreateCustomCall");

  return absl::make_unique<HloCustomCallInstruction>(
      shape, operands, to_apply, custom_call_target, std::move(opaque),
      api_version);
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateCustomCall(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    absl::Span<HloComputation* const> called_computations,
    absl::string_view custom_call_target, std::string opaque,
    CustomCallApiVersion api_version) {
   std::vector<std::string> mht_92_v;
   mht_92_v.push_back("custom_call_target: \"" + std::string(custom_call_target.data(), custom_call_target.size()) + "\"");
   mht_92_v.push_back("opaque: \"" + opaque + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_92(mht_92_v, 2329, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CreateCustomCall");

  return absl::make_unique<HloCustomCallInstruction>(
      shape, operands, called_computations, custom_call_target,
      std::move(opaque), api_version);
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateCustomCall(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    absl::string_view custom_call_target,
    absl::Span<const Shape> operand_shapes_with_layout, std::string opaque,
    CustomCallApiVersion api_version) {
   std::vector<std::string> mht_93_v;
   mht_93_v.push_back("custom_call_target: \"" + std::string(custom_call_target.data(), custom_call_target.size()) + "\"");
   mht_93_v.push_back("opaque: \"" + opaque + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_93(mht_93_v, 2344, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CreateCustomCall");

  return absl::make_unique<HloCustomCallInstruction>(
      shape, operands, custom_call_target, std::move(opaque),
      operand_shapes_with_layout, api_version);
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateTuple(
    absl::Span<HloInstruction* const> elements) {
   std::vector<std::string> mht_94_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_94(mht_94_v, 2354, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CreateTuple");

  std::vector<Shape> element_shapes;
  element_shapes.reserve(elements.size());
  for (auto element : elements) {
    element_shapes.push_back(element->shape());
  }
  Shape tuple_shape = ShapeUtil::MakeTupleShape(element_shapes);
  return CreateVariadic(tuple_shape, HloOpcode::kTuple, elements);
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateGather(
    const Shape& shape, HloInstruction* operand, HloInstruction* start_indices,
    const GatherDimensionNumbers& gather_dim_numbers,
    absl::Span<const int64_t> slice_sizes, bool indices_are_sorted) {
   std::vector<std::string> mht_95_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_95(mht_95_v, 2370, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CreateGather");

  return absl::make_unique<HloGatherInstruction>(
      shape, operand, start_indices, gather_dim_numbers, slice_sizes,
      indices_are_sorted);
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateScatter(
    const Shape& shape, HloInstruction* operand,
    HloInstruction* scatter_indices, HloInstruction* updates,
    HloComputation* update_computation,
    const ScatterDimensionNumbers& scatter_dim_numbers, bool indices_are_sorted,
    bool unique_indices) {
   std::vector<std::string> mht_96_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_96(mht_96_v, 2384, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CreateScatter");

  return absl::make_unique<HloScatterInstruction>(
      shape, operand, scatter_indices, updates, update_computation,
      scatter_dim_numbers, indices_are_sorted, unique_indices);
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateDomain(
    const Shape& shape, HloInstruction* operand,
    std::unique_ptr<DomainMetadata> operand_side_metadata,
    std::unique_ptr<DomainMetadata> user_side_metadata) {
   std::vector<std::string> mht_97_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_97(mht_97_v, 2396, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CreateDomain");

  return absl::make_unique<HloDomainInstruction>(
      shape, operand, std::move(operand_side_metadata),
      std::move(user_side_metadata));
}

std::unique_ptr<HloInstruction> HloInstruction::CloneWithNewOperands(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
   std::vector<std::string> mht_98_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_98(mht_98_v, 2407, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CloneWithNewOperands");

  VLOG(3) << "CloneWithNewOperands:\n  " << ToString();
  VLOG(3) << "  new operands:";
  for (const HloInstruction* new_operand : new_operands) {
    VLOG(3) << "    %" << new_operand->name();
  }

  std::unique_ptr<HloInstruction> clone;
  // Explicitly call the factory for the instruction type. This is more robust
  // in the face of code changes than copying fields explicitly. This also
  // properly sets the user fields of the operands.
  switch (opcode_) {
    // Ops migrated to subclasses.
    // TODO(b/80131774): Remove this switch when migration is complete.
    case HloOpcode::kBatchNormTraining:
    case HloOpcode::kBatchNormInference:
    case HloOpcode::kBatchNormGrad:
    case HloOpcode::kFft:
    case HloOpcode::kCompare:
    case HloOpcode::kAsyncStart:
    case HloOpcode::kAsyncUpdate:
    case HloOpcode::kAsyncDone:
    case HloOpcode::kCopyStart:
    case HloOpcode::kSend:
    case HloOpcode::kSendDone:
    case HloOpcode::kRecv:
    case HloOpcode::kRecvDone:
    case HloOpcode::kReverse:
    case HloOpcode::kConcatenate:
    case HloOpcode::kReduce:
    case HloOpcode::kTranspose:
    case HloOpcode::kBroadcast:
    case HloOpcode::kReshape:
    case HloOpcode::kDynamicReshape:
    case HloOpcode::kMap:
    case HloOpcode::kSlice:
    case HloOpcode::kConstant:
    case HloOpcode::kTrace:
    case HloOpcode::kFusion:
    case HloOpcode::kRng:
    case HloOpcode::kRngBitGenerator:
    case HloOpcode::kRngGetAndUpdateState:
    case HloOpcode::kParameter:
    case HloOpcode::kGetTupleElement:
    case HloOpcode::kReducePrecision:
    case HloOpcode::kAllGather:
    case HloOpcode::kAllGatherStart:
    case HloOpcode::kAllReduce:
    case HloOpcode::kReduceScatter:
    case HloOpcode::kAllReduceStart:
    case HloOpcode::kAllToAll:
    case HloOpcode::kCollectivePermute:
    case HloOpcode::kCollectivePermuteStart:
    case HloOpcode::kInfeed:
    case HloOpcode::kOutfeed:
    case HloOpcode::kConvolution:
    case HloOpcode::kCustomCall:
    case HloOpcode::kReduceWindow:
    case HloOpcode::kSelectAndScatter:
    case HloOpcode::kPad:
    case HloOpcode::kDynamicSlice:
    case HloOpcode::kSort:
    case HloOpcode::kGather:
    case HloOpcode::kScatter:
    case HloOpcode::kIota:
    case HloOpcode::kDot:
    case HloOpcode::kDomain:
    case HloOpcode::kGetDimensionSize:
    case HloOpcode::kSetDimensionSize:
    case HloOpcode::kTriangularSolve:
    case HloOpcode::kCholesky:
      clone = CloneWithNewOperandsImpl(shape, new_operands, context);
      break;
    // Unary ops.
    case HloOpcode::kAbs:
    case HloOpcode::kAllGatherDone:
    case HloOpcode::kAllReduceDone:
    case HloOpcode::kRoundNearestAfz:
    case HloOpcode::kBitcast:
    case HloOpcode::kCeil:
    case HloOpcode::kClz:
    case HloOpcode::kCollectivePermuteDone:
    case HloOpcode::kCopy:
    case HloOpcode::kOptimizationBarrier:
    case HloOpcode::kCopyDone:
    case HloOpcode::kCos:
    case HloOpcode::kExp:
    case HloOpcode::kExpm1:
    case HloOpcode::kImag:
    case HloOpcode::kIsFinite:
    case HloOpcode::kFloor:
    case HloOpcode::kLog:
    case HloOpcode::kLog1p:
    case HloOpcode::kNot:
    case HloOpcode::kNegate:
    case HloOpcode::kPopulationCount:
    case HloOpcode::kReal:
    case HloOpcode::kRsqrt:
    case HloOpcode::kLogistic:
    case HloOpcode::kSign:
    case HloOpcode::kSin:
    case HloOpcode::kSqrt:
    case HloOpcode::kCbrt:
    case HloOpcode::kTanh:
      CHECK_EQ(new_operands.size(), 1);
      clone = CreateUnary(shape, opcode_, new_operands[0]);
      break;
    // Binary ops.
    case HloOpcode::kAdd:
    case HloOpcode::kAtan2:
    case HloOpcode::kComplex:
    case HloOpcode::kDivide:
    case HloOpcode::kMultiply:
    case HloOpcode::kSubtract:
    case HloOpcode::kMaximum:
    case HloOpcode::kMinimum:
    case HloOpcode::kPower:
    case HloOpcode::kRemainder:
    case HloOpcode::kAnd:
    case HloOpcode::kOr:
    case HloOpcode::kXor:
    case HloOpcode::kShiftLeft:
    case HloOpcode::kShiftRightArithmetic:
    case HloOpcode::kShiftRightLogical:
      CHECK_EQ(new_operands.size(), 2);
      clone = CreateBinary(shape, opcode_, new_operands[0], new_operands[1]);
      break;
    // Ternary ops.
    case HloOpcode::kClamp:
    case HloOpcode::kSelect:
    case HloOpcode::kTupleSelect:
      CHECK_EQ(new_operands.size(), 3);
      clone = CreateTernary(shape, opcode_, new_operands[0], new_operands[1],
                            new_operands[2]);
      break;
    // Other supported ops.
    case HloOpcode::kCall:
      clone = CreateCall(shape, new_operands, to_apply());
      break;
    case HloOpcode::kConvert:
      CHECK_EQ(new_operands.size(), 1);
      clone = CreateConvert(shape, new_operands[0]);
      break;
    case HloOpcode::kBitcastConvert:
      CHECK_EQ(new_operands.size(), 1);
      clone = CreateBitcastConvert(shape, new_operands[0]);
      break;
    case HloOpcode::kDynamicUpdateSlice:
      clone = CreateDynamicUpdateSlice(shape, new_operands[0], new_operands[1],
                                       new_operands.subspan(2));
      break;
    case HloOpcode::kTuple:
      clone = CreateTuple(new_operands);
      *clone->mutable_shape() = shape;
      break;
    case HloOpcode::kWhile:
      CHECK_EQ(new_operands.size(), 1);
      clone =
          CreateWhile(shape, while_condition(), while_body(), new_operands[0]);
      break;
    case HloOpcode::kConditional:
      CHECK_EQ(new_operands.size(), branch_count() + 1);
      clone = CreateConditional(shape, new_operands[0],
                                absl::MakeSpan(branch_computations()),
                                new_operands.subspan(1));
      break;
    case HloOpcode::kAfterAll:
      if (new_operands.empty()) {
        clone = CreateToken();
      } else {
        clone = CreateAfterAll(new_operands);
      }
      break;
    case HloOpcode::kAddDependency:
      CHECK_EQ(new_operands.size(), 2);
      clone = CreateAddDependency(new_operands[0], new_operands[1]);
      break;
    case HloOpcode::kReplicaId:
      CHECK_EQ(new_operands.size(), 0);
      clone = CreateReplicaId(shape);
      break;
    case HloOpcode::kPartitionId:
      CHECK_EQ(new_operands.size(), 0);
      clone = CreatePartitionId(shape);
      break;
  }
  // SetupDerivedInstruction will setup the precision_config_ field.
  SetupDerivedInstruction(clone.get());
  clone->set_parent(parent_);
  clone->set_outer_dimension_partitions(outer_dimension_partitions_);
  clone->backend_config_ = backend_config_.Clone();
  // The new instruction's name will be uniquified when it's added to a
  // computation.
  clone->SetAndSanitizeName(name());
  if (context != nullptr) {
    context->MapInstruction(this, clone.get());
    clone->ReplaceCalledComputations([&](HloComputation* callee) {
      return callee->parent() != context->module()
                 ? context->module()->DeepCloneComputation(callee, context)
                 : callee;
    });
  }
  return clone;
}

void HloInstruction::DetachFromOperandsAndUsers() {
   std::vector<std::string> mht_99_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_99(mht_99_v, 2615, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::DetachFromOperandsAndUsers");

  if (cleaned_up_) {
    return;
  }
  cleaned_up_ = true;
  // Detach from operands. An instruction may be repeated as an operand. To
  // avoid calling RemoveUser twice on the same operand, check before remove.
  for (int64_t operand_num = 0; operand_num < operand_count(); ++operand_num) {
    HloInstruction* operand = operands_[operand_num];
    if (operand == nullptr) {
      continue;
    }
    if (operand->user_map_.find(this) != operand->user_map_.end()) {
      operand->RemoveUser(this);
    }
    operands_[operand_num] = nullptr;
  }

  // Update users. Set `nullptr` to the corresponding operand slot for users.
  for (auto& user : this->users()) {
    for (int i = 0; i < user->operand_count(); ++i) {
      if (user->operands_[i] == this) {
        user->operands_[i] = nullptr;
      }
    }
  }
}

std::unique_ptr<HloInstruction> HloInstruction::CloneWithNewShape(
    const Shape& shape, const std::string& suffix,
    HloCloneContext* context) const {
   std::vector<std::string> mht_100_v;
   mht_100_v.push_back("suffix: \"" + suffix + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_100(mht_100_v, 2649, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CloneWithNewShape");

  std::unique_ptr<HloInstruction> clone =
      CloneWithNewOperands(shape, operands_, context);
  if (suffix.empty()) {
    clone->name_ = name();
  } else {
    // If an instruction is cloned multiple times avoid names like
    // foo.suffix.suffix.suffix. Instead of repeating the suffix add a numeric
    // suffix. Specifically, the clone of foo.suffix is named foo.suffix2, the
    // clone of foo.suffix2 is named foo.suffix3 and so on.
    const std::string dot_suffix = "." + suffix;
    size_t index = name().rfind(dot_suffix);
    if (index == std::string::npos) {
      // Existing name does not include ".suffix".
      clone->name_ = name() + dot_suffix;
    } else {
      // Existing name includes ".suffix". Determine if substring after
      // ".suffix" is numeric and should be replaced with an incremented number.
      std::string after_suffix = name().substr(index + dot_suffix.size());
      if (after_suffix.empty()) {
        // Existing name ends in ".suffix". New name should end in ".suffix2".
        clone->name_ = name() + "2";
      } else {
        // If names ends with .suffix[0-9]+ then replace with a suffix with the
        // numeric value incremented.
        int64_t numeric_suffix;
        if (absl::SimpleAtoi(after_suffix, &numeric_suffix)) {
          clone->name_ =
              StrCat(name().substr(0, index), dot_suffix, numeric_suffix + 1);
        } else {
          // Substring after ".suffix" is non-numeric.
          clone->name_ = name() + dot_suffix;
        }
      }
    }
  }
  return clone;
}

std::unique_ptr<HloInstruction> HloInstruction::Clone(
    const std::string& suffix, HloCloneContext* context) const {
   std::vector<std::string> mht_101_v;
   mht_101_v.push_back("suffix: \"" + suffix + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_101(mht_101_v, 2693, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::Clone");

  std::unique_ptr<HloInstruction> clone =
      CloneWithNewShape(shape_, suffix, context);
  return clone;
}

std::pair<const HloInstruction*, ShapeIndex>
HloInstruction::LatestNonGteAncestorAndIndex() const {
  const HloInstruction* hlo = this;
  ShapeIndex index;
  while (hlo->opcode() == HloOpcode::kGetTupleElement) {
    index.push_back(hlo->tuple_index());
    hlo = hlo->operand(0);
  }

  // We built up index in the reverse order from what we want.
  std::reverse(index.begin(), index.end());

  return {hlo, index};
}

const HloInstruction* HloInstruction::LatestNonGteAncestor() const {
   std::vector<std::string> mht_102_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_102(mht_102_v, 2717, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::LatestNonGteAncestor");

  const HloInstruction* hlo = this;
  while (hlo->opcode() == HloOpcode::kGetTupleElement) {
    hlo = hlo->operand(0);
  }
  return hlo;
}

const HloInstruction* HloInstruction::operand(int64_t i) const {
   std::vector<std::string> mht_103_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_103(mht_103_v, 2728, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::operand");

  return operands_.at(i);
}

HloInstruction* HloInstruction::mutable_operand(int64_t i) {
   std::vector<std::string> mht_104_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_104(mht_104_v, 2735, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::mutable_operand");

  CHECK(operands_[i] != nullptr);
  return operands_.at(i);
}

int64_t HloInstruction::operand_index(const HloInstruction* target) const {
   std::vector<std::string> mht_105_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_105(mht_105_v, 2743, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::operand_index");

  for (int64_t i = 0; i < operand_count(); ++i) {
    if (target == operand(i)) {
      return i;
    }
  }
  LOG(FATAL) << "target was not an operand: " << target->ToString();
}

HloInstruction::InstructionVector HloInstruction::unique_operands() const {
   std::vector<std::string> mht_106_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_106(mht_106_v, 2755, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::unique_operands");

  InstructionVector unique;
  absl::flat_hash_set<const HloInstruction*> seen;
  for (HloInstruction* operand : operands()) {
    if (seen.insert(operand).second) {
      unique.push_back(operand);
    }
  }
  return unique;
}

Status HloInstruction::AddControlDependencyTo(HloInstruction* instruction) {
   std::vector<std::string> mht_107_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_107(mht_107_v, 2769, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::AddControlDependencyTo");

  TF_RET_CHECK(instruction->parent() == parent());
  if (!absl::c_linear_search(control_successors_, instruction)) {
    control_successors_.push_back(instruction);
    TF_RET_CHECK(
        !absl::c_linear_search(instruction->control_predecessors_, this));
    instruction->control_predecessors_.push_back(this);
  }
  return Status::OK();
}

Status HloInstruction::RemoveControlDependencyTo(HloInstruction* instruction) {
   std::vector<std::string> mht_108_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_108(mht_108_v, 2783, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::RemoveControlDependencyTo");

  TF_RET_CHECK(instruction->parent() == parent());
  TF_RETURN_IF_ERROR(EraseElementFromVector(&control_successors_, instruction));
  TF_RETURN_IF_ERROR(
      EraseElementFromVector(&instruction->control_predecessors_, this));
  return Status::OK();
}

Status HloInstruction::DropAllControlDeps() {
   std::vector<std::string> mht_109_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_109(mht_109_v, 2794, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::DropAllControlDeps");

  for (auto* ctrl_succ : control_successors_) {
    TF_RETURN_IF_ERROR(
        EraseElementFromVector(&ctrl_succ->control_predecessors_, this));
  }
  for (auto* ctrl_pred : control_predecessors_) {
    TF_RETURN_IF_ERROR(
        EraseElementFromVector(&ctrl_pred->control_successors_, this));
  }
  control_successors_.clear();
  control_predecessors_.clear();
  return Status::OK();
}

Status HloInstruction::CopyAllControlDepsFrom(const HloInstruction* inst) {
   std::vector<std::string> mht_110_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_110(mht_110_v, 2811, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::CopyAllControlDepsFrom");

  for (auto* ctrl_pred : inst->control_predecessors()) {
    TF_RETURN_IF_ERROR(ctrl_pred->AddControlDependencyTo(this));
  }

  for (auto* ctrl_succ : inst->control_successors()) {
    TF_RETURN_IF_ERROR(this->AddControlDependencyTo(ctrl_succ));
  }

  return Status::OK();
}

bool HloInstruction::IdenticalInternal(
    const HloInstruction& other,
    const std::function<bool(const HloInstruction*, const HloInstruction*)>&
        eq_operands,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
        eq_computations,
    bool layout_sensitive, bool ignore_channel_id_values) const {
   std::vector<std::string> mht_111_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_111(mht_111_v, 2832, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::IdenticalInternal");

  // An instruction is always identical to itself.
  if (this == &other) {
    return true;
  }

  // Identical instruction must have the same opcode, shape, and identical
  // operands.
  if (opcode() != other.opcode()) {
    return false;
  }
  if (!(layout_sensitive ? ShapeUtil::Equal(shape(), other.shape())
                         : ShapeUtil::Compatible(shape(), other.shape()))) {
    return false;
  }
  if (operands().size() != other.operands().size()) {
    return false;
  }

  // Use an explicit loop rather than ContainerEquals, because copying around
  // std::functions may be too expensive in some cases.
  for (size_t i = 0; i < operands().size(); ++i) {
    if (!eq_operands(operand(i), other.operand(i))) {
      return false;
    }
  }

  if (backend_config_ != other.backend_config_) {
    return false;
  }

  if (ignore_channel_id_values) {
    if (auto channel_inst = DynCast<HloChannelInstruction>(this)) {
      return channel_inst->IdenticalSlowPathIgnoringChannelIdValues(
          other, eq_computations);
    }
  }
  return IdenticalSlowPath(other, eq_computations);
}

void HloInstruction::AppendOperand(HloInstruction* operand) {
   std::vector<std::string> mht_112_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_112(mht_112_v, 2875, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::AppendOperand");

  if (operand->parent() != nullptr) {
    DCHECK(!operand->parent()->IsMarkedAsDead(operand))
        << "Operand " << operand->name() << " is already marked dead";
  }
  operands_.push_back(operand);
  operand->AddUser(this);
}

void HloInstruction::RemoveOperandsAtAscendingIndices(
    absl::Span<const int> ascending_indices) {
   std::vector<std::string> mht_113_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_113(mht_113_v, 2888, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::RemoveOperandsAtAscendingIndices");

  if (ascending_indices.empty()) {
    return;
  }
  int next_index = 0;
  int removed_count = 0;
  for (int to_remove : ascending_indices) {
    while (next_index < to_remove) {
      operands_[next_index - removed_count] = operands_[next_index];
      ++next_index;
    }
    CHECK_LT(to_remove, operands_.size());
    ++removed_count;
    ++next_index;
  }
  while (next_index < operands_.size()) {
    operands_[next_index - removed_count] = operands_[next_index];
    ++next_index;
  }
  CHECK_EQ(removed_count, ascending_indices.size());
  operands_.resize(operands_.size() - removed_count);
}

void HloInstruction::AddUser(HloInstruction* user) {
   std::vector<std::string> mht_114_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_114(mht_114_v, 2914, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::AddUser");

  if (!ContainsKey(user_map_, user)) {
    user_map_.emplace(user, users_.size());
    users_.push_back(user);
  }
}

int64_t HloInstruction::UserId(HloInstruction* user) {
   std::vector<std::string> mht_115_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_115(mht_115_v, 2924, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::UserId");

  auto result = user_map_.find(user);
  CHECK(result != user_map_.end());
  return result->second;
}

bool HloInstruction::HasConstantOperand() const {
   std::vector<std::string> mht_116_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_116(mht_116_v, 2933, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::HasConstantOperand");

  for (const HloInstruction* operand : operands_) {
    if (operand->IsConstant()) {
      return true;
    }
  }
  return false;
}

bool HloInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
        eq_computations) const {
   std::vector<std::string> mht_117_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_117(mht_117_v, 2948, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::IdenticalSlowPath");

  // Perform opcode specific checks.
  switch (opcode()) {
    // The result of these instructions only depend upon their opcode and
    // operands.
    case HloOpcode::kAbs:
    case HloOpcode::kAllGatherDone:
    case HloOpcode::kAllReduceDone:
    case HloOpcode::kAtan2:
    case HloOpcode::kAdd:
    case HloOpcode::kBitcast:
    case HloOpcode::kBitcastConvert:
    case HloOpcode::kCeil:
    case HloOpcode::kClamp:
    case HloOpcode::kClz:
    case HloOpcode::kCollectivePermuteDone:
    case HloOpcode::kComplex:
    case HloOpcode::kConvert:
    case HloOpcode::kCopy:
    case HloOpcode::kCopyStart:
    case HloOpcode::kCopyDone:
    case HloOpcode::kCos:
    case HloOpcode::kDivide:
    case HloOpcode::kDynamicUpdateSlice:
    case HloOpcode::kExp:
    case HloOpcode::kExpm1:
    case HloOpcode::kFloor:
    case HloOpcode::kImag:
    case HloOpcode::kIsFinite:
    case HloOpcode::kLog:
    case HloOpcode::kLog1p:
    case HloOpcode::kAnd:
    case HloOpcode::kNot:
    case HloOpcode::kOr:
    case HloOpcode::kXor:
    case HloOpcode::kMaximum:
    case HloOpcode::kMinimum:
    case HloOpcode::kMultiply:
    case HloOpcode::kNegate:
    case HloOpcode::kOptimizationBarrier:
    case HloOpcode::kPartitionId:
    case HloOpcode::kPopulationCount:
    case HloOpcode::kPower:
    case HloOpcode::kReal:
    case HloOpcode::kRemainder:
    case HloOpcode::kReshape:
    case HloOpcode::kDynamicReshape:
    case HloOpcode::kReplicaId:
    case HloOpcode::kRoundNearestAfz:
    case HloOpcode::kRsqrt:
    case HloOpcode::kSelect:
    case HloOpcode::kShiftLeft:
    case HloOpcode::kShiftRightArithmetic:
    case HloOpcode::kShiftRightLogical:
    case HloOpcode::kLogistic:
    case HloOpcode::kSign:
    case HloOpcode::kSin:
    case HloOpcode::kSqrt:
    case HloOpcode::kCbrt:
    case HloOpcode::kSubtract:
    case HloOpcode::kTanh:
    case HloOpcode::kTuple:
    case HloOpcode::kTupleSelect:
      return true;

    // This opcode has complex or special behavior so just return false.
    case HloOpcode::kAfterAll:
    case HloOpcode::kAddDependency:
      return false;

    // Remaining instructions with special values.
    case HloOpcode::kCall:
      return eq_computations(to_apply(), other.to_apply());
    case HloOpcode::kConditional:
      for (int j = 0; j < branch_count(); ++j) {
        if (!eq_computations(branch_computation(j),
                             other.branch_computation(j))) {
          return false;
        }
      }
      return true;
    case HloOpcode::kWhile:
      return (eq_computations(while_body(), other.while_body()) &&
              eq_computations(while_condition(), other.while_condition()));

    // Ops migrated to subclasses should never come to this line.
    // TODO(b/80131774): Remove this switch when migration is complete.
    case HloOpcode::kAsyncStart:
    case HloOpcode::kAsyncUpdate:
    case HloOpcode::kAsyncDone:
    case HloOpcode::kBatchNormTraining:
    case HloOpcode::kBatchNormInference:
    case HloOpcode::kBatchNormGrad:
    case HloOpcode::kFft:
    case HloOpcode::kCompare:
    case HloOpcode::kSend:
    case HloOpcode::kSendDone:
    case HloOpcode::kRecv:
    case HloOpcode::kRecvDone:
    case HloOpcode::kReverse:
    case HloOpcode::kConcatenate:
    case HloOpcode::kReduce:
    case HloOpcode::kSort:
    case HloOpcode::kTranspose:
    case HloOpcode::kBroadcast:
    case HloOpcode::kMap:
    case HloOpcode::kSlice:
    case HloOpcode::kConstant:
    case HloOpcode::kIota:
    case HloOpcode::kTrace:
    case HloOpcode::kFusion:
    case HloOpcode::kRng:
    case HloOpcode::kRngBitGenerator:
    case HloOpcode::kRngGetAndUpdateState:
    case HloOpcode::kParameter:
    case HloOpcode::kGetTupleElement:
    case HloOpcode::kReducePrecision:
    case HloOpcode::kInfeed:
    case HloOpcode::kOutfeed:
    case HloOpcode::kAllGather:
    case HloOpcode::kAllGatherStart:
    case HloOpcode::kAllReduce:
    case HloOpcode::kReduceScatter:
    case HloOpcode::kAllReduceStart:
    case HloOpcode::kAllToAll:
    case HloOpcode::kCollectivePermute:
    case HloOpcode::kCollectivePermuteStart:
    case HloOpcode::kConvolution:
    case HloOpcode::kCustomCall:
    case HloOpcode::kReduceWindow:
    case HloOpcode::kSelectAndScatter:
    case HloOpcode::kPad:
    case HloOpcode::kDynamicSlice:
    case HloOpcode::kGather:
    case HloOpcode::kScatter:
    case HloOpcode::kDot:
    case HloOpcode::kDomain:
    case HloOpcode::kGetDimensionSize:
    case HloOpcode::kSetDimensionSize:
    case HloOpcode::kTriangularSolve:
    case HloOpcode::kCholesky:
      LOG(FATAL) << "Base class impl called for opcode with subclass: "
                 << opcode();
  }
  return false;
}

void HloInstruction::RemoveUser(HloInstruction* user) {
   std::vector<std::string> mht_118_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_118(mht_118_v, 3098, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::RemoveUser");

  auto map_it = user_map_.find(user);
  CHECK(map_it != user_map_.end());

  const int64_t index = map_it->second;
  CHECK_EQ(users_[index], user);

  // Move the last user into the position of the removed user.
  users_[index] = users_.back();
  user_map_[users_.back()] = index;

  // Remove the user from the map and drop the last slot from the vector what
  // have been moved to the position of the original user.
  user_map_.erase(map_it);
  users_.pop_back();
}

Status HloInstruction::ReplaceUseWith(HloInstruction* user,
                                      HloInstruction* new_producer) {
   std::vector<std::string> mht_119_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_119(mht_119_v, 3119, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::ReplaceUseWith");

  TF_RET_CHECK(
      ShapeUtil::CompatibleIgnoringFpPrecision(shape(), new_producer->shape()))
      << "this shape: " << ShapeUtil::HumanString(shape())
      << ", replacement shape: "
      << ShapeUtil::HumanString(new_producer->shape());
  return ReplaceUseWithDifferentShape(user, new_producer);
}

Status HloInstruction::ReplaceUseWithDifferentShape(
    HloInstruction* user, HloInstruction* new_producer) {
   std::vector<std::string> mht_120_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_120(mht_120_v, 3132, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::ReplaceUseWithDifferentShape");

  VLOG(3) << "Replacing uses of " << name() << " in " << user->name()
          << " with " << new_producer->name();

  RemoveUser(user);

  TF_RET_CHECK(absl::c_count(user->operands_, this) >= 0);
  std::replace(user->operands_.begin(), user->operands_.end(), this,
               new_producer);
  new_producer->AddUser(user);
  // Custom fusions may not be able to handle deduplicated operands.
  if (user->opcode() == HloOpcode::kFusion) {
    TF_RETURN_IF_ERROR(
        Cast<HloFusionInstruction>(user)->DeduplicateFusionOperands());
  }
  return Status::OK();
}

Status HloInstruction::ReplaceOperandWith(int64_t operand_num,
                                          HloInstruction* new_operand) {
   std::vector<std::string> mht_121_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_121(mht_121_v, 3154, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::ReplaceOperandWith");

  auto old_operand = operand(operand_num);
  TF_RET_CHECK(ShapeUtil::CompatibleIgnoringFpPrecision(old_operand->shape(),
                                                        new_operand->shape()))
      << old_operand->shape() << " is not compatible with "
      << new_operand->shape();
  return ReplaceOperandWithDifferentShape(operand_num, new_operand);
}

Status HloInstruction::ReplaceOperandWithDifferentShape(
    int64_t operand_num, HloInstruction* new_operand) {
   std::vector<std::string> mht_122_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_122(mht_122_v, 3167, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::ReplaceOperandWithDifferentShape");

  TF_RET_CHECK(operand_num >= 0);
  TF_RET_CHECK(operand_num < operand_count());
  HloInstruction* old_operand = mutable_operand(operand_num);
  if (old_operand == new_operand) {
    return Status::OK();
  }

  operands_[operand_num] = new_operand;

  VLOG(3) << "Replacing operand " << operand_num << " of " << name() << " with "
          << new_operand->name() << ", was " << old_operand->name();

  if (!absl::c_linear_search(operands_, old_operand)) {
    old_operand->RemoveUser(this);
  }
  new_operand->AddUser(this);
  return Status::OK();
}

Status HloInstruction::ReplaceUsesWith(absl::Span<HloInstruction* const> users,
                                       HloInstruction* new_producer) {
   std::vector<std::string> mht_123_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_123(mht_123_v, 3191, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::ReplaceUsesWith");

  TF_RET_CHECK(
      ShapeUtil::CompatibleIgnoringFpPrecision(shape(), new_producer->shape()))
      << shape() << " is not compatible with " << new_producer->shape();
  return ReplaceAllUsesWithDifferentShape(users, new_producer);
}

Status HloInstruction::ReplaceAllUsesWithDifferentShape(
    absl::Span<HloInstruction* const> users, HloInstruction* new_producer) {
   std::vector<std::string> mht_124_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_124(mht_124_v, 3202, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::ReplaceAllUsesWithDifferentShape");

  for (HloInstruction* user : users) {
    TF_RETURN_IF_ERROR(ReplaceUseWithDifferentShape(user, new_producer));
  }

  if (parent_ && parent_->root_instruction() == this) {
    parent_->set_root_instruction(new_producer,
                                  /*accept_different_shape=*/true);
  }
  return Status::OK();
}

Status HloInstruction::ReplaceAllUsesWith(HloInstruction* new_producer) {
   std::vector<std::string> mht_125_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_125(mht_125_v, 3217, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::ReplaceAllUsesWith");

  TF_RET_CHECK(
      ShapeUtil::CompatibleIgnoringFpPrecision(shape(), new_producer->shape()))
      << shape() << " is not compatible with " << new_producer->shape();
  return ReplaceAllUsesWithDifferentShape(new_producer);
}

Status HloInstruction::ReplaceAllUsesWithDifferentShape(
    HloInstruction* new_producer) {
   std::vector<std::string> mht_126_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_126(mht_126_v, 3228, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::ReplaceAllUsesWithDifferentShape");

  bool new_producer_is_user = false;
  for (HloInstruction* user : users()) {
    if (user == new_producer) {
      // It's possible that new_producer is a user of this instruction as might
      // be the case when replacing an instruction with a kCopy of itself. In
      // this case, don't do the replacement to avoid creating a cycle in the
      // graph. new_producer remains the only user of this instruction.
      new_producer_is_user = true;
    } else {
      std::replace(user->operands_.begin(), user->operands_.end(), this,
                   new_producer);
      new_producer->AddUser(user);
      if (user->opcode() == HloOpcode::kFusion) {
        TF_RETURN_IF_ERROR(
            Cast<HloFusionInstruction>(user)->DeduplicateFusionOperands());
      }
    }
  }
  users_.clear();
  user_map_.clear();
  if (new_producer_is_user) {
    AddUser(new_producer);
  }
  if (parent_ && parent_->root_instruction() == this) {
    parent_->set_root_instruction(new_producer,
                                  /*accept_different_shape=*/true);
  }

  return Status::OK();
}

bool HloInstruction::IsEffectiveBitcast() const {
   std::vector<std::string> mht_127_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_127(mht_127_v, 3263, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::IsEffectiveBitcast");

  return opcode_ == HloOpcode::kBitcast ||
         (opcode_ == HloOpcode::kTranspose &&
          ShapeUtil::TransposeIsBitcast(operand(0)->shape(), shape(),
                                        dimensions()));
}

HloComputation* HloInstruction::to_apply() const {
   std::vector<std::string> mht_128_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_128(mht_128_v, 3273, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::to_apply");

  switch (opcode_) {
    case HloOpcode::kCall:
    case HloOpcode::kMap:
    case HloOpcode::kReduceWindow:
    case HloOpcode::kReduce:
    case HloOpcode::kAllReduce:
    case HloOpcode::kReduceScatter:
    case HloOpcode::kAllReduceStart:
    case HloOpcode::kScatter:
    case HloOpcode::kSort:
    case HloOpcode::kCustomCall:
      CHECK_EQ(called_computations_.size(), 1);
      return called_computations_[0];
    default:
      LOG(FATAL) << "Invalid opcode for to_apply(): "
                 << HloOpcodeString(opcode());
  }
}

void HloInstruction::set_to_apply(HloComputation* computation) {
   std::vector<std::string> mht_129_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_129(mht_129_v, 3296, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::set_to_apply");

  // Don't allow changing the computation for fused instructions so we don't
  // have to recompute called_instructions for the entire fusion instruction.
  CHECK(!IsFused());
  switch (opcode_) {
    case HloOpcode::kCall:
    case HloOpcode::kMap:
    case HloOpcode::kReduceWindow:
    case HloOpcode::kReduce:
    case HloOpcode::kAllReduce:
    case HloOpcode::kAllReduceStart:
    case HloOpcode::kScatter:
    case HloOpcode::kSort:
    case HloOpcode::kCustomCall:
      CHECK_EQ(called_computations_.size(), 1);
      called_computations_[0] = computation;
      break;
    default:
      LOG(FATAL) << "Invalid opcode for to_apply(): "
                 << HloOpcodeString(opcode());
  }
}

HloComputation* HloInstruction::while_condition() const {
   std::vector<std::string> mht_130_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_130(mht_130_v, 3322, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::while_condition");

  CHECK_EQ(HloOpcode::kWhile, opcode_);
  return called_computations_[kConditionComputationIndex];
}

HloComputation* HloInstruction::while_body() const {
   std::vector<std::string> mht_131_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_131(mht_131_v, 3330, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::while_body");

  CHECK_EQ(HloOpcode::kWhile, opcode_);
  return called_computations_[kBodyComputationIndex];
}

void HloInstruction::set_while_condition(HloComputation* computation) {
   std::vector<std::string> mht_132_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_132(mht_132_v, 3338, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::set_while_condition");

  // Don't allow changing the computation for fused instructions so we don't
  // have to recompute called_instructions for the entire fusion instruction.
  CHECK(!IsFused());
  CHECK_EQ(HloOpcode::kWhile, opcode_);
  called_computations_[kConditionComputationIndex] = computation;
}

void HloInstruction::set_while_body(HloComputation* computation) {
   std::vector<std::string> mht_133_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_133(mht_133_v, 3349, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::set_while_body");

  // Don't allow changing the computation for fused instructions so we don't
  // have to recompute called_instructions for the entire fusion instruction.
  CHECK(!IsFused());
  CHECK_EQ(HloOpcode::kWhile, opcode_);
  called_computations_[kBodyComputationIndex] = computation;
}

HloInstruction* HloInstruction::while_init() const {
   std::vector<std::string> mht_134_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_134(mht_134_v, 3360, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::while_init");

  CHECK_EQ(HloOpcode::kWhile, opcode_);
  return operands_[0];
}

HloComputation* HloInstruction::true_computation() const {
   std::vector<std::string> mht_135_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_135(mht_135_v, 3368, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::true_computation");

  CHECK_EQ(HloOpcode::kConditional, opcode_);
  CHECK_EQ(PRED, operand(0)->shape().element_type());
  return called_computations_[kTrueComputationIndex];
}

HloComputation* HloInstruction::false_computation() const {
   std::vector<std::string> mht_136_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_136(mht_136_v, 3377, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::false_computation");

  CHECK_EQ(HloOpcode::kConditional, opcode_);
  CHECK_EQ(PRED, operand(0)->shape().element_type());
  return called_computations_[kFalseComputationIndex];
}

const std::vector<HloComputation*>& HloInstruction::branch_computations()
    const {
   std::vector<std::string> mht_137_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_137(mht_137_v, 3387, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::branch_computations");

  CHECK(HloOpcode::kConditional == opcode_);
  return called_computations_;
}

int HloInstruction::branch_count() const {
   std::vector<std::string> mht_138_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_138(mht_138_v, 3395, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::branch_count");

  CHECK(HloOpcode::kConditional == opcode_);
  return called_computations_.size();
}

HloComputation* HloInstruction::branch_computation(int b) const {
   std::vector<std::string> mht_139_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_139(mht_139_v, 3403, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::branch_computation");

  CHECK(HloOpcode::kConditional == opcode_);
  CHECK_GE(b, 0);
  CHECK_LT(b, called_computations_.size());
  return called_computations_[b];
}

void HloInstruction::set_branch_computation(int b,
                                            HloComputation* computation) {
   std::vector<std::string> mht_140_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_140(mht_140_v, 3414, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::set_branch_computation");

  // Don't allow changing the computation for fused instructions so we don't
  // have to recompute called_instructions for the entire fusion instruction.
  CHECK(!IsFused());
  CHECK_EQ(HloOpcode::kConditional, opcode_);
  called_computations_[b] = computation;
}

std::string HloInstruction::SignatureString() const {
   std::vector<std::string> mht_141_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_141(mht_141_v, 3425, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::SignatureString");

  std::string operands =
      StrJoin(operands_, ", ", [](std::string* out, HloInstruction* operand) {
        StrAppend(out, ShapeUtil::HumanString(operand->shape()));
      });
  return StrCat("(", operands, ") -> ", ShapeUtil::HumanString(shape()));
}

std::string PrintName(const std::string& name, bool print_ids) {
   std::vector<std::string> mht_142_v;
   mht_142_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_142(mht_142_v, 3437, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "PrintName");

  if (print_ids) {
    return name;
  } else {
    auto dot_position = name.find_first_of('.');
    return name.substr(0, dot_position);
  }
}

namespace {

using DFSStack = absl::InlinedVector<std::pair<int, HloInstruction*>, 16>;

std::string PrintNameInternal(const std::string& name,
                              const HloPrintOptions& options) {
   std::vector<std::string> mht_143_v;
   mht_143_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_143(mht_143_v, 3455, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "PrintNameInternal");

  return StrCat(options.print_percent() ? "%" : "",
                PrintName(name, options.print_ids()));
}

void PrintCycle(const HloInstruction* child, DFSStack* dfs_stack) {
   std::vector<std::string> mht_144_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_144(mht_144_v, 3463, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "PrintCycle");

  // This set contains HloInstructions from the top of `DFSStack` that might
  // belong to the cycle, i.e. if  DFSStack :=[back,...,child,...,top], then
  // `subgraph` := {child,...,top}.
  absl::flat_hash_set<const HloInstruction*> subgraph;
  while (!dfs_stack->empty() && dfs_stack->back().second != child) {
    subgraph.insert(dfs_stack->back().second);
    dfs_stack->pop_back();
  }
  // Start dfs at `child` and find a cycle with all nodes in `subgraph`.
  absl::flat_hash_set<const HloInstruction*> visited;
  absl::InlinedVector<const HloInstruction*, 16> dfs;
  dfs.push_back(child);
  while (!dfs.empty()) {
    bool found_next_instr = false;
    for (const auto& user : dfs.back()->users()) {
      if (user == child) {
        dfs.push_back(child);
        LOG(INFO) << "\n\nDirected cycle:\n  "
                  << absl::StrJoin(
                         dfs, "\n  ",
                         [](std::string* out, const HloInstruction* instr) {
                           out->append(instr->name());
                         });
        return;
      }
      if (!subgraph.contains(user) || visited.contains(user)) {
        continue;
      }
      visited.insert(user);
      dfs.push_back(user);
      found_next_instr = true;
    }
    if (!found_next_instr) {
      dfs.pop_back();
    }
  }
}

}  // namespace

std::string HloInstruction::ToString(const HloPrintOptions& options) const {
   std::vector<std::string> mht_145_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_145(mht_145_v, 3507, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::ToString");

  CanonicalNameMap new_map;
  return ToStringWithCanonicalNameMap(options, &new_map);
}

bool HloInstruction::IsOpElementwise(HloOpcode opcode) {
   std::vector<std::string> mht_146_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_146(mht_146_v, 3515, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::IsOpElementwise");

  switch (opcode) {
    // Unary elementwise operations.
    case HloOpcode::kAbs:
    case HloOpcode::kRoundNearestAfz:
    case HloOpcode::kCeil:
    case HloOpcode::kClz:
    case HloOpcode::kConvert:
    case HloOpcode::kBitcastConvert:
    case HloOpcode::kCopy:
    case HloOpcode::kCos:
    case HloOpcode::kExp:
    case HloOpcode::kExpm1:
    case HloOpcode::kFloor:
    case HloOpcode::kImag:
    case HloOpcode::kIsFinite:
    case HloOpcode::kLog:
    case HloOpcode::kLog1p:
    case HloOpcode::kNot:
    case HloOpcode::kNegate:
    case HloOpcode::kPopulationCount:
    case HloOpcode::kReal:
    case HloOpcode::kReducePrecision:
    case HloOpcode::kRsqrt:
    case HloOpcode::kLogistic:
    case HloOpcode::kSign:
    case HloOpcode::kSin:
    case HloOpcode::kSqrt:
    case HloOpcode::kCbrt:
    case HloOpcode::kTanh:
      return true;

    // Binary elementwise operations, the same as in IsElementwiseBinary().
    case HloOpcode::kAdd:
    case HloOpcode::kAtan2:
    case HloOpcode::kCompare:
    case HloOpcode::kComplex:
    case HloOpcode::kDivide:
    case HloOpcode::kMaximum:
    case HloOpcode::kMinimum:
    case HloOpcode::kMultiply:
    case HloOpcode::kPower:
    case HloOpcode::kRemainder:
    case HloOpcode::kSubtract:
    case HloOpcode::kAnd:
    case HloOpcode::kOr:
    case HloOpcode::kXor:
    case HloOpcode::kShiftLeft:
    case HloOpcode::kShiftRightArithmetic:
    case HloOpcode::kShiftRightLogical:
      return true;

    // Ternary elementwise operations.
    case HloOpcode::kSelect:
    case HloOpcode::kClamp:
      return true;

    default:
      return false;
  }
}

bool HloInstruction::IsElementwiseImpl(
    const absl::optional<int64_t>& operand_idx) const {
   std::vector<std::string> mht_147_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_147(mht_147_v, 3581, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::IsElementwiseImpl");

  if (opcode_ == HloOpcode::kDynamicUpdateSlice) {
    return operand_idx.has_value() && operand_idx.value() == 0;
  }
  if (opcode_ == HloOpcode::kBitcastConvert &&
      primitive_util::BitWidth(shape_.element_type()) !=
          primitive_util::BitWidth(operands_[0]->shape().element_type())) {
    return false;
  }
  return IsOpElementwise(opcode_);
}

bool HloInstruction::IsCrossModuleAllReduce() const {
   std::vector<std::string> mht_148_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_148(mht_148_v, 3596, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::IsCrossModuleAllReduce");

  return opcode() == HloOpcode::kAllReduce && channel_id();
}

bool HloInstruction::IsCrossReplicaAllReduce() const {
   std::vector<std::string> mht_149_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_149(mht_149_v, 3603, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::IsCrossReplicaAllReduce");

  return opcode() == HloOpcode::kAllReduce && !channel_id();
}

std::string HloInstruction::ToStringWithCanonicalNameMap(
    const HloPrintOptions& options,
    CanonicalNameMap* canonical_name_map) const {
   std::vector<std::string> mht_150_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_150(mht_150_v, 3612, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::ToStringWithCanonicalNameMap");

  std::string result = "";

  // Logic to print the instruction name (e.g. "%foo = ").
  if (options.canonicalize_instruction_names()) {
    if (options.is_in_nested_computation()) {
      // If we are canonicalizing instruction names and this is a top-level
      // HloInstruction::ToString() call, don't print an instruction name.
      DCHECK(!options.print_percent());  // no need to call PrintNameInternal
      StrAppend(&result, canonical_name_map->LookupOrInsert(name()), " = ");
    }
  } else {
    StrAppend(&result, PrintNameInternal(name(), options), " = ");
  }

  if (options.print_result_shape()) {
    // Print shape.
    if (options.include_layout_in_shapes()) {
      StrAppend(&result, ShapeUtil::HumanStringWithLayout(shape()), " ");
    } else {
      StrAppend(&result, ShapeUtil::HumanString(shape()), " ");
    }
  }

  // Print opcode, operand(s).
  if (options.syntax_sugar_async_ops() && HloOpcodeIsAsync(opcode())) {
    std::string suffix = [&]() {
   std::vector<std::string> mht_151_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_151(mht_151_v, 3641, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "lambda");

      switch (opcode()) {
        case HloOpcode::kAsyncStart:
          return "-start";
        case HloOpcode::kAsyncUpdate:
          return "-update";
        default:
          CHECK(opcode() == HloOpcode::kAsyncDone)
              << "Unexpected async opcode: " << HloOpcodeString(opcode());
          return "-done";
      }
    }();
    StrAppend(&result, HloOpcodeString(async_wrapped_opcode()), suffix);
  } else {
    StrAppend(&result, HloOpcodeString(opcode()));
  }
  StrAppend(&result, "(",
            OperandsToStringWithCanonicalNameMap(options, canonical_name_map),
            ")");

  // Print additional attributes. If an instruction contains a subcomputation,
  // the subcomputation is also printed here.
  for (const std::string& extra : ExtraAttributesToString(options)) {
    StrAppend(&result, ", ", extra);
  }

  if (options.print_metadata() &&
      (!metadata_.op_type().empty() || !metadata_.op_name().empty() ||
       !metadata_.source_file().empty())) {
    StrAppend(&result, ", metadata={", xla::OpMetadataToString(metadata_), "}");
  }
  if (options.print_backend_config() && !backend_config_.empty()) {
    StrAppend(&result, ", backend_config=\"",
              CEscape(backend_config_.GetRawString()), "\"");
  }
  return result;
}

std::string HloInstruction::OperandsToString(
    const HloPrintOptions& options) const {
   std::vector<std::string> mht_152_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_152(mht_152_v, 3683, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::OperandsToString");

  CanonicalNameMap new_map;
  return OperandsToStringWithCanonicalNameMap(options, &new_map);
}

std::string HloInstruction::OperandsToStringWithCanonicalNameMap(
    const HloPrintOptions& options,
    CanonicalNameMap* canonical_name_map) const {
   std::vector<std::string> mht_153_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_153(mht_153_v, 3693, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::OperandsToStringWithCanonicalNameMap");

  std::string operands;
  absl::Span<HloInstruction* const> slice(operands_);
  const int64_t kMaxOperandsToShowIfCompact = 4;
  if (options.compact_operands() &&
      slice.size() > kMaxOperandsToShowIfCompact) {
    slice.remove_suffix(slice.size() - kMaxOperandsToShowIfCompact);
  }
  for (int64_t i = 0; i < slice.size(); ++i) {
    HloInstruction* operand = slice[i];
    if (i != 0) {
      StrAppend(&operands, ", ");
      if (options.print_operand_index_annotation_interval() != 0 &&
          i % options.print_operand_index_annotation_interval() == 0) {
        StrAppend(&operands, absl::StrFormat("/*index=%lld*/", i));
      }
    }
    // If operand is already been deleted, put `null` to the string output.
    if (operand == nullptr) {
      StrAppend(&operands, "null ");
      continue;
    }
    std::vector<std::string> str;
    if (options.print_operand_shape()) {
      if (options.include_layout_in_shapes()) {
        str.push_back(ShapeUtil::HumanStringWithLayout(operand->shape()));
      } else {
        str.push_back(ShapeUtil::HumanString(operand->shape()));
      }
    }
    if (options.canonicalize_instruction_names()) {
      if (options.is_in_nested_computation()) {
        // In a top-level HloInstruction::ToString() call, the operand name is
        // not part of the canonical string.
        DCHECK(!options.print_percent());  // no need to call PrintNameInternal
        str.push_back(canonical_name_map->LookupOrInsert(operand->name()));
      }
    } else if (options.print_operand_names()) {
      str.push_back(PrintNameInternal(operand->name(), options));
    }
    StrAppend(&operands, StrJoin(str, " "));
  }
  const int64_t remaining = operands_.size() - slice.size();
  if (slice.size() != operands_.size()) {
    StrAppend(&operands, ", ...(+", remaining, ")");
  }
  return operands;
}

namespace {

bool IsSequentialCall(HloOpcode opcode) {
   std::vector<std::string> mht_154_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_154(mht_154_v, 3747, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "IsSequentialCall");

  switch (opcode) {
    case HloOpcode::kCall:
    case HloOpcode::kConditional:
    case HloOpcode::kWhile:
      return true;
    default:
      return false;
  }
}

}  // namespace

std::vector<std::string> HloInstruction::ExtraAttributesToString(
    const HloPrintOptions& options) const {
   std::vector<std::string> mht_155_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_155(mht_155_v, 3764, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::ExtraAttributesToString");

  std::vector<std::string> extra = options.print_extra_attributes()
                                       ? ExtraAttributesToStringImpl(options)
                                       : std::vector<std::string>();

  const auto subcomputation_mode = options.print_subcomputation_mode();
  if (subcomputation_mode ==
      HloPrintOptions::PrintSubcomputationMode::kNameOnly) {
    if (opcode() == HloOpcode::kWhile) {
      extra.push_back(StrCat(
          "condition=", PrintNameInternal(while_condition()->name(), options)));
      extra.push_back(
          StrCat("body=", PrintNameInternal(while_body()->name(), options)));
    } else if (opcode() == HloOpcode::kSelectAndScatter) {
      extra.push_back(
          StrCat("select=", PrintNameInternal(select()->name(), options)));
      extra.push_back(
          StrCat("scatter=", PrintNameInternal(scatter()->name(), options)));
    } else if (opcode() == HloOpcode::kConditional) {
      if (operand(0)->shape().element_type() == PRED) {
        extra.push_back(
            StrCat("true_computation=",
                   PrintNameInternal(true_computation()->name(), options)));
        extra.push_back(
            StrCat("false_computation=",
                   PrintNameInternal(false_computation()->name(), options)));
      } else {
        extra.push_back(StrCat(
            "branch_computations={",
            StrJoin(branch_computations(), ", ",
                    [&](std::string* out, const HloComputation* computation) {
                      StrAppend(
                          out, PrintNameInternal(computation->name(), options));
                    }),
            "}"));
      }
    } else if (opcode() == HloOpcode::kCall || opcode() == HloOpcode::kMap ||
               opcode() == HloOpcode::kReduceWindow ||
               opcode() == HloOpcode::kReduce ||
               opcode() == HloOpcode::kAllReduce ||
               opcode() == HloOpcode::kReduceScatter ||
               opcode() == HloOpcode::kAllReduceStart ||
               opcode() == HloOpcode::kScatter ||
               opcode() == HloOpcode::kSort) {
      extra.push_back(
          StrCat("to_apply=", PrintNameInternal(to_apply()->name(), options)));
    } else if (opcode() == HloOpcode::kCustomCall) {
      if (!called_computations().empty()) {
        extra.push_back(StrCat(
            "called_computations={",
            StrJoin(called_computations(), ", ",
                    [&](std::string* out, const HloComputation* computation) {
                      StrAppend(
                          out, PrintNameInternal(computation->name(), options));
                    }),
            "}"));
      }
    } else if (HloOpcodeIsAsync(opcode())) {
      if (!options.syntax_sugar_async_ops()) {
        extra.push_back(StrCat(
            "calls=",
            PrintNameInternal(async_wrapped_computation()->name(), options)));
      }
    } else if (!called_computations().empty()) {
      extra.push_back(StrCat(
          "calls=",
          StrJoin(called_computations(), ", ",
                  [&](std::string* out, const HloComputation* computation) {
                    StrAppend(out,
                              PrintNameInternal(computation->name(), options));
                  })));
    }
  } else if ((subcomputation_mode ==
              HloPrintOptions::PrintSubcomputationMode::kFullBodies) ||
             (subcomputation_mode == HloPrintOptions::PrintSubcomputationMode::
                                         kNonSequentialBodies &&
              !IsSequentialCall(opcode()))) {
    HloPrintOptions new_options = options;
    new_options.set_is_in_nested_computation(true);
    switch (opcode()) {
      case HloOpcode::kWhile:
        extra.push_back(
            StrCat("condition=\n", while_condition()->ToString(new_options)));
        extra.push_back(StrCat("body=\n", while_body()->ToString(new_options)));
        break;
      case HloOpcode::kSelectAndScatter:
        extra.push_back(StrCat("select=\n", select()->ToString(new_options)));
        extra.push_back(StrCat("scatter=\n", scatter()->ToString(new_options)));
        break;
      case HloOpcode::kConditional:
        if (operand(0)->shape().element_type() == PRED) {
          extra.push_back(StrCat("true_computation=\n",
                                 true_computation()->ToString(new_options)));
          extra.push_back(StrCat("false_computation=\n",
                                 false_computation()->ToString(new_options)));
        } else {
          extra.push_back(StrCat(
              "branch_computations={\n",
              StrJoin(branch_computations(), ",\n",
                      [&](std::string* out, const HloComputation* computation) {
                        StrAppend(out, computation->ToString(new_options));
                      }),
              "\n}"));
        }
        break;
      case HloOpcode::kCall:
      case HloOpcode::kMap:
      case HloOpcode::kReduceWindow:
      case HloOpcode::kReduce:
      case HloOpcode::kAllReduce:
      case HloOpcode::kAllReduceStart:
      case HloOpcode::kScatter:
      case HloOpcode::kSort:
        extra.push_back(
            StrCat("to_apply=\n", to_apply()->ToString(new_options)));
        break;
      default:
        if (!called_computations().empty()) {
          extra.push_back(StrCat(
              "calls=\n",
              StrJoin(called_computations(), ", ",
                      [&](std::string* out, const HloComputation* computation) {
                        StrAppend(out, computation->ToString(new_options));
                      })));
        }
        break;
    }
  }

  if (has_sharding()) {
    extra.push_back(
        StrCat("sharding=", sharding().ToString(options.print_metadata())));
  }
  if (!frontend_attributes_.map().empty()) {
    extra.push_back(StrCat("frontend_attributes=",
                           FrontendAttributesToString(frontend_attributes_)));
  }
  if (!outer_dimension_partitions_.empty()) {
    extra.push_back(absl::StrFormat("outer_dimension_partitions={%s}",
                                    StrJoin(outer_dimension_partitions_, ",")));
  }

  if (options.print_control_dependencies() && !control_predecessors_.empty()) {
    extra.push_back(StrCat("control-predecessors={",
                           StrJoin(control_predecessors_, ", ",
                                   [&](std::string* out, HloInstruction* pre) {
                                     StrAppend(out, PrintNameInternal(
                                                        pre->name(), options));
                                   }),
                           "}"));
  }

  return extra;
}

std::string HloInstruction::ToShortString() const {
   std::vector<std::string> mht_156_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_156(mht_156_v, 3922, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::ToShortString");

  return StrCat("%", name(), " = ", HloOpcodeString(opcode()), "(",
                StrJoin(operands_, ", ",
                        [](std::string* out, HloInstruction* operand) {
                          StrAppend(out, "%", operand->name());
                        }),
                ")");
}

HloInstructionProto HloInstruction::ToProto() const {
   std::vector<std::string> mht_157_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_157(mht_157_v, 3934, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::ToProto");

  HloInstructionProto proto;
  CHECK(unique_id_ != -1)
      << "This instruction does not have a valid id. Please make sure the "
         "instruction is inside a module before dumping it.";
  proto.set_id(unique_id_);
  proto.set_name(name_);
  proto.set_opcode(HloOpcodeString(opcode_));
  *proto.mutable_shape() = shape_.ToProto();
  for (const HloInstruction* operand : operands_) {
    proto.add_operand_ids(operand->unique_id());
  }
  for (const HloInstruction* control : control_predecessors_) {
    proto.add_control_predecessor_ids(control->unique_id());
  }

  *proto.mutable_metadata() = metadata_;
  proto.set_backend_config(backend_config_.GetRawString());
  if (opcode() != HloOpcode::kFusion) {
    for (const HloComputation* computation : called_computations_) {
      proto.add_called_computation_ids(computation->unique_id());
    }
  }

  if (has_sharding()) {
    *proto.mutable_sharding() = sharding().ToProto();
  }
  if (!outer_dimension_partitions_.empty()) {
    for (const auto& idx : outer_dimension_partitions_) {
      proto.mutable_outer_dimension_partitions()->Add(idx);
    }
  }

  *proto.mutable_frontend_attributes() = frontend_attributes_;

  return proto;
}

std::string HloInstruction::ToCategory() const {
   std::vector<std::string> mht_158_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_158(mht_158_v, 3975, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::ToCategory");

  if (opcode() == HloOpcode::kTranspose || opcode() == HloOpcode::kCopy ||
      opcode() == HloOpcode::kReshape ||
      opcode() == HloOpcode::kDynamicReshape) {
    return "data formatting";
  }

  if (IsElementwise()) {
    return "non-fusion elementwise";
  }

  return HloOpcodeString(opcode());
}

HloInstruction* HloInstruction::tracing() const {
   std::vector<std::string> mht_159_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_159(mht_159_v, 3992, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::tracing");
 return trace_instruction_; }

void HloInstruction::set_tracing(HloInstruction* trace_instruction) {
   std::vector<std::string> mht_160_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_160(mht_160_v, 3997, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::set_tracing");

  trace_instruction_ = trace_instruction;
}

bool HloInstruction::IsFused() const {
   std::vector<std::string> mht_161_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_161(mht_161_v, 4004, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::IsFused");

  return parent_ != nullptr && parent_->IsFusionComputation();
}

bool HloInstruction::IsCustomCall(absl::string_view target) const {
   std::vector<std::string> mht_162_v;
   mht_162_v.push_back("target: \"" + std::string(target.data(), target.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_162(mht_162_v, 4012, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::IsCustomCall");

  return opcode() == HloOpcode::kCustomCall && custom_call_target() == target;
}

bool HloInstruction::IsInputFusion() const {
   std::vector<std::string> mht_163_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_163(mht_163_v, 4019, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::IsInputFusion");

  return opcode() == HloOpcode::kFusion && fusion_kind() == FusionKind::kInput;
}

bool HloInstruction::IsLoopFusion() const {
   std::vector<std::string> mht_164_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_164(mht_164_v, 4026, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::IsLoopFusion");

  return opcode() == HloOpcode::kFusion && fusion_kind() == FusionKind::kLoop;
}

bool HloInstruction::IsOutputFusion() const {
   std::vector<std::string> mht_165_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_165(mht_165_v, 4033, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::IsOutputFusion");

  return opcode() == HloOpcode::kFusion && fusion_kind() == FusionKind::kOutput;
}

bool HloInstruction::IsCustomFusion() const {
   std::vector<std::string> mht_166_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_166(mht_166_v, 4040, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::IsCustomFusion");

  return opcode() == HloOpcode::kFusion && fusion_kind() == FusionKind::kCustom;
}

bool HloInstruction::IsFusible() const {
   std::vector<std::string> mht_167_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_167(mht_167_v, 4047, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::IsFusible");

  // Instructions which are traced should not be fused.
  if (tracing()) {
    return false;
  }
  // Some kinds of instructions don't make sense to fuse.
  switch (opcode_) {
    case HloOpcode::kDomain:
    case HloOpcode::kParameter:
    case HloOpcode::kWhile:
    case HloOpcode::kConditional:
    case HloOpcode::kCall:
      return false;
    // Fusions are always fusible.
    case HloOpcode::kFusion:
    // Side effecting reduce and reduce window would be invalid HLO.
    case HloOpcode::kMap:
    case HloOpcode::kReduce:
    case HloOpcode::kReduceWindow:
      return true;
    case HloOpcode::kRng:
      return user_count() <= 1;
    // Side effecting instructions cannot be fused.
    default:
      return !HasSideEffect();
  }
}

HloInstruction::HloInstruction(HloOpcode opcode, const Shape& shape)
    : unique_id_(-1),
      opcode_(opcode),
      shape_(shape),
      name_(HloOpcodeString(opcode)),
      marked_as_dead_(false) {
   std::vector<std::string> mht_168_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_168(mht_168_v, 4083, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::HloInstruction");

  TF_DCHECK_OK(ShapeUtil::ValidateShapeWithOptionalLayout(shape_));
}

template <typename HloInstructionPtr>
Status HloInstruction::Visit(DfsHloVisitorBase<HloInstructionPtr>* visitor) {
   std::vector<std::string> mht_169_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_169(mht_169_v, 4091, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::Visit");

  switch (opcode_) {
    case HloOpcode::kAbs:
      return visitor->HandleAbs(this);
    case HloOpcode::kAtan2:
      return visitor->HandleAtan2(this);
    case HloOpcode::kRoundNearestAfz:
      return visitor->HandleRound(this);
    case HloOpcode::kBatchNormTraining:
      return visitor->HandleBatchNormTraining(this);
    case HloOpcode::kBatchNormInference:
      return visitor->HandleBatchNormInference(this);
    case HloOpcode::kBatchNormGrad:
      return visitor->HandleBatchNormGrad(this);
    case HloOpcode::kLogistic:
      return visitor->HandleLogistic(this);
    case HloOpcode::kSign:
      return visitor->HandleSign(this);
    case HloOpcode::kConstant:
      return visitor->HandleConstant(this);
    case HloOpcode::kGetTupleElement:
      return visitor->HandleGetTupleElement(this);
    case HloOpcode::kParameter:
      return visitor->HandleParameter(this);
    case HloOpcode::kCompare:
      return visitor->HandleCompare(this);
    case HloOpcode::kComplex:
      return visitor->HandleComplex(this);
    case HloOpcode::kAdd:
      return visitor->HandleAdd(this);
    case HloOpcode::kDivide:
      return visitor->HandleDivide(this);
    case HloOpcode::kSubtract:
      return visitor->HandleSubtract(this);
    case HloOpcode::kMaximum:
      return visitor->HandleMaximum(this);
    case HloOpcode::kMinimum:
      return visitor->HandleMinimum(this);
    case HloOpcode::kAnd:
      return visitor->HandleAnd(this);
    case HloOpcode::kOr:
      return visitor->HandleOr(this);
    case HloOpcode::kXor:
      return visitor->HandleXor(this);
    case HloOpcode::kShiftLeft:
      return visitor->HandleShiftLeft(this);
    case HloOpcode::kShiftRightArithmetic:
      return visitor->HandleShiftRightArithmetic(this);
    case HloOpcode::kShiftRightLogical:
      return visitor->HandleShiftRightLogical(this);
    case HloOpcode::kConcatenate:
      return visitor->HandleConcatenate(this);
    case HloOpcode::kConvert:
      return visitor->HandleConvert(this);
    case HloOpcode::kBitcastConvert:
      return visitor->HandleBitcastConvert(this);
    case HloOpcode::kCopy:
      return visitor->HandleCopy(this);
    case HloOpcode::kMultiply:
      return visitor->HandleMultiply(this);
    case HloOpcode::kDot:
      return visitor->HandleDot(this);
    case HloOpcode::kPower:
      return visitor->HandlePower(this);
    case HloOpcode::kRemainder:
      return visitor->HandleRemainder(this);
    case HloOpcode::kSelect:
      return visitor->HandleSelect(this);
    case HloOpcode::kTupleSelect:
      return visitor->HandleTupleSelect(this);
    case HloOpcode::kConvolution:
      return visitor->HandleConvolution(this);
    case HloOpcode::kFft:
      return visitor->HandleFft(this);
    case HloOpcode::kAllGather:
      return visitor->HandleAllGather(this);
    case HloOpcode::kAllGatherStart:
      return visitor->HandleAllGatherStart(this);
    case HloOpcode::kAllGatherDone:
      return visitor->HandleAllGatherDone(this);
    case HloOpcode::kAllReduce:
      return visitor->HandleAllReduce(this);
    case HloOpcode::kReduceScatter:
      return visitor->HandleReduceScatter(this);
    case HloOpcode::kAllReduceStart:
      return visitor->HandleAllReduceStart(this);
    case HloOpcode::kAllReduceDone:
      return visitor->HandleAllReduceDone(this);
    case HloOpcode::kAllToAll:
      return visitor->HandleAllToAll(this);
    case HloOpcode::kCollectivePermute:
      return visitor->HandleCollectivePermute(this);
    case HloOpcode::kCollectivePermuteStart:
      return visitor->HandleCollectivePermuteStart(this);
    case HloOpcode::kCollectivePermuteDone:
      return visitor->HandleCollectivePermuteDone(this);
    case HloOpcode::kReplicaId:
      return visitor->HandleReplicaId(this);
    case HloOpcode::kPartitionId:
      return visitor->HandlePartitionId(this);
    case HloOpcode::kTuple:
      return visitor->HandleTuple(this);
    case HloOpcode::kMap:
      return visitor->HandleMap(this);
    case HloOpcode::kClamp:
      return visitor->HandleClamp(this);
    case HloOpcode::kReduce:
      return visitor->HandleReduce(this);
    case HloOpcode::kReduceWindow:
      return visitor->HandleReduceWindow(this);
    case HloOpcode::kSelectAndScatter:
      return visitor->HandleSelectAndScatter(this);
    case HloOpcode::kNegate:
      return visitor->HandleNegate(this);
    case HloOpcode::kExp:
      return visitor->HandleExp(this);
    case HloOpcode::kExpm1:
      return visitor->HandleExpm1(this);
    case HloOpcode::kFloor:
      return visitor->HandleFloor(this);
    case HloOpcode::kCeil:
      return visitor->HandleCeil(this);
    case HloOpcode::kClz:
      return visitor->HandleClz(this);
    case HloOpcode::kLog:
      return visitor->HandleLog(this);
    case HloOpcode::kLog1p:
      return visitor->HandleLog1p(this);
    case HloOpcode::kTanh:
      return visitor->HandleTanh(this);
    case HloOpcode::kCos:
      return visitor->HandleCos(this);
    case HloOpcode::kSin:
      return visitor->HandleSin(this);
    case HloOpcode::kSqrt:
      return visitor->HandleSqrt(this);
    case HloOpcode::kCbrt:
      return visitor->HandleCbrt(this);
    case HloOpcode::kRsqrt:
      return visitor->HandleRsqrt(this);
    case HloOpcode::kReal:
      return visitor->HandleReal(this);
    case HloOpcode::kImag:
      return visitor->HandleImag(this);
    case HloOpcode::kIsFinite:
      return visitor->HandleIsFinite(this);
    case HloOpcode::kNot:
      return visitor->HandleNot(this);
    case HloOpcode::kPopulationCount:
      return visitor->HandlePopulationCount(this);
    case HloOpcode::kBitcast:
      return visitor->HandleBitcast(this);
    case HloOpcode::kBroadcast:
      return visitor->HandleBroadcast(this);
    case HloOpcode::kPad:
      return visitor->HandlePad(this);
    case HloOpcode::kReshape:
      return visitor->HandleReshape(this);
    case HloOpcode::kDynamicReshape:
      return visitor->HandleDynamicReshape(this);
    case HloOpcode::kTranspose:
      return visitor->HandleTranspose(this);
    case HloOpcode::kReverse:
      return visitor->HandleReverse(this);
    case HloOpcode::kReducePrecision:
      return visitor->HandleReducePrecision(this);
    case HloOpcode::kSlice:
      return visitor->HandleSlice(this);
    case HloOpcode::kDynamicSlice:
      return visitor->HandleDynamicSlice(this);
    case HloOpcode::kDynamicUpdateSlice:
      return visitor->HandleDynamicUpdateSlice(this);
    case HloOpcode::kSort:
      return visitor->HandleSort(this);
    case HloOpcode::kInfeed:
      return visitor->HandleInfeed(this);
    case HloOpcode::kOutfeed:
      return visitor->HandleOutfeed(this);
    case HloOpcode::kRng:
      return visitor->HandleRng(this);
    case HloOpcode::kRngBitGenerator:
      return visitor->HandleRngBitGenerator(this);
    case HloOpcode::kRngGetAndUpdateState:
      return visitor->HandleRngGetAndUpdateState(this);
    case HloOpcode::kWhile:
      return visitor->HandleWhile(this);
    case HloOpcode::kFusion:
      return visitor->HandleFusion(this);
    case HloOpcode::kCall:
      return visitor->HandleCall(this);
    case HloOpcode::kConditional:
      return visitor->HandleConditional(this);
    case HloOpcode::kCustomCall:
      return visitor->HandleCustomCall(this);
    case HloOpcode::kAsyncStart:
      return visitor->HandleAsyncStart(this);
    case HloOpcode::kAsyncUpdate:
      return visitor->HandleAsyncUpdate(this);
    case HloOpcode::kAsyncDone:
      return visitor->HandleAsyncDone(this);
    case HloOpcode::kCopyStart:
      return visitor->HandleCopyStart(this);
    case HloOpcode::kCopyDone:
      return visitor->HandleCopyDone(this);
    case HloOpcode::kRecv:
      return visitor->HandleRecv(this);
    case HloOpcode::kRecvDone:
      return visitor->HandleRecvDone(this);
    case HloOpcode::kSend:
      return visitor->HandleSend(this);
    case HloOpcode::kSendDone:
      return visitor->HandleSendDone(this);
    case HloOpcode::kGather:
      return visitor->HandleGather(this);
    case HloOpcode::kScatter:
      return visitor->HandleScatter(this);
    case HloOpcode::kDomain:
      return visitor->HandleDomain(this);
    case HloOpcode::kAfterAll:
      return visitor->HandleAfterAll(this);
    case HloOpcode::kAddDependency:
      return visitor->HandleAddDependency(this);
    case HloOpcode::kIota:
      return visitor->HandleIota(this);
    case HloOpcode::kGetDimensionSize:
      return visitor->HandleGetDimensionSize(this);
    case HloOpcode::kSetDimensionSize:
      return visitor->HandleSetDimensionSize(this);
    case HloOpcode::kTriangularSolve:
      return visitor->HandleTriangularSolve(this);
    case HloOpcode::kCholesky:
      return visitor->HandleCholesky(this);
    case HloOpcode::kOptimizationBarrier:
      return visitor->HandleOptimizationBarrier(this);

    // These opcodes are not handled here.
    case HloOpcode::kTrace:
      return Status::OK();
  }
  return InternalError(
      "Unhandled HloOpcode for DfsHloVisitor: %s. This should not happen - "
      "please file a bug for XLA.",
      HloOpcodeString(opcode_));
}

// Explicit instantiations.
template Status HloInstruction::Visit(DfsHloVisitor* visitor);
template Status HloInstruction::Visit(ConstDfsHloVisitor* visitor);

// Push "child" onto the dfs_stack if not already visited.  Returns false if a
// cycle was detected, and true otherwise.
template <typename Visitor>
inline bool PushDFSChild(Visitor* visitor, DFSStack* dfs_stack,
                         HloInstruction* child) {
   std::vector<std::string> mht_170_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_170(mht_170_v, 4347, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "PushDFSChild");

  CHECK(child != nullptr);
  const int id = child->unique_id();
  CHECK_GE(id, 0) << "instruction may not have a parent computation";
  switch (visitor->GetVisitState(id)) {
    case Visitor::kVisiting:
      return false;

    case Visitor::kVisited:
      // Nothing to do
      return true;

    case Visitor::kNotVisited:
      dfs_stack->push_back(std::make_pair(id, child));
      return true;
  }
}

using InternalCompareFunction =
    std::function<bool(std::pair<int, const HloInstruction*>,
                       std::pair<int, const HloInstruction*>)>;
template <typename Visitor>
static Status PostOrderDFS(HloInstruction* root, Visitor* visitor,
                           const InternalCompareFunction* operand_order,
                           bool ignore_control_predecessors) {
   std::vector<std::string> mht_171_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_171(mht_171_v, 4374, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "PostOrderDFS");

  visitor->ReserveVisitStates(root->parent()->instruction_count());

  // dfs_stack holds pairs of <HloInstruction*->unique_id(), HloInstruction*>.
  //
  // We need to keep track of both the id and the instruction because
  // instructions can get deleted while they are on the stack, so we
  // can't always use the (potentially dead) instruction object to grab
  // its id.
  DFSStack dfs_stack;
  dfs_stack.emplace_back(root->unique_id(), root);

  do {
    DCHECK(!dfs_stack.empty());

    int current_id = dfs_stack.back().first;
    HloInstruction* current_node = dfs_stack.back().second;
    CHECK_GE(current_id, 0) << current_id << ": " << current_node
                            << ": instruction may not have parent computation";
    typename Visitor::VisitState visit_state =
        visitor->GetVisitState(current_id);
    if (visit_state == Visitor::kVisited) {
      dfs_stack.pop_back();
      VLOG(3) << "Not visiting HLO (id = " << current_id
              << ") as it was already visited.";
      continue;
    }

    if (visit_state == Visitor::kVisiting) {
      dfs_stack.pop_back();

      TF_RETURN_IF_ERROR(visitor->Preprocess(current_node));
      VLOG(2) << "Visiting HLO %" << current_node->name();
      TF_RETURN_IF_ERROR(current_node->Visit(visitor));
      visitor->SetVisitState(current_id, Visitor::kVisited);
      TF_RETURN_IF_ERROR(visitor->Postprocess(current_node));
      continue;
    }

    visitor->SetVisitState(current_id, Visitor::kVisiting);

    const size_t old_dfs_stack_size = dfs_stack.size();
    for (HloInstruction* child : current_node->operands()) {
      if (!ABSL_PREDICT_TRUE(PushDFSChild(visitor, &dfs_stack, child))) {
        PrintCycle(child, &dfs_stack);
        return FailedPrecondition(
            "A cycle is detected while visiting instruction %s",
            current_node->ToString());
      }
    }

    if (!ignore_control_predecessors) {
      for (HloInstruction* child : current_node->control_predecessors()) {
        if (!ABSL_PREDICT_TRUE(PushDFSChild(visitor, &dfs_stack, child))) {
          PrintCycle(child, &dfs_stack);
          return FailedPrecondition(
              "A cycle is detected while visiting instruction %s",
              current_node->ToString());
        }
      }
    }

    if (operand_order != nullptr) {
      std::sort(dfs_stack.begin() + old_dfs_stack_size, dfs_stack.end(),
                *operand_order);
    }

    // This makes the traversal order the same as what you'd expect
    // out of a recursive algorithm.
    std::reverse(dfs_stack.begin() + old_dfs_stack_size, dfs_stack.end());
  } while (!dfs_stack.empty());

  return Status::OK();
}

template <typename HloInstructionPtr>
Status HloInstruction::Accept(DfsHloVisitorBase<HloInstructionPtr>* visitor,
                              bool call_finish_visit,
                              bool ignore_control_predecessors) {
   std::vector<std::string> mht_172_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_172(mht_172_v, 4455, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::Accept");

  VLOG(3) << "HloInstruction::Accept(%" << name() << ")";
  TF_RETURN_IF_ERROR(
      PostOrderDFS(this, visitor, nullptr, ignore_control_predecessors));
  if (call_finish_visit) {
    TF_RETURN_IF_ERROR(visitor->FinishVisit(this));
  }
  return Status::OK();
}

// Explicit instantiations.
template Status HloInstruction::Accept(DfsHloVisitor*, bool, bool);
template Status HloInstruction::Accept(ConstDfsHloVisitor*, bool, bool);

Status HloInstruction::AcceptWithOperandOrder(
    DfsHloVisitor* visitor, const CompareFunction& operand_order,
    bool call_finish_visit) {
   std::vector<std::string> mht_173_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_173(mht_173_v, 4474, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::AcceptWithOperandOrder");

  VLOG(2) << "HloInstruction::AcceptWithOperandOrder(%" << name() << ")";
  InternalCompareFunction func = [&operand_order](
                                     std::pair<int, const HloInstruction*> a,
                                     std::pair<int, const HloInstruction*> b) {
   std::vector<std::string> mht_174_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_174(mht_174_v, 4481, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "lambda");

    // Call the client's comparison function on the actual HloInstruction*
    // objects (ignoring the internal ids we also have in our stack entries)
    return operand_order(a.second, b.second);
  };
  TF_RETURN_IF_ERROR(PostOrderDFS(this, visitor, &func,
                                  /*ignore_control_predecessors=*/false));
  if (call_finish_visit) {
    VLOG(3) << "HloInstruction::AcceptWithOperandOrder BEFORE FINISH VISIT";
    TF_RETURN_IF_ERROR(visitor->FinishVisit(this));
    VLOG(3) << "HloInstruction::AcceptWithOperandOrder AFTER FINISH VISIT";
  }
  VLOG(2) << "HloInstruction::AcceptWithOperandOrder EXIT";
  return Status::OK();
}

const Shape& HloInstruction::shape() const {
   std::vector<std::string> mht_175_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_175(mht_175_v, 4500, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::shape");
 return shape_; }

absl::InlinedVector<int64_t, 4> HloInstruction::OperandIndices(
    const HloInstruction* operand) const {
  absl::InlinedVector<int64_t, 4> result;
  for (int64_t i = 0; i < operand_count(); ++i) {
    if (this->operand(i) == operand) {
      result.push_back(i);
    }
  }
  return result;
}

bool HloInstruction::IsElementwiseBinary() const {
   std::vector<std::string> mht_176_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_176(mht_176_v, 4516, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::IsElementwiseBinary");

  return IsElementwise() && operand_count() == 2;
}

bool HloInstruction::IsElementwise() const {
   std::vector<std::string> mht_177_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_177(mht_177_v, 4523, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::IsElementwise");

  return IsElementwiseImpl(absl::nullopt);
}

bool HloInstruction::IsElementwiseOnOperand(int64_t operand_idx) const {
   std::vector<std::string> mht_178_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_178(mht_178_v, 4530, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::IsElementwiseOnOperand");

  return IsElementwiseImpl(operand_idx);
}

namespace {

// Indicates how an instruction uses a value (such as an operand).
//
// Does it (a) not use it, (b) use it, or (c) use it multiple times?
enum class UseKind { kReuse = 0, kUse = 1, kNoUse = 2 };

// A helper class for memoized, recursive computation of HloOpcode::kFusion
// in HloInstruction::OperandElementUse below.
class FusionReusesParamElements {
 public:
  static UseKind Compute(int64_t i, const HloInstruction& hlo) {
   std::vector<std::string> mht_179_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_179(mht_179_v, 4548, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "Compute");

    absl::flat_hash_map<const HloInstruction*, UseKind> memoization_cache;
    return ComputeInternal(i, hlo, &memoization_cache);
  }

 private:
  static UseKind ComputeInternal(
      int64_t outer_param_num, const HloInstruction& hlo,
      absl::flat_hash_map<const HloInstruction*, UseKind>* cache);
};

}  // namespace

// Returns how this instruction uses elements of its operand at operand_num.
static UseKind OperandElementUse(const HloInstruction& instr,
                                 int64_t operand_num) {
   std::vector<std::string> mht_180_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_180(mht_180_v, 4566, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "OperandElementUse");

  switch (instr.opcode()) {
    case HloOpcode::kBitcast:
    case HloOpcode::kConcatenate:
    case HloOpcode::kReshape:
    case HloOpcode::kReverse:
    case HloOpcode::kSlice:
    case HloOpcode::kTranspose:
    case HloOpcode::kGather:
      return UseKind::kUse;
    case HloOpcode::kPad:
      // Pad reuses the padding value but not the padded array elements.
      return operand_num > 0 ? UseKind::kReuse : UseKind::kUse;
    case HloOpcode::kReduce:
      // Reduce reuses the init values but not the operand array elements.
      return operand_num >= Cast<HloReduceInstruction>(&instr)->input_count()
                 ? UseKind::kReuse
                 : UseKind::kUse;
    case HloOpcode::kFusion:
      // Uses the memoizing, recursive computation defined above.
      return FusionReusesParamElements::Compute(operand_num,
                                                *instr.fused_expression_root());
    case HloOpcode::kDot:
      // Matrix-vector dots do not reuse the matrix operand.
      if (instr.shape().dimensions_size() <= 1) {
        if ((operand_num == 0 && instr.operand(1)->shape().rank() <= 1) ||
            (operand_num == 1 && instr.operand(0)->shape().rank() <= 1)) {
          return UseKind::kUse;
        }
      }
      return UseKind::kReuse;
    case HloOpcode::kDynamicUpdateSlice:
      // Dynamic-update-slice reuses only start_indices.
      if (operand_num == 0 || operand_num == 1) {
        return UseKind::kUse;
      }
      return UseKind::kReuse;
    default:
      return instr.IsElementwise() ? UseKind::kUse : UseKind::kReuse;
  }
}

UseKind FusionReusesParamElements::ComputeInternal(
    int64_t outer_param_num, const HloInstruction& hlo,
    absl::flat_hash_map<const HloInstruction*, UseKind>* cache) {
   std::vector<std::string> mht_181_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_181(mht_181_v, 4613, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "FusionReusesParamElements::ComputeInternal");

  if (auto hlo_param = DynCast<HloParameterInstruction>(&hlo)) {
    if (hlo_param->parameter_number() == outer_param_num) {
      return UseKind::kUse;
    }
  }

  auto p = cache->emplace(&hlo, UseKind::kNoUse);
  auto value_it = p.first;
  const bool key_is_new = p.second;

  if (!key_is_new) {
    return value_it->second;
  }

  // Our dataflow graph has no loops, so we don't need the fixed point
  // computation.
  for (int64_t operand_num = 0; operand_num < hlo.operands().size();
       ++operand_num) {
    UseKind old_val = value_it->second;

    // Compute updated value.
    UseKind new_val = [&] {
      // How does the HLO use this operand.
      UseKind hlo_use = OperandElementUse(hlo, operand_num);

      // If the HLO does not use the outer operand, return previous value.
      if (hlo_use == UseKind::kNoUse) {
        return old_val;
      }

      UseKind operand_use =
          ComputeInternal(outer_param_num, *hlo.operand(operand_num), cache);

      // If the operand does not use the outer operand, return the previous
      // value.
      if (operand_use == UseKind::kNoUse) {
        return old_val;
      }

      // Meet operator on a lattice:
      //
      //   kReuse < kUse < kNoUse.
      return std::min({old_val, hlo_use, operand_use});
    }();

    value_it = cache->find(&hlo);
    value_it->second = new_val;
    // Fold() minimizes the UseKind value. If it is already minimum, we do not
    // have to check all the remaining operands.
    if (new_val == UseKind::kReuse) {
      break;
    }
  }
  return value_it->second;
}

bool HloInstruction::ReusesOperandElements(int64_t i) const {
   std::vector<std::string> mht_182_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_182(mht_182_v, 4673, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::ReusesOperandElements");

  return OperandElementUse(*this, i) == UseKind::kReuse;
}

std::tuple<bool, std::vector<int64_t>, std::vector<int64_t>>
HloInstruction::ReshapeMerelyInsertsOrDeletes1SizedDimensions() const {
   std::vector<std::string> mht_183_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_183(mht_183_v, 4681, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::ReshapeMerelyInsertsOrDeletes1SizedDimensions");

  if (HloOpcode::kReshape != opcode_) {
    return std::make_tuple(false, std::vector<int64_t>(),
                           std::vector<int64_t>());
  }
  return ShapeUtil::InsertedOrDeleted1SizedDimensions(operand(0)->shape_,
                                                      shape_);
}

std::string ToString(HloInstruction::FusionKind kind) {
   std::vector<std::string> mht_184_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_184(mht_184_v, 4693, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "ToString");

  switch (kind) {
    case HloInstruction::FusionKind::kLoop:
      return "kLoop";
    case HloInstruction::FusionKind::kInput:
      return "kInput";
    case HloInstruction::FusionKind::kOutput:
      return "kOutput";
    case HloInstruction::FusionKind::kCustom:
      return "kCustom";
  }
}

StatusOr<HloInstruction::FusionKind> StringToFusionKind(
    const std::string& kind_name) {
  if (kind_name == "kLoop") {
    return HloInstruction::FusionKind::kLoop;
  }
  if (kind_name == "kInput") {
    return HloInstruction::FusionKind::kInput;
  }
  if (kind_name == "kOutput") {
    return HloInstruction::FusionKind::kOutput;
  }
  if (kind_name == "kCustom") {
    return HloInstruction::FusionKind::kCustom;
  }
  return InvalidArgument("Unknown fusion kind: %s", kind_name);
}

std::string FrontendAttributesToString(
    const FrontendAttributes& frontend_attributes) {
   std::vector<std::string> mht_185_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_185(mht_185_v, 4727, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "FrontendAttributesToString");

  std::vector<std::pair<std::string, std::string>> sorted_attributes(
      frontend_attributes.map().begin(), frontend_attributes.map().end());
  absl::c_sort(sorted_attributes);
  // Frontend attribute is a comma-separated list of attribute="value" pairs,
  // e.g., frontend_attributes={name="value_a",type="int32_t"}.
  const auto formatter = [](std::string* out,
                            const std::pair<std::string, std::string>& item) {
   std::vector<std::string> mht_186_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_186(mht_186_v, 4737, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "lambda");

    absl::StrAppend(out, item.first, "=\"", item.second, "\"");
  };
  return absl::StrFormat("{%s}",
                         absl::StrJoin(sorted_attributes, ",", formatter));
}

std::string PaddingConfigToString(const PaddingConfig& padding) {
   std::vector<std::string> mht_187_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_187(mht_187_v, 4747, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "PaddingConfigToString");

  bool has_interior_padding =
      absl::c_any_of(padding.dimensions(),
                     [](const PaddingConfig::PaddingConfigDimension& dim) {
                       return dim.interior_padding() != 0;
                     });
  return StrJoin(
      padding.dimensions(), "x",
      [&](std::string* out, const PaddingConfig::PaddingConfigDimension& dim) {
        StrAppend(
            out, dim.edge_padding_low(), "_", dim.edge_padding_high(),
            has_interior_padding ? StrCat("_", dim.interior_padding()) : "");
      });
}

std::string RandomDistributionToString(const RandomDistribution& distribution) {
   std::vector<std::string> mht_188_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_188(mht_188_v, 4765, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "RandomDistributionToString");

  return absl::AsciiStrToLower(RandomDistribution_Name(distribution));
}
std::string RandomAlgorithmToString(const RandomAlgorithm& algorithm) {
   std::vector<std::string> mht_189_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_189(mht_189_v, 4771, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "RandomAlgorithmToString");

  return absl::AsciiStrToLower(RandomAlgorithm_Name(algorithm));
}

std::string PrecisionToString(const PrecisionConfig::Precision& precision) {
   std::vector<std::string> mht_190_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_190(mht_190_v, 4778, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "PrecisionToString");

  return absl::AsciiStrToLower(PrecisionConfig::Precision_Name(precision));
}

static std::string CustomCallScheduleToString(
    const CustomCallSchedule& schedule) {
   std::vector<std::string> mht_191_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_191(mht_191_v, 4786, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "CustomCallScheduleToString");

  return absl::AsciiStrToLower(CustomCallSchedule_Name(schedule));
}

static std::string CustomCallApiVersionToString(
    const CustomCallApiVersion& schedule) {
   std::vector<std::string> mht_192_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_192(mht_192_v, 4794, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "CustomCallApiVersionToString");

  return absl::AsciiStrToLower(CustomCallApiVersion_Name(schedule));
}

std::string DotDimensionNumbersToString(const DotDimensionNumbers& dnums) {
   std::vector<std::string> mht_193_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_193(mht_193_v, 4801, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "DotDimensionNumbersToString");

  std::vector<std::string> result;
  if (!dnums.lhs_batch_dimensions().empty()) {
    result.push_back(StrCat("lhs_batch_dims={",
                            StrJoin(dnums.lhs_batch_dimensions(), ","), "}"));
  }
  result.push_back(StrCat("lhs_contracting_dims={",
                          StrJoin(dnums.lhs_contracting_dimensions(), ","),
                          "}"));

  if (!dnums.rhs_batch_dimensions().empty()) {
    result.push_back(StrCat("rhs_batch_dims={",
                            StrJoin(dnums.rhs_batch_dimensions(), ","), "}"));
  }
  result.push_back(StrCat("rhs_contracting_dims={",
                          StrJoin(dnums.rhs_contracting_dimensions(), ","),
                          "}"));

  return StrJoin(result, ", ");
}

std::string ConvolutionDimensionNumbersToString(
    const ConvolutionDimensionNumbers& dnums) {
   std::vector<std::string> mht_194_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_194(mht_194_v, 4826, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "ConvolutionDimensionNumbersToString");

  auto len_required = [](int64_t a, int64_t b, absl::Span<const int64_t> cs) {
   std::vector<std::string> mht_195_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_195(mht_195_v, 4830, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "lambda");

    return std::max({a, b, cs.empty() ? 0 : *absl::c_max_element(cs)}) + 1;
  };

  // lhs_dims[i] is the symbol of the logical dimension i for the lhs
  // operand. E.g. if batch has dimension number 2, then lhs_dims[2] == "b".
  std::vector<std::string> lhs_dims(
      len_required(dnums.input_batch_dimension(),
                   dnums.input_feature_dimension(),
                   dnums.input_spatial_dimensions()),
      "?");
  lhs_dims[dnums.input_batch_dimension()] = 'b';
  lhs_dims[dnums.input_feature_dimension()] = 'f';
  for (int64_t i = 0; i < dnums.input_spatial_dimensions().size(); ++i) {
    lhs_dims[dnums.input_spatial_dimensions(i)] = StrCat(i);
  }

  std::vector<std::string> rhs_dims(
      len_required(dnums.kernel_input_feature_dimension(),
                   dnums.kernel_output_feature_dimension(),
                   dnums.kernel_spatial_dimensions()),
      "?");
  rhs_dims[dnums.kernel_input_feature_dimension()] = "i";
  rhs_dims[dnums.kernel_output_feature_dimension()] = "o";
  for (int64_t i = 0; i < dnums.kernel_spatial_dimensions().size(); ++i) {
    rhs_dims[dnums.kernel_spatial_dimensions(i)] = StrCat(i);
  }

  std::vector<std::string> output_dims(
      len_required(dnums.output_batch_dimension(),
                   dnums.output_feature_dimension(),
                   dnums.output_spatial_dimensions()),
      "?");
  output_dims[dnums.output_batch_dimension()] = 'b';
  output_dims[dnums.output_feature_dimension()] = 'f';
  for (int64_t i = 0; i < dnums.output_spatial_dimensions().size(); ++i) {
    output_dims[dnums.output_spatial_dimensions(i)] = StrCat(i);
  }

  return StrCat(StrJoin(lhs_dims, ""), "_", StrJoin(rhs_dims, ""), "->",
                StrJoin(output_dims, ""));
}

std::string ReplicaGroupsToString(
    absl::Span<const ReplicaGroup> replica_groups) {
   std::vector<std::string> mht_196_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_196(mht_196_v, 4877, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "ReplicaGroupsToString");

  std::vector<std::string> replica_group_str;
  replica_group_str.reserve(replica_groups.size());
  for (const ReplicaGroup& group : replica_groups) {
    replica_group_str.push_back(
        StrCat("{", StrJoin(group.replica_ids(), ","), "}"));
  }
  return StrCat("{", StrJoin(replica_group_str, ","), "}");
}

StatusOr<RandomAlgorithm> StringToRandomAlgorithm(const std::string& name) {
  static absl::flat_hash_map<std::string, RandomAlgorithm>* map = [] {
    static auto* map = new absl::flat_hash_map<std::string, RandomAlgorithm>;
    for (int i = 0; i < RandomAlgorithm_ARRAYSIZE; i++) {
      if (RandomAlgorithm_IsValid(i)) {
        auto value = static_cast<RandomAlgorithm>(i);
        (*map)[RandomAlgorithmToString(value)] = value;
      }
    }
    return map;
  }();
  auto found = map->find(absl::AsciiStrToLower(name));
  if (found == map->end()) {
    return InvalidArgument("Unknown algorithm");
  }
  return found->second;
}

StatusOr<RandomDistribution> StringToRandomDistribution(
    const std::string& name) {
  static absl::flat_hash_map<std::string, RandomDistribution>* map = [] {
    static auto* map = new absl::flat_hash_map<std::string, RandomDistribution>;
    for (int i = 0; i < RandomDistribution_ARRAYSIZE; i++) {
      if (RandomDistribution_IsValid(i)) {
        auto value = static_cast<RandomDistribution>(i);
        (*map)[RandomDistributionToString(value)] = value;
      }
    }
    return map;
  }();
  auto found = map->find(absl::AsciiStrToLower(name));
  if (found == map->end()) {
    return InvalidArgument("Unknown distribution");
  }
  return found->second;
}

StatusOr<PrecisionConfig::Precision> StringToPrecision(
    const std::string& name) {
  static absl::flat_hash_map<std::string, PrecisionConfig::Precision>* map =
      [] {
        static auto* map =
            new absl::flat_hash_map<std::string, PrecisionConfig::Precision>;
        for (int i = 0; i < PrecisionConfig::Precision_ARRAYSIZE; i++) {
          if (PrecisionConfig::Precision_IsValid(i)) {
            auto value = static_cast<PrecisionConfig::Precision>(i);
            (*map)[PrecisionToString(value)] = value;
          }
        }
        return map;
      }();
  auto found = map->find(absl::AsciiStrToLower(name));
  if (found == map->end()) {
    return InvalidArgument("Unknown distribution");
  }
  return found->second;
}

StatusOr<CustomCallSchedule> StringToCustomCallSchedule(
    absl::string_view name) {
  static const absl::flat_hash_map<std::string, CustomCallSchedule>* map = [] {
    static auto* map = new absl::flat_hash_map<std::string, CustomCallSchedule>;
    for (int i = 0; i < CustomCallSchedule_ARRAYSIZE; i++) {
      if (CustomCallSchedule_IsValid(i)) {
        auto value = static_cast<CustomCallSchedule>(i);
        (*map)[CustomCallScheduleToString(value)] = value;
      }
    }
    return map;
  }();
  auto found = map->find(absl::AsciiStrToLower(name));
  if (found == map->end()) {
    return InvalidArgument("Unknown schedule");
  }
  return found->second;
}

StatusOr<CustomCallApiVersion> StringToCustomCallApiVersion(
    absl::string_view name) {
  static const absl::flat_hash_map<std::string, CustomCallApiVersion>* map =
      [] {
        static auto* map =
            new absl::flat_hash_map<std::string, CustomCallApiVersion>;
        for (int i = 0; i < CustomCallApiVersion_ARRAYSIZE; i++) {
          if (CustomCallApiVersion_IsValid(i)) {
            auto value = static_cast<CustomCallApiVersion>(i);
            (*map)[CustomCallApiVersionToString(value)] = value;
          }
        }
        return map;
      }();
  auto found = map->find(absl::AsciiStrToLower(name));
  if (found == map->end()) {
    return InvalidArgument("Unknown API version");
  }
  return found->second;
}

std::ostream& operator<<(std::ostream& os, HloInstruction::FusionKind kind) {
   std::vector<std::string> mht_197_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_197(mht_197_v, 4988, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "operator<<");

  return os << ToString(kind);
}

bool HloPtrComparator::operator()(const HloInstruction* const& lhs,
                                  const HloInstruction* const& rhs) const {
  if (rhs == nullptr) {
    // Nothing compares less than nullptr.
    return false;
  }
  if (lhs == nullptr) {
    return true;
  }
  auto lhs_module = lhs->GetModule();
  auto rhs_module = rhs->GetModule();
  CHECK((lhs_module == nullptr && rhs_module == nullptr) ||
        (lhs_module != nullptr && rhs_module != nullptr));
  if (lhs_module != nullptr &&
      lhs_module->unique_id() != rhs_module->unique_id()) {
    return lhs_module->unique_id() < rhs_module->unique_id();
  }
  return lhs->unique_id() < rhs->unique_id();
}

Status HloInstruction::GetBackendConfigInternal(
    tensorflow::protobuf::Message* proto) const {
   std::vector<std::string> mht_198_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_198(mht_198_v, 5016, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::GetBackendConfigInternal");

  proto->Clear();

  if (auto* proto_ptr = backend_config_.GetProtoPtr()) {
    proto->CopyFrom(*proto_ptr);
    return Status::OK();
  }

  auto& raw_string = raw_backend_config_string();
  // Empty string does not parse as valid JSON, but it's a valid backend config,
  // corresponding to the empty proto.
  if (raw_string.empty()) {
    return Status::OK();
  }
  TF_RETURN_IF_ERROR(tensorflow::HumanReadableJsonToProto(raw_string, proto));
  backend_config_.SetProto(*proto);
  return Status::OK();
}

const std::string& HloInstruction::BackendConfigRep::GetRawString() const {
   std::vector<std::string> mht_199_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_199(mht_199_v, 5038, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::BackendConfigRep::GetRawString");

  if (proto_ && raw_string_.empty()) {
    raw_string_ = BackendConfigToRawString(*proto_).ValueOrDie();
  }
  return raw_string_;
}

HloInstruction::BackendConfigRep HloInstruction::BackendConfigRep::Clone()
    const {
   std::vector<std::string> mht_200_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_200(mht_200_v, 5049, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::BackendConfigRep::Clone");

  // Prefer cloning protobuf, raw_string_ will be lazily generated if accessed.
  BackendConfigRep cloned;
  if (auto* proto = GetProtoPtr()) {
    cloned.SetProto(*proto);
  } else {
    cloned.raw_string_ = raw_string_;
  }
  return cloned;
}

HloInstruction::BackendConfigRep& HloInstruction::BackendConfigRep::operator=(
    std::string raw_string) {
   std::vector<std::string> mht_201_v;
   mht_201_v.push_back("raw_string: \"" + raw_string + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_201(mht_201_v, 5065, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "=");

  raw_string_ = std::move(raw_string);
  proto_.reset();
  return *this;
}

HloInstruction::BackendConfigRep& HloInstruction::BackendConfigRep::operator=(
    const tensorflow::protobuf::Message& proto) {
   std::vector<std::string> mht_202_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_202(mht_202_v, 5075, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "=");

  SetProto(proto);
  raw_string_.clear();
  return *this;
}

void HloInstruction::BackendConfigRep::SetProto(
    const tensorflow::protobuf::Message& proto) {
   std::vector<std::string> mht_203_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_203(mht_203_v, 5085, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::BackendConfigRep::SetProto");

  proto_.reset(proto.New());
  proto_->CopyFrom(proto);
}

bool HloInstruction::BackendConfigRep::operator==(
    const BackendConfigRep& other) const {
  auto* proto_a = GetProtoPtr();
  auto* proto_b = other.GetProtoPtr();
  if (proto_a != nullptr && proto_b != nullptr) {
    using ::tensorflow::protobuf::util::MessageDifferencer;
    return MessageDifferencer::Equals(*proto_a, *proto_b);
  }
  // TODO(b/225956414): Consider canonicalizing raw string form.
  return GetRawString() == other.GetRawString();
}

/* static */ StatusOr<std::string> HloInstruction::BackendConfigToRawString(
    const tensorflow::protobuf::Message& proto) {
   std::vector<std::string> mht_204_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_204(mht_204_v, 5106, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::BackendConfigToRawString");

  std::string ret;
  // Pass ignore_accuracy_loss = true because estimated_cycles field can be
  // INT64_MAX. If ignore_accuracy_loss = false and estimated_cycles =
  // INT64_MAX, JsonFormat will return an error status, although there is no
  // accuracy loss for int64_t.
  TF_RETURN_IF_ERROR(tensorflow::ProtoToHumanReadableJson(
      proto, &ret, /*ignore_accuracy_loss=*/true));
  return ret;
}

const PrecisionConfig& HloInstruction::precision_config() const {
   std::vector<std::string> mht_205_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_205(mht_205_v, 5120, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::precision_config");

  if (auto* convolution = DynCast<HloConvolutionInstruction>(this)) {
    return convolution->precision_config();
  }
  if (auto* dot = DynCast<HloDotInstruction>(this)) {
    return dot->precision_config();
  }

  if (auto* custom_call = DynCast<HloCustomCallInstruction>(this)) {
    return custom_call->precision_config();
  }
  LOG(FATAL) << "Unimplemented method.";
}

PrecisionConfig* HloInstruction::mutable_precision_config() {
   std::vector<std::string> mht_206_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_206(mht_206_v, 5137, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::mutable_precision_config");

  if (auto* convolution = DynCast<HloConvolutionInstruction>(this)) {
    return convolution->mutable_precision_config();
  }
  if (auto* dot = DynCast<HloDotInstruction>(this)) {
    return dot->mutable_precision_config();
  }
  LOG(FATAL) << "Unimplemented method.";
}

HloModule* HloInstruction::GetModule() const {
   std::vector<std::string> mht_207_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_207(mht_207_v, 5150, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::GetModule");

  if (parent_) {
    return parent_->parent();
  }
  return nullptr;
}

void HloInstruction::UniquifyName(NameUniquer* name_uniquer) {
   std::vector<std::string> mht_208_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_208(mht_208_v, 5160, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::UniquifyName");

  std::string parent_str = parent() == nullptr ? "noparent" : parent()->name();
  name_ = name_uniquer->GetUniqueName(name_);
}

void HloInstruction::set_outer_dimension_partitions(
    const std::vector<int64_t>& outer_dimension_partitions) {
   std::vector<std::string> mht_209_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_209(mht_209_v, 5169, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::set_outer_dimension_partitions");

  outer_dimension_partitions_ = outer_dimension_partitions;
}

// TODO(b/80131774): Remove these temporary methods after transition.
int64_t HloInstruction::feature_index() const {
   std::vector<std::string> mht_210_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_210(mht_210_v, 5177, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::feature_index");

  return Cast<HloBatchNormInstruction>(this)->feature_index();
}

float HloInstruction::epsilon() const {
   std::vector<std::string> mht_211_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_211(mht_211_v, 5184, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::epsilon");

  return Cast<HloBatchNormInstruction>(this)->epsilon();
}

FftType HloInstruction::fft_type() const {
   std::vector<std::string> mht_212_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_212(mht_212_v, 5191, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::fft_type");

  return Cast<HloFftInstruction>(this)->fft_type();
}

const std::vector<int64_t>& HloInstruction::fft_length() const {
   std::vector<std::string> mht_213_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_213(mht_213_v, 5198, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::fft_length");

  return Cast<HloFftInstruction>(this)->fft_length();
}

int64_t HloInstruction::concatenate_dimension() const {
   std::vector<std::string> mht_214_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_214(mht_214_v, 5205, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::concatenate_dimension");

  return Cast<HloConcatenateInstruction>(this)->concatenate_dimension();
}

int64_t HloInstruction::dimension() const {
   std::vector<std::string> mht_215_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_215(mht_215_v, 5212, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::dimension");

  if (auto set_size = DynCast<HloSetDimensionSizeInstruction>(this)) {
    return set_size->dimension();
  }
  return Cast<HloGetDimensionSizeInstruction>(this)->dimension();
}

int64_t HloInstruction::inferred_dimension() const {
   std::vector<std::string> mht_216_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_216(mht_216_v, 5222, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::inferred_dimension");

  return Cast<HloReshapeInstruction>(this)->inferred_dimension();
}

bool HloInstruction::IsRank2Transpose() const {
   std::vector<std::string> mht_217_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_217(mht_217_v, 5229, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::IsRank2Transpose");

  auto transpose = DynCast<HloTransposeInstruction>(this);
  return transpose != nullptr && transpose->IsRank2Transpose();
}

int64_t HloInstruction::slice_starts(int64_t dimension) const {
   std::vector<std::string> mht_218_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_218(mht_218_v, 5237, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::slice_starts");

  return Cast<HloSliceInstruction>(this)->slice_starts(dimension);
}

const std::vector<int64_t>& HloInstruction::slice_starts() const {
   std::vector<std::string> mht_219_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_219(mht_219_v, 5244, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::slice_starts");

  return Cast<HloSliceInstruction>(this)->slice_starts();
}

std::vector<int64_t>* HloInstruction::mutable_slice_starts() {
   std::vector<std::string> mht_220_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_220(mht_220_v, 5251, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::mutable_slice_starts");

  return Cast<HloSliceInstruction>(this)->mutable_slice_starts();
}

int64_t HloInstruction::slice_limits(int64_t dimension) const {
   std::vector<std::string> mht_221_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_221(mht_221_v, 5258, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::slice_limits");

  return Cast<HloSliceInstruction>(this)->slice_limits(dimension);
}

const std::vector<int64_t>& HloInstruction::slice_limits() const {
   std::vector<std::string> mht_222_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_222(mht_222_v, 5265, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::slice_limits");

  return Cast<HloSliceInstruction>(this)->slice_limits();
}

std::vector<int64_t>* HloInstruction::mutable_slice_limits() {
   std::vector<std::string> mht_223_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_223(mht_223_v, 5272, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::mutable_slice_limits");

  return Cast<HloSliceInstruction>(this)->mutable_slice_limits();
}

int64_t HloInstruction::slice_strides(int64_t dimension) const {
   std::vector<std::string> mht_224_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_224(mht_224_v, 5279, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::slice_strides");

  return Cast<HloSliceInstruction>(this)->slice_strides(dimension);
}

const std::vector<int64_t>& HloInstruction::slice_strides() const {
   std::vector<std::string> mht_225_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_225(mht_225_v, 5286, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::slice_strides");

  return Cast<HloSliceInstruction>(this)->slice_strides();
}

std::vector<int64_t>* HloInstruction::mutable_slice_strides() {
   std::vector<std::string> mht_226_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_226(mht_226_v, 5293, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::mutable_slice_strides");

  return Cast<HloSliceInstruction>(this)->mutable_slice_strides();
}

const Literal& HloInstruction::literal() const {
   std::vector<std::string> mht_227_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_227(mht_227_v, 5300, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::literal");

  return Cast<HloConstantInstruction>(this)->literal();
}

bool HloInstruction::IsConstant() const {
   std::vector<std::string> mht_228_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_228(mht_228_v, 5307, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::IsConstant");

  return DynCast<HloConstantInstruction>(this) != nullptr;
}

void HloInstruction::RelayoutConstant(const Layout& new_layout,
                                      const ShapeIndex& shape_index) {
   std::vector<std::string> mht_229_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_229(mht_229_v, 5315, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::RelayoutConstant");

  Cast<HloConstantInstruction>(this)->RelayoutConstant(new_layout, shape_index);
}

std::string HloInstruction::TracingTag() const {
   std::vector<std::string> mht_230_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_230(mht_230_v, 5322, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::TracingTag");

  return Cast<HloTraceInstruction>(this)->TracingTag();
}

HloInstruction* HloInstruction::AddFusionOperand(HloInstruction* new_operand) {
   std::vector<std::string> mht_231_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_231(mht_231_v, 5329, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::AddFusionOperand");

  return Cast<HloFusionInstruction>(this)->AddFusionOperand(new_operand);
}

// Delegates to HloFusionInstruction::MergeFusionInstruction.
void HloInstruction::MergeFusionInstruction(
    HloInstruction* instruction_to_merge) {
   std::vector<std::string> mht_232_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_232(mht_232_v, 5338, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::MergeFusionInstruction");

  return Cast<HloFusionInstruction>(this)->MergeFusionInstruction(
      Cast<HloFusionInstruction>(instruction_to_merge));
}

// Delegates to HloFusionInstruction::MergeFusionInstructionIntoMultiOutput.
void HloInstruction::MergeFusionInstructionIntoMultiOutput(
    HloInstruction* instruction_to_merge) {
   std::vector<std::string> mht_233_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_233(mht_233_v, 5348, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::MergeFusionInstructionIntoMultiOutput");

  return Cast<HloFusionInstruction>(this)
      ->MergeFusionInstructionIntoMultiOutput(
          Cast<HloFusionInstruction>(instruction_to_merge));
}

HloInstruction* HloInstruction::FuseInstruction(
    HloInstruction* instruction_to_fuse) {
   std::vector<std::string> mht_234_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_234(mht_234_v, 5358, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::FuseInstruction");

  return Cast<HloFusionInstruction>(this)->FuseInstruction(instruction_to_fuse);
}

HloInstruction* HloInstruction::FuseInstructionIntoMultiOutput(
    HloInstruction* instruction_to_fuse) {
   std::vector<std::string> mht_235_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_235(mht_235_v, 5366, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::FuseInstructionIntoMultiOutput");

  return Cast<HloFusionInstruction>(this)->FuseInstructionIntoMultiOutput(
      instruction_to_fuse);
}

HloComputation* HloInstruction::fused_instructions_computation() const {
   std::vector<std::string> mht_236_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_236(mht_236_v, 5374, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::fused_instructions_computation");

  return Cast<HloFusionInstruction>(this)->fused_instructions_computation();
}

HloInstruction* HloInstruction::fused_expression_root() const {
   std::vector<std::string> mht_237_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_237(mht_237_v, 5381, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::fused_expression_root");

  return Cast<HloFusionInstruction>(this)->fused_expression_root();
}

const tensorflow::gtl::iterator_range<UnwrappingIterator<
    std::list<std::unique_ptr<HloInstruction>>::const_iterator>>
HloInstruction::fused_instructions() const {
   std::vector<std::string> mht_238_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_238(mht_238_v, 5390, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::fused_instructions");

  return Cast<HloFusionInstruction>(this)->fused_instructions();
}

const tensorflow::gtl::iterator_range<
    UnwrappingIterator<std::list<std::unique_ptr<HloInstruction>>::iterator>>
HloInstruction::fused_instructions() {
   std::vector<std::string> mht_239_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_239(mht_239_v, 5399, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::fused_instructions");

  return Cast<HloFusionInstruction>(this)->fused_instructions();
}

int64_t HloInstruction::fused_instruction_count() const {
   std::vector<std::string> mht_240_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_240(mht_240_v, 5406, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::fused_instruction_count");

  return Cast<HloFusionInstruction>(this)->fused_instruction_count();
}

HloInstruction* HloInstruction::fused_parameter(
    int64_t parameter_number) const {
   std::vector<std::string> mht_241_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_241(mht_241_v, 5414, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::fused_parameter");

  return Cast<HloFusionInstruction>(this)->fused_parameter(parameter_number);
}

const std::vector<HloInstruction*>& HloInstruction::fused_parameters() const {
   std::vector<std::string> mht_242_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_242(mht_242_v, 5421, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::fused_parameters");

  return Cast<HloFusionInstruction>(this)->fused_parameters();
}

const bool HloInstruction::IsMultiOutputFusion() const {
   std::vector<std::string> mht_243_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_243(mht_243_v, 5428, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::IsMultiOutputFusion");

  const HloFusionInstruction* fusion = DynCast<HloFusionInstruction>(this);
  return fusion != nullptr && fusion->IsMultiOutputFusion();
}

HloInstruction::FusionKind HloInstruction::fusion_kind() const {
   std::vector<std::string> mht_244_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_244(mht_244_v, 5436, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::fusion_kind");

  return Cast<HloFusionInstruction>(this)->fusion_kind();
}

void HloInstruction::set_fusion_kind(FusionKind kind) {
   std::vector<std::string> mht_245_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_245(mht_245_v, 5443, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::set_fusion_kind");

  return Cast<HloFusionInstruction>(this)->set_fusion_kind(kind);
}

RandomDistribution HloInstruction::random_distribution() const {
   std::vector<std::string> mht_246_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_246(mht_246_v, 5450, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::random_distribution");

  return Cast<HloRngInstruction>(this)->random_distribution();
}

int64_t HloInstruction::parameter_number() const {
   std::vector<std::string> mht_247_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_247(mht_247_v, 5457, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::parameter_number");

  return Cast<HloParameterInstruction>(this)->parameter_number();
}

void HloInstruction::set_parameter_replicated_at_leaf_buffers(
    absl::Span<const bool> parameter_replicated_at_leaf_buffers) {
   std::vector<std::string> mht_248_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_248(mht_248_v, 5465, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::set_parameter_replicated_at_leaf_buffers");

  return Cast<HloParameterInstruction>(this)
      ->set_parameter_replicated_at_leaf_buffers(
          parameter_replicated_at_leaf_buffers);
}

void HloInstruction::set_parameter_replicated_at_leaf_buffers(
    const std::vector<bool>& parameter_replicated_at_leaf_buffers) {
   std::vector<std::string> mht_249_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_249(mht_249_v, 5475, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::set_parameter_replicated_at_leaf_buffers");

  return Cast<HloParameterInstruction>(this)
      ->set_parameter_replicated_at_leaf_buffers(
          parameter_replicated_at_leaf_buffers);
}

const absl::optional<std::vector<bool>>&
HloInstruction::parameter_replicated_at_leaf_buffers() const {
   std::vector<std::string> mht_250_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_250(mht_250_v, 5485, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::parameter_replicated_at_leaf_buffers");

  return Cast<HloParameterInstruction>(this)
      ->parameter_replicated_at_leaf_buffers();
}

int64_t HloInstruction::tuple_index() const {
   std::vector<std::string> mht_251_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_251(mht_251_v, 5493, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::tuple_index");

  return Cast<HloGetTupleElementInstruction>(this)->tuple_index();
}

void HloInstruction::set_tuple_index(int64_t new_tuple_index) {
   std::vector<std::string> mht_252_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_252(mht_252_v, 5500, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::set_tuple_index");

  return Cast<HloGetTupleElementInstruction>(this)->set_tuple_index(
      new_tuple_index);
}

int32_t HloInstruction::exponent_bits() const {
   std::vector<std::string> mht_253_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_253(mht_253_v, 5508, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::exponent_bits");

  return Cast<HloReducePrecisionInstruction>(this)->exponent_bits();
}

int32_t HloInstruction::mantissa_bits() const {
   std::vector<std::string> mht_254_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_254(mht_254_v, 5515, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::mantissa_bits");

  return Cast<HloReducePrecisionInstruction>(this)->mantissa_bits();
}

std::string HloInstruction::infeed_config() const {
   std::vector<std::string> mht_255_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_255(mht_255_v, 5522, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::infeed_config");

  return Cast<HloInfeedInstruction>(this)->infeed_config();
}

void HloInstruction::set_infeed_config(const std::string& config) {
   std::vector<std::string> mht_256_v;
   mht_256_v.push_back("config: \"" + config + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_256(mht_256_v, 5530, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::set_infeed_config");

  return Cast<HloInfeedInstruction>(this)->set_infeed_config(config);
}

const Shape& HloInstruction::outfeed_shape() const {
   std::vector<std::string> mht_257_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_257(mht_257_v, 5537, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::outfeed_shape");

  return Cast<HloOutfeedInstruction>(this)->outfeed_shape();
}

Shape* HloInstruction::mutable_outfeed_shape() {
   std::vector<std::string> mht_258_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_258(mht_258_v, 5544, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::mutable_outfeed_shape");

  return Cast<HloOutfeedInstruction>(this)->mutable_outfeed_shape();
}

const std::string& HloInstruction::outfeed_config() const {
   std::vector<std::string> mht_259_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_259(mht_259_v, 5551, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::outfeed_config");

  return Cast<HloOutfeedInstruction>(this)->outfeed_config();
}

void HloInstruction::set_outfeed_config(const std::string& config) {
   std::vector<std::string> mht_260_v;
   mht_260_v.push_back("config: \"" + config + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_260(mht_260_v, 5559, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::set_outfeed_config");

  return Cast<HloOutfeedInstruction>(this)->set_outfeed_config(config);
}

const std::vector<ReplicaGroup>& HloInstruction::replica_groups() const {
   std::vector<std::string> mht_261_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_261(mht_261_v, 5566, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::replica_groups");

  return Cast<HloCollectiveInstruction>(this)->replica_groups();
}

const std::vector<std::pair<int64_t, int64_t>>&
HloInstruction::source_target_pairs() const {
  return Cast<HloCollectivePermuteInstruction>(this)->source_target_pairs();
}

absl::optional<int64_t> HloInstruction::channel_id() const {
   std::vector<std::string> mht_262_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_262(mht_262_v, 5578, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::channel_id");

  return Cast<HloChannelInstruction>(this)->channel_id();
}

void HloInstruction::set_channel_id(const absl::optional<int64_t>& channel_id) {
   std::vector<std::string> mht_263_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_263(mht_263_v, 5585, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::set_channel_id");

  return Cast<HloChannelInstruction>(this)->set_channel_id(channel_id);
}

const ConvolutionDimensionNumbers&
HloInstruction::convolution_dimension_numbers() const {
   std::vector<std::string> mht_264_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_264(mht_264_v, 5593, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::convolution_dimension_numbers");

  if (auto convolution = DynCast<HloConvolutionInstruction>(this)) {
    return convolution->convolution_dimension_numbers();
  }
  if (auto custom_call = DynCast<HloCustomCallInstruction>(this)) {
    return custom_call->convolution_dimension_numbers();
  }
  LOG(FATAL) << "Unimplemented method.";
}

void HloInstruction::set_convolution_dimension_numbers(
    const ConvolutionDimensionNumbers& dnums) {
   std::vector<std::string> mht_265_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_265(mht_265_v, 5607, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::set_convolution_dimension_numbers");

  if (auto convolution = DynCast<HloConvolutionInstruction>(this)) {
    convolution->set_convolution_dimension_numbers(dnums);
  } else if (auto custom_call = DynCast<HloCustomCallInstruction>(this)) {
    custom_call->set_convolution_dimension_numbers(dnums);
  } else {
    LOG(FATAL) << "Unimplemented method.";
  }
}

int64_t HloInstruction::feature_group_count() const {
   std::vector<std::string> mht_266_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_266(mht_266_v, 5620, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::feature_group_count");

  if (auto convolution = DynCast<HloConvolutionInstruction>(this)) {
    return convolution->feature_group_count();
  }
  return Cast<HloCustomCallInstruction>(this)->feature_group_count();
}

void HloInstruction::set_feature_group_count(int64_t feature_group_count) {
   std::vector<std::string> mht_267_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_267(mht_267_v, 5630, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::set_feature_group_count");

  if (auto convolution = DynCast<HloConvolutionInstruction>(this)) {
    return convolution->set_feature_group_count(feature_group_count);
  }
  Cast<HloCustomCallInstruction>(this)->set_feature_group_count(
      feature_group_count);
}

int64_t HloInstruction::batch_group_count() const {
   std::vector<std::string> mht_268_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_268(mht_268_v, 5641, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::batch_group_count");

  if (auto convolution = DynCast<HloConvolutionInstruction>(this)) {
    return convolution->batch_group_count();
  }
  return Cast<HloCustomCallInstruction>(this)->batch_group_count();
}

void HloInstruction::set_batch_group_count(int64_t batch_group_count) {
   std::vector<std::string> mht_269_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_269(mht_269_v, 5651, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::set_batch_group_count");

  if (auto convolution = DynCast<HloConvolutionInstruction>(this)) {
    return convolution->set_batch_group_count(batch_group_count);
  }
  Cast<HloCustomCallInstruction>(this)->set_batch_group_count(
      batch_group_count);
}

HloComputation* HloInstruction::select() const {
   std::vector<std::string> mht_270_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_270(mht_270_v, 5662, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::select");

  return Cast<HloSelectAndScatterInstruction>(this)->select();
}

HloComputation* HloInstruction::scatter() const {
   std::vector<std::string> mht_271_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_271(mht_271_v, 5669, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::scatter");

  return Cast<HloSelectAndScatterInstruction>(this)->scatter();
}

void HloInstruction::set_select(HloComputation* computation) {
   std::vector<std::string> mht_272_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_272(mht_272_v, 5676, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::set_select");

  return Cast<HloSelectAndScatterInstruction>(this)->set_select(computation);
}

void HloInstruction::set_scatter(HloComputation* computation) {
   std::vector<std::string> mht_273_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_273(mht_273_v, 5683, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::set_scatter");

  return Cast<HloSelectAndScatterInstruction>(this)->set_scatter(computation);
}

const std::string& HloInstruction::custom_call_target() const {
   std::vector<std::string> mht_274_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_274(mht_274_v, 5690, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::custom_call_target");

  return Cast<HloCustomCallInstruction>(this)->custom_call_target();
}
void HloInstruction::set_custom_call_target(absl::string_view target) {
   std::vector<std::string> mht_275_v;
   mht_275_v.push_back("target: \"" + std::string(target.data(), target.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_275(mht_275_v, 5697, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::set_custom_call_target");

  Cast<HloCustomCallInstruction>(this)->set_custom_call_target(target);
}

const PaddingConfig& HloInstruction::padding_config() const {
   std::vector<std::string> mht_276_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_276(mht_276_v, 5704, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::padding_config");

  return Cast<HloPadInstruction>(this)->padding_config();
}

PaddingType HloInstruction::padding_type() const {
   std::vector<std::string> mht_277_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_277(mht_277_v, 5711, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::padding_type");

  return Cast<HloCustomCallInstruction>(this)->padding_type();
}

PaddingConfig* HloInstruction::mutable_padding_config() {
   std::vector<std::string> mht_278_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_278(mht_278_v, 5718, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::mutable_padding_config");

  return Cast<HloPadInstruction>(this)->mutable_padding_config();
}

int64_t HloInstruction::slice_sizes(int64_t dimension) const {
   std::vector<std::string> mht_279_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_279(mht_279_v, 5725, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::slice_sizes");

  return Cast<HloDynamicSliceInstruction>(this)->slice_sizes(dimension);
}

const std::vector<int64_t>& HloInstruction::dynamic_slice_sizes() const {
   std::vector<std::string> mht_280_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_280(mht_280_v, 5732, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::dynamic_slice_sizes");

  return Cast<HloDynamicSliceInstruction>(this)->dynamic_slice_sizes();
}

const std::vector<std::vector<int64_t>>&
HloInstruction::dynamic_slice_sizes_list() const {
   std::vector<std::string> mht_281_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_281(mht_281_v, 5740, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::dynamic_slice_sizes_list");

  return Cast<HloCollectivePermuteInstruction>(this)
      ->dynamic_slice_sizes_list();
}

const GatherDimensionNumbers& HloInstruction::gather_dimension_numbers() const {
   std::vector<std::string> mht_282_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_282(mht_282_v, 5748, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::gather_dimension_numbers");

  return Cast<HloGatherInstruction>(this)->gather_dimension_numbers();
}

absl::Span<const int64_t> HloInstruction::gather_slice_sizes() const {
   std::vector<std::string> mht_283_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_283(mht_283_v, 5755, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::gather_slice_sizes");

  return Cast<HloGatherInstruction>(this)->gather_slice_sizes();
}

const ScatterDimensionNumbers& HloInstruction::scatter_dimension_numbers()
    const {
   std::vector<std::string> mht_284_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_284(mht_284_v, 5763, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::scatter_dimension_numbers");

  return Cast<HloScatterInstruction>(this)->scatter_dimension_numbers();
}

const DotDimensionNumbers& HloInstruction::dot_dimension_numbers() const {
   std::vector<std::string> mht_285_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_285(mht_285_v, 5770, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::dot_dimension_numbers");

  return Cast<HloDotInstruction>(this)->dot_dimension_numbers();
}

const DomainMetadata& HloInstruction::operand_side_metadata() const {
   std::vector<std::string> mht_286_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_286(mht_286_v, 5777, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::operand_side_metadata");

  return Cast<HloDomainInstruction>(this)->operand_side_metadata();
}

const DomainMetadata& HloInstruction::user_side_metadata() const {
   std::vector<std::string> mht_287_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_287(mht_287_v, 5784, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::user_side_metadata");

  return Cast<HloDomainInstruction>(this)->user_side_metadata();
}

bool HloInstruction::IsAsynchronous() const {
   std::vector<std::string> mht_288_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_288(mht_288_v, 5791, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::IsAsynchronous");

  return opcode() == HloOpcode::kAsyncStart ||
         opcode() == HloOpcode::kAsyncUpdate ||
         opcode() == HloOpcode::kAsyncDone;
}

HloComputation* HloInstruction::async_wrapped_computation() const {
   std::vector<std::string> mht_289_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_289(mht_289_v, 5800, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::async_wrapped_computation");

  CHECK(IsAsynchronous());
  return called_computations()[0];
}

// HloInstruction* HloInstruction::async_chain_start() const {
//   return Cast<HloAsyncInstruction>(this)->async_chain_start();
// }

HloInstruction* HloInstruction::async_wrapped_instruction() const {
   std::vector<std::string> mht_290_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_290(mht_290_v, 5812, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::async_wrapped_instruction");

  return Cast<HloAsyncInstruction>(this)->async_wrapped_instruction();
}

HloOpcode HloInstruction::async_wrapped_opcode() const {
   std::vector<std::string> mht_291_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_291(mht_291_v, 5819, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::async_wrapped_opcode");

  return Cast<HloAsyncInstruction>(this)->async_wrapped_opcode();
}

bool HloInstruction::is_cross_program_prefetch() const {
   std::vector<std::string> mht_292_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_292(mht_292_v, 5826, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::is_cross_program_prefetch");

  return Cast<HloCopyStartInstruction>(this)->is_cross_program_prefetch();
}

ComparisonDirection HloInstruction::comparison_direction() const {
   std::vector<std::string> mht_293_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_293(mht_293_v, 5833, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::comparison_direction");

  return Cast<HloCompareInstruction>(this)->direction();
}

const TriangularSolveOptions& HloInstruction::triangular_solve_options() const {
   std::vector<std::string> mht_294_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_294(mht_294_v, 5840, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::triangular_solve_options");

  return Cast<HloTriangularSolveInstruction>(this)->triangular_solve_options();
}

const CholeskyOptions& HloInstruction::cholesky_options() const {
   std::vector<std::string> mht_295_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_instructionDTcc mht_295(mht_295_v, 5847, "", "./tensorflow/compiler/xla/service/hlo_instruction.cc", "HloInstruction::cholesky_options");

  return Cast<HloCholeskyInstruction>(this)->cholesky_options();
}

}  // namespace xla
