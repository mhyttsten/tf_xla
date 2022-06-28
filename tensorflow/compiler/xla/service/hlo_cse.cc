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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cseDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cseDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cseDTcc() {
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

#include "tensorflow/compiler/xla/service/hlo_cse.h"

#include <functional>
#include <list>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_domain_map.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {

namespace {

template <bool kIsLayoutSensitive>
struct ConstantKey {
  template <typename H>
  friend H AbslHashValue(H h, const ConstantKey& key) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cseDTcc mht_0(mht_0_v, 217, "", "./tensorflow/compiler/xla/service/hlo_cse.cc", "AbslHashValue");

    h = H::combine(std::move(h), key.domain);
    return Literal::Hash<H, kIsLayoutSensitive, /*kByteLimit=*/64>(
        std::move(h), key.hlo->literal());
  }
  friend bool operator==(const ConstantKey& lhs, const ConstantKey& rhs) {
    return lhs.domain == rhs.domain &&
           (kIsLayoutSensitive ? Shape::Equal()
                               : Shape::Equal().IgnoreLayout())(
               lhs.hlo->shape(), rhs.hlo->shape()) &&
           lhs.hlo->literal() == rhs.hlo->literal();
  }
  HloConstantInstruction* hlo;
  int64_t domain;
};

template <bool kIsLayoutSensitive>
struct IotaKey {
  template <typename H>
  friend H AbslHashValue(H h, const IotaKey<kIsLayoutSensitive>& key) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cseDTcc mht_1(mht_1_v, 239, "", "./tensorflow/compiler/xla/service/hlo_cse.cc", "AbslHashValue");

    h = H::combine(std::move(h), key.domain, key.hlo->iota_dimension());
    return Shape::Hash<H, kIsLayoutSensitive>(std::move(h), key.hlo->shape());
  }
  friend bool operator==(const IotaKey<kIsLayoutSensitive>& lhs,
                         const IotaKey<kIsLayoutSensitive>& rhs) {
    return lhs.domain == rhs.domain &&
           (kIsLayoutSensitive ? Shape::Equal()
                               : Shape::Equal().IgnoreLayout())(
               lhs.hlo->shape(), rhs.hlo->shape()) &&
           lhs.hlo->iota_dimension() == rhs.hlo->iota_dimension();
  }
  HloIotaInstruction* hlo;
  int64_t domain;
};

// Find and combine identical constants. Constants are identical if they have
// the same type and value.
//
// While we're here, also combine identical iota instructions, since they need
// similar treatment.
template <bool kIsLayoutSensitive>
StatusOr<bool> CombineConstants(HloComputation* computation) {
  TF_ASSIGN_OR_RETURN(auto domain_map, HloDomainMap::Create(computation, ""));
  // Map from the literal hash of a constant or the shape hash of an iota all
  // equivalent instructions. This avoids extreme quadratic behavior with many
  // scalar constants.
  absl::flat_hash_set<ConstantKey<kIsLayoutSensitive>> constants;
  absl::flat_hash_set<IotaKey<kIsLayoutSensitive>> iotas;
  int64_t combined = 0;
  auto inst_it = computation->instructions().begin();
  while (inst_it != computation->instructions().end()) {
    HloInstruction* instruction = *inst_it;

    // Advance list iterator before loop body because iterator may be
    // invalidated due to deletion.
    ++inst_it;

    HloInstruction* match = nullptr;
    if (auto* constant_inst = DynCast<HloConstantInstruction>(instruction)) {
      auto insert_result = constants.insert(ConstantKey<kIsLayoutSensitive>{
          constant_inst, domain_map->GetDomainId(instruction)});
      if (!insert_result.second) {
        match = insert_result.first->hlo;
      }
    }
    if (auto* iota_inst = DynCast<HloIotaInstruction>(instruction)) {
      auto insert_result = iotas.insert(IotaKey<kIsLayoutSensitive>{
          iota_inst, domain_map->GetDomainId(instruction)});
      if (!insert_result.second) {
        match = insert_result.first->hlo;
      }
    }

    if (match != nullptr) {
      // Match found, replace this instruction with the one in the set.
      TF_CHECK_OK(instruction->ReplaceAllUsesWith(match));
      TF_CHECK_OK(computation->RemoveInstruction(instruction));
      ++combined;
    }
  }
  VLOG(4) << "Combined " << combined << " constants and iotas in "
          << computation->name() << " computation";
  return combined > 0;
}

// An instruction is considered to be equivalent to another only if they
// share the exact same set of operands.
struct CseKey {
  template <typename H>
  friend H AbslHashValue(H h, const CseKey& key) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cseDTcc mht_2(mht_2_v, 312, "", "./tensorflow/compiler/xla/service/hlo_cse.cc", "AbslHashValue");

    auto instruction = key.hlo;
    h = H::combine(std::move(h), instruction->opcode(),
                   instruction->shape().dimensions());
    auto window_hash = [](H h, const Window& window) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cseDTcc mht_3(mht_3_v, 319, "", "./tensorflow/compiler/xla/service/hlo_cse.cc", "lambda");

      const auto& window_dims = window.dimensions();
      for (const auto& window_dim : window_dims) {
        h = H::combine(std::move(h), window_dim.size(), window_dim.stride(),
                       window_dim.padding_low(), window_dim.padding_high(),
                       window_dim.window_dilation(), window_dim.base_dilation(),
                       window_dim.window_reversal());
      }
      return H::combine(std::move(h), window_dims.size());
    };
    for (auto operand : instruction->operands()) {
      h = H::combine(std::move(h), operand->unique_id());
    }
    for (auto c : instruction->called_computations()) {
      h = H::combine(std::move(h), c->root_instruction()->opcode());
    }
    switch (instruction->opcode()) {
      case HloOpcode::kSlice:
        return H::combine(std::move(h), instruction->slice_starts(),
                          instruction->slice_strides());
      case HloOpcode::kPad: {
        const auto& padding_dims = instruction->padding_config().dimensions();
        for (const auto& padding_dim : padding_dims) {
          h = H::combine(std::move(h), padding_dim.edge_padding_low(),
                         padding_dim.edge_padding_high(),
                         padding_dim.interior_padding());
        }
        h = H::combine(std::move(h), padding_dims.size());
        return std::move(h);
      }
      case HloOpcode::kDot: {
        const auto& dot_dimension_numbers =
            instruction->dot_dimension_numbers();
        h = H::combine(
            std::move(h),
            absl::MakeSpan(dot_dimension_numbers.lhs_contracting_dimensions()),
            absl::MakeSpan(dot_dimension_numbers.rhs_contracting_dimensions()),
            absl::MakeSpan(dot_dimension_numbers.lhs_batch_dimensions()),
            absl::MakeSpan(dot_dimension_numbers.rhs_batch_dimensions()));
        return std::move(h);
      }
      case HloOpcode::kConvolution: {
        const auto& conv_dimension_numbers =
            instruction->convolution_dimension_numbers();
        h = H::combine(
            std::move(h), conv_dimension_numbers.input_batch_dimension(),
            conv_dimension_numbers.input_feature_dimension(),
            absl::MakeSpan(conv_dimension_numbers.input_spatial_dimensions()),
            conv_dimension_numbers.kernel_input_feature_dimension(),
            conv_dimension_numbers.kernel_output_feature_dimension(),
            absl::MakeSpan(conv_dimension_numbers.kernel_spatial_dimensions()),
            conv_dimension_numbers.output_batch_dimension(),
            conv_dimension_numbers.output_feature_dimension(),
            absl::MakeSpan(conv_dimension_numbers.output_spatial_dimensions()));
        return window_hash(std::move(h), instruction->window());
      }
      case HloOpcode::kReduceWindow:
        return window_hash(std::move(h), instruction->window());
      case HloOpcode::kConcatenate:
      case HloOpcode::kBroadcast:
      case HloOpcode::kTranspose:
      case HloOpcode::kReduce:
        return H::combine(std::move(h), instruction->dimensions());
      case HloOpcode::kGetTupleElement:
        return H::combine(std::move(h), instruction->tuple_index());
      default:
        return std::move(h);
    }
  }
  HloInstruction* hlo;
};

}  // namespace

StatusOr<bool> HloCSE::Run(HloModule* module) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cseDTcc mht_4(mht_4_v, 396, "", "./tensorflow/compiler/xla/service/hlo_cse.cc", "HloCSE::Run");

  bool changed = false;

  const std::function<bool(const HloInstruction*, const HloInstruction*)>
      eq_instructions = std::equal_to<const HloInstruction*>();
  const std::function<bool(const HloComputation*, const HloComputation*)>
      eq_computations = [](const HloComputation* lhs,
                           const HloComputation* rhs) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cseDTcc mht_5(mht_5_v, 406, "", "./tensorflow/compiler/xla/service/hlo_cse.cc", "lambda");
 return *lhs == *rhs; };

  auto cse_equal = [&](const CseKey& lhs, const CseKey& rhs) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cseDTcc mht_6(mht_6_v, 411, "", "./tensorflow/compiler/xla/service/hlo_cse.cc", "lambda");

    return lhs.hlo->Identical(*rhs.hlo, eq_instructions, eq_computations,
                              is_layout_sensitive_);
  };

  for (auto* computation : module->computations()) {
    if (only_fusion_computations_ && !computation->IsFusionComputation()) {
      continue;
    }

    TF_ASSIGN_OR_RETURN(bool combined,
                        is_layout_sensitive_
                            ? CombineConstants<true>(computation)
                            : CombineConstants<false>(computation));
    changed |= combined;

    // HLO instructions are grouped into equivalency classes by using the
    // cse_equal predicate defined above. This set holds a representative
    // instruction for each class.
    absl::flat_hash_set<CseKey, absl::Hash<CseKey>, decltype(cse_equal)>
        representatives(/*N=*/computation->instruction_count() + 1,
                        absl::Hash<CseKey>{}, cse_equal);
    for (auto instruction : computation->MakeInstructionPostOrder()) {
      // If the instruction has zero operands (constants, parameters, etc.) skip
      // over it.
      if (instruction->operand_count() == 0 &&
          instruction->opcode() != HloOpcode::kPartitionId &&
          instruction->opcode() != HloOpcode::kReplicaId) {
        continue;
      }
      // Skip instructions which have side effects.
      if (instruction->HasSideEffect()) {
        continue;
      }

      auto pair = representatives.insert(CseKey{instruction});
      if (!pair.second) {
        HloInstruction* equivalent_instruction = pair.first->hlo;
        TF_RETURN_IF_ERROR(
            instruction->ReplaceAllUsesWith(equivalent_instruction));
        TF_RETURN_IF_ERROR(computation->RemoveInstruction(instruction));
        changed = true;
        continue;
      }
    }
  }
  return changed;
}

}  // namespace xla
