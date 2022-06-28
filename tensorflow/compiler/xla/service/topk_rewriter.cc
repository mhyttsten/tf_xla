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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStopk_rewriterDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStopk_rewriterDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStopk_rewriterDTcc() {
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

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/topk_rewriter.h"

#include "absl/algorithm/container.h"
#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace xla {

static bool IsNanSafeGt(HloComputation* comp) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStopk_rewriterDTcc mht_0(mht_0_v, 196, "", "./tensorflow/compiler/xla/service/topk_rewriter.cc", "IsNanSafeGt");

  namespace m = match;
  auto match_bitcast_f32 = [](int64_t parameter_number) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStopk_rewriterDTcc mht_1(mht_1_v, 201, "", "./tensorflow/compiler/xla/service/topk_rewriter.cc", "lambda");

    auto param = m::Parameter(parameter_number)
                     .WithShape(m::Shape().WithElementType(F32));
    auto param_s32 =
        m::BitcastConvert(param).WithShape(m::Shape().WithElementType(S32));
    auto param_u32 =
        m::BitcastConvert(param).WithShape(m::Shape().WithElementType(U32));
    return m::Select(
        m::Lt(param_s32, m::ConstantScalar(0)),
        m::BitcastConvert(
            m::Subtract(m::ConstantScalar(std::numeric_limits<int32_t>::max()),
                        param_u32))
            .WithShape(m::Shape().WithElementType(S32)),
        param_s32);
  };

  auto match_bitcast_f32_with_convert = [](int64_t parameter_number) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStopk_rewriterDTcc mht_2(mht_2_v, 220, "", "./tensorflow/compiler/xla/service/topk_rewriter.cc", "lambda");

    auto param = m::Parameter(parameter_number)
                     .WithShape(m::Shape().WithElementType(F32));
    auto param_s32 =
        m::BitcastConvert(param).WithShape(m::Shape().WithElementType(S32));
    auto param_u32 =
        m::BitcastConvert(param).WithShape(m::Shape().WithElementType(U32));
    auto max_u32 =
        m::Convert(m::ConstantScalar(std::numeric_limits<int32_t>::max()))
            .WithShape(m::Shape().WithElementType(U32));
    return m::Select(m::Lt(param_s32, m::ConstantScalar(0)),
                     m::BitcastConvert(m::Subtract(max_u32, param_u32))
                         .WithShape(m::Shape().WithElementType(S32)),
                     param_s32);
  };

  auto match_bitcast_bf16 = [](int64_t parameter_number) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStopk_rewriterDTcc mht_3(mht_3_v, 239, "", "./tensorflow/compiler/xla/service/topk_rewriter.cc", "lambda");

    auto param = m::Convert(m::Parameter(parameter_number)
                                .WithShape(m::Shape().WithElementType(BF16)))
                     .WithShape(m::Shape().WithElementType(F32));
    auto param_s32 =
        m::BitcastConvert(param).WithShape(m::Shape().WithElementType(S32));
    auto param_u32 =
        m::BitcastConvert(param).WithShape(m::Shape().WithElementType(U32));
    return m::Select(
        m::Lt(param_s32, m::ConstantScalar(0)),
        m::BitcastConvert(
            m::Subtract(m::ConstantScalar(std::numeric_limits<int32_t>::max()),
                        param_u32))
            .WithShape(m::Shape().WithElementType(S32)),
        param_s32);
  };

  auto match_bitcast_bf16_with_convert = [](int64_t parameter_number) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStopk_rewriterDTcc mht_4(mht_4_v, 259, "", "./tensorflow/compiler/xla/service/topk_rewriter.cc", "lambda");

    auto param = m::Convert(m::Parameter(parameter_number)
                                .WithShape(m::Shape().WithElementType(BF16)))
                     .WithShape(m::Shape().WithElementType(F32));
    auto param_s32 =
        m::BitcastConvert(param).WithShape(m::Shape().WithElementType(S32));
    auto param_u32 =
        m::BitcastConvert(param).WithShape(m::Shape().WithElementType(U32));
    auto max_u32 =
        m::Convert(m::ConstantScalar(std::numeric_limits<int32_t>::max()))
            .WithShape(m::Shape().WithElementType(U32));
    return m::Select(m::Lt(param_s32, m::ConstantScalar(0)),
                     m::BitcastConvert(m::Subtract(max_u32, param_u32))
                         .WithShape(m::Shape().WithElementType(S32)),
                     param_s32);
  };

  auto match_s32 = [](int64_t parameter_number) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStopk_rewriterDTcc mht_5(mht_5_v, 279, "", "./tensorflow/compiler/xla/service/topk_rewriter.cc", "lambda");

    auto param = m::Parameter(parameter_number)
                     .WithShape(m::Shape().WithElementType(S32));
    return param;
  };

  return Match(comp->root_instruction(),
               m::Gt(match_bitcast_f32(0), match_bitcast_f32(1))) ||
         Match(comp->root_instruction(),
               m::Gt(match_bitcast_bf16(0), match_bitcast_bf16(1))) ||
         Match(comp->root_instruction(),
               m::Gt(match_bitcast_f32_with_convert(0),
                     match_bitcast_f32_with_convert(1))) ||
         Match(comp->root_instruction(),
               m::Gt(match_bitcast_bf16_with_convert(0),
                     match_bitcast_bf16_with_convert(1))) ||
         Match(comp->root_instruction(), m::Gt(match_s32(0), match_s32(1)));
}

absl::optional<int64_t> TopkRewriter::SortIsInTopK(HloInstruction* inst) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStopk_rewriterDTcc mht_6(mht_6_v, 301, "", "./tensorflow/compiler/xla/service/topk_rewriter.cc", "TopkRewriter::SortIsInTopK");

  HloSortInstruction* sort = DynCast<HloSortInstruction>(inst);
  if (sort == nullptr) {
    return absl::nullopt;
  }
  if (sort->operand_count() != 1 && sort->operand_count() != 2) {
    return absl::nullopt;
  }
  HloInstruction* data = sort->mutable_operand(0);

  if (sort->operand_count() == 2) {
    HloIotaInstruction* iota =
        DynCast<HloIotaInstruction>(sort->mutable_operand(1));
    if (iota == nullptr || iota->shape().rank() != data->shape().rank() ||
        iota->shape().element_type() != S32 ||
        iota->opcode() != HloOpcode::kIota ||
        iota->iota_dimension() != sort->sort_dimension()) {
      return absl::nullopt;
    }
  }
  if (!IsNanSafeGt(sort->to_apply())) {
    return absl::nullopt;
  }
  const int64_t sort_dim = sort->sort_dimension();
  const int64_t batch_dim = sort_dim == 1 ? 0 : 1;
  const bool has_batch = data->shape().rank() == 2;

  bool supported = true;
  absl::optional<int64_t> k;
  for (HloInstruction* user : sort->users()) {
    const HloInstruction* slice = user;
    if (sort->operand_count() == 2) {
      if (user->opcode() != HloOpcode::kGetTupleElement ||
          user->user_count() != 1) {
        supported = false;
        break;
      }
      slice = user->users()[0];
    }
    if (slice->opcode() != HloOpcode::kSlice) {
      // Non-slice user means we are not doing a TopK
      supported = false;
      break;
    }
    if (absl::c_any_of(slice->slice_starts(), [](int x) { return x != 0; }) ||
        absl::c_any_of(slice->slice_strides(), [](int x) { return x != 1; })) {
      // Strided slice or slicing at the beginning isn't supported.
      supported = false;
      break;
    }
    if (has_batch && slice->slice_limits(batch_dim) !=
                         slice->operand(0)->shape().dimensions(batch_dim)) {
      // Slicing along the batch dimension isn't supported.
      supported = false;
      break;
    }
    if (k == absl::nullopt) {
      k = slice->slice_limits(sort_dim);
    } else if (k != slice->slice_limits(sort_dim)) {
      // Different k for the different operands isn't supported.
      supported = false;
      break;
    }
  }
  if (k == absl::nullopt || !supported) {
    return absl::nullopt;
  }
  return k;
}

StatusOr<bool> TopkRewriter::TransformToCustomCall(HloModule* module) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStopk_rewriterDTcc mht_7(mht_7_v, 374, "", "./tensorflow/compiler/xla/service/topk_rewriter.cc", "TopkRewriter::TransformToCustomCall");

  bool changed = false;
  for (HloComputation* comp : module->computations()) {
    for (HloInstruction* inst : comp->MakeInstructionPostOrder()) {
      // Check if sort is in TopK.
      absl::optional<int64_t> k = SortIsInTopK(inst);
      if (!k) {
        continue;
      }

      HloSortInstruction* sort = DynCast<HloSortInstruction>(inst);
      HloInstruction* data = sort->mutable_operand(0);
      const PrimitiveType element_type = data->shape().element_type();

      if ((data->shape().rank() != 1 && data->shape().rank() != 2) ||
          (element_type != F32 && element_type != BF16)) {
        continue;
      }

      const int64_t sort_dim = sort->sort_dimension();
      const int64_t batch_dim = sort_dim == 1 ? 0 : 1;
      const bool has_batch = data->shape().rank() == 2;

      // Profitability check.
      if (!is_profitable_to_convert_(sort, *k)) {
        continue;
      }

      const int64_t batch_size =
          has_batch ? sort->operand(0)->shape().dimensions(batch_dim) : 1;
      const int64_t input_size = sort->operand(0)->shape().dimensions(sort_dim);
      HloInstruction* input = sort->mutable_operand(0);
      if (has_batch && sort_dim == 0) {
        input = comp->AddInstruction(HloInstruction::CreateTranspose(
            ShapeUtil::MakeShape(element_type, {batch_size, input_size}), input,
            {1, 0}));
      }

      Shape topk_shape =
          has_batch ? ShapeUtil::MakeTupleShape(
                          {ShapeUtil::MakeShape(element_type,
                                                {batch_size, k.value()}),
                           ShapeUtil::MakeShape(S32, {batch_size, k.value()})})
                    : ShapeUtil::MakeTupleShape(
                          {ShapeUtil::MakeShape(element_type, {k.value()}),
                           ShapeUtil::MakeShape(S32, {k.value()})});
      HloInstruction* topk = comp->AddInstruction(
          HloInstruction::CreateCustomCall(topk_shape, {input}, "TopK"));
      HloInstruction* value_gte =
          comp->AddInstruction(HloInstruction::CreateGetTupleElement(
              topk->shape().tuple_shapes(0), topk, 0));
      HloInstruction* index_gte =
          comp->AddInstruction(HloInstruction::CreateGetTupleElement(
              topk->shape().tuple_shapes(1), topk, 1));

      if (has_batch && sort_dim == 0) {
        value_gte = comp->AddInstruction(HloInstruction::CreateTranspose(
            ShapeUtil::MakeShape(element_type, {k.value(), batch_size}),
            value_gte, {1, 0}));
        index_gte = comp->AddInstruction(HloInstruction::CreateTranspose(
            ShapeUtil::MakeShape(S32, {k.value(), batch_size}), index_gte,
            {1, 0}));
      }

      for (HloInstruction* user : sort->users()) {
        if (sort->operand_count() == 2) {
          HloInstruction* gte = user;
          for (HloInstruction* slice : gte->users()) {
            if (gte->tuple_index() == 0) {
              TF_RETURN_IF_ERROR(slice->ReplaceAllUsesWith(value_gte));
            } else if (gte->tuple_index() == 1) {
              TF_RETURN_IF_ERROR(slice->ReplaceAllUsesWith(index_gte));
            } else {
              LOG(FATAL) << "Sort with more than 2 output isn't supported in "
                            "topk rewriter";
            }
          }
        } else {
          TF_RETURN_IF_ERROR(user->ReplaceAllUsesWith(value_gte));
        }
      }
      changed = true;
    }
  }
  return changed;
}

StatusOr<bool> TopkRewriter::Run(HloModule* module) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStopk_rewriterDTcc mht_8(mht_8_v, 464, "", "./tensorflow/compiler/xla/service/topk_rewriter.cc", "TopkRewriter::Run");

  bool changed = false;
  TF_ASSIGN_OR_RETURN(auto transform_to_customcall_changed,
                      TransformToCustomCall(module));
  changed |= transform_to_customcall_changed;
  return changed;
}

}  // namespace xla
