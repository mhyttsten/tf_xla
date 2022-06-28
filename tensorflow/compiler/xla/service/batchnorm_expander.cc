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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbatchnorm_expanderDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbatchnorm_expanderDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbatchnorm_expanderDTcc() {
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

#include "tensorflow/compiler/xla/service/batchnorm_expander.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

namespace {

using absl::optional;

// BatchNormExpanderVisitor traverses the HLO computation and rewrites BatchNorm
// operations into smaller operations.
class BatchNormExpanderVisitor : public DfsHloRewriteVisitor {
 public:
  Status HandleBatchNormTraining(HloInstruction* batch_norm) override;

  Status HandleBatchNormInference(HloInstruction* batch_norm) override;

  Status HandleBatchNormGrad(HloInstruction* batch_norm) override;

  // Runs the visitor on a computation.
  static bool Run(HloComputation* computation, bool rewrite_training_op,
                  bool rewrite_inference_op, bool rewrite_grad_op);

  ~BatchNormExpanderVisitor() override = default;

 private:
  explicit BatchNormExpanderVisitor(HloComputation* computation,
                                    bool rewrite_training_op,
                                    bool rewrite_inference_op,
                                    bool rewrite_grad_op)
      : computation_(computation),
        rewrite_training_op_(rewrite_training_op),
        rewrite_inference_op_(rewrite_inference_op),
        rewrite_grad_op_(rewrite_grad_op) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbatchnorm_expanderDTcc mht_0(mht_0_v, 239, "", "./tensorflow/compiler/xla/service/batchnorm_expander.cc", "BatchNormExpanderVisitor");
}

  HloComputation* GetOrCreateScalarAddComputation(
      PrimitiveType primitive_type) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbatchnorm_expanderDTcc mht_1(mht_1_v, 245, "", "./tensorflow/compiler/xla/service/batchnorm_expander.cc", "GetOrCreateScalarAddComputation");

    HloComputation::Builder b("scalar_add_computation");
    Shape shape = ShapeUtil::MakeShape(primitive_type, {});
    auto scalar_lhs = b.AddInstruction(
        HloInstruction::CreateParameter(0, shape, "scalar_lhs"));
    auto scalar_rhs = b.AddInstruction(
        HloInstruction::CreateParameter(1, shape, "scalar_rhs"));
    auto scalar_op = b.AddInstruction(HloInstruction::CreateBinary(
        shape, HloOpcode::kAdd, scalar_lhs, scalar_rhs));
    return computation_->parent()->AddEmbeddedComputation(b.Build(scalar_op));
  }

  std::unique_ptr<HloInstruction> Rsqrt(
      HloInstruction* operand,
      const std::function<HloInstruction*(std::unique_ptr<HloInstruction>)>&
          add_instruction) {
    return HloInstruction::CreateUnary(operand->shape(), HloOpcode::kRsqrt,
                                       operand);
  }

  std::unique_ptr<HloInstruction> Mean(
      HloInstruction* element_count, HloInstruction* operand,
      const std::function<HloInstruction*(std::unique_ptr<HloInstruction>)>&
          add_instruction) {
    auto broadcast = add_instruction(
        HloInstruction::CreateBroadcast(operand->shape(), element_count, {}));
    return HloInstruction::CreateBinary(operand->shape(), HloOpcode::kDivide,
                                        operand, broadcast);
  }

  std::unique_ptr<HloInstruction> DynamicElementCountPerFeature(
      HloInstruction* operand, int64_t feature_index,
      const std::function<HloInstruction*(std::unique_ptr<HloInstruction>)>&
          add_instruction) {
    auto elements_per_feature_s32 = add_instruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(1)));

    for (int64_t i = 0; i < operand->shape().rank(); ++i) {
      if (i == feature_index) {
        continue;
      }
      auto dynamic_dimension_size =
          add_instruction(HloInstruction::CreateGetDimensionSize(
              ShapeUtil::MakeShape(S32, {}), operand, i));
      elements_per_feature_s32 = add_instruction(HloInstruction::CreateBinary(
          ShapeUtil::MakeShape(S32, {}), HloOpcode::kMultiply,
          dynamic_dimension_size, elements_per_feature_s32));
    }

    return HloInstruction::CreateConvert(
        ShapeUtil::MakeShape(operand->shape().element_type(), {}),
        elements_per_feature_s32);
  }

  // Current HloComputation instance the BatchNormExpander is
  // traversing.
  HloComputation* computation_;

  bool rewrite_training_op_;
  bool rewrite_inference_op_;
  bool rewrite_grad_op_;
};

}  // namespace

bool BatchNormExpanderVisitor::Run(HloComputation* computation,
                                   bool rewrite_training_op,
                                   bool rewrite_inference_op,
                                   bool rewrite_grad_op) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbatchnorm_expanderDTcc mht_2(mht_2_v, 316, "", "./tensorflow/compiler/xla/service/batchnorm_expander.cc", "BatchNormExpanderVisitor::Run");

  BatchNormExpanderVisitor visitor(
      computation,
      /*rewrite_training_op=*/rewrite_training_op,
      /*rewrite_inference_op=*/rewrite_inference_op,
      /*rewrite_grad_op=*/rewrite_grad_op);
  TF_CHECK_OK(computation->Accept(&visitor));
  return visitor.changed();
}

Status BatchNormExpanderVisitor::HandleBatchNormTraining(
    HloInstruction* batch_norm) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbatchnorm_expanderDTcc mht_3(mht_3_v, 330, "", "./tensorflow/compiler/xla/service/batchnorm_expander.cc", "BatchNormExpanderVisitor::HandleBatchNormTraining");

  if (!rewrite_training_op_) {
    return Status::OK();
  }

  std::vector<HloInstruction*> added_instructions;
  auto add = [&](std::unique_ptr<HloInstruction> inst) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbatchnorm_expanderDTcc mht_4(mht_4_v, 339, "", "./tensorflow/compiler/xla/service/batchnorm_expander.cc", "lambda");

    HloInstruction* added_inst = computation_->AddInstruction(std::move(inst));
    added_inst->set_metadata(batch_norm->metadata());
    added_instructions.push_back(added_inst);
    return added_inst;
  };
  auto add_binary = [&](const Shape& shape, const HloOpcode opcode,
                        HloInstruction* a, HloInstruction* b) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbatchnorm_expanderDTcc mht_5(mht_5_v, 349, "", "./tensorflow/compiler/xla/service/batchnorm_expander.cc", "lambda");

    return add(HloInstruction::CreateBinary(shape, opcode, a, b));
  };
  int64_t instruction_count_before = computation_->instruction_count();

  // Expand batch norm training into smaller HLO ops.
  HloInstruction* operand = batch_norm->mutable_operand(0);
  const Shape operand_shape = operand->shape();
  PrimitiveType ptype = operand_shape.element_type();
  int64_t feature_index = batch_norm->feature_index();

  HloInstruction* scale = batch_norm->mutable_operand(1);
  HloInstruction* offset = batch_norm->mutable_operand(2);
  const Shape feature_shape = scale->shape();

  auto zero_literal = LiteralUtil::CreateR0(0.0f);
  TF_ASSIGN_OR_RETURN(zero_literal, zero_literal.Convert(ptype));
  auto zero = add(HloInstruction::CreateConstant(std::move(zero_literal)));

  auto epsilon_literal = LiteralUtil::CreateR0(batch_norm->epsilon());
  TF_ASSIGN_OR_RETURN(epsilon_literal, epsilon_literal.Convert(ptype));
  auto epsilon = add(HloInstruction::CreateBroadcast(
      operand_shape,
      add(HloInstruction::CreateConstant(std::move(epsilon_literal))), {}));
  std::vector<int64_t> dimensions_without_feature;
  const int64_t rank = operand_shape.rank();
  dimensions_without_feature.reserve(rank - 1);

  for (int64_t i = 0; i < rank; ++i) {
    if (i != feature_index) {
      dimensions_without_feature.push_back(i);
    }
  }

  auto elements_per_feature =
      add(DynamicElementCountPerFeature(operand, feature_index, add));

  auto scale_broadcasted = add(
      HloInstruction::CreateBroadcast(operand_shape, scale, {feature_index}));

  auto offset_broadcasted = add(
      HloInstruction::CreateBroadcast(operand_shape, offset, {feature_index}));

  HloComputation* add_reduce_computation =
      GetOrCreateScalarAddComputation(ptype);

  // X^2.
  auto operand_squared =
      add_binary(operand_shape, HloOpcode::kMultiply, operand, operand);
  // Sum[X].
  auto sum = add(HloInstruction::CreateReduce(feature_shape, operand, zero,
                                              dimensions_without_feature,
                                              add_reduce_computation));

  // Sum[X^2].
  auto squared_sum = add(HloInstruction::CreateReduce(
      feature_shape, operand_squared, zero, dimensions_without_feature,
      add_reduce_computation));

  // E[X].
  auto mean = add(Mean(elements_per_feature, sum, add));

  auto mean_broadcasted = add(
      HloInstruction::CreateBroadcast(operand_shape, mean, {feature_index}));

  // E[X^2].
  auto square_mean = add(Mean(elements_per_feature, squared_sum, add));

  // E^2[X].
  auto mean_square =
      add_binary(feature_shape, HloOpcode::kMultiply, mean, mean);

  // Var[X].
  auto var =
      add_binary(feature_shape, HloOpcode::kSubtract, square_mean, mean_square);

  auto var_broadcasted =
      add(HloInstruction::CreateBroadcast(operand_shape, var, {feature_index}));

  // Var[X] + epsilon.
  auto var_add_epsilon =
      add_binary(operand_shape, HloOpcode::kAdd, var_broadcasted, epsilon);

  // 1 / Sqrt[Var[X] + epsilon].
  auto rsqrt_var_add_epsilon = add(Rsqrt(var_add_epsilon, add));

  // X - E[X].
  auto operand_minus_mean = add_binary(operand_shape, HloOpcode::kSubtract,
                                       operand, mean_broadcasted);

  // (X - E[X]) / Sqrt[Var[X] + epsilon].
  auto normalized = add_binary(operand_shape, HloOpcode::kMultiply,
                               operand_minus_mean, rsqrt_var_add_epsilon);

  // (X - E[X]) / Sqrt[Var[X] + epsilon] * scale.
  auto scaled_normalized = add_binary(operand_shape, HloOpcode::kMultiply,
                                      normalized, scale_broadcasted);

  // (X - E[X]) / Sqrt[Var[X] + epsilon] * scale + offset.
  auto shifted_normalized = add_binary(operand_shape, HloOpcode::kAdd,
                                       scaled_normalized, offset_broadcasted);

  auto tuple = HloInstruction::CreateTuple({shifted_normalized, mean, var});

  if (batch_norm->has_sharding()) {
    int64_t instruction_count_after = computation_->instruction_count();
    CHECK_EQ(instruction_count_after,
             instruction_count_before + added_instructions.size());
    const HloSharding& sharding = batch_norm->sharding();
    HloSharding operand_sharding =
        sharding.GetAsShapeTree(batch_norm->shape()).element({0});
    optional<int64_t> unique_device = batch_norm->sharding_unique_device();
    HloSharding default_sharding =
        unique_device.has_value()
            ? HloSharding::AssignDevice(unique_device.value())
            : HloSharding::Replicate();
    for (HloInstruction* inst : added_instructions) {
      if (ShapeUtil::Equal(inst->shape(), operand_shape)) {
        inst->set_sharding(operand_sharding);
      } else {
        inst->set_sharding(default_sharding);
      }
    }
    tuple->set_sharding(sharding);
  }
  TF_CHECK_OK(ReplaceWithNewInstruction(batch_norm, std::move(tuple)));
  return Status::OK();
}

Status BatchNormExpanderVisitor::HandleBatchNormInference(
    HloInstruction* batch_norm) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbatchnorm_expanderDTcc mht_6(mht_6_v, 482, "", "./tensorflow/compiler/xla/service/batchnorm_expander.cc", "BatchNormExpanderVisitor::HandleBatchNormInference");

  if (!rewrite_inference_op_) {
    return Status::OK();
  }
  // Expand batch norm inference into smaller HLO ops.
  HloInstruction* operand = batch_norm->mutable_operand(0);
  const Shape operand_shape = operand->shape();
  int64_t feature_index = batch_norm->feature_index();
  PrimitiveType ptype = operand_shape.element_type();

  HloInstruction* scale = batch_norm->mutable_operand(1);
  HloInstruction* offset = batch_norm->mutable_operand(2);
  HloInstruction* mean = batch_norm->mutable_operand(3);
  HloInstruction* var = batch_norm->mutable_operand(4);
  const Shape feature_shape = scale->shape();

  auto epsilon_literal = LiteralUtil::CreateR0(batch_norm->epsilon());
  TF_ASSIGN_OR_RETURN(epsilon_literal, epsilon_literal.Convert(ptype));
  auto epsilon = computation_->AddInstruction(HloInstruction::CreateBroadcast(
      feature_shape,
      computation_->AddInstruction(
          HloInstruction::CreateConstant(std::move(epsilon_literal))),
      {}));

  std::vector<int64_t> dimensions_without_feature;
  const int64_t rank = operand_shape.rank();
  dimensions_without_feature.reserve(rank - 1);

  for (int64_t i = 0; i < rank; ++i) {
    if (i != feature_index) {
      dimensions_without_feature.push_back(i);
    }
  }

  std::vector<HloInstruction*> added_instructions;
  auto add = [&](std::unique_ptr<HloInstruction> inst) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbatchnorm_expanderDTcc mht_7(mht_7_v, 520, "", "./tensorflow/compiler/xla/service/batchnorm_expander.cc", "lambda");

    HloInstruction* added_inst = computation_->AddInstruction(std::move(inst));
    added_inst->set_metadata(batch_norm->metadata());
    added_instructions.push_back(added_inst);
    return added_inst;
  };
  auto add_binary = [&](const Shape& shape, const HloOpcode opcode,
                        HloInstruction* a, HloInstruction* b) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbatchnorm_expanderDTcc mht_8(mht_8_v, 530, "", "./tensorflow/compiler/xla/service/batchnorm_expander.cc", "lambda");

    return add(HloInstruction::CreateBinary(shape, opcode, a, b));
  };
  auto feature_broadcast = [&](HloInstruction* a) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbatchnorm_expanderDTcc mht_9(mht_9_v, 536, "", "./tensorflow/compiler/xla/service/batchnorm_expander.cc", "lambda");

    return add(
        HloInstruction::CreateBroadcast(operand_shape, a, {feature_index}));
  };

  int64_t instruction_count_before = computation_->instruction_count();
  auto true_scale = add_binary(
      feature_shape, HloOpcode::kMultiply, scale,
      add(Rsqrt(add_binary(feature_shape, HloOpcode::kAdd, var, epsilon),
                add)));
  auto true_shift = add_binary(
      feature_shape, HloOpcode::kSubtract, offset,
      add_binary(feature_shape, HloOpcode::kMultiply, mean, true_scale));

  auto shifted_normalized =
      add_binary(operand_shape, HloOpcode::kAdd,
                 add_binary(operand_shape, HloOpcode::kMultiply, operand,
                            feature_broadcast(true_scale)),
                 feature_broadcast(true_shift));

  int64_t instruction_count_after = computation_->instruction_count();
  CHECK_EQ(instruction_count_after,
           instruction_count_before + added_instructions.size());
  if (batch_norm->has_sharding()) {
    const HloSharding& sharding = batch_norm->sharding();
    optional<int64_t> unique_device = batch_norm->sharding_unique_device();
    HloSharding default_sharding =
        unique_device.has_value()
            ? HloSharding::AssignDevice(unique_device.value())
            : HloSharding::Replicate();
    for (HloInstruction* inst : added_instructions) {
      if (ShapeUtil::Equal(inst->shape(), operand_shape)) {
        inst->set_sharding(sharding);
      } else {
        inst->set_sharding(default_sharding);
      }
    }
    shifted_normalized->set_sharding(sharding);
  }
  TF_CHECK_OK(ReplaceInstruction(batch_norm, shifted_normalized));
  return Status::OK();
}

Status BatchNormExpanderVisitor::HandleBatchNormGrad(
    HloInstruction* batch_norm) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbatchnorm_expanderDTcc mht_10(mht_10_v, 583, "", "./tensorflow/compiler/xla/service/batchnorm_expander.cc", "BatchNormExpanderVisitor::HandleBatchNormGrad");

  // Use the following formulas to calculate gradients:
  // scale_grad =
  //   sum(output_grad * (activation - mean(activation))) * rsqrt(var + epsilon)
  //
  // offset_grad =
  //   sum(output_grad)
  //
  // activation_grad =
  //   1/N * scale * rsqrt(var + epsilon) *
  //   (N * output_grad - sum(output_grad) - (activation - mean(activation)) *
  //   sum(output_grad * (activation - mean(activation))) / (variance +
  //   epsilon))
  if (!rewrite_grad_op_) {
    return Status::OK();
  }
  std::vector<HloInstruction*> added_instructions;
  auto add = [&](std::unique_ptr<HloInstruction> inst) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbatchnorm_expanderDTcc mht_11(mht_11_v, 603, "", "./tensorflow/compiler/xla/service/batchnorm_expander.cc", "lambda");

    HloInstruction* added_inst = computation_->AddInstruction(std::move(inst));
    added_inst->set_metadata(batch_norm->metadata());
    added_instructions.push_back(added_inst);
    return added_inst;
  };
  auto add_binary = [&](const Shape& shape, const HloOpcode opcode,
                        HloInstruction* a, HloInstruction* b) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbatchnorm_expanderDTcc mht_12(mht_12_v, 613, "", "./tensorflow/compiler/xla/service/batchnorm_expander.cc", "lambda");

    return add(HloInstruction::CreateBinary(shape, opcode, a, b));
  };
  int64_t instruction_count_before = computation_->instruction_count();

  HloInstruction* activation = batch_norm->mutable_operand(0);
  const Shape activation_shape = activation->shape();
  PrimitiveType ptype = activation_shape.element_type();
  HloInstruction* scale = batch_norm->mutable_operand(1);
  const Shape feature_shape = scale->shape();
  HloInstruction* mean = batch_norm->mutable_operand(2);
  HloInstruction* variance = batch_norm->mutable_operand(3);
  HloInstruction* grad_output = batch_norm->mutable_operand(4);

  int64_t feature_index = batch_norm->feature_index();

  auto elements_per_feature =
      add(DynamicElementCountPerFeature(activation, feature_index, add));

  auto zero_literal = LiteralUtil::CreateR0(0.0f);
  TF_ASSIGN_OR_RETURN(zero_literal, zero_literal.Convert(ptype));
  auto zero = add(HloInstruction::CreateConstant(std::move(zero_literal)));

  auto epsilon_literal = LiteralUtil::CreateR0(batch_norm->epsilon());
  TF_ASSIGN_OR_RETURN(epsilon_literal, epsilon_literal.Convert(ptype));
  auto epsilon_scalar =
      add(HloInstruction::CreateConstant(std::move(epsilon_literal)));
  auto epsilon_activation = add(
      HloInstruction::CreateBroadcast(activation_shape, epsilon_scalar, {}));
  auto epsilon_feature =
      add(HloInstruction::CreateBroadcast(feature_shape, epsilon_scalar, {}));

  std::vector<int64_t> dimensions_without_feature;
  const int64_t rank = activation_shape.rank();
  dimensions_without_feature.reserve(rank - 1);

  for (int64_t i = 0; i < rank; ++i) {
    if (i != feature_index) {
      dimensions_without_feature.push_back(i);
    }
  }

  auto scale_broadcasted = add(HloInstruction::CreateBroadcast(
      activation_shape, scale, {feature_index}));
  auto variance_broadcasted = add(HloInstruction::CreateBroadcast(
      activation_shape, variance, {feature_index}));

  // E[X].
  auto mean_broadcasted = add(
      HloInstruction::CreateBroadcast(activation_shape, mean, {feature_index}));

  // rsqrt[Var[X] + epsilon].
  auto rsqrt_var_add_epsilon_broadcasted =
      add(Rsqrt(add_binary(activation_shape, HloOpcode::kAdd,
                           variance_broadcasted, epsilon_activation),
                add));

  auto rsqrt_var_add_epsilon = add(Rsqrt(
      add_binary(feature_shape, HloOpcode::kAdd, variance, epsilon_feature),
      add));

  // X - E[X].
  auto activation_minus_mean = add_binary(
      activation_shape, HloOpcode::kSubtract, activation, mean_broadcasted);

  // Grad[Y] * (X - E[X]).
  auto grad_output_times_activation_minus_mean =
      add_binary(activation_shape, HloOpcode::kMultiply, grad_output,
                 activation_minus_mean);

  HloComputation* add_reduce_computation =
      GetOrCreateScalarAddComputation(ptype);

  // sum(Grad[Y] * (X - E[X])).
  auto sum_grad_output_times_activation_minus_mean =
      add(HloInstruction::CreateReduce(
          feature_shape, grad_output_times_activation_minus_mean, zero,
          dimensions_without_feature, add_reduce_computation));

  // Grad[beta] = Sum(Grad[Y]).
  auto grad_beta = add(HloInstruction::CreateReduce(
      feature_shape, grad_output, zero, dimensions_without_feature,
      add_reduce_computation));

  // Grad[scale] = Sum(Grad[Y] * (X - E[X]) * rsqrt[Var[X] + epsilon]).
  auto grad_scale = add_binary(feature_shape, HloOpcode::kMultiply,
                               sum_grad_output_times_activation_minus_mean,
                               rsqrt_var_add_epsilon);

  // I2 = Sum(Grad[Y])
  auto i2 = add(HloInstruction::CreateBroadcast(activation_shape, grad_beta,
                                                {feature_index}));

  // I3 = Sum(Grad[Y] * (X - E[X]))
  auto i3 = add(HloInstruction::CreateBroadcast(
      activation_shape, sum_grad_output_times_activation_minus_mean,
      {feature_index}));

  // I4 = (X - E[X]) * I3
  auto i4 = add_binary(activation_shape, HloOpcode::kMultiply, i3,
                       activation_minus_mean);

  // I5 = I4 / (Var[X] + epsilon)
  auto i5 = add_binary(activation_shape, HloOpcode::kDivide, i4,
                       add_binary(activation_shape, HloOpcode::kAdd,
                                  variance_broadcasted, epsilon_activation));

  // scale * rsqrt[Var[X] + epsilon] * 1/N
  auto scale_times_rsqrt_var_add_epsilon =
      add_binary(activation_shape, HloOpcode::kMultiply, scale_broadcasted,
                 rsqrt_var_add_epsilon_broadcasted);

  scale_times_rsqrt_var_add_epsilon =
      add(Mean(elements_per_feature, scale_times_rsqrt_var_add_epsilon, add));

  auto i1 = add_binary(activation_shape, HloOpcode::kMultiply, grad_output,
                       add(HloInstruction::CreateBroadcast(
                           activation_shape, elements_per_feature, {})));

  // I6 = I1 - I2 - I5
  auto i6 = add_binary(
      activation_shape, HloOpcode::kSubtract,
      add_binary(activation_shape, HloOpcode::kSubtract, i1, i2), i5);

  // Grad[X] = scale * rsqrt[Var[X] + epsilon] * 1/N * I6.
  auto grad_activation = add_binary(activation_shape, HloOpcode::kMultiply,
                                    scale_times_rsqrt_var_add_epsilon, i6);
  auto tuple =
      HloInstruction::CreateTuple({grad_activation, grad_scale, grad_beta});
  if (batch_norm->has_sharding()) {
    const HloSharding& sharding = batch_norm->sharding();
    int64_t instruction_count_after = computation_->instruction_count();
    CHECK_EQ(instruction_count_after,
             instruction_count_before + added_instructions.size());
    HloSharding activation_sharding =
        sharding.GetAsShapeTree(batch_norm->shape()).element({0});
    auto unique_device = batch_norm->sharding_unique_device();
    HloSharding default_sharding =
        unique_device.has_value()
            ? HloSharding::AssignDevice(unique_device.value())
            : HloSharding::Replicate();
    for (HloInstruction* inst : added_instructions) {
      if (ShapeUtil::Equal(inst->shape(), activation_shape)) {
        inst->set_sharding(activation_sharding);
      } else {
        inst->set_sharding(default_sharding);
      }
    }
    tuple->set_sharding(sharding);
  }

  TF_CHECK_OK(ReplaceWithNewInstruction(batch_norm, std::move(tuple)));

  return Status::OK();
}

StatusOr<bool> BatchNormExpander::Run(HloModule* module) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbatchnorm_expanderDTcc mht_13(mht_13_v, 772, "", "./tensorflow/compiler/xla/service/batchnorm_expander.cc", "BatchNormExpander::Run");

  XLA_VLOG_LINES(2, "BatchNormExpander::Run(), before:\n" + module->ToString());
  bool changed = false;
  for (HloComputation* computation : module->MakeNonfusionComputations()) {
    if (BatchNormExpanderVisitor::Run(computation, rewrite_training_op_,
                                      rewrite_inference_op_,
                                      rewrite_grad_op_)) {
      changed = true;
    }
  }
  XLA_VLOG_LINES(2, "BatchNormExpander::Run(), after:\n" + module->ToString());
  return changed;
}

}  // namespace xla
