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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSrng_expanderDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSrng_expanderDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSrng_expanderDTcc() {
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

/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/rng_expander.h"

#include <random>

#include "tensorflow/compiler/xla/client/lib/prng.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"

namespace xla {

namespace {

int64_t GlobalRandomValue() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSrng_expanderDTcc mht_0(mht_0_v, 199, "", "./tensorflow/compiler/xla/service/rng_expander.cc", "GlobalRandomValue");

  static auto* mu = new absl::Mutex();
  static std::mt19937_64 rng{42};
  absl::MutexLock l(mu);
  return rng();
}

int64_t GetNumberOf32bitUnits(const Shape& shape) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSrng_expanderDTcc mht_1(mht_1_v, 209, "", "./tensorflow/compiler/xla/service/rng_expander.cc", "GetNumberOf32bitUnits");

  int64_t bit_width = primitive_util::BitWidth(shape.element_type());
  CHECK(bit_width == 32 || bit_width == 64);
  int64_t num_elems = ShapeUtil::ElementsIn(shape);
  return num_elems * (bit_width / 32);
}

StatusOr<HloInstruction*> ConvertSmallFpRngToF32Rng(HloInstruction* rng) {
  CHECK_EQ(rng->opcode(), HloOpcode::kRng);
  PrimitiveType primitive_type = rng->shape().element_type();
  CHECK(primitive_type == F16 || primitive_type == BF16);

  std::vector<HloInstruction*> new_operands;
  absl::c_transform(rng->operands(), std::back_inserter(new_operands),
                    [&](HloInstruction* operand) {
                      CHECK_EQ(operand->shape().element_type(), primitive_type);
                      return MakeConvertToHlo(operand, F32);
                    });

  Shape shape = ShapeUtil::ChangeElementType(rng->shape(), F32);
  HloComputation* computation = rng->parent();
  HloCloneContext context(computation->parent());
  HloInstruction* new_rng = computation->AddInstruction(
      rng->CloneWithNewOperands(shape, new_operands, &context));
  TF_RETURN_IF_ERROR(new_rng->CopyAllControlDepsFrom(rng));

  TF_RETURN_IF_ERROR(
      rng->ReplaceAllUsesWith(MakeConvertToHlo(new_rng, primitive_type)));
  TF_RETURN_IF_ERROR(rng->DropAllControlDeps());

  // Since rng is a side-effecting instruction, we can't rely on DCE to remove
  // it.
  TF_RETURN_IF_ERROR(computation->RemoveInstruction(rng));

  return new_rng;
}

StatusOr<HloComputation*> GetComputationForRng(HloInstruction* rng) {
  XlaBuilder builder("rng");
  const Shape u64_shape = ShapeUtil::MakeShape(xla::U64, {});
  const Shape u128_shape = ShapeUtil::MakeShape(xla::U64, {2});
  const Shape& result_shape = rng->shape();

  XlaOp key = Parameter(&builder, 0, u64_shape, "key");
  XlaOp state = Parameter(&builder, 1, u128_shape, "state");
  XlaOp a_or_mean =
      Parameter(&builder, 2, rng->operand(0)->shape(), "a_or_mean");
  XlaOp b_or_sigma =
      Parameter(&builder, 3, rng->operand(1)->shape(), "b_or_sigma");

  auto generator = [](xla::XlaOp key, xla::XlaOp state,
                      const xla::Shape& shape) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSrng_expanderDTcc mht_2(mht_2_v, 263, "", "./tensorflow/compiler/xla/service/rng_expander.cc", "lambda");

    return PhiloxBitGenerator(key, state, shape);
  };

  XlaOp result;
  if (rng->random_distribution() == RNG_NORMAL) {
    result =
        NormalFloatingPointDistribution(key, state, generator, result_shape)
            .value;
    // Transform standard normal distribution to normal distribution with the
    // given mean and standard deviation.
    result = a_or_mean + (b_or_sigma * result);
  } else {
    CHECK_EQ(rng->random_distribution(), RNG_UNIFORM);
    if (primitive_util::IsFloatingPointType(result_shape.element_type())) {
      result = UniformFloatingPointDistribution(
                   key, state, generator, a_or_mean, b_or_sigma, result_shape)
                   .value;
    } else {
      result = UniformIntDistribution(key, state, generator, a_or_mean,
                                      b_or_sigma, result_shape)
                   .value;
    }
  }

  TF_ASSIGN_OR_RETURN(XlaComputation xla_computation, builder.Build());

  TF_ASSIGN_OR_RETURN(ProgramShape program_shape,
                      xla_computation.GetProgramShape());
  HloModuleConfig config(program_shape);
  TF_ASSIGN_OR_RETURN(auto new_module, HloModule::CreateFromProto(
                                           xla_computation.proto(), config));
  HloModule* module = rng->parent()->parent();
  HloCloneContext context(module);
  return module->DeepCloneComputation(new_module->entry_computation(),
                                      &context);
}

}  // namespace

bool RngExpander::InstructionMatchesPattern(HloInstruction* instruction) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSrng_expanderDTcc mht_3(mht_3_v, 306, "", "./tensorflow/compiler/xla/service/rng_expander.cc", "RngExpander::InstructionMatchesPattern");

  return instruction->opcode() == HloOpcode::kRng;
}

StatusOr<HloInstruction*> RngExpander::ExpandInstruction(HloInstruction* rng) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSrng_expanderDTcc mht_4(mht_4_v, 313, "", "./tensorflow/compiler/xla/service/rng_expander.cc", "RngExpander::ExpandInstruction");

  VLOG(2) << "Expand rng instruction " << rng->ToString();
  PrimitiveType old_primitive_type = rng->shape().element_type();
  if (primitive_util::BitWidth(old_primitive_type) < 32) {
    TF_ASSIGN_OR_RETURN(rng, ConvertSmallFpRngToF32Rng(rng));
  }
  HloComputation*& rng_computation = expanded_rng_instructions_[std::make_tuple(
      rng->random_distribution(), rng->shape(), rng->operand(0)->shape(),
      rng->operand(1)->shape())];
  if (!rng_computation) {
    TF_ASSIGN_OR_RETURN(rng_computation, GetComputationForRng(rng));
  }
  HloComputation* computation = rng->parent();

  // A random number generated by the per module random number generator.
  int64_t module_random_value = rng->GetModule()->RandomNew64();

  // A value specified by the configuration or generated by a global random
  // number generator.
  int64_t module_config_seed = rng->parent()->parent()->config().seed();
  int64_t global_random_value =
      module_config_seed != 0 ? module_config_seed : GlobalRandomValue();

  // Construct the key using the two random values above.
  HloInstruction* key = MakeR0ConstantHlo<uint64_t>(
      computation, module_random_value ^ global_random_value);

  const Shape u128_shape = ShapeUtil::MakeShape(xla::U64, {2});
  HloInstruction* state =
      computation->AddInstruction(HloInstruction::CreateRngGetAndUpdateState(
          u128_shape, GetNumberOf32bitUnits(rng->shape())));

  VLOG(2) << "Rng key " << key->ToString();
  VLOG(2) << "Rng state " << state->ToString();

  HloInstruction* new_rng =
      computation->AddInstruction(HloInstruction::CreateCall(
          rng->shape(),
          {key, state, rng->mutable_operand(0), rng->mutable_operand(1)},
          rng_computation));

  TF_RETURN_IF_ERROR(new_rng->CopyAllControlDepsFrom(rng));

  TF_RETURN_IF_ERROR(rng->ReplaceAllUsesWith(new_rng));
  TF_RETURN_IF_ERROR(rng->DropAllControlDeps());

  // Since rng is a side-effecting instruction, we can't rely on DCE to remove
  // it.
  TF_RETURN_IF_ERROR(computation->RemoveInstruction(rng));

  // Returns nullptr to OpExpanderPass::Run to indicate the old rng instruction
  // has been replaced with the new rng instruction.
  return nullptr;
}

}  // namespace xla
