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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSrng_bit_generator_expanderDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSrng_bit_generator_expanderDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSrng_bit_generator_expanderDTcc() {
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

#include "tensorflow/compiler/xla/service/rng_bit_generator_expander.h"

#include "tensorflow/compiler/xla/client/lib/prng.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace xla {
namespace {

XlaOp GetPhiloxStateOp(XlaOp input_state, const Shape& state_shape) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSrng_bit_generator_expanderDTcc mht_0(mht_0_v, 204, "", "./tensorflow/compiler/xla/service/rng_bit_generator_expander.cc", "GetPhiloxStateOp");

  if (state_shape.dimensions(0) >= 3) {
    return Slice(input_state, {1}, {3}, {1});
  }
  return Rev(input_state, {0});
}

XlaOp GetPhiloxOutputStateOp(XlaOp output_state, const Shape& state_shape) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSrng_bit_generator_expanderDTcc mht_1(mht_1_v, 214, "", "./tensorflow/compiler/xla/service/rng_bit_generator_expander.cc", "GetPhiloxOutputStateOp");

  if (state_shape.dimensions(0) < 3) {
    output_state = Slice(output_state, {0}, {1}, {1});
  }
  return output_state;
}

}  // namespace

bool RngBitGeneratorExpander::InstructionMatchesPattern(
    HloInstruction* instruction) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSrng_bit_generator_expanderDTcc mht_2(mht_2_v, 227, "", "./tensorflow/compiler/xla/service/rng_bit_generator_expander.cc", "RngBitGeneratorExpander::InstructionMatchesPattern");

  return instruction->opcode() == HloOpcode::kRngBitGenerator;
}

StatusOr<HloComputation*> RngBitGeneratorExpander::GetGeneratorComputation(
    const Shape& data_shape, const Shape& state_shape,
    RandomAlgorithm algorithm, HloModule* module) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSrng_bit_generator_expanderDTcc mht_3(mht_3_v, 236, "", "./tensorflow/compiler/xla/service/rng_bit_generator_expander.cc", "RngBitGeneratorExpander::GetGeneratorComputation");

  RngGeneratorKey cache_key{data_shape, state_shape, algorithm, module};
  auto it = computation_cache_.find(cache_key);
  if (it != computation_cache_.end()) {
    return it->second;
  }

  XlaBuilder builder("rng");
  XlaOp state_param = Parameter(&builder, 0, state_shape, "state");
  XlaOp key_op = Reshape(Slice(state_param, {0}, {1}, {1}), {});
  RngOutput output;
  switch (algorithm) {
    case RandomAlgorithm::RNG_THREE_FRY:
      output = ThreeFryBitGenerator(key_op, Slice(state_param, {1}, {2}, {1}),
                                    data_shape);
      break;
    case RandomAlgorithm::RNG_PHILOX:
      output = PhiloxBitGenerator(
          key_op, GetPhiloxStateOp(state_param, state_shape), data_shape);
      output.state = GetPhiloxOutputStateOp(output.state, state_shape);
      break;
    default:
      return Unimplemented("Unsupported random algorthm: %s",
                           RandomAlgorithm_Name(algorithm));
  }

  XlaOp final_state =
      ConcatInDim(&builder, {Reshape(key_op, {1}), output.state}, 0);
  Tuple(&builder, {final_state, output.value});
  TF_ASSIGN_OR_RETURN(XlaComputation xla_computation, builder.Build());

  TF_ASSIGN_OR_RETURN(ProgramShape program_shape,
                      xla_computation.GetProgramShape());
  HloModuleConfig config(program_shape);
  TF_ASSIGN_OR_RETURN(auto new_module, HloModule::CreateFromProto(
                                           xla_computation.proto(), config));
  HloCloneContext context(module);
  HloComputation* new_computation =
      module->DeepCloneComputation(new_module->entry_computation(), &context);
  computation_cache_.emplace(cache_key, new_computation);
  return new_computation;
}

StatusOr<HloInstruction*> RngBitGeneratorExpander::ExpandInstruction(
    HloInstruction* hlo) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSrng_bit_generator_expanderDTcc mht_4(mht_4_v, 283, "", "./tensorflow/compiler/xla/service/rng_bit_generator_expander.cc", "RngBitGeneratorExpander::ExpandInstruction");

  HloRngBitGeneratorInstruction* rng = Cast<HloRngBitGeneratorInstruction>(hlo);
  RandomAlgorithm algorithm = rng->algorithm();
  if (algorithm == RandomAlgorithm::RNG_DEFAULT) {
    algorithm = default_algorithm_;
  }

  HloModule* module = hlo->parent()->parent();
  const Shape& data_shape = rng->shape().tuple_shapes(1);
  const Shape& state_shape = rng->operand(0)->shape();
  TF_ASSIGN_OR_RETURN(
      HloComputation * generator_computation,
      GetGeneratorComputation(data_shape, state_shape, algorithm, module));
  return hlo->parent()->AddInstruction(HloInstruction::CreateCall(
      ShapeUtil::MakeTupleShape({state_shape, data_shape}),
      {hlo->mutable_operand(0)}, generator_computation));
}

}  // namespace xla
