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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbitcast_dtypes_expanderDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbitcast_dtypes_expanderDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbitcast_dtypes_expanderDTcc() {
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

/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/bitcast_dtypes_expander.h"

#include "absl/algorithm/container.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/client/lib/broadcast.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

StatusOr<HloInstruction*> BitcastDtypesExpander::ExpandInstruction(
    HloInstruction* instruction) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbitcast_dtypes_expanderDTcc mht_0(mht_0_v, 208, "", "./tensorflow/compiler/xla/service/bitcast_dtypes_expander.cc", "BitcastDtypesExpander::ExpandInstruction");

  HloInstruction* input = instruction->mutable_operand(0);
  const Shape& from_shape = input->shape();
  const Shape& to_shape = instruction->shape();

  int input_bit_width = primitive_util::BitWidth(from_shape.element_type());
  int output_bit_width = primitive_util::BitWidth(to_shape.element_type());

  PrimitiveType input_logical_type =
      primitive_util::UnsignedIntegralTypeForBitWidth(input_bit_width);
  PrimitiveType output_logical_type =
      primitive_util::UnsignedIntegralTypeForBitWidth(output_bit_width);

  if (input_bit_width == output_bit_width) {
    return instruction;
  }

  std::string name =
      absl::StrFormat("xla.bitcast_convert_%s_2_%s", from_shape.ToString(),
                      to_shape.ToString());

  // Note: we are duplicating a hack from `cholesky_expander` to build a
  // computation using XlaBuilder.
  HloModule* module = instruction->parent()->parent();
  HloComputation*& computation =
      computation_cache_.emplace(name, nullptr).first->second;
  if (!computation) {
    XlaBuilder b(name);
    XlaOp input = Parameter(&b, 0, instruction->operand(0)->shape(), "a");

    if (input_bit_width > output_bit_width) {
      std::vector<int64_t> broadcasted_input_shape(
          from_shape.dimensions().begin(), from_shape.dimensions().end());
      std::vector<int64_t> reshaped_input_shape(from_shape.dimensions().begin(),
                                                from_shape.dimensions().end());
      broadcasted_input_shape.push_back(input_bit_width / output_bit_width);
      reshaped_input_shape.push_back(1);
      int64_t output_bit_width_mask = (1l << output_bit_width) - 1;

      TF_ASSIGN_OR_RETURN(input,
                          BroadcastTo(Reshape(input, reshaped_input_shape),
                                      broadcasted_input_shape));
      input = BitcastConvertType(input, input_logical_type);
      TF_ASSIGN_OR_RETURN(Shape input_shape, b.GetShape(input));
      XlaOp iota = Iota(&b, input_shape, input_shape.dimensions_size() - 1);
      XlaOp iota_m = Mul(ScalarLike(input, output_bit_width), iota);
      input = And(ShiftRightLogical(input, iota_m),
                  ScalarLike(input, output_bit_width_mask));
      input = ConvertElementType(input, output_logical_type);
    } else if (input_bit_width < output_bit_width) {
      input = BitcastConvertType(input, input_logical_type);
      input = ConvertElementType(input, output_logical_type);

      // Shift bits and OR them together to reduce the inner dimension.
      XlaOp iota_m = Mul(
          ConstantR0WithType(&b, output_logical_type, input_bit_width),
          Iota(&b,
               ShapeUtil::ChangeElementType(from_shape, output_logical_type),
               from_shape.rank() - 1));
      input = ShiftLeft(input, iota_m);
      input = Reduce(input, Zero(&b, output_logical_type),
                     CreateScalarOrComputation(output_logical_type, &b),
                     {from_shape.rank() - 1});
    }

    BitcastConvertType(input, to_shape.element_type());

    TF_ASSIGN_OR_RETURN(XlaComputation xla_computation, b.Build());
    TF_ASSIGN_OR_RETURN(ProgramShape program_shape,
                        xla_computation.GetProgramShape());
    HloModuleConfig config(program_shape);
    TF_ASSIGN_OR_RETURN(auto new_module, HloModule::CreateFromProto(
                                             xla_computation.proto(), config));
    HloCloneContext context(module);
    computation =
        module->DeepCloneComputation(new_module->entry_computation(), &context);
  }

  return instruction->parent()->AddInstruction(HloInstruction::CreateCall(
      instruction->shape(), instruction->operands(), computation));
}

bool BitcastDtypesExpander::InstructionMatchesPattern(
    HloInstruction* instruction) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbitcast_dtypes_expanderDTcc mht_1(mht_1_v, 294, "", "./tensorflow/compiler/xla/service/bitcast_dtypes_expander.cc", "BitcastDtypesExpander::InstructionMatchesPattern");

  return instruction->opcode() == HloOpcode::kBitcastConvert &&
         primitive_util::BitWidth(instruction->shape().element_type()) !=
             primitive_util::BitWidth(
                 instruction->operand(0)->shape().element_type());
}

}  // namespace xla
