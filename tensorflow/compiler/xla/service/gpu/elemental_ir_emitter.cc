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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSelemental_ir_emitterDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSelemental_ir_emitterDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSelemental_ir_emitterDTcc() {
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

#include "tensorflow/compiler/xla/service/gpu/elemental_ir_emitter.h"

#include <stddef.h>

#include <vector>

#include "tensorflow/core/platform/logging.h"
// IWYU pragma: no_include "llvm/IR/Attributes.gen.inc"
// IWYU pragma: no_include "llvm/IR/Intrinsics.gen.inc"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/APInt.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/service/gpu/backend_configs.pb.h"
#include "tensorflow/compiler/xla/service/gpu/target_util.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/llvm_ir/ir_array.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_loop.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/service/llvm_ir/math_ops.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/window_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {
namespace gpu {

using absl::StrAppend;
using llvm_ir::IrArray;
using llvm_ir::IrName;
using llvm_ir::SetToFirstInsertPoint;

namespace {
// Returns whether operand is a floating-point literal with the given value.
bool IsFPLiteralWithValue(const HloInstruction* operand, float value) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSelemental_ir_emitterDTcc mht_0(mht_0_v, 230, "", "./tensorflow/compiler/xla/service/gpu/elemental_ir_emitter.cc", "IsFPLiteralWithValue");

  if (operand->opcode() == HloOpcode::kConstant &&
      operand->literal().IsAllFloat(value)) {
    return true;
  }
  return operand->opcode() == HloOpcode::kBroadcast &&
         IsFPLiteralWithValue(operand->operand(0), value);
}
}  // namespace

GpuElementalIrEmitter::GpuElementalIrEmitter(
    const HloModuleConfig& hlo_module_config, llvm::Module* module,
    llvm::IRBuilder<>* b, NestedComputer compute_nested)
    : ElementalIrEmitter(module, b),
      hlo_module_config_(hlo_module_config),
      compute_nested_(std::move(compute_nested)) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSelemental_ir_emitterDTcc mht_1(mht_1_v, 248, "", "./tensorflow/compiler/xla/service/gpu/elemental_ir_emitter.cc", "GpuElementalIrEmitter::GpuElementalIrEmitter");
}

StatusOr<llvm::Value*> GpuElementalIrEmitter::EmitDeviceMathCall(
    TargetDeviceFunctionID funcid, absl::Span<llvm::Value* const> operands,
    absl::Span<const PrimitiveType> input_types, PrimitiveType output_type,
    absl::string_view name) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSelemental_ir_emitterDTcc mht_2(mht_2_v, 257, "", "./tensorflow/compiler/xla/service/gpu/elemental_ir_emitter.cc", "GpuElementalIrEmitter::EmitDeviceMathCall");

  // Device functions dont have f16 math functions, so we convert the operands
  // to f32 before calling the function and then convert the result back to f16.
  bool cast_result_to_fp16 = false;
  std::vector<llvm::Value*> converted_operands(operands.begin(),
                                               operands.end());
  std::vector<PrimitiveType> converted_input_types(input_types.begin(),
                                                   input_types.end());
  switch (output_type) {
    case F16:
      cast_result_to_fp16 = true;
      for (int64_t i = 0; i < operands.size(); ++i) {
        if (input_types[i] == F16) {
          converted_operands[i] =
              FPCast(converted_operands[i], b()->getFloatTy());
          converted_input_types[i] = F32;
        }
      }
      output_type = F32;
      ABSL_FALLTHROUGH_INTENDED;
    case F32:
      break;
    case F64:
      break;
    default:
      return Unimplemented("Bad type for device math call: %s",
                           PrimitiveType_Name(output_type));
  }
  const std::string& munged_callee =
      ObtainDeviceFunctionName(funcid, output_type, b());
  llvm::Value* result = EmitMathCall(munged_callee, converted_operands,
                                     converted_input_types, output_type, name)
                            .ValueOrDie();
  if (cast_result_to_fp16) {
    result = FPCast(result, b()->getHalfTy());
  }
  return result;
}

StatusOr<llvm::Value*> GpuElementalIrEmitter::EmitLlvmIntrinsicMathCall(
    const std::string& callee_name, absl::Span<llvm::Value* const> operands,
    absl::Span<const PrimitiveType> input_types, PrimitiveType output_type) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("callee_name: \"" + callee_name + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSelemental_ir_emitterDTcc mht_3(mht_3_v, 302, "", "./tensorflow/compiler/xla/service/gpu/elemental_ir_emitter.cc", "GpuElementalIrEmitter::EmitLlvmIntrinsicMathCall");

  // llvm intrinsics differentiate between half/float/double functions via
  // the suffixes ".f16", ".f32" and ".f64".
  std::string munged_callee = callee_name;
  switch (output_type) {
    case F16:
      StrAppend(&munged_callee, ".f16");
      break;
    case F32:
      StrAppend(&munged_callee, ".f32");
      break;
    case F64:
      StrAppend(&munged_callee, ".f64");
      break;
    default:
      return Unimplemented("Bad type for llvm intrinsic math call: %s",
                           PrimitiveType_Name(output_type));
  }
  return EmitMathCall(munged_callee, operands, input_types, output_type);
}

StatusOr<llvm::Value*> GpuElementalIrEmitter::EmitMathCall(
    const std::string& callee_name, absl::Span<llvm::Value* const> operands,
    absl::Span<const PrimitiveType> input_types, PrimitiveType output_type,
    absl::string_view name) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("callee_name: \"" + callee_name + "\"");
   mht_4_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSelemental_ir_emitterDTcc mht_4(mht_4_v, 331, "", "./tensorflow/compiler/xla/service/gpu/elemental_ir_emitter.cc", "GpuElementalIrEmitter::EmitMathCall");

  // Binary math functions transform are of type [T] -> T.
  for (PrimitiveType input_type : input_types) {
    if (output_type != input_type) {
      return Unimplemented("Input type != output type: %s != %s",
                           PrimitiveType_Name(input_type),
                           PrimitiveType_Name(output_type));
    }
  }

  return EmitDeviceFunctionCall(
      callee_name, operands, input_types, output_type,
      {llvm::Attribute::ReadNone, llvm::Attribute::NoUnwind}, b(), name);
}

llvm_ir::IrArray::Index GpuElementalIrEmitter::GetSourceIndexOfBitcast(
    const llvm_ir::IrArray::Index& index, const HloInstruction* hlo) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSelemental_ir_emitterDTcc mht_5(mht_5_v, 350, "", "./tensorflow/compiler/xla/service/gpu/elemental_ir_emitter.cc", "GpuElementalIrEmitter::GetSourceIndexOfBitcast");

  Shape shape = hlo->shape();
  Shape operand_shape = hlo->operand(0)->shape();

  // Decode the layout of the shape from the Protobugs attached to
  // backend_config_.
  BitcastBackendConfig bitcast_config;
  CHECK(bitcast_config.ParseFromString(hlo->raw_backend_config_string()));

  *shape.mutable_layout() =
      xla::Layout::CreateFromProto(bitcast_config.result_layout());
  *operand_shape.mutable_layout() =
      xla::Layout::CreateFromProto(bitcast_config.source_layout());
  return index.SourceIndexOfBitcast(shape, operand_shape, b());
}

StatusOr<llvm::Value*> GpuElementalIrEmitter::EmitFloatBinaryOp(
    const HloInstruction* op, llvm::Value* lhs_value, llvm::Value* rhs_value) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSelemental_ir_emitterDTcc mht_6(mht_6_v, 370, "", "./tensorflow/compiler/xla/service/gpu/elemental_ir_emitter.cc", "GpuElementalIrEmitter::EmitFloatBinaryOp");

  PrimitiveType lhs_input_type = op->operand(0)->shape().element_type();
  PrimitiveType rhs_input_type = op->operand(1)->shape().element_type();
  PrimitiveType output_type = op->shape().element_type();
  HloOpcode opcode = op->opcode();

  if (hlo_module_config_.debug_options().xla_gpu_enable_fast_min_max() &&
      (opcode == HloOpcode::kMaximum || opcode == HloOpcode::kMinimum)) {
    return llvm_ir::EmitCallToIntrinsic(
        opcode == HloOpcode::kMaximum ? llvm::Intrinsic::maxnum
                                      : llvm::Intrinsic::minnum,
        {lhs_value, rhs_value}, {lhs_value->getType()}, b());
  }

  switch (op->opcode()) {
    case HloOpcode::kRemainder: {
      return EmitDeviceMathCall(TargetDeviceFunctionID::kFmod,
                                {lhs_value, rhs_value},
                                {lhs_input_type, rhs_input_type}, output_type);
    }
    case HloOpcode::kPower: {
      return EmitPowerOp(op, lhs_value, rhs_value);
    }
    default:
      return ElementalIrEmitter::EmitFloatBinaryOp(op, lhs_value, rhs_value);
  }
}

StatusOr<llvm::Value*> GpuElementalIrEmitter::EmitPowerOp(
    const HloInstruction* op, llvm::Value* lhs_value, llvm::Value* rhs_value) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSelemental_ir_emitterDTcc mht_7(mht_7_v, 402, "", "./tensorflow/compiler/xla/service/gpu/elemental_ir_emitter.cc", "GpuElementalIrEmitter::EmitPowerOp");

  CHECK_EQ(op->opcode(), HloOpcode::kPower);
  PrimitiveType lhs_input_type = op->operand(0)->shape().element_type();
  PrimitiveType rhs_input_type = op->operand(1)->shape().element_type();
  PrimitiveType output_type = op->shape().element_type();
  return EmitDeviceMathCall(TargetDeviceFunctionID::kPow,
                            {lhs_value, rhs_value},
                            {lhs_input_type, rhs_input_type}, output_type);
}

StatusOr<llvm::Value*> GpuElementalIrEmitter::EmitLog(PrimitiveType prim_type,
                                                      llvm::Value* value) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSelemental_ir_emitterDTcc mht_8(mht_8_v, 416, "", "./tensorflow/compiler/xla/service/gpu/elemental_ir_emitter.cc", "GpuElementalIrEmitter::EmitLog");

  return EmitDeviceMathCall(TargetDeviceFunctionID::kLog, {value}, {prim_type},
                            prim_type);
}

StatusOr<llvm::Value*> GpuElementalIrEmitter::EmitLog1p(PrimitiveType prim_type,
                                                        llvm::Value* value) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSelemental_ir_emitterDTcc mht_9(mht_9_v, 425, "", "./tensorflow/compiler/xla/service/gpu/elemental_ir_emitter.cc", "GpuElementalIrEmitter::EmitLog1p");

  return EmitDeviceMathCall(TargetDeviceFunctionID::kLog1p, {value},
                            {prim_type}, prim_type);
}

StatusOr<llvm::Value*> GpuElementalIrEmitter::EmitSin(PrimitiveType prim_type,
                                                      llvm::Value* value) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSelemental_ir_emitterDTcc mht_10(mht_10_v, 434, "", "./tensorflow/compiler/xla/service/gpu/elemental_ir_emitter.cc", "GpuElementalIrEmitter::EmitSin");

  return EmitDeviceMathCall(TargetDeviceFunctionID::kSin, {value}, {prim_type},
                            prim_type);
}

StatusOr<llvm::Value*> GpuElementalIrEmitter::EmitCos(PrimitiveType prim_type,
                                                      llvm::Value* value) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSelemental_ir_emitterDTcc mht_11(mht_11_v, 443, "", "./tensorflow/compiler/xla/service/gpu/elemental_ir_emitter.cc", "GpuElementalIrEmitter::EmitCos");

  return EmitDeviceMathCall(TargetDeviceFunctionID::kCos, {value}, {prim_type},
                            prim_type);
}

StatusOr<llvm::Value*> GpuElementalIrEmitter::EmitExp(
    PrimitiveType prim_type, llvm::Value* value, absl::string_view /*name*/) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSelemental_ir_emitterDTcc mht_12(mht_12_v, 452, "", "./tensorflow/compiler/xla/service/gpu/elemental_ir_emitter.cc", "GpuElementalIrEmitter::EmitExp");

  return EmitDeviceMathCall(TargetDeviceFunctionID::kExp, {value}, {prim_type},
                            prim_type);
}

StatusOr<llvm::Value*> GpuElementalIrEmitter::EmitExpm1(PrimitiveType prim_type,
                                                        llvm::Value* value) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSelemental_ir_emitterDTcc mht_13(mht_13_v, 461, "", "./tensorflow/compiler/xla/service/gpu/elemental_ir_emitter.cc", "GpuElementalIrEmitter::EmitExpm1");

  return EmitDeviceMathCall(TargetDeviceFunctionID::kExpm1, {value},
                            {prim_type}, prim_type);
}

StatusOr<llvm::Value*> GpuElementalIrEmitter::EmitPow(PrimitiveType prim_type,
                                                      llvm::Value* lhs,
                                                      llvm::Value* rhs,
                                                      absl::string_view name) {
   std::vector<std::string> mht_14_v;
   mht_14_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSelemental_ir_emitterDTcc mht_14(mht_14_v, 473, "", "./tensorflow/compiler/xla/service/gpu/elemental_ir_emitter.cc", "GpuElementalIrEmitter::EmitPow");

  return EmitDeviceMathCall(TargetDeviceFunctionID::kPow, {lhs, rhs},
                            {prim_type, prim_type}, prim_type, name);
}

StatusOr<llvm::Value*> GpuElementalIrEmitter::EmitSqrt(PrimitiveType prim_type,
                                                       llvm::Value* value) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSelemental_ir_emitterDTcc mht_15(mht_15_v, 482, "", "./tensorflow/compiler/xla/service/gpu/elemental_ir_emitter.cc", "GpuElementalIrEmitter::EmitSqrt");

  return EmitDeviceMathCall(TargetDeviceFunctionID::kSqrt, {value}, {prim_type},
                            prim_type);
}

StatusOr<llvm::Value*> GpuElementalIrEmitter::EmitRsqrt(PrimitiveType prim_type,
                                                        llvm::Value* value) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSelemental_ir_emitterDTcc mht_16(mht_16_v, 491, "", "./tensorflow/compiler/xla/service/gpu/elemental_ir_emitter.cc", "GpuElementalIrEmitter::EmitRsqrt");

  return EmitDeviceMathCall(TargetDeviceFunctionID::kRsqrt, {value},
                            {prim_type}, prim_type);
}

StatusOr<llvm::Value*> GpuElementalIrEmitter::EmitAtan2(
    PrimitiveType prim_type, llvm::Value* lhs, llvm::Value* rhs,
    absl::string_view name) {
   std::vector<std::string> mht_17_v;
   mht_17_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSelemental_ir_emitterDTcc mht_17(mht_17_v, 502, "", "./tensorflow/compiler/xla/service/gpu/elemental_ir_emitter.cc", "GpuElementalIrEmitter::EmitAtan2");

  return EmitDeviceMathCall(TargetDeviceFunctionID::kAtan2, {lhs, rhs},
                            {prim_type, prim_type}, prim_type, name);
}

StatusOr<llvm::Value*> GpuElementalIrEmitter::EmitTanh(PrimitiveType prim_type,
                                                       llvm::Value* value) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSelemental_ir_emitterDTcc mht_18(mht_18_v, 511, "", "./tensorflow/compiler/xla/service/gpu/elemental_ir_emitter.cc", "GpuElementalIrEmitter::EmitTanh");

  // When F64 is being requested, assume performance is less important and use
  // the more numerically precise tanh function.
  if (prim_type == F64) {
    return EmitDeviceMathCall(TargetDeviceFunctionID::kTanh, {value},
                              {prim_type}, prim_type);
  }

  // Emit a fast approximation of tanh instead of calling __nv_tanh.
  // __nv_tanh is particularly bad because it contains branches, thus
  // preventing LLVM's load-store vectorizer from working its magic across a
  // function which contains tanh calls.
  //
  // This routine isn't numerically precise, but it's good enough for ML.

  // Upcast F16 to F32 if necessary.
  llvm::Type* type = prim_type == F16 ? b()->getFloatTy() : value->getType();
  llvm::Value* input = FPCast(value, type);

  // If |value| >= kMaxValue, tanh() is set to -1.0 or 1.0.
  constexpr double kMaxValue = 20.0;
  auto max_value = llvm::ConstantFP::get(type, kMaxValue);
  llvm::Value* abs_value =
      llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::fabs, {input}, {type}, b());

  llvm::Value* fast_tanh = llvm_ir::EmitFastTanh(b(), input);
  auto one = llvm::ConstantFP::get(type, 1.0);
  auto one_with_sign = llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::copysign,
                                                    {one, input}, {type}, b());
  return FPCast(Select(FCmpULT(abs_value, max_value), fast_tanh, one_with_sign),
                value->getType(), "tanh");
}

StatusOr<llvm::Value*> GpuElementalIrEmitter::EmitComplexAbs(
    PrimitiveType prim_type, llvm::Value* value) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSelemental_ir_emitterDTcc mht_19(mht_19_v, 548, "", "./tensorflow/compiler/xla/service/gpu/elemental_ir_emitter.cc", "GpuElementalIrEmitter::EmitComplexAbs");

  return EmitDeviceMathCall(TargetDeviceFunctionID::kHypot,
                            {EmitExtractReal(value), EmitExtractImag(value)},
                            {prim_type, prim_type}, prim_type);
}

llvm::Value* GpuElementalIrEmitter::EmitThreadId() {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSelemental_ir_emitterDTcc mht_20(mht_20_v, 557, "", "./tensorflow/compiler/xla/service/gpu/elemental_ir_emitter.cc", "GpuElementalIrEmitter::EmitThreadId");

  llvm::Value* block_id = IntCast(
      EmitCallToTargetIntrinsic(TargetIntrinsicID::kBlockIdx, {}, {}, b()),
      b()->getIntNTy(128), /*isSigned=*/true, "block.id");
  llvm::Value* thread_id_in_block = IntCast(
      EmitCallToTargetIntrinsic(TargetIntrinsicID::kThreadIdx, {}, {}, b()),
      b()->getIntNTy(128), /*isSigned=*/true, "thread.id");
  llvm::Value* threads_per_block = IntCast(
      EmitCallToTargetIntrinsic(TargetIntrinsicID::kBlockDimx, {}, {}, b()),
      b()->getIntNTy(128), /*isSigned=*/true, "threads_per_block");
  return NSWAdd(NSWMul(block_id, threads_per_block), thread_id_in_block);
}

}  // namespace gpu
}  // namespace xla
