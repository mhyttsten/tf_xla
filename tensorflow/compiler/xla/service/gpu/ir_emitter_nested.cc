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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emitter_nestedDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emitter_nestedDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emitter_nestedDTcc() {
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

#include "tensorflow/compiler/xla/service/gpu/ir_emitter_nested.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_constants.h"
#include "tensorflow/compiler/xla/service/gpu/hlo_to_ir_bindings.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emitter_context.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/llvm_ir/buffer_assignment_util.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/service/llvm_ir/tuple_ops.h"
#include "tensorflow/compiler/xla/service/name_uniquer.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {
namespace gpu {

IrEmitterNested::IrEmitterNested(const HloModuleConfig& hlo_module_config,
                                 const HloComputation& nested_computation,
                                 IrEmitterContext* ir_emitter_context)
    : IrEmitter(hlo_module_config, ir_emitter_context, /*is_nested=*/true),
      nested_computation_(nested_computation) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emitter_nestedDTcc mht_0(mht_0_v, 216, "", "./tensorflow/compiler/xla/service/gpu/ir_emitter_nested.cc", "IrEmitterNested::IrEmitterNested");
}

StatusOr<std::unique_ptr<IrEmitterNested>> IrEmitterNested::Create(
    const HloModuleConfig& hlo_module_config,
    const HloComputation& nested_computation,
    IrEmitterContext* ir_emitter_context) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emitter_nestedDTcc mht_1(mht_1_v, 224, "", "./tensorflow/compiler/xla/service/gpu/ir_emitter_nested.cc", "IrEmitterNested::Create");

  std::unique_ptr<IrEmitterNested> emitter(new IrEmitterNested(
      hlo_module_config, nested_computation, ir_emitter_context));
  TF_RETURN_IF_ERROR(emitter->EmitConstants(nested_computation));
  return emitter;
}

// Nested function serves the same purpose on GPU as a thread-local function on
// a CPU.
Status IrEmitterNested::CodegenNestedComputation() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emitter_nestedDTcc mht_2(mht_2_v, 236, "", "./tensorflow/compiler/xla/service/gpu/ir_emitter_nested.cc", "IrEmitterNested::CodegenNestedComputation");

  std::vector<const HloInstruction*> io_hlos;
  std::vector<llvm::Type*> argument_types;
  std::vector<int64_t> argument_dereferenceable_bytes;
  const auto& params = nested_computation_.parameter_instructions();
  const auto n = params.size() + 1;
  io_hlos.reserve(n - 1);
  argument_types.reserve(n);
  argument_dereferenceable_bytes.reserve(n);
  for (const HloInstruction* param : params) {
    io_hlos.push_back(param);
    const Shape& param_shape = param->shape();
    argument_types.push_back(
        llvm_ir::ShapeToIrType(param_shape, module_)->getPointerTo());
    int64_t param_size =
        llvm_ir::ByteSizeOf(param_shape, module_->getDataLayout());
    argument_dereferenceable_bytes.push_back(param_size);
  }

  const HloInstruction* root = nested_computation_.root_instruction();
  {
    const Shape& root_shape = root->shape();
    argument_types.push_back(
        llvm_ir::ShapeToIrType(root_shape, module_)->getPointerTo());
    int64_t root_size = llvm_ir::ByteSizeOf(
        root_shape, ir_emitter_context_->llvm_module()->getDataLayout());
    argument_dereferenceable_bytes.push_back(root_size);
  }

  llvm::FunctionType* function_type =
      llvm::FunctionType::get(b_.getVoidTy(), argument_types, false);
  llvm::Function* function = llvm::Function::Create(
      function_type,                       // The function type.
      llvm::GlobalValue::InternalLinkage,  // The linkage type.
      ir_emitter_context_->name_uniquer()->GetUniqueName(
          llvm_ir::SanitizeFunctionName(
              nested_computation_.name())),  // The name of the function.
      ir_emitter_context_->llvm_module());   // The parent LLVM module.
  for (size_t arg_no = 0; arg_no < argument_dereferenceable_bytes.size();
       ++arg_no) {
    int64_t arg_size = argument_dereferenceable_bytes[arg_no];
    if (arg_size > 0) {
      function->addDereferenceableParamAttr(arg_no, arg_size);
    }
  }

  // TODO(b/65380986): Investigate if adding fast math flags for generated
  // kernels makes sense.

  llvm::BasicBlock* entry_bb =
      llvm::BasicBlock::Create(function->getContext(), "entry", function);
  // Emit a "return void" at entry_bb's end, and sets the insert point before
  // that return instruction.
  llvm::ReturnInst* ret_instr =
      llvm::ReturnInst::Create(function->getContext(), entry_bb);
  b_.SetInsertPoint(ret_instr);

  std::vector<const HloInstruction*> non_io_hlos;
  non_io_hlos.push_back(root);
  for (const auto* hlo : nested_computation_.instructions()) {
    if (hlo->opcode() != HloOpcode::kParameter &&
        hlo != nested_computation_.root_instruction()) {
      non_io_hlos.push_back(hlo);
    }
  }
  bindings_.EmitBasePointersForHlos(io_hlos, non_io_hlos);

  TF_RETURN_IF_ERROR(nested_computation_.root_instruction()->Accept(this));
  b_.SetInsertPoint(ret_instr);

  // Function epilogue: copy the output value back.
  {
    // TODO(cheshire) Duplication vs. EmitThreadLocalFunctionEpilogue
    const HloInstruction* root_instruction =
        nested_computation_.root_instruction();
    llvm::Value* root_value = bindings_.GetBasePointer(*root_instruction);
    const Shape& return_shape = root_instruction->shape();

    // Last argument is the out parameter.
    llvm::Argument* out_parameter = std::prev(function->arg_end(), 1);

    if (ShapeUtil::IsScalar(return_shape)) {
      llvm::Value* ret_value = Load(root_value, "load_ret_value");
      Store(ret_value,
            BitCast(out_parameter, root_value->getType(), "bitcast_ret_value"));
    } else {
      CHECK(return_shape.IsTuple());
      llvm::Type* tuple_type = llvm_ir::ShapeToIrType(return_shape, module_);
      llvm::Type* tuple_type_ptr = tuple_type->getPointerTo();
      llvm::Value* tuple_ptr = BitCast(out_parameter, tuple_type_ptr);

      for (int i = 0; i < return_shape.tuple_shapes_size(); i++) {
        const Shape& element_shape = return_shape.tuple_shapes(i);
        llvm::Value* destination =
            llvm_ir::EmitGetTupleElement(element_shape,
                                         /*index=*/i,
                                         /*alignment=*/1, tuple_ptr, &b_);
        llvm::Value* source =
            llvm_ir::EmitGetTupleElement(element_shape,
                                         /*index=*/i,
                                         /*alignment=*/1, root_value, &b_);
        Store(Load(source), destination);
      }
    }
  }
  b_.SetInsertPoint(ret_instr);
  emitted_function_ = function;
  return Status::OK();
}

Status IrEmitterNested::HandleParameter(HloInstruction* parameter) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emitter_nestedDTcc mht_3(mht_3_v, 349, "", "./tensorflow/compiler/xla/service/gpu/ir_emitter_nested.cc", "IrEmitterNested::HandleParameter");

  return Status::OK();
}

Status IrEmitterNested::EmitTargetElementLoop(
    const HloInstruction& hlo,
    const llvm_ir::ElementGenerator& element_generator) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emitter_nestedDTcc mht_4(mht_4_v, 358, "", "./tensorflow/compiler/xla/service/gpu/ir_emitter_nested.cc", "IrEmitterNested::EmitTargetElementLoop");

  // For MOF we give the loop emitter an array for every output it should
  // generate.
  if (hlo.shape().IsTuple()) {
    std::vector<llvm_ir::IrArray> target_arrays =
        ConstructIrArrayForOutputs(hlo);
    TF_RETURN_IF_ERROR(
        llvm_ir::LoopEmitter(element_generator, target_arrays, &b_).EmitLoop());
    llvm_ir::EmitTuple(GetIrArray(hlo, hlo), target_arrays, &b_);
    return Status::OK();
  }
  return llvm_ir::LoopEmitter(element_generator, GetIrArray(hlo, hlo), &b_)
      .EmitLoop();
}

Status IrEmitterNested::EmitConstants(const HloComputation& computation) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emitter_nestedDTcc mht_5(mht_5_v, 376, "", "./tensorflow/compiler/xla/service/gpu/ir_emitter_nested.cc", "IrEmitterNested::EmitConstants");

  for (HloInstruction* instr : computation.instructions()) {
    if (instr->opcode() != HloOpcode::kConstant) {
      continue;
    }
    Literal& literal = *Cast<HloConstantInstruction>(instr)->mutable_literal();
    const bool should_emit_initializer = ShouldEmitLiteralInLlvmIr(literal);
    llvm::ArrayType* global_type =
        llvm::ArrayType::get(b_.getInt8Ty(), literal.size_bytes());
    llvm::Constant* initializer =
        should_emit_initializer
            ? llvm_ir::ConvertLiteralToIrConstant(literal, module_)
            : llvm::ConstantAggregateZero::get(global_type);
    if (should_emit_initializer) {
      VLOG(3) << "Emitted initializer for constant with shape "
              << ShapeUtil::HumanString(literal.shape());
    }

    // These globals will be looked up by name by GpuExecutable so we need to
    // give them an external linkage.  Not all of their uses are visible in
    // the LLVM IR (e.g. TupleThunk) so we can't give then a linkage that
    // merely preserves their names (like available_externally), we also need
    // to ensure that they stick around even if they're "unused".
    //
    // We may have to be more clever here in the future if we notice that we're
    // keeping around too many globals because of their linkage.
    std::string global_name = llvm_ir::ConstantHloToGlobalName(*instr);

    llvm::GlobalVariable* global_for_const = new llvm::GlobalVariable(
        global_type, /*isConstant=*/should_emit_initializer,
        llvm::GlobalValue::ExternalLinkage,
        /*Initializer=*/initializer, global_name,
        /*TLMode=*/llvm::GlobalValue::NotThreadLocal,
        /*AddressSpace=*/0,
        /*isExternallyInitialized=*/false);
    global_for_const->setAlignment(llvm::Align(kConstantBufferAlignBytes));
    ir_emitter_context_->llvm_module()->getGlobalList().push_back(
        global_for_const);

    GpuExecutable::ConstantInfo info;
    info.symbol_name = global_name;

    if (!should_emit_initializer) {
      auto base = static_cast<const uint8_t*>(literal.untyped_data());
      info.content.assign(base, base + literal.size_bytes());
    }
    ir_emitter_context_->constants().push_back(std::move(info));
  }
  return Status::OK();
}

}  // namespace gpu
}  // namespace xla
