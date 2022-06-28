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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSfused_ir_emitterDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSfused_ir_emitterDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSfused_ir_emitterDTcc() {
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

#include "tensorflow/compiler/xla/service/llvm_ir/fused_ir_emitter.h"

#include <algorithm>
#include <functional>
#include <utility>

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "tensorflow/compiler/xla/service/elemental_ir_emitter.h"
#include "tensorflow/compiler/xla/service/fusion_node_indexing_evaluation.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/llvm_ir/ir_array.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/service/llvm_ir/tuple_ops.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/statusor.h"

namespace xla {

using llvm_ir::IrArray;

StatusOr<FusedIrEmitter::IndexedGenerator> FusedIrEmitter::DefaultAction(
    const HloInstruction& instruction) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSfused_ir_emitterDTcc mht_0(mht_0_v, 217, "", "./tensorflow/compiler/xla/service/llvm_ir/fused_ir_emitter.cc", "FusedIrEmitter::DefaultAction");

  IndexedGenerator generator = elemental_emitter_.MakeElementGenerator(
      &instruction, indexed_generators_);

  return StatusOr<IndexedGenerator>([&, generator = std::move(generator)](
                                        const IrArray::Index& index)
                                        -> StatusOr<llvm::Value*> {
    ValueCacheKey key{&instruction, index.multidim()};
    llvm::Value* value = value_cache_.insert({key, nullptr}).first->second;

    if (value != nullptr) {
      if (const auto* generated_instruction =
              llvm::dyn_cast<llvm::Instruction>(value)) {
        const llvm::BasicBlock* bb = generated_instruction->getParent();

        // Ideally, we should be able to reuse the cached generated value if it
        // dominates the current insertion block. However, the check for
        // dominance can be expensive and unreliable when the function is being
        // constructed.
        //
        // It's also worth experimenting what if we don't do caching at all.
        // LLVM's CSE or GVN should be able to easily merge common
        // subexpressions that would be regenerated without caching. But this
        // might increase the JIT compilation time.
        llvm::IRBuilder<>* b = elemental_emitter_.b();

        if (bb == b->GetInsertBlock()) {
          VLOG(3) << "The cached generated value is reused.";
          return value;
        }

        VLOG(3)
            << "The cached generated value can't be reused, because it is in "
               "a different BB ("
            << bb->getName().str() << ") from the current insertion block ("
            << b->GetInsertBlock()->getName().str() << ").";
      }
    }

    TF_ASSIGN_OR_RETURN(value, generator(index));
    value_cache_[std::move(key)] = value;
    return value;
  });
}

FusedIrEmitter::IndexedGenerator FusedIrEmitter::HandleConstant(
    const HloInstruction& constant) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSfused_ir_emitterDTcc mht_1(mht_1_v, 266, "", "./tensorflow/compiler/xla/service/llvm_ir/fused_ir_emitter.cc", "FusedIrEmitter::HandleConstant");

  llvm::Module* module = elemental_emitter_.module();
  llvm::IRBuilder<>* b = elemental_emitter_.b();

  llvm::Constant* initializer =
      llvm_ir::ConvertLiteralToIrConstant(constant.literal(), module);
  llvm::GlobalVariable* global = new llvm::GlobalVariable(
      *b->GetInsertBlock()->getModule(), initializer->getType(),
      /*isConstant=*/true,
      /*Linkage=*/llvm::GlobalValue::PrivateLinkage,
      /*Initializer=*/initializer,
      /*Name=*/"", /*InsertBefore=*/nullptr,
      /*TLMode=*/llvm::GlobalValue::NotThreadLocal,
      /*AddressSpace=*/0,
      /*isExternallyInitialized=*/false);
  global->setUnnamedAddr(llvm::GlobalVariable::UnnamedAddr::Global);

  llvm::Type* shape_type = llvm_ir::ShapeToIrType(constant.shape(), module);
  llvm::Constant* global_with_shape =
      llvm::ConstantExpr::getPointerBitCastOrAddrSpaceCast(
          global, shape_type->getPointerTo());

  IrArray array(global_with_shape, constant.shape());

  return [&, b, array = std::move(array)](const IrArray::Index& index) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSfused_ir_emitterDTcc mht_2(mht_2_v, 293, "", "./tensorflow/compiler/xla/service/llvm_ir/fused_ir_emitter.cc", "lambda");

    return array.EmitReadArrayElement(index, b, constant.name());
  };
}

StatusOr<FusedIrEmitter::IndexedGenerator> FusedIrEmitter::HandleTuple(
    const HloInstruction& tuple) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSfused_ir_emitterDTcc mht_3(mht_3_v, 302, "", "./tensorflow/compiler/xla/service/llvm_ir/fused_ir_emitter.cc", "FusedIrEmitter::HandleTuple");

  std::vector<llvm::Type*> element_ir_types;
  element_ir_types.reserve(tuple.operand_count());
  for (const HloInstruction* operand : tuple.operands()) {
    element_ir_types.push_back(llvm_ir::PrimitiveTypeToIrType(
        operand->shape().element_type(), elemental_emitter_.module()));
  }

  llvm::IRBuilder<>* b = elemental_emitter_.b();
  llvm::Type* type = llvm::StructType::get(b->getContext(), element_ir_types);

  return StatusOr<IndexedGenerator>(
      [&, b, type](const IrArray::Index& index) -> StatusOr<llvm::Value*> {
        llvm::Value* ret = llvm::UndefValue::get(type);
        for (size_t i = 0; i < tuple.operand_count(); ++i) {
          TF_ASSIGN_OR_RETURN(llvm::Value * value,
                              indexed_generators_.at(tuple.operand(i))(index));
          ret = b->CreateInsertValue(ret, value, i);
        }
        return ret;
      });
}

bool FusedIrEmitter::IsFusedIrEmitterInefficient(
    const HloInstruction& consumer, const HloInstruction& producer) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSfused_ir_emitterDTcc mht_4(mht_4_v, 329, "", "./tensorflow/compiler/xla/service/llvm_ir/fused_ir_emitter.cc", "FusedIrEmitter::IsFusedIrEmitterInefficient");

  if (consumer.opcode() != HloOpcode::kFusion) {
    return false;
  }
  FusionNodeIndexingEvaluation eval_consumer(&consumer);
  if (producer.opcode() != HloOpcode::kFusion) {
    return eval_consumer.CodeDuplicationTooHigh(&producer);
  }
  // If 'producer' is a fusion node as well, also evaluate it. Pass the
  // evaluated duplication of the fusion node if it is merged into consumer.
  FusionNodeIndexingEvaluation eval_producer(
      &producer, eval_consumer.EvaluateEmittedInstructions(&producer));
  return eval_producer.MaxCodeDuplicationTooHigh();
}

StatusOr<FusedIrEmitter::IndexedGenerator> FusedIrEmitter::CreateGenerator(
    const HloInstruction& instruction) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSfused_ir_emitterDTcc mht_5(mht_5_v, 348, "", "./tensorflow/compiler/xla/service/llvm_ir/fused_ir_emitter.cc", "FusedIrEmitter::CreateGenerator");

  switch (instruction.opcode()) {
    case HloOpcode::kConstant:
      return HandleConstant(instruction);
    case HloOpcode::kGetTupleElement:
      return InternalError("Tuple parameters are not supported for fusion");
    case HloOpcode::kParameter:
      return InvalidArgument("Unbound parameter: %s", instruction.ToString());
    case HloOpcode::kTuple:
      return HandleTuple(instruction);
    default:
      return DefaultAction(instruction);
  }
}

StatusOr<FusedIrEmitter::IndexedGenerator> FusedIrEmitter::GetGenerator(
    const HloInstruction& instruction) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSfused_ir_emitterDTcc mht_6(mht_6_v, 367, "", "./tensorflow/compiler/xla/service/llvm_ir/fused_ir_emitter.cc", "FusedIrEmitter::GetGenerator");

  std::vector<const HloInstruction*> stack = {&instruction};
  while (!stack.empty()) {
    const HloInstruction& instr = *stack.back();
    stack.pop_back();

    IndexedGenerator& indexed_generator = indexed_generators_[&instr];
    if (indexed_generator != nullptr) continue;

    stack.insert(stack.end(), instr.operands().begin(), instr.operands().end());
    TF_ASSIGN_OR_RETURN(indexed_generator, CreateGenerator(instr));
  }
  return indexed_generators_[&instruction];
}

}  // namespace xla
