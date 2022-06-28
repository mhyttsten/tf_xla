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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emitterDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emitterDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emitterDTcc() {
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

#include "tensorflow/compiler/xla/service/gpu/ir_emitter.h"

#include "tensorflow/core/platform/logging.h"
// IWYU pragma: no_include "llvm/IR/Intrinsics.gen.inc"
#include "absl/algorithm/container.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/service/elemental_ir_emitter.h"
#include "tensorflow/compiler/xla/service/gpu/elemental_ir_emitter.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emitter_nested.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emitter_unnested.h"
#include "tensorflow/compiler/xla/service/gpu/launch_dimensions.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/llvm_ir/fused_ir_emitter.h"
#include "tensorflow/compiler/xla/service/llvm_ir/ir_array.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_loop.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/service/llvm_ir/loop_emitter.h"
#include "tensorflow/compiler/xla/service/llvm_ir/tuple_ops.h"
#include "tensorflow/compiler/xla/service/name_uniquer.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/window_util.h"
#include "tensorflow/core/lib/core/errors.h"

// Convenient function to cast the provided llvm::Value* using IRBuilder
// to default address space. This is useful in particular for generating
// IR for AMDGPU target, as its kernel variables are in address space 5
// instead of the default address space.
static llvm::Value* AddrCastToDefault(llvm::Value* arg, llvm::IRBuilder<>& b) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emitterDTcc mht_0(mht_0_v, 221, "", "./tensorflow/compiler/xla/service/gpu/ir_emitter.cc", "AddrCastToDefault");

  llvm::Type* arg_type = arg->getType();
  CHECK(arg_type->isPointerTy());
  if (arg_type->getPointerAddressSpace() != 0) {
    llvm::Type* generic_arg_type =
        arg_type->getPointerElementType()->getPointerTo(0);
    llvm::Value* addrspacecast_arg =
        b.CreateAddrSpaceCast(arg, generic_arg_type);
    return addrspacecast_arg;
  }
  return arg;
}

namespace xla {

using llvm_ir::IrName;
using llvm_ir::SetToFirstInsertPoint;

namespace gpu {

IrEmitter::IrEmitter(const HloModuleConfig& hlo_module_config,
                     IrEmitterContext* ir_emitter_context, bool is_nested)
    : ir_emitter_context_(ir_emitter_context),
      module_(ir_emitter_context->llvm_module()),
      b_(module_->getContext()),
      bindings_(&b_, module_, is_nested),
      hlo_module_config_(hlo_module_config) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emitterDTcc mht_1(mht_1_v, 250, "", "./tensorflow/compiler/xla/service/gpu/ir_emitter.cc", "IrEmitter::IrEmitter");
}

Status IrEmitter::DefaultAction(HloInstruction* hlo) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emitterDTcc mht_2(mht_2_v, 255, "", "./tensorflow/compiler/xla/service/gpu/ir_emitter.cc", "IrEmitter::DefaultAction");

  ElementalIrEmitter::HloToElementGeneratorMap operand_to_generator;
  for (const HloInstruction* operand : hlo->operands()) {
    operand_to_generator[operand] = [=](const llvm_ir::IrArray::Index& index) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emitterDTcc mht_3(mht_3_v, 261, "", "./tensorflow/compiler/xla/service/gpu/ir_emitter.cc", "lambda");

      return GetIrArray(*operand, *hlo)
          .EmitReadArrayElement(index, &b_, operand->name());
    };
  }
  return EmitTargetElementLoop(
      *hlo, GpuElementalIrEmitter(hlo_module_config_, module_, &b_,
                                  GetNestedComputer())
                .MakeElementGenerator(hlo, operand_to_generator));
}

Status IrEmitter::HandleConstant(HloInstruction* constant) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emitterDTcc mht_4(mht_4_v, 275, "", "./tensorflow/compiler/xla/service/gpu/ir_emitter.cc", "IrEmitter::HandleConstant");

  return Status::OK();
}

Status IrEmitter::HandleAddDependency(HloInstruction* add_dependency) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emitterDTcc mht_5(mht_5_v, 282, "", "./tensorflow/compiler/xla/service/gpu/ir_emitter.cc", "IrEmitter::HandleAddDependency");

  VLOG(2) << "HandleAddDependency: " << add_dependency->ToString();
  const HloInstruction* operand = add_dependency->operand(0);
  // Add_Dependency is a no-op, but we still want to bind it to an llvm::Value
  // sometimes, e.g., when it's operand is a constant or a bitcast of a
  // constant.
  if (bindings_.BoundToIrValue(*operand)) {
    bindings_.BindHloToIrValue(*add_dependency, GetBasePointer(*operand));
  }
  return Status::OK();
}

Status IrEmitter::HandleGetTupleElement(HloInstruction* get_tuple_element) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emitterDTcc mht_6(mht_6_v, 297, "", "./tensorflow/compiler/xla/service/gpu/ir_emitter.cc", "IrEmitter::HandleGetTupleElement");

  auto operand = get_tuple_element->operand(0);
  CHECK(bindings_.BoundToIrValue(*operand));
  bindings_.BindHloToIrValue(
      *get_tuple_element,
      llvm_ir::EmitGetTupleElement(
          get_tuple_element->shape(), get_tuple_element->tuple_index(),
          // TODO(b/26344050): tighten the alignment here
          // based on the real element type.
          /*alignment=*/1, GetBasePointer(*operand), &b_));
  return Status::OK();
}

Status IrEmitter::HandleSend(HloInstruction*) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emitterDTcc mht_7(mht_7_v, 313, "", "./tensorflow/compiler/xla/service/gpu/ir_emitter.cc", "IrEmitter::HandleSend");

  return Unimplemented("Send is not implemented on GPU");
}

Status IrEmitter::HandleSendDone(HloInstruction*) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emitterDTcc mht_8(mht_8_v, 320, "", "./tensorflow/compiler/xla/service/gpu/ir_emitter.cc", "IrEmitter::HandleSendDone");

  return Unimplemented("Send-Done is not implemented on GPU");
}

Status IrEmitter::HandleRecv(HloInstruction*) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emitterDTcc mht_9(mht_9_v, 327, "", "./tensorflow/compiler/xla/service/gpu/ir_emitter.cc", "IrEmitter::HandleRecv");

  return Unimplemented("Recv is not implemented on GPU");
}

Status IrEmitter::HandleRecvDone(HloInstruction*) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emitterDTcc mht_10(mht_10_v, 334, "", "./tensorflow/compiler/xla/service/gpu/ir_emitter.cc", "IrEmitter::HandleRecvDone");

  return Unimplemented("Recv-done is not implemented on GPU");
}

Status IrEmitter::HandleScatter(HloInstruction*) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emitterDTcc mht_11(mht_11_v, 341, "", "./tensorflow/compiler/xla/service/gpu/ir_emitter.cc", "IrEmitter::HandleScatter");

  return Unimplemented("Scatter is not implemented on GPUs.");
}

Status IrEmitter::HandleTuple(HloInstruction* tuple) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emitterDTcc mht_12(mht_12_v, 348, "", "./tensorflow/compiler/xla/service/gpu/ir_emitter.cc", "IrEmitter::HandleTuple");

  std::vector<llvm::Value*> base_ptrs;
  for (const HloInstruction* operand : tuple->operands()) {
    base_ptrs.push_back(GetBasePointer(*operand));
  }
  llvm_ir::EmitTuple(GetIrArray(*tuple, *tuple), base_ptrs, &b_);
  return Status::OK();
}

Status IrEmitter::EmitCallToNestedComputation(
    const HloComputation& nested_computation,
    absl::Span<llvm::Value* const> operands, llvm::Value* output) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emitterDTcc mht_13(mht_13_v, 362, "", "./tensorflow/compiler/xla/service/gpu/ir_emitter.cc", "IrEmitter::EmitCallToNestedComputation");

  TF_RET_CHECK(nested_computation.num_parameters() > 0);
  llvm::Function*& emitted_function =
      computation_to_ir_function_[&nested_computation];
  if (emitted_function == nullptr) {
    TF_ASSIGN_OR_RETURN(
        auto ir_emitter_nested,
        IrEmitterNested::Create(hlo_module_config_, nested_computation,
                                ir_emitter_context_));
    TF_RETURN_IF_ERROR(ir_emitter_nested->CodegenNestedComputation());
    emitted_function = ir_emitter_nested->GetEmittedFunction();
  }

  // Operands are in default address space for non-AMDGPU target.
  // However for AMDGPU target, addrspacecast alloca variables from
  // addrspace 5 to addrspace 0 is needed.
  std::vector<llvm::Value*> arguments;
  absl::c_transform(
      operands, std::back_inserter(arguments),
      [this](llvm::Value* arg) { return AddrCastToDefault(arg, b_); });

  llvm::Value* casted_output = AddrCastToDefault(output, b_);
  arguments.push_back(casted_output);

  Call(emitted_function, arguments);

  return Status::OK();
}

bool IrEmitter::MaybeEmitDirectAtomicOperation(
    const HloComputation& computation, llvm::Value* output_address,
    llvm::Value* source_address) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emitterDTcc mht_14(mht_14_v, 396, "", "./tensorflow/compiler/xla/service/gpu/ir_emitter.cc", "IrEmitter::MaybeEmitDirectAtomicOperation");

  CHECK_EQ(2, computation.num_parameters());

  HloOpcode root_opcode = computation.root_instruction()->opcode();
  PrimitiveType element_type =
      computation.root_instruction()->shape().element_type();
  bool is_atomic_integral = element_type == S32 || element_type == U32 ||
                            element_type == S64 || element_type == U64;
  llvm::Value* source = Load(source_address, "source");

  // Just passing along RHS -> atomic store.
  if (computation.instruction_count() == 2 &&
      root_opcode == HloOpcode::kParameter &&
      (element_type == F32 || is_atomic_integral) &&
      computation.root_instruction()->parameter_number() == 1) {
    llvm::StoreInst* store = Store(source, output_address);
    store->setAtomic(llvm::AtomicOrdering::Unordered);
    // Derive a minimum alignment from the type. The optimizer can increase it
    // later.
    store->setAlignment(
        llvm::Align(ShapeUtil::ByteSizeOfPrimitiveType(element_type)));
    return true;
  }

  if (computation.instruction_count() != 3) {
    // We special-case only computations with one computing instruction for now.
    // Such computation has exactly three instructions given it has two
    // parameters.
    return false;
  }

  if (root_opcode == HloOpcode::kAdd) {
    llvm::Triple target_triple = llvm::Triple(module_->getTargetTriple());
    // NVPTX supports atomicAdd on F32 and integer types.
    if (target_triple.isNVPTX()) {
      // "atom.add.f64 requires sm_60 or higher."
      // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-atom
      bool f64_atomic_add_supported =
          ir_emitter_context_->cuda_compute_capability().IsAtLeast(6);
      bool atomic_add_supported =
          element_type == F32 ||
          (f64_atomic_add_supported && element_type == F64);
      if (atomic_add_supported) {
        AtomicRMW(llvm::AtomicRMWInst::FAdd, output_address, source,
                  llvm::MaybeAlign(),
                  llvm::AtomicOrdering::SequentiallyConsistent);
        return true;
      }
    }

    if (IsEmittingForAMDGPU() &&
        (element_type == F32)) /* is atomic add supported? */ {
      EmitAMDGPUAtomicAdd(output_address, source);
      return true;
    }

    if (is_atomic_integral) {
      // integral + integral
      AtomicRMW(
          llvm::AtomicRMWInst::Add, output_address, source, llvm::MaybeAlign(),
          llvm::AtomicOrdering::SequentiallyConsistent, DetermineSyncScope());
      return true;
    }
  }

  // NVPTX supports atomicMax and atomicMin only on integer types.
  if (root_opcode == HloOpcode::kMaximum && is_atomic_integral) {
    // max(integral, integral)
    auto opcode = primitive_util::IsSignedIntegralType(element_type)
                      ? llvm::AtomicRMWInst::Max
                      : llvm::AtomicRMWInst::UMax;
    AtomicRMW(opcode, output_address, source, llvm::MaybeAlign(),
              llvm::AtomicOrdering::SequentiallyConsistent,
              DetermineSyncScope());
    return true;
  }

  if (root_opcode == HloOpcode::kMinimum && is_atomic_integral) {
    // min(integral, integral)
    auto opcode = primitive_util::IsSignedIntegralType(element_type)
                      ? llvm::AtomicRMWInst::Min
                      : llvm::AtomicRMWInst::UMin;
    AtomicRMW(opcode, output_address, source, llvm::MaybeAlign(),
              llvm::AtomicOrdering::SequentiallyConsistent,
              DetermineSyncScope());
    return true;
  }

  return false;
}

// Implements atomic binary operations using atomic compare-and-swap
// (atomicCAS) as follows:
//   1. Reads the value from the memory pointed to by output_address and
//     records it as old_output.
//   2. Uses old_output as one of the source operand to perform the binary
//     operation and stores the result in new_output.
//   3. Calls atomicCAS which implements compare-and-swap as an atomic
//     operation. In particular, atomicCAS reads the value from the memory
//     pointed to by output_address, and compares the value with old_output. If
//     the two values equal, new_output is written to the same memory location
//     and true is returned to indicate that the atomic operation succeeds.
//     Otherwise, the new value read from the memory is returned. In this case,
//     the new value is copied to old_output, and steps 2. and 3. are repeated
//     until atomicCAS succeeds.
//
// On Nvidia GPUs, atomicCAS can only operate on 32 bit and 64 bit integers. If
// the element type of the binary operation is 32 bits or 64 bits, the integer
// type of the same size is used for the atomicCAS operation. On the other hand,
// if the element type is smaller than 32 bits, int32_t is used for the
// atomicCAS operation. In this case, atomicCAS reads and writes 32 bit values
// from the memory, which is larger than the memory size required by the
// original atomic binary operation. We mask off the last two bits of the
// output_address and use the result as an address to read the 32 bit values
// from the memory. This can avoid out of bound memory accesses if tensor
// buffers are 4 byte aligned and have a size of 4N, an assumption that the
// runtime can guarantee.
//
// The pseudo code is shown below. Variables *_address are pointers to a memory
// region with a size equal to the size of the atomicCAS operation, with the
// exception that new_output_address is a pointer to a memory region with a size
// equal to the element size of the binary operation.
//
//   element_size = sizeof(element_type);
//   atomic_size = max(32, element_size);
//   cas_new_output_address = alloca(atomic_size);
//   cas_old_output_address = alloca(atomic_size);
//   if (atomic_size != element_size) {
//     atomic_address = output_address & ((int64_t)(-4));
//     new_output_address = cas_new_output_address + (output_address & 3);
//   } else {
//     atomic_address = output_address;
//     new_output_address = cas_new_output_address;
//   }
//
//   *cas_old_output_address = *atomic_address;
//   do {
//     *cas_new_output_address = *cas_old_output_address;
//     *new_output_address = operation(*new_output_address, *source_address);
//     (*cas_old_output_address, success) =
//       atomicCAS(atomic_address, *cas_old_output_address,
//       *cas_new_output_address);
//   } while (!success);
//
Status IrEmitter::EmitAtomicOperationUsingCAS(const HloComputation& computation,
                                              llvm::Value* output_address,
                                              llvm::Value* source_address) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emitterDTcc mht_15(mht_15_v, 545, "", "./tensorflow/compiler/xla/service/gpu/ir_emitter.cc", "IrEmitter::EmitAtomicOperationUsingCAS");

  llvm::PointerType* output_address_type =
      llvm::dyn_cast<llvm::PointerType>(output_address->getType());
  CHECK_NE(output_address_type, nullptr);

  // element_type is the data type for the binary operation.
  llvm::Type* element_type = output_address_type->getPointerElementType();
  int element_size = llvm_ir::GetSizeInBits(element_type);

  int atomic_size = (element_size < 32) ? 32 : element_size;
  llvm::Type* atomic_type = b_.getIntNTy(atomic_size);
  llvm::Type* atomic_address_type =
      atomic_type->getPointerTo(output_address_type->getPointerAddressSpace());

  // cas_old_output_address and cas_new_output_address point to the scratch
  // memory where we store the old and new values for the repeated atomicCAS
  // operations.
  llvm::Value* cas_old_output_address = llvm_ir::EmitAllocaAtFunctionEntry(
      atomic_type, "cas_old_output_address", &b_);
  llvm::Value* cas_new_output_address = llvm_ir::EmitAllocaAtFunctionEntry(
      atomic_type, "cas_new_output_address", &b_);

  // Emit preparation code to the preheader.
  llvm::BasicBlock* loop_preheader_bb = b_.GetInsertBlock();

  llvm::Value* atomic_memory_address;
  // binop_output_address points to the scratch memory that stores the
  // result of the binary operation.
  llvm::Value* binop_output_address;
  if (element_size < 32) {
    // Assume the element size is an integer number of bytes.
    CHECK_EQ((element_size % sizeof(char)), 0);
    llvm::Type* address_int_type =
        module_->getDataLayout().getIntPtrType(output_address_type);
    atomic_memory_address = PtrToInt(output_address, address_int_type);
    llvm::Value* mask = llvm::ConstantInt::get(address_int_type, 3);
    llvm::Value* offset = And(atomic_memory_address, mask);
    mask = llvm::ConstantInt::get(address_int_type, -4);
    atomic_memory_address = And(atomic_memory_address, mask);
    atomic_memory_address =
        IntToPtr(atomic_memory_address, atomic_address_type);
    binop_output_address =
        Add(PtrToInt(cas_new_output_address, address_int_type), offset);
    binop_output_address = IntToPtr(
        binop_output_address,
        llvm::PointerType::get(
            element_type,
            cas_new_output_address->getType()->getPointerAddressSpace()));
  } else {
    atomic_memory_address = b_.CreatePointerBitCastOrAddrSpaceCast(
        output_address, atomic_address_type);
    binop_output_address = b_.CreatePointerBitCastOrAddrSpaceCast(
        cas_new_output_address,
        llvm::PointerType::get(
            element_type,
            cas_new_output_address->getType()->getPointerAddressSpace()));
  }

  // Use the value from the memory that atomicCAS operates on to initialize
  // cas_old_output.
  llvm::Value* cas_old_output = Load(atomic_memory_address, "cas_old_output");
  Store(cas_old_output, cas_old_output_address);

  llvm::BasicBlock* loop_exit_bb = loop_preheader_bb->splitBasicBlock(
      b_.GetInsertPoint(), "atomic_op_loop_exit");
  llvm::BasicBlock* loop_body_bb = llvm::BasicBlock::Create(
      b_.getContext(), "atomic_op_loop_body", b_.GetInsertBlock()->getParent());
  b_.SetInsertPoint(loop_body_bb);
  // Change preheader's successor from loop_exit_bb to loop_body_bb.
  loop_preheader_bb->getTerminator()->setSuccessor(0, loop_body_bb);

  // Emit the body of the loop that repeatedly invokes atomicCAS.
  //
  // Use cas_old_output to initialize cas_new_output.
  cas_old_output = Load(cas_old_output_address, "cas_old_output");
  Store(cas_old_output, cas_new_output_address);
  // Emits code to calculate new_output = operation(old_output, source);
  TF_RETURN_IF_ERROR(EmitCallToNestedComputation(
      computation, {binop_output_address, source_address},
      binop_output_address));

  llvm::Value* cas_new_output = Load(cas_new_output_address, "cas_new_output");

  // If cas_new_output == cas_old_output, we're not asking for anything to
  // change, so we're done here!
  llvm::Value* old_eq_new = ICmpEQ(cas_old_output, cas_new_output);
  llvm::BasicBlock* loop_cas_bb = llvm::BasicBlock::Create(
      b_.getContext(), "atomic_op_loop_cas", b_.GetInsertBlock()->getParent());
  CondBr(old_eq_new, loop_exit_bb, loop_cas_bb);
  b_.SetInsertPoint(loop_cas_bb);

  // Emit code to perform the atomicCAS operation
  // (cas_old_output, success) = atomicCAS(memory_address, cas_old_output,
  //                                       cas_new_output);
  llvm::Value* ret_value = AtomicCmpXchg(
      atomic_memory_address, cas_old_output, cas_new_output, llvm::MaybeAlign(),
      llvm::AtomicOrdering::SequentiallyConsistent,
      llvm::AtomicOrdering::SequentiallyConsistent, DetermineSyncScope());

  // Extract the memory value returned from atomicCAS and store it as
  // cas_old_output.
  Store(ExtractValue(ret_value, 0, "cas_old_output"), cas_old_output_address);
  // Extract the success bit returned from atomicCAS and generate a
  // conditional branch on the success bit.
  CondBr(ExtractValue(ret_value, 1, "success"), loop_exit_bb, loop_body_bb);

  // Set the insertion point to the exit basic block so that the caller of
  // this method can continue emitting code to the right place.
  SetToFirstInsertPoint(loop_exit_bb, &b_);
  return Status::OK();
}

Status IrEmitter::EmitAtomicOperationForNestedComputation(
    const HloComputation& computation, llvm::Value* output_address,
    llvm::Value* source_address) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emitterDTcc mht_16(mht_16_v, 662, "", "./tensorflow/compiler/xla/service/gpu/ir_emitter.cc", "IrEmitter::EmitAtomicOperationForNestedComputation");

  if (computation.num_parameters() != 2) {
    // TODO(b/30258929): We only accept binary computations so far.
    return Unimplemented(
        "We only support atomic functions with exactly two parameters, but "
        "computation %s has %d.",
        computation.name(), computation.num_parameters());
  }

  if (MaybeEmitDirectAtomicOperation(computation, output_address,
                                     source_address)) {
    return Status::OK();
  }

  return EmitAtomicOperationUsingCAS(computation, output_address,
                                     source_address);
}

bool IrEmitter::IsEmittingForAMDGPU() const {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emitterDTcc mht_17(mht_17_v, 683, "", "./tensorflow/compiler/xla/service/gpu/ir_emitter.cc", "IrEmitter::IsEmittingForAMDGPU");

  llvm::Triple target_triple = llvm::Triple(module_->getTargetTriple());
  return target_triple.isAMDGPU();
}

void IrEmitter::EmitAMDGPUAtomicAdd(llvm::Value* output_address,
                                    llvm::Value* source) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emitterDTcc mht_18(mht_18_v, 692, "", "./tensorflow/compiler/xla/service/gpu/ir_emitter.cc", "IrEmitter::EmitAMDGPUAtomicAdd");

  CHECK(IsEmittingForAMDGPU());
  auto output_address_type =
      llvm::dyn_cast<llvm::PointerType>(output_address->getType());
  CHECK_NE(output_address_type, nullptr);

  auto output_ptr =
      (output_address_type->getPointerAddressSpace() != 3)
          ?
          // the compiler will only generate a global_atomic_fadd if the pointer
          // is in global addrspace (1)
          b_.CreateAddrSpaceCast(
              output_address, llvm::PointerType::get(
                                  output_address_type->getPointerElementType(),
                                  /*AddressSpace=*/1))
          :
          // adds to shared memory are always atomic.
          output_address;

  AtomicRMW(llvm::AtomicRMWInst::FAdd, output_ptr, source, llvm::MaybeAlign(),
            llvm::AtomicOrdering::SequentiallyConsistent,
            b_.getContext().getOrInsertSyncScopeID("agent"));
}

llvm::SyncScope::ID IrEmitter::DetermineSyncScope() const {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emitterDTcc mht_19(mht_19_v, 719, "", "./tensorflow/compiler/xla/service/gpu/ir_emitter.cc", "IrEmitter::DetermineSyncScope");

  return (IsEmittingForAMDGPU())
             ? b_.getContext().getOrInsertSyncScopeID("agent")
             : llvm::SyncScope::System;
}

Status IrEmitter::HandleTupleSelect(HloInstruction* tuple_select) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emitterDTcc mht_20(mht_20_v, 728, "", "./tensorflow/compiler/xla/service/gpu/ir_emitter.cc", "IrEmitter::HandleTupleSelect");

  return InternalError(
      "Dynamic selection of tuples is not supported. Please file a bug against "
      "XLA/GPU if you need it");
}

namespace {
llvm::Value* Real(llvm::Value* x, llvm::IRBuilder<>* b) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emitterDTcc mht_21(mht_21_v, 738, "", "./tensorflow/compiler/xla/service/gpu/ir_emitter.cc", "Real");

  return b->CreateExtractValue(x, {0});
}

llvm::Value* Imag(llvm::Value* x, llvm::IRBuilder<>* b) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emitterDTcc mht_22(mht_22_v, 745, "", "./tensorflow/compiler/xla/service/gpu/ir_emitter.cc", "Imag");

  return b->CreateExtractValue(x, {1});
}

std::pair<llvm::Value*, llvm::Value*> MultiplyComplex(llvm::Value* lhs_value,
                                                      llvm::Value* rhs_value,
                                                      llvm::IRBuilder<>* b) {
  llvm::Value* lhs_real = Real(lhs_value, b);
  llvm::Value* lhs_imag = Imag(lhs_value, b);
  llvm::Value* rhs_real = Real(rhs_value, b);
  llvm::Value* rhs_imag = Imag(rhs_value, b);
  llvm::Value* real_result1 = b->CreateFMul(lhs_real, rhs_real);
  llvm::Value* real_result2 = b->CreateFMul(lhs_imag, rhs_imag);
  llvm::Value* real_result = b->CreateFSub(real_result1, real_result2);
  llvm::Value* imag_result1 = b->CreateFMul(lhs_real, rhs_imag);
  llvm::Value* imag_result2 = b->CreateFMul(lhs_imag, rhs_real);
  llvm::Value* imag_result = b->CreateFAdd(imag_result1, imag_result2);
  return {real_result, imag_result};
}
}  // namespace

Status IrEmitter::HandleConvolution(HloInstruction* convolution) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emitterDTcc mht_23(mht_23_v, 769, "", "./tensorflow/compiler/xla/service/gpu/ir_emitter.cc", "IrEmitter::HandleConvolution");

  if (ShapeUtil::IsZeroElementArray(convolution->shape())) {
    // Emit no code for an empty output.
    return Status::OK();
  }
  // TODO(b/31409998): Support convolution with dilation.
  return Unimplemented(
      "Hit a case for convolution that is not implemented on GPU.");
}

Status IrEmitter::HandleFft(HloInstruction* fft) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emitterDTcc mht_24(mht_24_v, 782, "", "./tensorflow/compiler/xla/service/gpu/ir_emitter.cc", "IrEmitter::HandleFft");

  if (ShapeUtil::IsZeroElementArray(fft->shape())) {
    // Emit no code for an empty output.
    return Status::OK();
  }
  return Unimplemented("Hit a case for fft that is not implemented on GPU.");
}

Status IrEmitter::HandleAllReduce(HloInstruction* crs) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emitterDTcc mht_25(mht_25_v, 793, "", "./tensorflow/compiler/xla/service/gpu/ir_emitter.cc", "IrEmitter::HandleAllReduce");

  return Unimplemented(
      "AllReduce cannot be nested inside of fusion, map, etc.");
}

Status IrEmitter::HandleParameter(HloInstruction* parameter) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emitterDTcc mht_26(mht_26_v, 801, "", "./tensorflow/compiler/xla/service/gpu/ir_emitter.cc", "IrEmitter::HandleParameter");

  return Status::OK();
}

Status IrEmitter::HandleFusion(HloInstruction* fusion) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emitterDTcc mht_27(mht_27_v, 808, "", "./tensorflow/compiler/xla/service/gpu/ir_emitter.cc", "IrEmitter::HandleFusion");

  // kFusion for library calls should be handled by
  // IrEmitterUnnested::HandleFusion.
  CHECK_EQ(HloInstruction::FusionKind::kLoop, fusion->fusion_kind());
  GpuElementalIrEmitter elemental_emitter(hlo_module_config_, module_, &b_,
                                          GetNestedComputer());
  FusedIrEmitter fused_emitter(elemental_emitter);
  BindFusionArguments(fusion, &fused_emitter);
  TF_ASSIGN_OR_RETURN(auto generator, fused_emitter.GetGenerator(
                                          *fusion->fused_expression_root()));
  return EmitTargetElementLoop(*fusion, generator);
}

Status IrEmitter::HandleCall(HloInstruction* call) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emitterDTcc mht_28(mht_28_v, 824, "", "./tensorflow/compiler/xla/service/gpu/ir_emitter.cc", "IrEmitter::HandleCall");

  std::vector<llvm::Value*> operand_addresses;
  for (HloInstruction* operand : call->operands()) {
    operand_addresses.push_back(GetBasePointer(*operand));
  }
  return EmitCallToNestedComputation(*call->to_apply(), operand_addresses,
                                     GetBasePointer(*call));
}

Status IrEmitter::HandleCustomCall(HloInstruction*) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emitterDTcc mht_29(mht_29_v, 836, "", "./tensorflow/compiler/xla/service/gpu/ir_emitter.cc", "IrEmitter::HandleCustomCall");

  return Unimplemented("custom-call");
}

Status IrEmitter::HandleInfeed(HloInstruction*) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emitterDTcc mht_30(mht_30_v, 843, "", "./tensorflow/compiler/xla/service/gpu/ir_emitter.cc", "IrEmitter::HandleInfeed");

  // TODO(b/30467474): Implement infeed on GPU.
  return Unimplemented("Infeed is not supported on GPU.");
}

Status IrEmitter::HandleOutfeed(HloInstruction*) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emitterDTcc mht_31(mht_31_v, 851, "", "./tensorflow/compiler/xla/service/gpu/ir_emitter.cc", "IrEmitter::HandleOutfeed");

  // TODO(b/34359662): Implement outfeed on GPU.
  return Unimplemented("Outfeed is not supported on GPU.");
}

Status IrEmitter::HandleBatchNormInference(HloInstruction*) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emitterDTcc mht_32(mht_32_v, 859, "", "./tensorflow/compiler/xla/service/gpu/ir_emitter.cc", "IrEmitter::HandleBatchNormInference");

  return Unimplemented(
      "The GPU backend does not implement BatchNormInference directly.  It "
      "should be lowered before IR emission to HLO-soup using "
      "BatchNormRewriter.");
}

Status IrEmitter::HandleBatchNormTraining(HloInstruction*) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emitterDTcc mht_33(mht_33_v, 869, "", "./tensorflow/compiler/xla/service/gpu/ir_emitter.cc", "IrEmitter::HandleBatchNormTraining");

  return Unimplemented(
      "The GPU backend does not implement BatchNormTraining directly.  It "
      "should be lowered before IR emission to HLO-soup using "
      "BatchNormRewriter.");
}

Status IrEmitter::HandleBatchNormGrad(HloInstruction*) {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emitterDTcc mht_34(mht_34_v, 879, "", "./tensorflow/compiler/xla/service/gpu/ir_emitter.cc", "IrEmitter::HandleBatchNormGrad");

  return Unimplemented(
      "The GPU backend does not implement BatchNormGrad directly.  It should "
      "be lowered before IR emission to HLO-soup using BatchNormRewriter.");
}

StatusOr<std::vector<llvm::Value*>> IrEmitter::ComputeNestedElement(
    const HloComputation& computation,
    absl::Span<llvm::Value* const> parameter_elements) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emitterDTcc mht_35(mht_35_v, 890, "", "./tensorflow/compiler/xla/service/gpu/ir_emitter.cc", "IrEmitter::ComputeNestedElement");

  std::vector<llvm::Value*> parameter_buffers;
  for (llvm::Value* parameter_element : parameter_elements) {
    parameter_buffers.push_back(llvm_ir::EmitAllocaAtFunctionEntry(
        parameter_element->getType(), "parameter_buffer", &b_));
    Store(parameter_element, parameter_buffers.back());
  }

  return ComputeNestedElementFromAddrs(computation, parameter_buffers);
}

StatusOr<std::vector<llvm::Value*>> IrEmitter::ComputeNestedElementFromAddrs(
    const HloComputation& computation,
    absl::Span<llvm::Value* const> parameter_elements_addrs) {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emitterDTcc mht_36(mht_36_v, 906, "", "./tensorflow/compiler/xla/service/gpu/ir_emitter.cc", "IrEmitter::ComputeNestedElementFromAddrs");

  const Shape& return_shape = computation.root_instruction()->shape();
  llvm::Value* return_buffer = llvm_ir::EmitAllocaAtFunctionEntry(
      llvm_ir::ShapeToIrType(return_shape, module_), "return_buffer", &b_);

  std::vector<llvm::Value*> allocas_for_returned_scalars;
  if (!return_shape.IsTuple()) {
    allocas_for_returned_scalars.push_back(return_buffer);
  } else {
    allocas_for_returned_scalars =
        llvm_ir::EmitTupleAllocasAtFunctionEntry(return_shape, &b_);
    llvm_ir::IrArray tuple_array(return_buffer, return_shape);

    EmitTuple(tuple_array, allocas_for_returned_scalars, &b_);
  }

  TF_RETURN_IF_ERROR(EmitCallToNestedComputation(
      computation, parameter_elements_addrs, return_buffer));

  std::vector<llvm::Value*> returned_scalars;
  returned_scalars.reserve(allocas_for_returned_scalars.size());
  for (llvm::Value* addr : allocas_for_returned_scalars) {
    returned_scalars.push_back(Load(addr));
  }
  return returned_scalars;
}

std::vector<llvm_ir::IrArray> IrEmitter::ConstructIrArrayForOutputs(
    const HloInstruction& hlo) {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emitterDTcc mht_37(mht_37_v, 937, "", "./tensorflow/compiler/xla/service/gpu/ir_emitter.cc", "IrEmitter::ConstructIrArrayForOutputs");

  std::vector<llvm_ir::IrArray> output_arrays;
  if (hlo.shape().IsTuple()) {
    int64_t num_outputs = ShapeUtil::TupleElementCount(hlo.shape());
    output_arrays.reserve(num_outputs);
    for (int64_t i = 0; i < num_outputs; ++i) {
      output_arrays.push_back(GetIrArray(hlo, hlo, {i}));
    }
  } else {
    output_arrays.push_back(GetIrArray(hlo, hlo));
  }
  return output_arrays;
}

void IrEmitter::BindFusionArguments(const HloInstruction* fusion,
                                    FusedIrEmitter* fused_emitter) {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emitterDTcc mht_38(mht_38_v, 955, "", "./tensorflow/compiler/xla/service/gpu/ir_emitter.cc", "IrEmitter::BindFusionArguments");

  for (int i = 0; i < fusion->operand_count(); i++) {
    const HloInstruction* operand = fusion->operand(i);
    fused_emitter->BindGenerator(
        *fusion->fused_parameter(i),
        [this, operand, fusion](llvm_ir::IrArray::Index index) {
          return GetIrArray(*operand, *fusion)
              .EmitReadArrayElement(index, &b_, operand->name());
        });
  }
}

}  // namespace gpu
}  // namespace xla
