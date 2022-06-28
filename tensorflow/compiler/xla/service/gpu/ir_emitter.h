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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_IR_EMITTER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_IR_EMITTER_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emitterDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emitterDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emitterDTh() {
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


#include <functional>
#include <map>
#include <memory>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Value.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/gpu/elemental_ir_emitter.h"
#include "tensorflow/compiler/xla/service/gpu/hlo_to_ir_bindings.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emitter_context.h"
#include "tensorflow/compiler/xla/service/gpu/kernel_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/thunk.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/llvm_ir/fused_ir_emitter.h"
#include "tensorflow/compiler/xla/service/llvm_ir/ir_array.h"
#include "tensorflow/compiler/xla/service/llvm_ir/ir_builder_mixin.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_loop.h"
#include "tensorflow/compiler/xla/service/llvm_ir/loop_emitter.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {
namespace gpu {

// Abstract base class for translating HLO graphs to LLVM IR for a GPU.
//
// There are two concrete subclasses of IrEmitter: IrEmitterNested and
// IrEmitterUnnested.  In the unnested variety, each HLO gets its own kernel
// function, whereas in the nested version the whole computation is emitted as
// one *non-kernel* function.
//
// In XLA, kernel functions never call other kernel functions.  This means that
// if we have a kernel -- e.g. implementing a kReduce HLO -- that wants to use
// an HLO computation as a "subroutine" -- e.g. the HLO computation that
// specifies how to reduce two elements -- then the subroutine computation must
// be emitted using IrEmitterNested.
//
// Fusion nodes are a special case.  A fusion node is emitted using
// IrEmitterUnnested, but the code is generated using FusedIrEmitter, which is
// not a subclass of gpu::IrEmitter, and in fact is better understood as an IR
// generator generator.  See comments on that class.
class IrEmitter : public DfsHloVisitorWithDefault,
                  public IrBuilderMixin<IrEmitter> {
 public:
  IrEmitter(const IrEmitter&) = delete;
  IrEmitter& operator=(const IrEmitter&) = delete;

  Status DefaultAction(HloInstruction* hlo) override;
  Status HandleConstant(HloInstruction* constant) override;
  Status HandleGetTupleElement(HloInstruction* get_tuple_element) override;
  Status HandleConvolution(HloInstruction* convolution) override;
  Status HandleFft(HloInstruction* fft) override;
  Status HandleAllReduce(HloInstruction* crs) override;
  Status HandleInfeed(HloInstruction* infeed) override;
  Status HandleOutfeed(HloInstruction* outfeed) override;
  Status HandleSend(HloInstruction* send) override;
  Status HandleSendDone(HloInstruction* send_done) override;
  Status HandleRecv(HloInstruction* recv) override;
  Status HandleRecvDone(HloInstruction* recv_done) override;
  Status HandleParameter(HloInstruction* parameter) override;
  Status HandleTuple(HloInstruction* tuple) override;
  Status HandleScatter(HloInstruction* scatter) override;
  Status HandleTupleSelect(HloInstruction* tuple_select) override;
  Status HandleFusion(HloInstruction* fusion) override;
  Status HandleCall(HloInstruction* call) override;
  Status HandleCustomCall(HloInstruction* custom_call) override;
  Status HandleBatchNormInference(HloInstruction* batch_norm) override;
  Status HandleBatchNormTraining(HloInstruction* batch_norm) override;
  Status HandleBatchNormGrad(HloInstruction* batch_norm) override;
  Status HandleAddDependency(HloInstruction* add_dependency) override;

  Status FinishVisit(HloInstruction* root) override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emitterDTh mht_0(mht_0_v, 267, "", "./tensorflow/compiler/xla/service/gpu/ir_emitter.h", "FinishVisit");
 return Status::OK(); }

  llvm::IRBuilder<>* builder() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emitterDTh mht_1(mht_1_v, 272, "", "./tensorflow/compiler/xla/service/gpu/ir_emitter.h", "builder");
 return &b_; }

 protected:
  // Constructs an IrEmitter with the given IrEmitter context.
  // ir_emitter_context is owned by the caller and should outlive the IrEmitter
  // object.
  explicit IrEmitter(const HloModuleConfig& hlo_module_config,
                     IrEmitterContext* ir_emitter_context, bool is_nested);

  // Helper for calling HloToIrBindings::GetIrArray.
  //
  // Gets the IrArray which contains inst.  This array has metadata that makes
  // it valid only within the IR that implements consumer.  If you are
  // implementing an HLO and want to get its own output buffer, call
  // GetIrArray(hlo, hlo).
  llvm_ir::IrArray GetIrArray(const HloInstruction& inst,
                              const HloInstruction& consumer,
                              const ShapeIndex& shape_index = {}) {
    return bindings_.GetIrArray(inst, consumer, shape_index);
  }
  // A convenient helper for calling HloToIrBindings::GetBasePointer.
  llvm::Value* GetBasePointer(const HloInstruction& inst,
                              ShapeIndexView shape_index = {}) const {
    return bindings_.GetBasePointer(inst, shape_index);
  }

  // Generates the IrArray for each output of an hlo instruction and returns
  // a vector containing such IrArrays.
  std::vector<llvm_ir::IrArray> ConstructIrArrayForOutputs(
      const HloInstruction& hlo);

  // Emit a singlethreaded or multithreaded loop that computes every element in
  // the result of the given HLO instruction. This produces a series of nested
  // loops (e.g. one for each dimension of the `hlo`'s shape). The body of the
  // inner-most loop is provided by the body_emitter function.
  virtual Status EmitTargetElementLoop(
      const HloInstruction& hlo,
      const llvm_ir::ElementGenerator& body_emitter) = 0;

  // Emits a call in IR to the given nested computation with the given operands
  // and output. If no IR function has been previously emitted for the
  // computation, also emits such a function.
  Status EmitCallToNestedComputation(const HloComputation& nested_computation,
                                     absl::Span<llvm::Value* const> operands,
                                     llvm::Value* output);

  // Emits an atomic operation that implements `nested_computation` in the
  // sequentially consistent memory model. `output_address` and `source_address`
  // are the arguments of the nested computation. For example,
  // atomicAdd(output_address, *source_address).
  Status EmitAtomicOperationForNestedComputation(
      const HloComputation& nested_computation, llvm::Value* output_address,
      llvm::Value* source_address);

  GpuElementalIrEmitter::NestedComputer GetNestedComputer() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emitterDTh mht_2(mht_2_v, 329, "", "./tensorflow/compiler/xla/service/gpu/ir_emitter.h", "GetNestedComputer");

    return std::bind(&IrEmitter::ComputeNestedElement, this,
                     std::placeholders::_1, std::placeholders::_2);
  }

  StatusOr<std::vector<llvm::Value*>> ComputeNestedElement(
      const HloComputation& computation,
      absl::Span<llvm::Value* const> parameter_elements);

  StatusOr<std::vector<llvm::Value*>> ComputeNestedElementFromAddrs(
      const HloComputation& computation,
      absl::Span<llvm::Value* const> parameter_elements_addrs);

  IrEmitterContext* ir_emitter_context_;
  llvm::Module* module_;

  // The following fields track the IR emission state. According to LLVM memory
  // management rules, their memory is owned by the module.
  llvm::IRBuilder<> b_;

  // Mapping from HLO to its underlying LLVM value.
  HloToIrBindings bindings_;

  // Hlo configuration data used during code generation.
  const HloModuleConfig& hlo_module_config_;

  // Bind all argument IrArrays of `fusion` to `fused_emitter`.
  void BindFusionArguments(const HloInstruction* fusion,
                           FusedIrEmitter* fused_emitter);

 private:
  // A helper method for EmitAtomicOperationForNestedComputation. Certain
  // computations, such as floating-point addition and integer maximization, can
  // be simply implemented using an LLVM atomic instruction. If "computation" is
  // one of this kind, emits code to do that and returns true; otherwise,
  // returns false.
  bool MaybeEmitDirectAtomicOperation(const HloComputation& computation,
                                      llvm::Value* output_address,
                                      llvm::Value* source_address);

  // A helper method for EmitAtomicOperationForNestedComputation. It implements
  // binary atomic operations using atomicCAS with special handling to support
  // small data types.
  Status EmitAtomicOperationUsingCAS(const HloComputation& computation,
                                     llvm::Value* output_address,
                                     llvm::Value* source_address);

  // A helper method for HandleSort(). It adds the inner comparison loop where
  // we compare elements pointed to by 'keys_index' and 'compare_keys_index'.
  void EmitCompareLoop(int64_t dimension_to_sort,
                       const llvm_ir::IrArray::Index& keys_index,
                       const llvm_ir::IrArray::Index& compare_keys_index,
                       const llvm_ir::IrArray& keys_array);

  // Emits an atomic operation that implements `nested_computation` in the
  // sequentially consistent memory model. `output_address` and `source_address`
  // are the arguments of the nested computation. For example,
  // atomicAdd(output_address, *source_address).
  StatusOr<llvm::Function*> EmitAtomicFunctionForNestedComputation(
      const HloComputation& nested_computation, llvm::Type* element_ir_type);

  // A convenience method to determine whether or not IR is emitted for AMDGPU.
  bool IsEmittingForAMDGPU() const;

  // Emits atomic add operation for AMD GPU.
  void EmitAMDGPUAtomicAdd(llvm::Value* output_address, llvm::Value* source);

  // A convenience method to determine the proper sync scope for an atomic op.
  llvm::SyncScope::ID DetermineSyncScope() const;

  // Map nested computations to emitted IR functions. This serves as a cache so
  // that IrEmitter does not emit multiple functions for the same
  // HloComputation.
  std::map<const HloComputation*, llvm::Function*> computation_to_ir_function_;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_IR_EMITTER_H_
