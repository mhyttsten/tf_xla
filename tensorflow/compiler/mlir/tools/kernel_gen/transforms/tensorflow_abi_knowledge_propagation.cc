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
class MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPStensorflow_abi_knowledge_propagationDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPStensorflow_abi_knowledge_propagationDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPStensorflow_abi_knowledge_propagationDTcc() {
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

// This file contains the analysis and transformation to rewrite kernel
// functions such that information about alignment, aliasing and zero offsets
// steming from the tf_framework uses is propagated.

#include <cstdint>
#include <memory>

#include "llvm/ADT/Bitfields.h"
#include "llvm/ADT/DenseMap.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/GPU/GPUDialect.h"  // from @llvm-project
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/ir/tf_framework_ops.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/passes.h"

namespace mlir {
namespace kernel_gen {
namespace transforms {
namespace {

#define GEN_PASS_CLASSES
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/kernel_gen_passes.h.inc"

struct PropagateTfAbiKnowledgeToKernelsPass
    : public PropagateTfAbiKnowledgeToKernelsBase<
          PropagateTfAbiKnowledgeToKernelsPass> {
  void runOnOperation() override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPStensorflow_abi_knowledge_propagationDTcc mht_0(mht_0_v, 216, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/tensorflow_abi_knowledge_propagation.cc", "runOnOperation");

    FuncOp function = getOperation();
    llvm::SmallVector<Value, 4> worklist;
    // We currently only handle entry functions and do not propagate across
    // functions.
    if (function->getAttrOfType<mlir::UnitAttr>(
            tf_framework::TFFrameworkDialect::kTFEntryAttrName)) {
      // For all operands of this function, we know they are aligned. Also, by
      // construction of kernel generator, we know that there is no offset and
      // the inner stride is one.
      // TODO(herhut): Insert asserts in debug mode to check this.
      for (auto argument : function.getArguments()) {
        if (argument.getType().isa<BaseMemRefType>()) {
          worklist.push_back(argument);
          allocated_by_tf_runtime.insert(argument);
          offset_is_zero.insert(argument);
          inner_stride_is_constant.insert({argument, 1});
        }
      }
    }

    // For locally allocated values, we know they are aligned and have offset
    // zero. Further, they also do not alias with other memrefs, except in
    // benign ways. This is by construction and ensured by the reuse analysis.
    function.walk([&](tf_framework::TFAllocOp op) {
      Value allocated = op.getResult();
      worklist.push_back(allocated);
      no_alias.insert(allocated);
      allocated_by_tf_runtime.insert(allocated);
      offset_is_zero.insert(allocated);
      inner_stride_is_constant.insert({allocated, 1});
    });

    // Next, take what we have and propagate it through known operations.
    propagateThroughUses(worklist);

    // Now look at launches and make use of the knowledge we have.
    function.walk([&](gpu::LaunchFuncOp launch) {
      auto module = launch->getParentOfType<ModuleOp>();
      auto kernel = module.lookupSymbol<LLVM::LLVMFuncOp>(launch.kernel());

      if (!kernel || kernel.isExternal()) return;

      // Count the position of kernel operands independently, as they do not
      // coincide with laucnh operands as memref parameters get expanded when
      // lowered to llvm.
      int kernel_p = 0;
      OpBuilder b = OpBuilder::atBlockBegin(&kernel.getBody().front());
      llvm::SmallDenseMap<int64_t, Value> constants;
      auto loc = kernel.getLoc();
      for (auto operand : launch.operands()) {
        auto memref = operand.getType().dyn_cast<MemRefType>();
        if (!memref) {
          // Scalar argument, advance kernel position by one.
          kernel_p++;
          continue;
        }
        if (allocated_by_tf_runtime.contains(operand)) {
          // This was allocated by the tf runtime, so the two pointers in the
          // descriptor coincide. Rewrite the kernel accordingly.
          Value alloc_ptr = kernel.getArgument(kernel_p);
          Value align_ptr = kernel.getArgument(kernel_p + 1);
          alloc_ptr.replaceAllUsesWith(align_ptr);
          kernel.setArgAttr(
              kernel_p + 1, LLVM::LLVMDialect::getAlignAttrName(),
              b.getIndexAttr(
                  tf_framework::TFFrameworkDialect::kAllocationAlignment));
        }
        if (offset_is_zero.contains(operand)) {
          Value offset = kernel.getArgument(kernel_p + 2);
          Value &zero = constants[0];
          if (!zero) {
            zero = b.create<LLVM::ConstantOp>(loc, offset.getType(),
                                              b.getIndexAttr(0));
          }
          offset.replaceAllUsesWith(zero);
        }
        auto const_stride = inner_stride_is_constant.find(operand);
        if (const_stride != inner_stride_is_constant.end()) {
          // The stride is the last argument belonging to this memref.
          Value inner_stride =
              kernel.getArgument(kernel_p + 2 + memref.getRank() * 2);
          Value &stride_val = constants[const_stride->second];
          if (!stride_val) {
            stride_val = b.create<LLVM::ConstantOp>(
                loc, inner_stride.getType(),
                b.getIndexAttr(const_stride->second));
          }
          inner_stride.replaceAllUsesWith(stride_val);
        }
        if (no_alias.contains(operand)) {
          // TODO(herhut): We also need to check whether any of the other args
          //     are aliases. This is currently never the case by construction
          //     but we could use the alias analysis from buffer placement here
          //     to make sure.
          // Add the no_alias attribute to the corresponding pointer.
          kernel.setArgAttr(kernel_p + 1,
                            LLVM::LLVMDialect::getNoAliasAttrName(),
                            b.getUnitAttr());
        }
        // Advance base, aligned, offset, strides and sizes many arguments.
        kernel_p += memref.getRank() * 2 + 3;
      }
    });
  }

 private:
  void propagateThroughUses(SmallVectorImpl<Value> &worklist) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPStensorflow_abi_knowledge_propagationDTcc mht_1(mht_1_v, 326, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/tensorflow_abi_knowledge_propagation.cc", "propagateThroughUses");

    while (!worklist.empty()) {
      Value candidate = worklist.pop_back_val();
      for (auto user : candidate.getUsers()) {
        if (isa<memref::CastOp, memref::ReshapeOp>(user)) {
          // Reshape and Cast propagate alignment, offset and innermost stride.
          // TODO(herhut): This should be a trait.
          Value result = user->getResult(0);
          if (allocated_by_tf_runtime.contains(candidate)) {
            allocated_by_tf_runtime.insert(result);
          }
          auto const_stride = inner_stride_is_constant.find(candidate);
          if (const_stride != inner_stride_is_constant.end()) {
            inner_stride_is_constant.insert({result, const_stride->second});
          }
          if (offset_is_zero.contains(candidate)) {
            offset_is_zero.insert(result);
          }
          worklist.push_back(result);
        }
        if (auto cast = dyn_cast<memref::ReinterpretCastOp>(user)) {
          // Check that we have offset 0.
          Value result = cast.result();
          if (!cast.isDynamicOffset(0) && cast.getStaticOffset(0) == 0) {
            offset_is_zero.insert(result);
          }
          if (allocated_by_tf_runtime.contains(candidate)) {
            allocated_by_tf_runtime.insert(result);
          }
          size_t last_stride = cast.getResultRank() - 1;
          // TODO(herhut): Remove this once canonicalization handles this.
          if (cast.isDynamicStride(last_stride)) {
            auto dyn_stride = cast.getDynamicStride(last_stride)
                                  .getDefiningOp<arith::ConstantIndexOp>();
            if (dyn_stride) {
              inner_stride_is_constant.insert({result, dyn_stride.value()});
            }
          } else {
            inner_stride_is_constant.insert(
                {result, cast.getStaticStride(last_stride)});
          }
          worklist.push_back(result);
        }
      }
    }
  }

  // Set of values that were allocated by the tf runtime and hence are aligned.
  llvm::SmallPtrSet<Value, 8> allocated_by_tf_runtime;
  // Set of values that are known to not have an offset of 0.
  llvm::SmallPtrSet<Value, 8> offset_is_zero;
  // Set of values that are known to have a constant stride.
  llvm::SmallDenseMap<Value, int64_t, 8> inner_stride_is_constant;
  // Set of values we know do not alias other values.
  llvm::SmallPtrSet<Value, 8> no_alias;
};

}  // namespace

std::unique_ptr<OperationPass<FuncOp>>
CreatePropagateTfAbiKnowledgeToKernels() {
  return std::make_unique<PropagateTfAbiKnowledgeToKernelsPass>();
}

}  // namespace transforms
}  // namespace kernel_gen
}  // namespace mlir
