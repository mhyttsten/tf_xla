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
class MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSbuffer_reuse_passDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSbuffer_reuse_passDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSbuffer_reuse_passDTcc() {
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

#include <cstddef>
#include <vector>

#include "llvm/ADT/None.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Analysis/BufferViewFlowAnalysis.h"  // from @llvm-project
#include "mlir/Analysis/Liveness.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/IR/AffineMap.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/ir/tf_framework_ops.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/passes.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/rewriters.h"

constexpr llvm::StringRef
    mlir::kernel_gen::tf_framework::TFAllocOp::kReuseOutputAttrName;
constexpr llvm::StringRef
    mlir::kernel_gen::tf_framework::TFAllocOp::kReuseInputCandidatesAttrName;
constexpr llvm::StringRef
    mlir::kernel_gen::tf_framework::TFFrameworkDialect::kTFEntryAttrName;

namespace mlir {
namespace kernel_gen {
namespace transforms {
namespace {

class BufferReuseAnalysis {
 public:
  explicit BufferReuseAnalysis(FuncOp f) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSbuffer_reuse_passDTcc mht_0(mht_0_v, 219, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/buffer_reuse_pass.cc", "BufferReuseAnalysis");
 build(f); }

  static constexpr int32_t kIndexAmbiguous = -1;

  Optional<SmallVector<int32_t, 2>> get_reuse_candiates(memref::AllocOp op) {
    auto it = reuse_candidates_.find(op);
    if (it == reuse_candidates_.end()) return llvm::None;
    return it->second;
  }

  Optional<int32_t> get_output_index(memref::AllocOp op) {
    auto it = output_indices_.find(op);
    if (it == output_indices_.end()) return llvm::None;
    return it->second;
  }

 private:
  void build(FuncOp &f) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSbuffer_reuse_passDTcc mht_1(mht_1_v, 239, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/buffer_reuse_pass.cc", "build");

    BufferViewFlowAnalysis aliases(f);
    find_output_indices(f, aliases);
    find_reuse_candiates(f, aliases);
  }

  void find_output_indices(FuncOp &f, BufferViewFlowAnalysis &aliases) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSbuffer_reuse_passDTcc mht_2(mht_2_v, 248, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/buffer_reuse_pass.cc", "find_output_indices");

    f.walk([&](memref::AllocOp alloc_op) {
      int32_t output_index = kIndexAmbiguous;
      int count_return_uses = 0;
      auto buffer_aliases = aliases.resolve(alloc_op.getResult());
      for (Value alias : buffer_aliases) {
        for (auto &use : alias.getUses()) {
          if (isa<func::ReturnOp>(use.getOwner())) {
            int32_t index = use.getOperandNumber();
            if (count_return_uses++ == 0)
              output_index = index;
            else if (output_index != index)
              output_index = kIndexAmbiguous;
          }
        }
      }
      output_indices_[alloc_op] = output_index;
    });
  }

  void find_reuse_candiates(FuncOp &f, BufferViewFlowAnalysis &aliases) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSbuffer_reuse_passDTcc mht_3(mht_3_v, 271, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/buffer_reuse_pass.cc", "find_reuse_candiates");

    Liveness liveness(f);
    f.walk([&](Block *block) {
      find_reuse_candiates(block, aliases, liveness.getLiveness(block),
                           f.getArguments());
    });
  }

  void find_reuse_candiates(Block *block, BufferViewFlowAnalysis &aliases,
                            const LivenessBlockInfo *liveness,
                            ArrayRef<BlockArgument> arguments) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSbuffer_reuse_passDTcc mht_4(mht_4_v, 284, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/buffer_reuse_pass.cc", "find_reuse_candiates");

    for (Operation &op : *block) {
      auto alloc_op = dyn_cast<memref::AllocOp>(op);
      if (!alloc_op) continue;

      // Find first use of the newly allocated buffer within this block.
      Value new_buffer = alloc_op.getResult();
      Operation *first_reuse = find_first_use_in_block(new_buffer, block);
      assert((first_reuse == nullptr || first_reuse->getBlock() == block) &&
             "Expected first use in same block if found.");

      // Find reuse candidates for the regarded allocation.
      SmallVector<int32_t, 2> local_reuse_candidates;
      for (BlockArgument old_buffer : arguments) {
        if (!old_buffer.getType().isa<BaseMemRefType>()) continue;

        // Lifetime criterion: Only reuse buffers that are no longer used on
        // first reuse, i.e. they are no longer alive.
        bool lifetimes_compatible = true;
        for (Value old_buffer_alias : aliases.resolve(old_buffer)) {
          if (first_reuse == nullptr) {
            // If the first use is beyond the end of this block we look at the
            // block end. An argument buffer that is already reusable there is
            // certainly reusable at any later actual use. Otherwise, lifetimes
            // are incompatible.
            if (liveness->isLiveOut(old_buffer_alias)) {
              lifetimes_compatible = false;
              break;
            }
          } else {
            // A buffer is reusable if
            //   i)  its last use is before the point of reuse, or
            //   ii) its last use is also its first reuse and the operation
            //       allows for local reuse.
            // Otherwise, lifetimes are incompatible.
            Operation *last_use =
                liveness->getEndOperation(old_buffer_alias, &block->front());
            assert(last_use != nullptr && last_use->getBlock() == block &&
                   "Expected last use in same block.");
            if (first_reuse->isBeforeInBlock(last_use)) {
              lifetimes_compatible = false;
              break;
            }
            if (first_reuse == last_use &&
                !can_reuse_locally(first_reuse, old_buffer_alias, new_buffer)) {
              lifetimes_compatible = false;
              break;
            }
          }
        }

        if (lifetimes_compatible) {
          // All criteria are fulfilled 🙂.
          int32_t old_buffer_index = old_buffer.getArgNumber();
          local_reuse_candidates.push_back(old_buffer_index);
        }
      }

      reuse_candidates_[&op] = local_reuse_candidates;
    }
  }

  Operation *find_first_use_in_block(Value value, Block *block) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSbuffer_reuse_passDTcc mht_5(mht_5_v, 349, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/buffer_reuse_pass.cc", "find_first_use_in_block");

    Operation *first_use = nullptr;
    for (Operation *op : value.getUsers()) {
      Operation *ancestor_op = block->findAncestorOpInBlock(*op);
      if (ancestor_op == nullptr) continue;
      if (first_use == nullptr || ancestor_op->isBeforeInBlock(first_use))
        first_use = ancestor_op;
    }
    return first_use;
  }

  std::vector<Value> get_buffer_arguments(FuncOp &f) {
    std::vector<Value> buffer_arguments;
    for (BlockArgument arg : f.getArguments()) {
      if (arg.getType().isa<BaseMemRefType>()) buffer_arguments.push_back(arg);
    }
    return buffer_arguments;
  }

  bool can_reuse_locally(Operation *op, Value old_buffer, Value new_buffer) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSbuffer_reuse_passDTcc mht_6(mht_6_v, 371, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/buffer_reuse_pass.cc", "can_reuse_locally");

    // For now, we support only memrefs with the same memory layout.
    auto old_buffer_ty = old_buffer.getType().dyn_cast<MemRefType>();
    auto new_buffer_ty = old_buffer.getType().dyn_cast<MemRefType>();
    if (!old_buffer_ty || !new_buffer_ty ||
        old_buffer_ty.getLayout() != new_buffer_ty.getLayout())
      return false;

    if (auto generic_op = dyn_cast<linalg::GenericOp>(op)) {
      SmallVector<OpOperand *> op_operands =
          generic_op.getInputAndOutputOperands();
      auto old_it = llvm::find_if(op_operands, [&](OpOperand *op_operand) {
        return op_operand->get() == old_buffer;
      });
      auto new_it = llvm::find_if(op_operands, [&](OpOperand *op_operand) {
        return op_operand->get() == new_buffer;
      });
      assert(old_it != op_operands.end() && new_it != op_operands.end() &&
             "Expect `old/new_buffer` to be operand of `op`.");

      auto is_projection = [](AffineMap map) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSbuffer_reuse_passDTcc mht_7(mht_7_v, 394, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/buffer_reuse_pass.cc", "lambda");

        // Allow dropping dimensions but no permutations.
        int64_t i = -1;
        for (AffineExpr expr : map.getResults()) {
          auto dim_expr = expr.dyn_cast<AffineDimExpr>();
          if (!dim_expr || dim_expr.getPosition() <= i) return false;
          i = dim_expr.getPosition();
        }
        return true;
      };

      // If `linalg.generic` indexing maps are the same for input and output
      // buffer then the last use of the input buffer happens before its first
      // reuse (per memory location). Since we know that the inputs and outputs
      // have the same size we also know that when one side has an identity map
      // and the other side only drops dimensions, these dimensions have to be
      // of size 1.
      AffineMap old_indexing_map = generic_op.getTiedIndexingMap(*old_it);
      AffineMap new_indexing_map = generic_op.getTiedIndexingMap(*new_it);
      return (old_indexing_map == new_indexing_map &&
              old_indexing_map.isProjectedPermutation()) ||
             (old_indexing_map.isIdentity() &&
              is_projection(new_indexing_map)) ||
             (is_projection(old_indexing_map) && new_indexing_map.isIdentity());
    }
    return false;
  }

  DenseMap<Operation *, SmallVector<int32_t, 2>> reuse_candidates_;
  DenseMap<Operation *, int32_t> output_indices_;
};

#define GEN_PASS_CLASSES
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/kernel_gen_passes.h.inc"

struct BufferReusePass : public BufferReusePassBase<BufferReusePass> {
  void runOnOperation() override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSbuffer_reuse_passDTcc mht_8(mht_8_v, 433, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/buffer_reuse_pass.cc", "runOnOperation");

    if (!getOperation()->getAttrOfType<UnitAttr>(
            tf_framework::TFFrameworkDialect::kTFEntryAttrName))
      return;

    BufferReuseAnalysis analysis(getOperation());

    // Annotate IR with reuse candidates and output indices per allocation.
    Builder builder(&getContext());
    getOperation().walk([&](memref::AllocOp op) {
      if (auto output_index = analysis.get_output_index(op)) {
        auto attr = builder.getI32IntegerAttr(*output_index);
        op.getOperation()->setAttr(
            tf_framework::TFAllocOp::kReuseOutputAttrName, attr);
      }
      if (auto reuse_candiates = analysis.get_reuse_candiates(op)) {
        auto attr = builder.getI32ArrayAttr(*reuse_candiates);
        op.getOperation()->setAttr(
            tf_framework::TFAllocOp::kReuseInputCandidatesAttrName, attr);
      }
    });
  }
};

}  // namespace

std::unique_ptr<OperationPass<FuncOp>> CreateBufferReusePass() {
  return std::make_unique<BufferReusePass>();
}

}  // namespace transforms
}  // namespace kernel_gen
}  // namespace mlir
