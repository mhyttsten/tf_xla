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
class MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPScopy_cleanup_passDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPScopy_cleanup_passDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPScopy_cleanup_passDTcc() {
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

#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Interfaces/SideEffectInterfaces.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/passes.h"

namespace mlir {
namespace kernel_gen {
namespace transforms {
namespace {
#define GEN_PASS_CLASSES
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/kernel_gen_passes.h.inc"

// A pass to remove memref::AllocOps and memref::CopyOps ops.
//
// The idea behind this pass is to collect all patterns we are interested in in
// a single place. Eventually, this should be replaced by a generalized copy
// removal pass.

// Handles the pattern where an input operand of a linalg generic is copied
// even though the producer is not mutated.
void RemoveCopyIfTargetOnlyRead(FuncOp func) {
  llvm::SmallVector<memref::AllocOp, 8> allocs_to_remove;
  llvm::SmallVector<memref::CopyOp, 8> copies_to_remove;

  // Gather all allocs and copies which are only read and have an immutable
  // source.
  func->walk([&](memref::AllocOp op) {
    memref::CopyOp copy;
    MemoryEffectOpInterface reader;
    bool at_most_one_copy = true;
    bool at_most_one_read = true;
    for (auto user : op->getUsers()) {
      if (auto copy_user = dyn_cast<memref::CopyOp>(user)) {
        if (copy) {
          at_most_one_copy = false;
        } else {
          copy = copy_user;
        }
        continue;
      }
      if (auto effect_interface = cast<MemoryEffectOpInterface>(user)) {
        if (reader) {
          at_most_one_read = false;
        } else {
          reader = effect_interface;
        }
        SmallVector<MemoryEffects::EffectInstance, 2> effects;
        effect_interface.getEffectsOnValue(op.getResult(), effects);
        if (llvm::any_of(effects, [](MemoryEffects::EffectInstance it) {
              return !isa<MemoryEffects::Read>(it.getEffect());
            })) {
          at_most_one_read = false;
        }
        continue;
      }
      // We don't understand this use, be conservative.
      at_most_one_read = false;
    }
    if (!copy || !at_most_one_copy) return;
    if (!reader || !at_most_one_read) return;
    // The copy should have the alloc op as target.
    if (copy.getTarget() != op.getResult()) return;

    // The copy should be before the reading use.
    if (copy->getBlock() != reader->getBlock() ||
        !copy->isBeforeInBlock(reader)) {
      return;
    }

    // No write effects between copy and use. With aliasing information, this
    // could be made more precise but for now we have to be conservative. The
    // only thing we allow are writes to values that are allocated after the
    // copy, as the aliasing is clear in those cases.
    bool source_is_mutated = false;
    for (Operation *pos = copy->getNextNode(), *end = reader; pos != end;
         pos = pos->getNextNode()) {
      auto effect_interface = dyn_cast<MemoryEffectOpInterface>(pos);
      if (!effect_interface) {
        continue;
      }
      SmallVector<MemoryEffects::EffectInstance, 2> effects;
      effect_interface.getEffects<MemoryEffects::Write>(effects);
      for (auto effect : effects) {
        if (auto alloc = effect.getValue().getDefiningOp<memref::AllocOp>()) {
          if (alloc->getBlock() == copy->getBlock() &&
              copy->isBeforeInBlock(alloc)) {
            continue;
          }
        }
        source_is_mutated = true;
        break;
      }
    }
    if (source_is_mutated) return;

    op->replaceAllUsesWith(ValueRange{copy.getSource()});
    allocs_to_remove.push_back(op);
    copies_to_remove.push_back(copy);
  });
  llvm::for_each(allocs_to_remove, [](Operation *op) { op->erase(); });
  llvm::for_each(copies_to_remove, [](Operation *op) { op->erase(); });
}

// Handles the case where the last instructions of a function implements a copy
// back to a function argument.
void RemoveCopyIfTargetIsFunctionArg(FuncOp func) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPScopy_cleanup_passDTcc mht_0(mht_0_v, 296, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/copy_cleanup_pass.cc", "RemoveCopyIfTargetIsFunctionArg");

  // For now only support this on functions with a single block.
  if (!func.getBody().hasOneBlock()) return;

  llvm::SmallVector<memref::AllocOp> allocs_to_remove;
  llvm::SmallVector<memref::CopyOp> copies_to_remove;

  Block &body = func.getBody().front();
  for (auto &op : llvm::reverse(body.without_terminator())) {
    if (auto copy = dyn_cast<memref::CopyOp>(op)) {
      auto block_arg = copy.getTarget().dyn_cast<BlockArgument>();
      if (!block_arg) break;
      if (!isa<FuncOp>(block_arg.getOwner()->getParentOp()) ||
          !block_arg.hasOneUse())
        break;
      auto alloc = copy.getSource().getDefiningOp<memref::AllocOp>();
      if (!alloc) break;
      alloc->replaceAllUsesWith(ValueRange{block_arg});
      allocs_to_remove.push_back(alloc);
      copies_to_remove.push_back(copy);
      continue;
    }
    break;
  }
  llvm::for_each(allocs_to_remove, [](Operation *op) { op->erase(); });
  llvm::for_each(copies_to_remove, [](Operation *op) { op->erase(); });
}

}  // namespace

struct CopyCleanupPass : public CopyCleanupPassBase<CopyCleanupPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPScopy_cleanup_passDTcc mht_1(mht_1_v, 330, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/copy_cleanup_pass.cc", "getDependentDialects");

    registry.insert<memref::MemRefDialect>();
  }

  void runOnOperation() override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPScopy_cleanup_passDTcc mht_2(mht_2_v, 337, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/copy_cleanup_pass.cc", "runOnOperation");

    RemoveCopyIfTargetOnlyRead(getOperation());
    RemoveCopyIfTargetIsFunctionArg(getOperation());
  }
};

std::unique_ptr<OperationPass<FuncOp>> CreateCopyCleanupPass() {
  return std::make_unique<CopyCleanupPass>();
}

}  // namespace transforms
}  // namespace kernel_gen
}  // namespace mlir
