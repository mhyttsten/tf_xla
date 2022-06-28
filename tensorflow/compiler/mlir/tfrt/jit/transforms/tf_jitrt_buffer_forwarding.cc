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
class MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_buffer_forwardingDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_buffer_forwardingDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_buffer_forwardingDTcc() {
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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_passes.h"
namespace tensorflow {
namespace {

#define GEN_PASS_CLASSES
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_passes.h.inc"

// Returns true if all linalg.generic operation iterators are "parallel".
bool AllIteratorsAreParallel(mlir::linalg::GenericOp op) {
  return llvm::all_of(op.iterator_types(), [](mlir::Attribute attr) -> bool {
    auto str_attr = attr.dyn_cast<mlir::StringAttr>();
    return str_attr && str_attr.getValue() == "parallel";
  });
}

// Returns buffer inputs that can be safely used as buffer outputs.
llvm::SmallVector<mlir::OpOperand*> FindBufferForwardingCandidates(
    mlir::linalg::GenericOp op) {
  llvm::SmallVector<mlir::OpOperand*> candidates;

  for (mlir::OpOperand* input_buffer : op.getInputOperands()) {
    // Input must be a contiguous memref ...
    if (!IsContiguousMemref(input_buffer->get())) continue;

    // ... allocated in the same function.
    auto* alloc = input_buffer->get().getDefiningOp();
    if (!alloc || !mlir::isa<mlir::memref::AllocOp>(alloc)) continue;

    // Find input users that are after linalg.generic operation in the block.
    auto users = llvm::make_filter_range(alloc->getUsers(),
                                         [&](mlir::Operation* user) -> bool {
                                           return op->isBeforeInBlock(user);
                                         });

    // Input buffer must have exactly one user after linalg.generic.
    llvm::SmallVector<mlir::Operation*> input_users(users.begin(), users.end());
    if (input_users.size() != 1) continue;

    // And it must be a memref.dealloc operation.
    if (!mlir::isa<mlir::memref::DeallocOp>(input_users[0])) continue;

    // This input memref can be safely reused for the output.
    candidates.push_back(input_buffer);
  }

  return candidates;
}

struct ForwardingCandidate {
  mlir::Value memref;
  mlir::AffineMap indexing_map;
};

struct LinalgTrivialBufferForwardingPattern
    : public mlir::OpRewritePattern<mlir::linalg::GenericOp> {
  using OpRewritePattern<mlir::linalg::GenericOp>::OpRewritePattern;
  mlir::LogicalResult matchAndRewrite(
      mlir::linalg::GenericOp op,
      mlir::PatternRewriter& rewriter) const override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_buffer_forwardingDTcc mht_0(mht_0_v, 246, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_buffer_forwarding.cc", "matchAndRewrite");

    // With parallel iterators it is guaranteed that every value used once, and
    // it is safe to forward input to output.
    if (!AllIteratorsAreParallel(op))
      return rewriter.notifyMatchFailure(op, "all iterators must be parallel");

    // Find memrefs that potentially could be forwarded.
    llvm::SmallVector<mlir::OpOperand*> forwarding_candidates =
        FindBufferForwardingCandidates(op);
    if (forwarding_candidates.empty()) {
      return rewriter.notifyMatchFailure(
          op, "did not find any candidates for input forwarding");
    }

    // Inputs that were reused.
    llvm::DenseSet<mlir::OpOperand*> reused_inputs;

    // Try to match output buffers to forwarding candidates.
    for (mlir::OpOperand* output_buffer : op.getOutputOperands()) {
      // Output must be allocated in the same function.
      auto* alloc = output_buffer->get().getDefiningOp();
      if (!alloc || !mlir::isa<mlir::memref::AllocOp>(alloc)) continue;

      // We cannot forward the buffer if there are any users before the
      // linalg.generic op in the block.
      if (llvm::any_of(alloc->getUsers(), [op](mlir::Operation* user) {
            return user->isBeforeInBlock(op);
          })) {
        continue;
      }

      // Find compatible input buffer.
      for (mlir::OpOperand* input_buffer : forwarding_candidates) {
        if (reused_inputs.contains(input_buffer)) continue;

        // Memref types must match (dimensions and affine maps).
        if (input_buffer->get().getType() != output_buffer->get().getType())
          continue;

        mlir::AffineMap src_map = op.getTiedIndexingMap(input_buffer);
        mlir::AffineMap dst_map = op.getTiedIndexingMap(output_buffer);

        // Only support identity maps for the output for now.
        if (!dst_map.isIdentity()) continue;

        auto is_projection = [](mlir::AffineMap map) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_buffer_forwardingDTcc mht_1(mht_1_v, 294, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_buffer_forwarding.cc", "lambda");

          // Allow adding/dropping dimensions but no permutations.
          int64_t i = -1;
          for (mlir::AffineExpr expr : map.getResults()) {
            auto constant = expr.dyn_cast<mlir::AffineConstantExpr>();
            if (constant && constant.getValue() == 0) continue;
            auto dim_expr = expr.dyn_cast<mlir::AffineDimExpr>();
            if (!dim_expr || dim_expr.getPosition() <= i) return false;
            i = dim_expr.getPosition();
          }
          return true;
        };

        auto same_shape = [](mlir::Value src, mlir::Value dst) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_buffer_forwardingDTcc mht_2(mht_2_v, 310, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_buffer_forwarding.cc", "lambda");

          auto src_type = src.getType().cast<mlir::ShapedType>();
          auto dst_type = dst.getType().cast<mlir::ShapedType>();
          mlir::OperandRange src_operands =
              src.getDefiningOp<mlir::memref::AllocOp>().getDynamicSizes();
          mlir::OperandRange dst_operands =
              dst.getDefiningOp<mlir::memref::AllocOp>().getDynamicSizes();
          return src_type.getShape().equals(dst_type.getShape()) &&
                 std::equal(src_operands.begin(), src_operands.end(),
                            dst_operands.begin());
        };

        // A reuse is valid if the maps are the same or if the shape is the same
        // and the source is a projection map (in which case the ignored
        // dimensions must be 1 assuming that the operation reads the entire
        // input). Note that we already know that the destination map is an
        // identity map.
        if (src_map != dst_map &&
            !(is_projection(src_map) &&
              same_shape(input_buffer->get(), output_buffer->get()))) {
          continue;
        }

        // Find the input buffer dealloc operation.
        mlir::Operation* input_dealloc = *llvm::find_if(
            input_buffer->get().getUsers(), [](mlir::Operation* user) -> bool {
              return mlir::isa<mlir::memref::DeallocOp>(user);
            });

        // Deallocate output buffer instead of the input buffer.
        input_buffer->get().replaceUsesWithIf(
            output_buffer->get(), [&](mlir::OpOperand& operand) -> bool {
              return operand.getOwner() == input_dealloc;
            });

        // Forward users of output buffer to the input buffer, if they are after
        // linalg.generic operation in the block (or linalg.generic itself).
        output_buffer->get().replaceUsesWithIf(
            input_buffer->get(), [&](mlir::OpOperand& operand) -> bool {
              return operand.getOwner() != input_dealloc &&
                     !operand.getOwner()->isBeforeInBlock(op);
            });

        reused_inputs.insert(input_buffer);
        // We have found an input buffer which we can forward. No need to keep
        // looking for another input buffer to forward.
        break;
      }
    }

    return mlir::success(!reused_inputs.empty());
  }
};

// -------------------------------------------------------------------------- //
// Trivial buffer forwarding for the linalg.generic operations.
// -------------------------------------------------------------------------- //
struct LinalgTrivialBufferForwardingPass
    : public LinalgTrivialBufferForwardingBase<
          LinalgTrivialBufferForwardingPass> {
  void runOnOperation() override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_buffer_forwardingDTcc mht_3(mht_3_v, 373, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_buffer_forwarding.cc", "runOnOperation");

    mlir::func::FuncOp function = getOperation();
    mlir::MLIRContext* ctx = function.getContext();

    mlir::RewritePatternSet patterns(ctx);
    patterns.add<LinalgTrivialBufferForwardingPattern>(ctx);

    (void)mlir::applyPatternsAndFoldGreedily(function, std::move(patterns));
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateLinalgTrivialBufferForwardingPass() {
  return std::make_unique<LinalgTrivialBufferForwardingPass>();
}

}  // namespace tensorflow
