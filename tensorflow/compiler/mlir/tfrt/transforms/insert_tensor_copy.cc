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
class MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSinsert_tensor_copyDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSinsert_tensor_copyDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSinsert_tensor_copyDTcc() {
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
#include "tensorflow/compiler/mlir/tfrt/ir/tfrt_fallback.h"
#include "tensorflow/compiler/mlir/tfrt/ir/tfrt_fallback_async.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/passes.h"
#include "tfrt/basic_kernels/opdefs/basic_kernels.h"  // from @tf_runtime
#include "tfrt/basic_kernels/opdefs/tfrt_base.h"  // from @tf_runtime
#include "tfrt/compiler/stream_analysis.h"  // from @tf_runtime

namespace tensorflow {
namespace tfrt_compiler {
namespace {

// This pass inserts copy kernels for fallback tensors when they are passed to
// multiple threads, to avoid atomic contention on their refcounts.
class InsertFallbackTensorCopy
    : public mlir::PassWrapper<InsertFallbackTensorCopy,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  void getDependentDialects(mlir::DialectRegistry& registry) const override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSinsert_tensor_copyDTcc mht_0(mht_0_v, 200, "", "./tensorflow/compiler/mlir/tfrt/transforms/insert_tensor_copy.cc", "getDependentDialects");

    registry.insert<tfrt::fallback_async::FallbackAsyncDialect>();
  }

  llvm::StringRef getArgument() const final {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSinsert_tensor_copyDTcc mht_1(mht_1_v, 207, "", "./tensorflow/compiler/mlir/tfrt/transforms/insert_tensor_copy.cc", "getArgument");

    return "tfrt-insert-fallback-tensor-copy";
  }

  llvm::StringRef getDescription() const final {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSinsert_tensor_copyDTcc mht_2(mht_2_v, 214, "", "./tensorflow/compiler/mlir/tfrt/transforms/insert_tensor_copy.cc", "getDescription");

    return "Inserts copy kernels for fallback tensors when they are passed to "
           "multiple threads, to avoid atomic contention on refcounts.";
  }

 public:
  void runOnOperation() override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSinsert_tensor_copyDTcc mht_3(mht_3_v, 223, "", "./tensorflow/compiler/mlir/tfrt/transforms/insert_tensor_copy.cc", "runOnOperation");

    mlir::func::FuncOp func_op = getOperation();

    // Use stream analysis to know whether a value is passed to different
    // threads.
    tfrt::compiler::StreamAnalysis stream_analysis(func_op);

    auto builder = mlir::OpBuilder::atBlockBegin(&func_op.front());

    // Process function arguments first.
    for (auto arg : func_op.getArguments()) {
      if (!arg.getType().isa<tfrt::fallback::TFTensorType>()) continue;
      InsertFallbackTensorCopyForValue(arg, func_op->getLoc(), builder,
                                       stream_analysis);
    }

    // Then process each operations in the block.
    for (mlir::Operation& op : llvm::make_early_inc_range(func_op.front())) {
      if (llvm::isa<tfrt::fallback_async::ExecuteOp,
                    tfrt::fallback_async::ExecuteOpSeq>(&op)) {
        InsertFallbackTensorCopyForFallbackOp(&op, builder, stream_analysis);
      }
    }
  }

 private:
  void InsertFallbackTensorCopyForFallbackOp(
      mlir::Operation* op, mlir::OpBuilder& builder,
      const tfrt::compiler::StreamAnalysis& stream_analysis) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSinsert_tensor_copyDTcc mht_4(mht_4_v, 254, "", "./tensorflow/compiler/mlir/tfrt/transforms/insert_tensor_copy.cc", "InsertFallbackTensorCopyForFallbackOp");

    builder.setInsertionPointAfter(op);

    // Process each result value.
    for (auto result : op->getResults()) {
      if (!result.getType().isa<tfrt::fallback::TFTensorType>()) continue;
      InsertFallbackTensorCopyForValue(result, op->getLoc(), builder,
                                       stream_analysis);
    }
  }

  // Insert copy kernels to copy the result, and allocate new atomic refcount
  // if the value is going to be used by different streams/threads, in order to
  // avoid contention on the atomic counter.
  void InsertFallbackTensorCopyForValue(
      mlir::Value value, mlir::Location loc, mlir::OpBuilder& builder,
      const tfrt::compiler::StreamAnalysis& stream_analysis) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSinsert_tensor_copyDTcc mht_5(mht_5_v, 273, "", "./tensorflow/compiler/mlir/tfrt/transforms/insert_tensor_copy.cc", "InsertFallbackTensorCopyForValue");

    llvm::DenseMap<int, llvm::SmallVector<mlir::OpOperand*, 4>> stream_map;

    // Find out streams that use this value and the corresponding uses.
    for (mlir::OpOperand& use : value.getUses()) {
      // Skip return op as there should not be atomic contention on the return
      // op.
      if (llvm::isa<tfrt::compiler::ReturnOp>(use.getOwner())) continue;

      int stream_id = stream_analysis.GetStream(use.getOwner()).id();
      stream_map[stream_id].push_back(&use);
    }

    // Organize these uses into groups. If a stream has many uses of this value,
    // put these uses into one stream. Otherwise, streams with small number
    // of uses are grouped with each other to form groups with enough uses.
    constexpr int kCopyGroupThreshold = 16;
    llvm::SmallVector<llvm::SmallVector<mlir::OpOperand*, 4>, 4> small_copies;
    llvm::SmallVector<llvm::SmallVector<mlir::OpOperand*, 4>, 4> copies;
    for (const auto& iter : stream_map) {
      if (iter.second.size() >= kCopyGroupThreshold) {
        copies.push_back(iter.second);
      } else {
        if (small_copies.empty() ||
            small_copies.back().size() >= kCopyGroupThreshold) {
          small_copies.push_back(iter.second);
        } else {
          small_copies.back().append(iter.second.begin(), iter.second.end());
        }
      }
    }

    if (!small_copies.empty())
      copies.append(small_copies.begin(), small_copies.end());

    // If it is only used by one group, then we don't need to copy.
    if (copies.size() <= 1) return;

    // Remove one group from the candidates, as we can just use the original
    // value for this group.
    copies.pop_back();

    // For each stream, we will create one new value that replaces the uses in
    // that stream.

    assert(value.getType().isa<tfrt::fallback::TFTensorType>());

    // The number of results is the number candidate streams.
    llvm::SmallVector<mlir::Type, 4> result_types(copies.size(),
                                                  value.getType());
    assert(!result_types.empty());

    // Create the tfrt_fallback_async.copy_if_small kernel.
    auto copy_op = builder.create<tfrt::fallback_async::CopyIfSmallOp>(
        loc, result_types, value);

    // Finally, replaces all uses with the new value.
    for (int i = 0; i < copies.size(); ++i) {
      const auto& uses = copies[i];
      auto new_value = copy_op.getResult(i);
      for (auto* use : uses) {
        use->set(new_value);
      }
    }
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateInsertFallbackTensorCopyPass() {
  return std::make_unique<InsertFallbackTensorCopy>();
}

static mlir::PassRegistration<InsertFallbackTensorCopy> register_pass(
    CreateInsertFallbackTensorCopyPass);

}  // namespace tfrt_compiler
}  // namespace tensorflow
