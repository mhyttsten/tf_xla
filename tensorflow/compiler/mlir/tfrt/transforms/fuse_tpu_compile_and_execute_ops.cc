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
class MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSfuse_tpu_compile_and_execute_opsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSfuse_tpu_compile_and_execute_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSfuse_tpu_compile_and_execute_opsDTcc() {
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

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"

namespace tensorflow {
namespace {

// This pass rewrites tf._TPUCompileMlirOp and tf.TPUExecuteOp into a single
// tf.TPUCompileMlirAndExecuteOp. Also it removes the unnecessary
// TPUCompileSucceededAssertOp.
class FuseTpuCompileAndExecutePass
    : public mlir::PassWrapper<FuseTpuCompileAndExecutePass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
 public:
  llvm::StringRef getArgument() const final {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSfuse_tpu_compile_and_execute_opsDTcc mht_0(mht_0_v, 205, "", "./tensorflow/compiler/mlir/tfrt/transforms/fuse_tpu_compile_and_execute_ops.cc", "getArgument");

    return "tfrt-fuse-tpu-compile-and-execute-ops";
  }
  llvm::StringRef getDescription() const final {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSfuse_tpu_compile_and_execute_opsDTcc mht_1(mht_1_v, 211, "", "./tensorflow/compiler/mlir/tfrt/transforms/fuse_tpu_compile_and_execute_ops.cc", "getDescription");

    return "Fuse TPU Ops according to TFRT's requirements.";
  }

  void runOnOperation() override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSfuse_tpu_compile_and_execute_opsDTcc mht_2(mht_2_v, 218, "", "./tensorflow/compiler/mlir/tfrt/transforms/fuse_tpu_compile_and_execute_ops.cc", "runOnOperation");

    auto func = getOperation();

    // remove TPUCompileSucceededAssertOp
    func.walk([&](mlir::Operation *op) {
      if (llvm::isa<mlir::TF::TPUCompileSucceededAssertOp>(op)) {
        op->erase();
      }
    });

    // A map from an exec op to a struct containing the static shape tensor from
    // a SetDynamicDimensionBoundsOp and the operand index.
    llvm::SmallDenseMap<
        mlir::TF::TPUExecuteOp,
        llvm::SmallDenseMap<int, mlir::TF::SetStaticDimensionBoundsOp>>
        exec_to_static_shaped_operands_map;

    llvm::SmallVector<mlir::TF::TPUExecuteOp, 4> tpu_execute_ops;
    func.walk([&](mlir::Operation *op) {
      if (auto exec_op = llvm::dyn_cast<mlir::TF::TPUExecuteOp>(op)) {
        tpu_execute_ops.push_back(exec_op);
        // Collect any operands to this tf.Execute op that are defined by a
        // SetStaticDimensionBoundsOp along with the operand index.
        for (const auto &operand : llvm::enumerate(exec_op.getOperands())) {
          if (auto defining_op =
                  operand.value()
                      .getDefiningOp<mlir::TF::SetStaticDimensionBoundsOp>()) {
            exec_to_static_shaped_operands_map[exec_op][operand.index()] =
                defining_op;
          }
        }
      }
    });

    mlir::OpBuilder builder(&func.getBody());

    for (auto exec_op : tpu_execute_ops) {
      auto compile_cache_entry = exec_op.key();
      auto compile_op = ::llvm::dyn_cast<mlir::TF::_TPUCompileMlirOp>(
          compile_cache_entry.getDefiningOp());
      if (!compile_op) {
        exec_op.emitOpError("could not get the _TPUCompileMlirOp");
        signalPassFailure();
        return;
      }

      builder.setInsertionPointAfter(compile_op);
      llvm::SmallVector<mlir::Type, 4> output_types;
      output_types.push_back(mlir::RankedTensorType::get(
          {3}, builder.getType<mlir::TF::StringType>()));
      output_types.insert(output_types.end(), exec_op.getResultTypes().begin(),
                          exec_op.getResultTypes().end());
      llvm::SmallVector<int> static_shaped_operand_indices_attr;
      llvm::SmallVector<mlir::Value> static_shape_tensors;
      llvm::SmallVector<mlir::Value> exec_op_args;
      exec_op_args.resize(exec_op.args().size());

      auto &static_shaped_operands =
          exec_to_static_shaped_operands_map[exec_op];
      for (int i = 0; i < exec_op.args().size(); ++i) {
        auto iter = static_shaped_operands.find(i);
        if (iter != static_shaped_operands.end()) {
          static_shaped_operand_indices_attr.push_back(iter->first);
          static_shape_tensors.push_back(iter->second.static_shape());
          exec_op_args[i] = iter->second.input();
          // The first operand is the input tensor, while the second operand is
          // the static shape tensor, hence the drop_back here.
          iter->second->replaceAllUsesWith(
              mlir::ValueRange({iter->second.input()}));
          iter->second->erase();
        } else {
          exec_op_args[i] = exec_op->getOperand(i);
        }
      }

      auto producer_name =
          exec_op->getAttrOfType<mlir::StringAttr>("_producer_name");
      if (!producer_name)
        producer_name = mlir::StringAttr::get(&getContext(), "default");
      auto compile_and_execute_op =
          builder.create<mlir::TF::TPUCompileMlirAndExecuteOp>(
              exec_op.getLoc(), output_types, exec_op_args,
              static_shape_tensors,
              builder.getI32ArrayAttr(static_shaped_operand_indices_attr),
              compile_op.mlir_module(), compile_op.metadata(), producer_name);

      exec_op.replaceAllUsesWith(compile_and_execute_op.results());
      for (auto program_result : compile_op.program()) {
        program_result.replaceAllUsesWith(
            compile_and_execute_op.rendezvous_key_base());
      }

      assert(exec_op.use_empty());
      exec_op.erase();
      assert(compile_op.use_empty());
      compile_op.erase();
    }
  }
};

}  // namespace

namespace tfrt_compiler {

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateFuseTpuCompileAndExecutePass() {
  return std::make_unique<FuseTpuCompileAndExecutePass>();
}

static mlir::PassRegistration<FuseTpuCompileAndExecutePass>
    fuse_tpu_compile_and_execute_ops_pass;

}  // namespace tfrt_compiler

}  // namespace tensorflow
