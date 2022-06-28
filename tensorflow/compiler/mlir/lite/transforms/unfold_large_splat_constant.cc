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
class MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSunfold_large_splat_constantDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSunfold_large_splat_constantDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSunfold_large_splat_constantDTcc() {
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

#include <utility>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_n_z.h"

namespace mlir {
namespace TFL {
namespace {

// The threshold of constant bits to be unfolded (1Mb). If there is a splat
// constant with size equal or greater to this threshold, then it will be
// unfolded back to a regular `tfl.fill` operation.
constexpr int64_t kConstantSizeThresholdInBits = 1e+6;

// Pass which will replace large splat constant tensors to `tfl.Fill` op to
// reduce the size of the generated flatbuffer model size.
class UnfoldLargeSplatConstant
    : public PassWrapper<UnfoldLargeSplatConstant, OperationPass<ModuleOp>> {
 public:
  void getDependentDialects(DialectRegistry& registry) const override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSunfold_large_splat_constantDTcc mht_0(mht_0_v, 216, "", "./tensorflow/compiler/mlir/lite/transforms/unfold_large_splat_constant.cc", "getDependentDialects");

    registry.insert<TFL::TensorFlowLiteDialect>();
  }

  StringRef getArgument() const final {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSunfold_large_splat_constantDTcc mht_1(mht_1_v, 223, "", "./tensorflow/compiler/mlir/lite/transforms/unfold_large_splat_constant.cc", "getArgument");

    // This is the argument used to refer to the pass in
    // the textual format (on the commandline for example).
    return "tfl-unfold-large-splat-constant";
  }
  StringRef getDescription() const final {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSunfold_large_splat_constantDTcc mht_2(mht_2_v, 231, "", "./tensorflow/compiler/mlir/lite/transforms/unfold_large_splat_constant.cc", "getDescription");

    // This is a brief description of the pass.
    return "Unfold large splat constant tensors";
  }

  void runOnOperation() override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSunfold_large_splat_constantDTcc mht_3(mht_3_v, 239, "", "./tensorflow/compiler/mlir/lite/transforms/unfold_large_splat_constant.cc", "runOnOperation");

    auto module = getOperation();

    mlir::OpBuilder op_builder(&module.getBodyRegion());
    module.walk([&](mlir::arith::ConstantOp const_op) {
      MaybeUnfoldLargeSplatConstant(&op_builder, const_op);
    });
  }

 private:
  void MaybeUnfoldLargeSplatConstant(mlir::OpBuilder* op_builder,
                                     mlir::arith::ConstantOp const_op) const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSunfold_large_splat_constantDTcc mht_4(mht_4_v, 253, "", "./tensorflow/compiler/mlir/lite/transforms/unfold_large_splat_constant.cc", "MaybeUnfoldLargeSplatConstant");

    auto splat_elements_attr =
        const_op.getValue().dyn_cast<SplatElementsAttr>();
    if (!splat_elements_attr) {
      return;
    }
    auto element_type = splat_elements_attr.getType().getElementType();
    if (!(element_type.isF32() || element_type.isInteger(1) ||
          element_type.isInteger(32) || element_type.isInteger(64))) {
      return;
    }
    if (splat_elements_attr.getNumElements() *
            splat_elements_attr.getType().getElementTypeBitWidth() <
        kConstantSizeThresholdInBits) {
      return;
    }

    op_builder->setInsertionPoint(const_op);
    mlir::arith::ConstantOp fill_shape =
        op_builder->create<mlir::arith::ConstantOp>(
            const_op->getLoc(),
            DenseIntElementsAttr::get(
                RankedTensorType::get({splat_elements_attr.getType().getRank()},
                                      op_builder->getI64Type()),
                splat_elements_attr.getType().getShape()));
    mlir::arith::ConstantOp fill_value =
        op_builder->create<mlir::arith::ConstantOp>(
            const_op->getLoc(),
            DenseElementsAttr::get(
                RankedTensorType::get(
                    {}, splat_elements_attr.getType().getElementType()),
                splat_elements_attr.getSplatValue<Attribute>()));
    TFL::FillOp fill = op_builder->create<TFL::FillOp>(
        const_op->getLoc(), splat_elements_attr.getType(), fill_shape,
        fill_value);
    const_op->replaceAllUsesWith(fill);
    const_op->erase();
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateUnfoldLargeSplatConstantPass() {
  return std::make_unique<UnfoldLargeSplatConstant>();
}

static PassRegistration<UnfoldLargeSplatConstant> pass;

}  // namespace TFL
}  // namespace mlir
