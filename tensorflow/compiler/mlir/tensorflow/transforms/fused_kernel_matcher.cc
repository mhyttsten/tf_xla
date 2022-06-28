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
class MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSfused_kernel_matcherDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSfused_kernel_matcherDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSfused_kernel_matcherDTcc() {
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

#include <cstdio>
#include <iostream>

#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes_detail.h"

namespace mlir {

namespace TF {

namespace {

// Note: This implements the fusions performed in the old Remapper Grappler
// pass. That pass has specific cases for GPU and based on different
// target configurations on both CPU and GPU (Intel MKL, ROCm, etc.). This MLIR
// pass covers (some of) the general CPU case and at the moment does not account
// for any target-specific configurations.

// This pass is being ported over from the Grappler Remapper pass based on
// need/usage. File a bug to request porting over additional fusions.

// TODO(b/158265178): Support GPU-specific fusions.
// TODO(b/158266710): Support CPU MKL configurations.

// Optimizes TF computations by fusing subgraphs/nodes onto more efficient
// implementations to decrease the number of operations needed to perform a
// computation.
struct FusedKernelMatcherPass
    : public FusedKernelMatcherPassBase<FusedKernelMatcherPass> {
  void runOnOperation() override;
};

bool IsActivationFunction(Operation *op) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSfused_kernel_matcherDTcc mht_0(mht_0_v, 227, "", "./tensorflow/compiler/mlir/tensorflow/transforms/fused_kernel_matcher.cc", "IsActivationFunction");

  return isa<EluOp, ReluOp, Relu6Op>(op);
}

// Finds and returns an activation op that uses the result of `op`. If there are
// multiple such activations, one is returned (with no guarantee as to which
// one). If there are no activation functions that use the output, returns
// nullptr.
Operation *GetActivation(Value op) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSfused_kernel_matcherDTcc mht_1(mht_1_v, 238, "", "./tensorflow/compiler/mlir/tensorflow/transforms/fused_kernel_matcher.cc", "GetActivation");

  for (auto &use : op.getUses()) {
    if (IsActivationFunction(use.getOwner())) return use.getOwner();
  }
  return nullptr;
}

// Finds and returns a BiasAdd that uses the result of `op` as the `value`
// input. If there are multiple such BiasAdds, one is returned (with no
// guarantee as to which one). If there are no BiasAdds that use the output,
// returns a null BiasAddOp.
BiasAddOp GetBiasAdd(Value op) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSfused_kernel_matcherDTcc mht_2(mht_2_v, 252, "", "./tensorflow/compiler/mlir/tensorflow/transforms/fused_kernel_matcher.cc", "GetBiasAdd");

  for (auto &use : op.getUses()) {
    auto bias_add = dyn_cast_or_null<BiasAddOp>(use.getOwner());
    // If it's a BiasAdd, check that the conv op is the first input.
    if (bias_add && bias_add.value() == op) return bias_add;
  }
  // No BiasAddOps found among uses.
  return BiasAddOp();
}

// Performs a fusion of the following pattern(s), if possible:
//   <Contraction> + BiasAdd + <Activation> -> <FusedContraction>
//
// Note that fusion with activation is preferred, but a contraction and BiasAdd
// can also be replaced by a _FusedConv2D if there is no other activation
// function.
// i.e., this class also supports the following fusion:
//   <Contraction> + BiasAdd -> <FusedContraction>
//
// TODO(b/158266331): Support fusing activation chains of arbitrary length.
template <typename SrcOpT, typename FusedOpT>
class FuseContractionWithBiasAdd : public OpRewritePattern<SrcOpT> {
 public:
  using OpRewritePattern<SrcOpT>::OpRewritePattern;
  // Class users should override this method if there are any op-specific
  // compatibility requirements between the contraction op and the BiasAdd op.
  virtual bool AreFuseCompatible(SrcOpT contraction_op, BiasAddOp bias_add,
                                 PatternRewriter &rewriter) const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSfused_kernel_matcherDTcc mht_3(mht_3_v, 282, "", "./tensorflow/compiler/mlir/tensorflow/transforms/fused_kernel_matcher.cc", "AreFuseCompatible");

    return true;
  }

  LogicalResult matchAndRewrite(SrcOpT contraction,
                                PatternRewriter &rewriter) const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSfused_kernel_matcherDTcc mht_4(mht_4_v, 290, "", "./tensorflow/compiler/mlir/tensorflow/transforms/fused_kernel_matcher.cc", "matchAndRewrite");

    auto context = rewriter.getContext();

    // We do support fusion only if the contraction operation is inside one of
    // the expected operations with regions. Other operations can have semantics
    // that is not compatible with fusion (e.g. region compilation).
    if (!isa<FuncOp, IfOp, WhileOp>(contraction->getParentOp())) {
      return rewriter.notifyMatchFailure(
          contraction,
          "fused operation must be nested inside a function, If or While");
    }

    // If the contraction is used in multiple places, fusing it will only create
    // more contraction nodes, which is slower.
    if (!contraction.getResult().hasOneUse())
      return rewriter.notifyMatchFailure(contraction,
                                         "result is used by multiple ops");

    BiasAddOp bias_add = GetBiasAdd(contraction.getResult());
    if (!bias_add) {
      return rewriter.notifyMatchFailure(
          contraction, "does not feed into a tf.BiasAdd/tf.BiasAddV1 op");
    }

    if (!AreFuseCompatible(contraction, bias_add, rewriter)) {
      return rewriter.notifyMatchFailure(
          contraction, "cannot fuse with the subsequent BiasAdd op");
    }

    SmallVector<Location, 3> locations{contraction.getLoc(), bias_add.getLoc()};
    SmallVector<Attribute, 2> fused_ops{StringAttr::get(
        context, bias_add.getOperation()->getName().stripDialect())};

    // BiasAdd may or may not feed into an activation function.
    auto activation = GetActivation(bias_add);

    // If there is an activation, only fuse it if this is the only op to use the
    // result of the BiasAdd.
    bool fuse_activation = activation && bias_add.output().hasOneUse();
    Type result_type;

    // Include info about the activation function if applicable.
    if (fuse_activation) {
      locations.push_back(activation->getLoc());
      fused_ops.push_back(
          StringAttr::get(context, activation->getName().stripDialect()));
      result_type = activation->getResultTypes().front();
    } else {
      result_type = bias_add.getResult().getType();
    }

    auto fused_loc = rewriter.getFusedLoc(locations);

    // The fused contraction has the same operands as the original contraction
    // with `bias` from the BiasAddOp appended.
    SmallVector<Value, 4> operands(contraction.operand_begin(),
                                   contraction.operand_end());
    operands.push_back(bias_add.bias());

    // The fused contraction has the same attributes as the original
    // contraction, with two additions: the list of ops which have been fused
    // together; epsilon (only with FusedBatchNorm).
    std::vector<NamedAttribute> attrs = contraction->getAttrs();
    ArrayAttr fused_ops_attr = ArrayAttr::get(context, fused_ops);
    attrs.push_back(
        NamedAttribute(StringAttr::get(context, "fused_ops"), fused_ops_attr));
    // Epsilon is used only in fusions with the FusedBatchNorm op, so we zero it
    // here.
    Attribute epsilon = rewriter.getF32FloatAttr(0);
    attrs.push_back(
        NamedAttribute(StringAttr::get(context, "epsilon"), epsilon));

    // Insert fused operation right before the BiasAdd operation to guarantee
    // that bias value dominates the fused operation. We already verified that
    // original operation has a single use, so this is safe to do.
    auto *bias_add_op = bias_add.getOperation();
    if (bias_add_op) rewriter.setInsertionPoint(bias_add_op);

    Value fused_op = rewriter.create<FusedOpT>(fused_loc, result_type,
                                               ValueRange(operands), attrs);
    auto op_to_replace = fuse_activation ? activation : bias_add;
    rewriter.replaceOp(op_to_replace, ValueRange({fused_op}));
    return success();
  }
};

// Performs a fusion of the following pattern(s), if possible:
//   Conv2D + BiasAdd + <Activation> -> _FusedConv2D
class FuseConv2DBiasAdd
    : public FuseContractionWithBiasAdd<Conv2DOp, _FusedConv2DOp> {
 public:
  using FuseContractionWithBiasAdd<Conv2DOp,
                                   _FusedConv2DOp>::FuseContractionWithBiasAdd;
  // Verify that the Conv2D and BiasAdd data formats match. This is necessary
  // for the ops to fuse correctly, the fused Conv2D op has one data format
  // attribute which is shared.
  bool AreFuseCompatible(Conv2DOp conv, BiasAddOp bias_add,
                         PatternRewriter &rewriter) const override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSfused_kernel_matcherDTcc mht_5(mht_5_v, 390, "", "./tensorflow/compiler/mlir/tensorflow/transforms/fused_kernel_matcher.cc", "AreFuseCompatible");

    // Verify that the data formats match and are valid for fusion.
    if (conv.data_format() != bias_add.data_format()) {
      (void)rewriter.notifyMatchFailure(conv, [&](Diagnostic &diag) {
        diag << "data format does not match Conv2D data format ("
             << bias_add.data_format() << " vs " << conv.data_format() << ")";
      });
      return false;
    }
    // Verify the data type is supported.
    if (!conv.T().isF32() && !conv.T().isF64()) {
      (void)rewriter.notifyMatchFailure(conv, [&](Diagnostic &diag) {
        diag << "supported data types for _FusedConv2D are float and double, "
             << " but got " << conv.T();
      });
      return false;
    }
    return true;
  }
};

// Performs a fusion of the following pattern(s), if possible:
//   MatMulOp + BiasAdd + <Activation> -> _FusedMatMulOp
class FuseMatMulBiasAdd
    : public FuseContractionWithBiasAdd<MatMulOp, _FusedMatMulOp> {
  using FuseContractionWithBiasAdd<MatMulOp,
                                   _FusedMatMulOp>::FuseContractionWithBiasAdd;

  bool AreFuseCompatible(MatMulOp matmul, BiasAddOp bias_add,
                         PatternRewriter &rewriter) const override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSfused_kernel_matcherDTcc mht_6(mht_6_v, 422, "", "./tensorflow/compiler/mlir/tensorflow/transforms/fused_kernel_matcher.cc", "AreFuseCompatible");

    // FusedMatMul kernel supports limited set of data types.
    if (!matmul.T().isF32() && !matmul.T().isBF16()) {
      (void)rewriter.notifyMatchFailure(matmul, [&](Diagnostic &diag) {
        diag << "supported data types for _FusedMatMul are float and bfloat16, "
             << " but got " << matmul.T();
      });
      return false;
    }
    return true;
  }
};

void FusedKernelMatcherPass::runOnOperation() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSfused_kernel_matcherDTcc mht_7(mht_7_v, 438, "", "./tensorflow/compiler/mlir/tensorflow/transforms/fused_kernel_matcher.cc", "FusedKernelMatcherPass::runOnOperation");

  RewritePatternSet patterns(&getContext());
  auto func = getOperation();
  patterns.add<FuseConv2DBiasAdd, FuseMatMulBiasAdd>(&getContext());

  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}

}  // namespace

std::unique_ptr<OperationPass<FuncOp>> CreateFusedKernelMatcherPass() {
  return std::make_unique<FusedKernelMatcherPass>();
}

}  // namespace TF

}  // namespace mlir
