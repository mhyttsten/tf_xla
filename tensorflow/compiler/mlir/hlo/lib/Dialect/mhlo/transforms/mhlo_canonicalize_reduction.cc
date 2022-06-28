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
class MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSmhlo_canonicalize_reductionDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSmhlo_canonicalize_reductionDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSmhlo_canonicalize_reductionDTcc() {
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

// This file canonicalize reduction ops in hlo dialect to match the
// capacity of codegen backend.

#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/PassDetail.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

namespace mlir {
namespace mhlo {
namespace {

// All the reduce ops can be divided into following four types:
//  - a) column reduction, only reduce the most significant dimensions.
//  - b) row reduction, only reduce the least significant dimensions.
//  - c) reduce to scalar, all dimensions are reduced.
//  - d) others. (not support now, maybe use transpose to canonicalize)
//
// Currently we do following canonicalization to match the capacity of codegen
// backend.
//
// For case a):
// ====================================================================================
//   we convert all column reduction to rank-2 column reduction.
//   For example, suppose we have:
//   ```
//     func @test(%arg0: tensor<?x?x?xf32>) -> tensor<?x?xf32> {
//       ...
//       %2 = "mhlo.reduce"(%arg0, ...) ({...})
//         {dimensions = dense<[0]> : tensor<1xi64>} :
//         (tensor<?x?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
//       return %2 : tensor<?x?xf32>
//     }
//  ```
//   After conversion:
//     func @test(%arg0: tensor<?x?x?xf32>) -> tensor<?x?xf32> {
//       // [a, b, c] -> [a, b*c]
//       %1 = mhlo.dynamic_reshape(%arg0, ...) : (tensor<?x?x?xf32>,
//       tensor<2xi64>) -> tensor<?x?xf32> %2 = "mhlo.reduce"(%1, ...) ({...})
//         {dimensions = dense<[0]> : tensor<1xi64>} :
//         (tensor<?x?xf32>, tensor<f32>) -> tensor<?xf32>
//       %3 = "mhlo.dynamic_reshape"(%2, ...) : (tensor<?xf32>, tensor<1xi64>)
//       -> tensor<?x?f32> return %3 : tensor<?x?xf32>
//     }
//  ```
//
// For case b):
// ====================================================================================
//   we convert all row reduction to rank-2 row reduction.
//   For example, suppose we have:
//   ```
//     func @test(%arg0: tensor<?x?x?xf32>) -> tensor<?x?xf32> {
//       ...
//       %2 = "mhlo.reduce"(%arg0, ...) ({...})
//         {dimensions = dense<[2]> : tensor<1xi64>} :
//         (tensor<?x?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
//       return %2 : tensor<?x?xf32>
//     }
//  ```
//   After conversion:
//     func @test(%arg0: tensor<?x?x?xf32>) -> tensor<?x?xf32> {
//       // [a, b, c] -> [a*b, c]
//       %1 = mhlo.dynamic_reshape(%arg0, ...) : (tensor<?x?x?xf32>,
//       tensor<2xi64>) -> tensor<?x?xf32> %2 = "mhlo.reduce"(%1, ...) ({...})
//         {dimensions = dense<[1]> : tensor<1xi64>} :
//         (tensor<?x?xf32>, tensor<f32>) -> tensor<?xf32>
//       %3 = "mhlo.dynamic_reshape"(%2, ...) : (tensor<?xf32>, tensor<1xi64>)
//       -> tensor<?x?f32> return %3 : tensor<?x?xf32>
//     }
//  ```
//
// For case c):
// ====================================================================================
//   we convert all reduce-to-scalar to rank-2 column reduction.
//
//   For example, suppose we have:
//   ```
//     func @test(%arg0: tensor<?x?x?xf32>) -> tensor<f32> {
//       ...
//       %2 = "mhlo.reduce"(%arg0, ...) ({...})
//         {dimensions = dense<[0,1,2]> : tensor<3xi64>} :
//         (tensor<?x?x?xf32>, tensor<f32>) -> tensor<f32>
//       return %2 : tensor<f32>
//     }
//  ```
//   After conversion:
//     func @test(%arg0: tensor<?x?x?xf32>) -> tensor<f32> {
//       // [a, b, c] -> [a*b*c, 1]
//       %1 = mhlo.dynamic_reshape(%arg0, ...) : (tensor<?x?x?xf32>,
//       tensor<2xi64>) -> tensor<?x?xf32> %2 = "mhlo.reduce"(%1, ...) ({...})
//         {dimensions = dense<[0]> : tensor<1xi64>} :
//         (tensor<?x?xf32>, tensor<f32>) -> tensor<?xf32>
//       %3 = "mhlo.reshape"(%2, ...) : (tensor<?xf32>, tensor<1xi64>) ->
//       tensor<f32> return %3 : tensor<f32>
//     }
//  ```

struct HloCanonicalizeReductionPass
    : HloCanonicalizeReductionPassBase<HloCanonicalizeReductionPass> {
  void getDependentDialects(DialectRegistry& registry) const override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSmhlo_canonicalize_reductionDTcc mht_0(mht_0_v, 283, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/mhlo_canonicalize_reduction.cc", "getDependentDialects");

    registry.insert<tensor::TensorDialect>();
  }
  void runOnOperation() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSmhlo_canonicalize_reductionDTcc mht_1(mht_1_v, 289, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/mhlo_canonicalize_reduction.cc", "runOnOperation");

    getOperation().walk([&](ReduceOp op) {
      SmallVector<int64_t, 4> dims_to_reduce;
      DenseSet<int64_t> dims_to_reduce_set;
      for (auto dim : op.dimensions().getValues<APInt>()) {
        dims_to_reduce.push_back(dim.getSExtValue());
        dims_to_reduce_set.insert(dims_to_reduce.back());
      }

      // empty reduction is just a no-op, thus no need to do codegen.
      if (dims_to_reduce.empty()) return;

      // suppose reduce input is a ranked tensor
      auto ty = op.getOperand(0).getType().dyn_cast<RankedTensorType>();
      if (!ty) return signalPassFailure();
      int rank = ty.getRank();
      int ndims_to_reduce = dims_to_reduce.size();
      auto elem_ty = ty.getElementType();
      llvm::sort(dims_to_reduce);

      // skip case d) form since we don't support it.
      if ((dims_to_reduce.back() - dims_to_reduce[0]) !=
              (ndims_to_reduce - 1) ||
          (dims_to_reduce[0] != 0 && dims_to_reduce.back() != (rank - 1))) {
        return;
      }

      // rank 2 row/column reduction is already supported.
      if (rank == 2 && ndims_to_reduce == 1) {
        return;
      }

      SmallVector<int64_t, 4> dims_to_keep;
      for (int i = 0; i < rank; ++i) {
        if (!dims_to_reduce_set.count(i)) dims_to_keep.push_back(i);
      }

      OpBuilder b(op);
      auto loc = op.getLoc();
      // TODO(disc): uniformed shape_scalar_type with shape_derivation
      auto shape_scalar_type = b.getIntegerType(32);
      auto one = b.create<arith::ConstantIntOp>(loc, 1ll, shape_scalar_type);

      // funtion to get total elements in selected dimensions
      auto dim_prod = [&](ArrayRef<int64_t> dims) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSmhlo_canonicalize_reductionDTcc mht_2(mht_2_v, 336, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/mhlo_canonicalize_reduction.cc", "lambda");

        Value nelems = one;
        for (int64_t v : dims) {
          Value dim_index = b.create<tensor::DimOp>(loc, op.getOperand(0), v);
          nelems = b.create<arith::MulIOp>(
              loc, nelems,
              b.create<arith::IndexCastOp>(loc, shape_scalar_type, dim_index));
        }
        return nelems;
      };

      SmallVector<Value, 2> new_operand_dims;
      DenseIntElementsAttr attr;
      Value nelem_to_reduce = dim_prod(dims_to_reduce);
      Value nelem_to_keep = dim_prod(dims_to_keep);
      if (rank == ndims_to_reduce) {
        // case c) Reduce to scalar.
        // Currently we don't support reduce to scalar directly.
        // As a workaround, we convert the `reduce to scalar` to a rank 2
        // column reduction having following form:
        // Suppose nelems = ProdutionOp(ShapeOp(I)), We convert I into
        // shape `[nelems, 1]`.
        // TODO(disc): this may have performance issue. Implements a reduce to
        // scalar schedule if necessary.
        new_operand_dims.push_back(nelem_to_reduce);
        new_operand_dims.push_back(nelem_to_keep);
        attr = DenseIntElementsAttr::get(
            RankedTensorType::get({1}, b.getIntegerType(64)), {0ll});
      } else if (dims_to_reduce[0] == 0) {
        // case a) column reduction
        new_operand_dims.push_back(nelem_to_reduce);
        new_operand_dims.push_back(nelem_to_keep);
        attr = DenseIntElementsAttr::get(
            RankedTensorType::get({1}, b.getIntegerType(64)), {0ll});
      } else {
        // case b) row reduction
        new_operand_dims.push_back(nelem_to_keep);
        new_operand_dims.push_back(nelem_to_reduce);
        attr = DenseIntElementsAttr::get(
            RankedTensorType::get({1}, b.getIntegerType(64)), {1ll});
      }

      Value new_operand_shape =
          b.create<tensor::FromElementsOp>(loc, new_operand_dims);

      SmallVector<Value, 4> new_operands;
      for (Value operand : op.inputs()) {
        new_operands.push_back(b.create<DynamicReshapeOp>(
            loc,
            RankedTensorType::get(
                SmallVector<int64_t, 4>(new_operand_dims.size(),
                                        ShapedType::kDynamicSize),
                elem_ty),
            operand, new_operand_shape));
      }
      auto new_op =
          b.create<ReduceOp>(loc, new_operands, op.init_values(), attr);
      new_op.body().takeBody(op.body());

      SmallVector<Value, 4> new_results;
      if (dims_to_keep.empty()) {
        // case c) reduce to scalar
        // reshape rank 1 tensor with size 1 to a rank 0 tensor
        for (Value result : new_op.getResults()) {
          new_results.push_back(b.create<ReshapeOp>(
              loc, RankedTensorType::get({}, elem_ty), result));
        }
      } else {
        SmallVector<Value, 4> result_dims;
        for (int64_t i : dims_to_keep) {
          Value dim_index = b.create<tensor::DimOp>(loc, op.getOperand(0), i);
          result_dims.push_back(
              b.create<arith::IndexCastOp>(loc, shape_scalar_type, dim_index));
        }
        Value result_shape = b.create<tensor::FromElementsOp>(loc, result_dims);
        for (auto&& e : llvm::zip(op.getResults(), new_op.getResults())) {
          new_results.push_back(b.create<DynamicReshapeOp>(
              loc, std::get<0>(e).getType(), std::get<1>(e), result_shape));
        }
      }
      for (auto&& e : llvm::zip(op.getResults(), new_results)) {
        std::get<0>(e).replaceAllUsesWith(std::get<1>(e));
      }
      op.erase();
    });
  }
};

}  // namespace

std::unique_ptr<OperationPass<FuncOp>> createHloCanonicalizeReductionPass() {
  return std::make_unique<HloCanonicalizeReductionPass>();
}

}  // namespace mhlo
}  // namespace mlir
