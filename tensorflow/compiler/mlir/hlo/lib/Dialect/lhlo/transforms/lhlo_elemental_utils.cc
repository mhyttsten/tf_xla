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
class MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSlhloPStransformsPSlhlo_elemental_utilsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSlhloPStransformsPSlhlo_elemental_utilsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSlhloPStransformsPSlhlo_elemental_utilsDTcc() {
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

// This file provides basic utilities for the elemental lowering of
// each node

#include "mlir-hlo/Dialect/lhlo/transforms/lhlo_elemental_utils.h"

#include "llvm/Support/Debug.h"
#include "mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h"
#include "mlir-hlo/Dialect/lhlo/transforms/map_lmhlo_to_scalar_op.h"
#include "mlir-hlo/utils/codegen_utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

using mlir::memref::DimOp;
using mlir::memref::LoadOp;
using mlir::memref::StoreOp;

namespace mlir {
namespace lmhlo {

Value createLoadOrUseCachedValue(Location loc, OpBuilder* b, Value memref,
                                 ValueRange indices,
                                 OpBuilder::InsertPoint insert_point) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSlhloPStransformsPSlhlo_elemental_utilsDTcc mht_0(mht_0_v, 214, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/lhlo/transforms/lhlo_elemental_utils.cc", "createLoadOrUseCachedValue");

  // Check if there are any cached value that can be reused,
  // within the current Block. Alternatively we can do this for
  // all the Blocks that dominant this Block, but that will be
  // complicated anyway.
  std::vector<StoreOp> store_ops;
  insert_point.getBlock()->walk(
      insert_point.getBlock()->begin(), insert_point.getPoint(),
      [&](StoreOp store_op) {
        if (store_op.getOperation()->getBlock() != insert_point.getBlock())
          return;
        if ((store_op.getMemRef() == memref) &&
            (store_op.getIndices() == indices))
          store_ops.emplace_back(store_op);
      });
  if (!store_ops.empty()) return store_ops[0].getOperand(0);
  int rank = memref.getType().dyn_cast<MemRefType>().getRank();
  return rank > 0 ? b->create<LoadOp>(loc, memref, indices)
                  : b->create<LoadOp>(loc, memref);
}

DenseSet<Operation*> NoLoaderUser(SmallVectorImpl<Operation*>& ops) {
  SmallVector<Operation*, 4> worklist;
  DenseSet<Operation*> has_loader_ops;
  for (Operation* op : ops) {
    Value memref = cast<LmhloOp>(op).getResultBuffer();
    if (memref == nullptr) continue;
    for (auto* user : memref.getUsers()) {
      if (isa<memref::LoadOp>(user)) {
        worklist.push_back(op);
        has_loader_ops.insert(op);
      }
    }
  }

  while (!worklist.empty()) {
    Operation* op = worklist.pop_back_val();
    int num_operands = op->getNumOperands();
    for (int i = 0; i < num_operands - 1; ++i) {
      Value memref = op->getOperand(i);
      for (Operation* user : memref.getUsers()) {
        if ((!isa<LmhloOp>(user)) || has_loader_ops.count(user)) continue;
        if (cast<LmhloOp>(user).getResultBuffer() == memref) {
          worklist.push_back(user);
          has_loader_ops.insert(user);
        }
      }
    }
  }

  DenseSet<Operation*> no_loader_ops;
  for (Operation* op : ops)
    if (!has_loader_ops.count(op)) no_loader_ops.insert(op);
  return no_loader_ops;
}

void cleanUnusedLhloOps(Block* parent) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSlhloPStransformsPSlhlo_elemental_utilsDTcc mht_1(mht_1_v, 273, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/lhlo/transforms/lhlo_elemental_utils.cc", "cleanUnusedLhloOps");

  SmallVector<Operation*, 4> lhlo_ops;
  for (Operation& op : parent->getOperations()) {
    if (op.getDialect() == op.getContext()->getLoadedDialect("lmhlo") &&
        (!isa<lmhlo::TerminatorOp>(op)))
      lhlo_ops.push_back(&op);
  }
  const DenseSet<Operation*>& no_loader_user = NoLoaderUser(lhlo_ops);
  for (auto* lhlo_op : no_loader_user) lhlo_op->erase();
}

template <typename LHLO_OpTy>
Value elementalLower(OpBuilder* b, Location loc, LHLO_OpTy op,
                     ValueRange output_index, bool check_cache);

template <>
Value elementalLower<lmhlo::RealDynamicSliceOp>(OpBuilder* b, Location loc,
                                                lmhlo::RealDynamicSliceOp op,
                                                ValueRange output_index,
                                                bool check_cache) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSlhloPStransformsPSlhlo_elemental_utilsDTcc mht_2(mht_2_v, 295, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/lhlo/transforms/lhlo_elemental_utils.cc", "elementalLower<lmhlo::RealDynamicSliceOp>");

  Value start_indices_memref = op->getOperand(1);
  Value strides_memref = op->getOperand(3);
  int rank = output_index.size();
  SmallVector<Value, 4> input_index;
  for (int dim = 0; dim < rank; ++dim) {
    SmallVector<Value, 4> dim_index;
    dim_index.push_back(b->create<arith::ConstantOp>(
        loc, b->getIndexType(), b->getIntegerAttr(b->getIndexType(), dim)));
    auto start_index_load =
        b->create<LoadOp>(loc, start_indices_memref, ValueRange{dim_index});
    auto start_index =
        b->create<arith::IndexCastOp>(loc, b->getIndexType(), start_index_load);
    auto stride_load =
        b->create<LoadOp>(loc, strides_memref, ValueRange{dim_index});
    auto stride =
        b->create<arith::IndexCastOp>(loc, b->getIndexType(), stride_load);
    // input_dim = out_dim * stride + start_index
    auto input_dim = b->create<arith::AddIOp>(
        loc, b->create<arith::MulIOp>(loc, output_index[dim], stride),
        start_index);
    input_index.push_back(input_dim);
  }

  Value operand_memref = *(op->getOperands().begin());

  if (!check_cache) return b->create<LoadOp>(loc, operand_memref, input_index);
  return createLoadOrUseCachedValue(loc, b, operand_memref, input_index,
                                    b->saveInsertionPoint());
}

namespace {

template <typename T>
Value elementalLowerImplForBroadcastInDimOps(OpBuilder* b, Location loc,
                                             T broadcast_in_dim,
                                             ValueRange output_index,
                                             bool check_cache) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSlhloPStransformsPSlhlo_elemental_utilsDTcc mht_3(mht_3_v, 335, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/lhlo/transforms/lhlo_elemental_utils.cc", "elementalLowerImplForBroadcastInDimOps");

  auto broadcast_dimensions =
      broadcast_in_dim.broadcast_dimensions().template getValues<int64_t>();
  int out_rank = output_index.size();
  Value operand_memref = broadcast_in_dim->getOperand(0);
  SmallVector<Value, 4> input_index;
  for (int64_t dim = 0; dim < out_rank; ++dim) {
    auto it = std::find(broadcast_dimensions.begin(),
                        broadcast_dimensions.end(), dim);

    bool is_broadcast_dim = (it != broadcast_dimensions.end());
    if (is_broadcast_dim) {
      int input_dim = std::distance(broadcast_dimensions.begin(), it);
      int64_t static_dim_size =
          operand_memref.getType().cast<MemRefType>().getShape()[input_dim];
      if (static_dim_size == 1) {
        // we know this dim is to be broadcasted at compile time
        auto zero = b->create<arith::ConstantOp>(
            loc, b->getIndexType(), b->getIntegerAttr(b->getIndexType(), 0));
        input_index.push_back(zero);
      } else if (static_dim_size == ShapedType::kDynamicSize) {
        // we are not sure if this dim is to be broadcasted at compile time
        auto dim_size = b->create<DimOp>(loc, operand_memref, input_dim);
        auto one = b->create<arith::ConstantOp>(
            loc, b->getIndexType(), b->getIntegerAttr(b->getIndexType(), 1));
        auto zero = b->create<arith::ConstantOp>(
            loc, b->getIndexType(), b->getIntegerAttr(b->getIndexType(), 0));
        auto dim_size_is_1 = b->create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::eq, dim_size, one);
        input_index.push_back(b->create<mlir::arith::SelectOp>(
            loc, dim_size_is_1, zero, output_index[dim]));
      } else {
        // we know this dim is not to be broadcasted at compile time
        input_index.push_back(output_index[dim]);
      }
    }
  }

  if (!check_cache) {
    int rank = operand_memref.getType().dyn_cast<MemRefType>().getRank();
    return (rank > 0) ? b->create<LoadOp>(loc, operand_memref, input_index)
                      : b->create<LoadOp>(loc, operand_memref, ValueRange());
  }
  return createLoadOrUseCachedValue(loc, b, operand_memref, input_index,
                                    b->saveInsertionPoint());
}

}  // namespace

template <>
Value elementalLower<lmhlo::DynamicBroadcastInDimOp>(
    OpBuilder* b, Location loc, lmhlo::DynamicBroadcastInDimOp op,
    ValueRange output_index, bool check_cache) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSlhloPStransformsPSlhlo_elemental_utilsDTcc mht_4(mht_4_v, 390, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/lhlo/transforms/lhlo_elemental_utils.cc", "elementalLower<lmhlo::DynamicBroadcastInDimOp>");

  return elementalLowerImplForBroadcastInDimOps(b, loc, op, output_index,
                                                check_cache);
}

template <>
Value elementalLower<lmhlo::BroadcastInDimOp>(OpBuilder* b, Location loc,
                                              lmhlo::BroadcastInDimOp op,
                                              ValueRange output_index,
                                              bool check_cache) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSlhloPStransformsPSlhlo_elemental_utilsDTcc mht_5(mht_5_v, 402, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/lhlo/transforms/lhlo_elemental_utils.cc", "elementalLower<lmhlo::BroadcastInDimOp>");

  return elementalLowerImplForBroadcastInDimOps(b, loc, op, output_index,
                                                check_cache);
}

scf::ForOp createLoopAndSetInsPt(OpBuilder& b, Location loc, Value& var,
                                 Value lb, Value ub, Value step,
                                 ArrayRef<Value> init_values) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSlhloPStransformsPSlhlo_elemental_utilsDTcc mht_6(mht_6_v, 412, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/lhlo/transforms/lhlo_elemental_utils.cc", "createLoopAndSetInsPt");

  auto for_op = b.create<scf::ForOp>(loc, lb, ub, step, init_values);
  b.setInsertionPointToStart(for_op.getBody());
  var = for_op.getInductionVar();
  return for_op;
}

scf::ParallelOp createParallelAndSetInsPt(OpBuilder& b, Location loc,
                                          SmallVectorImpl<Value>& vars,
                                          ArrayRef<Value> lbs,
                                          ArrayRef<Value> ubs,
                                          ArrayRef<Value> steps,
                                          ArrayRef<Value> init_values) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSlhloPStransformsPSlhlo_elemental_utilsDTcc mht_7(mht_7_v, 427, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/lhlo/transforms/lhlo_elemental_utils.cc", "createParallelAndSetInsPt");

  auto par_op = b.create<scf::ParallelOp>(loc, lbs, ubs, steps, init_values,
                                          /*bodyBuilderFn=*/nullptr);
  b.setInsertionPointToStart(par_op.getBody());
  vars.append(par_op.getInductionVars().begin(),
              par_op.getInductionVars().end());
  return par_op;
}

// reinterpret_cast the input memref into 1D
memref::ReinterpretCastOp createMemRef1DReinterpretCast(OpBuilder& b,
                                                        Location loc,
                                                        Value memref) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSlhloPStransformsPSlhlo_elemental_utilsDTcc mht_8(mht_8_v, 442, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/lhlo/transforms/lhlo_elemental_utils.cc", "createMemRef1DReinterpretCast");

  auto memref_ty = memref.getType().cast<MemRefType>();
  assert(memref_ty.getLayout().isIdentity());
  Value size = codegen_utils::emitNumElementsComputation(b, loc, memref);
  Value stride = b.create<mlir::arith::ConstantOp>(
      loc, b.getIndexType(), b.getIntegerAttr(b.getIndexType(), 1));
  Value zero = b.create<mlir::arith::ConstantOp>(
      loc, b.getIndexType(), b.getIntegerAttr(b.getIndexType(), 0));
  auto memref_1d_type =
      MemRefType::get({ShapedType::kDynamicSize}, memref_ty.getElementType(),
                      b.getMultiDimIdentityMap(1), memref_ty.getMemorySpace());
  return b.create<memref::ReinterpretCastOp>(
      loc, memref_1d_type, memref, zero, ValueRange{size}, ValueRange{stride});
}

void createOffsetStore(OpBuilder& b, Location loc, Value res, Value memref,
                       Value offset) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSlhloPStransformsPSlhlo_elemental_utilsDTcc mht_9(mht_9_v, 461, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/lhlo/transforms/lhlo_elemental_utils.cc", "createOffsetStore");

  Value memref_1d = createMemRef1DReinterpretCast(b, loc, memref);
  b.create<memref::StoreOp>(loc, res, memref_1d, ValueRange{offset});
}

memref::LoadOp createOffsetLoad(OpBuilder& b, Location loc, Value memref,
                                Value offset) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSlhloPStransformsPSlhlo_elemental_utilsDTcc mht_10(mht_10_v, 470, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/lhlo/transforms/lhlo_elemental_utils.cc", "createOffsetLoad");

  Value memref_1d = createMemRef1DReinterpretCast(b, loc, memref);
  return b.create<memref::LoadOp>(loc, memref_1d, ValueRange{offset});
}

}  // namespace lmhlo
}  // namespace mlir
