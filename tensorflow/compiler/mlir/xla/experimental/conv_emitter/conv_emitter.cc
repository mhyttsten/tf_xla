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
class MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSexperimentalPSconv_emitterPSconv_emitterDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSexperimentalPSconv_emitterPSconv_emitterDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSexperimentalPSconv_emitterPSconv_emitterDTcc() {
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

/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// This is an explorative prototype emitter for convolution using MLIR.
// This prototype is still under construction.
// TODO(timshen): Fix the documentation once it's implemented.
//
// Goals:
// * Autotune-able tiling.
// * Autotune-able memory accesses.
// * Autotune-able lowering logic (from a portable program to thread-oriented
//   CUDA program).
// * Use milr::AffineExpr to analyze all accesses. It aims to algorithmically
//   find memory access strategies for given input layouts and tiling configs.

#include "tensorflow/compiler/mlir/xla/experimental/conv_emitter/conv_emitter.h"

#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/Dialect/Affine/LoopUtils.h"  // from @llvm-project
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/IR/AffineExpr.h"  // from @llvm-project
#include "mlir/IR/AffineMap.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/Transforms/RegionUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/xla/experimental/conv_emitter/conv_emitter_transforms.h"
#include "tensorflow/compiler/xla/permutation_util.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_type_conversion_util.h"
#include "tensorflow/compiler/xla/window_util.h"

namespace xla {
namespace experimental {
namespace {

using mlir::OpBuilder;

// Various extracted information for input shapes.
struct ShapeInfo {
  // Buffer dimensions in the order of NCHW.
  std::vector<int64_t> nchw_dimensions;

  // Buffer dimensions in the order of major to minor;
  std::vector<int64_t> physical_dimensions;

  // The affine map that takes NCHW indices, and maps to the physical order.
  mlir::AffineMap affine_map;

  mlir::Type element_type;
};

ShapeInfo GetShapeInfo(
    const Shape& shape, int64_t n_dim, int64_t c_dim,
    absl::Span<const tensorflow::protobuf_int64> spatial_dims,
    mlir::Builder builder) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSexperimentalPSconv_emitterPSconv_emitterDTcc mht_0(mht_0_v, 240, "", "./tensorflow/compiler/mlir/xla/experimental/conv_emitter/conv_emitter.cc", "GetShapeInfo");

  ShapeInfo shape_info;

  std::vector<int64_t> physical_to_logical(
      shape.layout().minor_to_major().rbegin(),
      shape.layout().minor_to_major().rend());

  std::vector<int64_t> nchw_to_logical;

  nchw_to_logical.push_back(n_dim);
  nchw_to_logical.push_back(c_dim);
  for (int64_t dim : spatial_dims) {
    nchw_to_logical.push_back(dim);
  }

  for (int64_t dim : nchw_to_logical) {
    shape_info.nchw_dimensions.push_back(shape.dimensions(dim));
  }

  for (int64_t dim : physical_to_logical) {
    shape_info.physical_dimensions.push_back(shape.dimensions(dim));
  }

  std::vector<mlir::AffineExpr> affine_exprs;
  // We want physical to nchw order.
  for (int64_t dim : ComposePermutations(InversePermutation(nchw_to_logical),
                                         physical_to_logical)) {
    affine_exprs.push_back(builder.getAffineDimExpr(dim));
  }

  shape_info.affine_map = mlir::AffineMap::get(
      /*dimCount=*/2 + spatial_dims.size(), /*symbolCount=*/0, affine_exprs,
      builder.getContext());

  shape_info.element_type = [&] {
    switch (shape.element_type()) {
      case xla::F16:
        return builder.getF16Type();
      case xla::F32:
        return builder.getF32Type();
      default:
        break;
    }
    CHECK(false);
  }();

  return shape_info;
}

void SetMemRef(mlir::Operation* op, mlir::Value memref) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSexperimentalPSconv_emitterPSconv_emitterDTcc mht_1(mht_1_v, 292, "", "./tensorflow/compiler/mlir/xla/experimental/conv_emitter/conv_emitter.cc", "SetMemRef");

  if (auto load = mlir::dyn_cast<mlir::AffineLoadOp>(op)) {
    load.setMemRef(memref);
  } else if (auto store = mlir::dyn_cast<mlir::AffineStoreOp>(op)) {
    store.setMemRef(memref);
  } else {
    CHECK(false);
  }
}

// Hoist operations out of `where`. [begin_op, end_op) must be the first
// operations of their parent loop, and `where` must be an ancestor of that
// parent loop.
//
// It always preserves the semantics of the program, therefore it may modify the
// hoisted operations or add extra loops at the hoisted place.
mlir::Operation* HoistAndFix(llvm::iplist<mlir::Operation>::iterator begin_op,
                             llvm::iplist<mlir::Operation>::iterator end_op,
                             mlir::AffineForOp where) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSexperimentalPSconv_emitterPSconv_emitterDTcc mht_2(mht_2_v, 313, "", "./tensorflow/compiler/mlir/xla/experimental/conv_emitter/conv_emitter.cc", "HoistAndFix");

  // All loops to hoist through.
  llvm::SmallVector<mlir::AffineForOp, 4> ancestors;
  getPerfectlyNestedLoops(ancestors, where);
  {
    int i;
    for (i = 0; i < ancestors.size(); i++) {
      if (&ancestors[i].getBody()->front() == &*begin_op) {
        break;
      }
    }
    CHECK(i < ancestors.size());
    ancestors.resize(i + 1);
  }

  std::vector<int64_t> ancestor_dimensions;
  for (auto ancestor : ancestors) {
    CHECK(IsSimpleLoop(ancestor));
    ancestor_dimensions.push_back(
        ancestor.getUpperBoundMap().getSingleConstantResult());
  }

  if (auto alloc = mlir::dyn_cast<mlir::memref::AllocOp>(begin_op)) {
    CHECK(std::next(begin_op) == end_op)
        << "alloc() needs to be hoisted by its own";

    OpBuilder builder(where);
    mlir::MemRefType type = alloc.getType();
    CHECK(type.getLayout().isIdentity());
    ancestor_dimensions.insert(ancestor_dimensions.end(),
                               type.getShape().begin(), type.getShape().end());
    mlir::MemRefType new_type =
        mlir::MemRefType::get(ancestor_dimensions, type.getElementType());
    auto new_alloc = builder.create<mlir::memref::AllocOp>(
        builder.getUnknownLoc(), new_type);

    std::vector<mlir::Value> indvars;
    for (auto ancestor : ancestors) {
      indvars.push_back(ancestor.getInductionVar());
    }
    for (auto& use : llvm::make_early_inc_range(alloc.getResult().getUses())) {
      mlir::Operation* owner = use.getOwner();
      BoundAffineMap affine_map = GetBoundAffineMapFrom(owner);
      affine_map.operands.insert(affine_map.operands.begin(), indvars.begin(),
                                 indvars.end());
      CHECK(affine_map.affine_map.isIdentity());
      affine_map.affine_map = mlir::AffineMap::getMultiDimIdentityMap(
          affine_map.operands.size(), builder.getContext());

      mlir::Operation* new_op =
          CloneWithNewAffineMap(owner, affine_map, OpBuilder(owner));
      SetMemRef(new_op, new_alloc);
      owner->replaceAllUsesWith(new_op);
      owner->erase();
    }
    alloc.erase();
    return new_alloc;
  }

  const bool any_op_is_loop_variant = [&] {
    for (mlir::Operation& op : llvm::make_range(begin_op, end_op)) {
      if (mlir::isa<mlir::AffineForOp, mlir::AffineStoreOp>(op)) {
        return true;
      }
    }
    return false;
  }();

  if (any_op_is_loop_variant) {
    auto builder = OpBuilder(where);
    std::vector<mlir::AffineForOp> new_loops;
    for (auto dim : ancestor_dimensions) {
      auto where =
          builder.create<mlir::AffineForOp>(builder.getUnknownLoc(), 0, dim);
      new_loops.push_back(where);
      builder = OpBuilder::atBlockTerminator(where.getBody());
    }
    for (mlir::Operation& op :
         llvm::make_early_inc_range(llvm::make_range(begin_op, end_op))) {
      op.moveBefore(&new_loops.back().getBody()->back());
    }
    CHECK_EQ(ancestors.size(), new_loops.size());
    for (int i = 0; i < ancestors.size(); i++) {
      replaceAllUsesInRegionWith(ancestors[i].getInductionVar(),
                                 new_loops[i].getInductionVar(),
                                 new_loops.back().region());
    }
    return new_loops.front();
  }
  CHECK(false);
}

mlir::Operation* HoistAndFix(mlir::Operation* op, mlir::AffineForOp where) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSexperimentalPSconv_emitterPSconv_emitterDTcc mht_3(mht_3_v, 408, "", "./tensorflow/compiler/mlir/xla/experimental/conv_emitter/conv_emitter.cc", "HoistAndFix");

  return HoistAndFix(op->getIterator(), std::next(op->getIterator()), where);
}

struct InitialMlirConvAnchors {
  std::vector<mlir::AffineForOp> cartesian_product_loops;
  std::vector<mlir::AffineForOp> reduction_loops;
  mlir::memref::AllocOp output_acc;
};

// Return the following IR with the anchors set to corresponding operations.
//   for (cartesian loops...) {
//     %output_acc = alloc() : memref(f32)
//     output_acc[] = 0
//     for (reduction loops...) {
//       output_acc[] += input[...] * filter[...]
//     }
//     output[...] = output_acc[]
//   }
StatusOr<InitialMlirConvAnchors> CreateNaiveMlirConv(
    mlir::Value input, mlir::Value filter, mlir::Value output,
    const ShapeInfo& input_shape_info, const ShapeInfo& filter_shape_info,
    const ShapeInfo& output_shape_info, const Window& window,
    OpBuilder builder) {
  CHECK(input_shape_info.element_type == builder.getF16Type());
  CHECK(filter_shape_info.element_type == builder.getF16Type());
  CHECK(output_shape_info.element_type == builder.getF16Type());

  auto location = mlir::UnknownLoc::get(builder.getContext());

  std::vector<mlir::AffineForOp> cartesian_product_loops =
      CreateNestedSimpleLoops(output_shape_info.nchw_dimensions, builder);

  builder =
      OpBuilder::atBlockTerminator(cartesian_product_loops.back().getBody());

  auto output_acc = builder.create<mlir::memref::AllocOp>(
      location, mlir::MemRefType::get({}, builder.getF32Type()));

  builder.create<mlir::AffineStoreOp>(
      location,
      builder.create<mlir::arith::ConstantOp>(
          location, mlir::FloatAttr::get(builder.getF32Type(), 0)),
      output_acc, llvm::ArrayRef<mlir::Value>());

  std::vector<mlir::AffineForOp> reduction_loops;
  reduction_loops = CreateNestedSimpleLoops(
      absl::MakeSpan(filter_shape_info.nchw_dimensions).subspan(1), builder);

  mlir::AffineForOp loop_n = cartesian_product_loops[0];
  mlir::AffineForOp loop_o = cartesian_product_loops[1];
  mlir::AffineForOp loop_c = reduction_loops[0];

  std::vector<mlir::Value> output_spatial_indvars;
  for (auto loop : absl::MakeSpan(cartesian_product_loops).subspan(2)) {
    output_spatial_indvars.push_back(loop.getInductionVar());
  }
  std::vector<mlir::Value> filter_spatial_indvars;
  for (auto loop : absl::MakeSpan(reduction_loops).subspan(1)) {
    filter_spatial_indvars.push_back(loop.getInductionVar());
  }
  int num_spatial_dims = output_spatial_indvars.size();
  CHECK_EQ(num_spatial_dims, filter_spatial_indvars.size());

  builder = OpBuilder::atBlockTerminator(reduction_loops.back().getBody());

  mlir::Value loaded_input = [&] {
    std::vector<mlir::AffineExpr> input_indices;
    input_indices.push_back(builder.getAffineDimExpr(0));
    input_indices.push_back(builder.getAffineDimExpr(1));

    // For spatial dimensions, generate input_index * stride + filter_index -
    // left_pad
    //
    // TODO(timshen): guard out-of-bound loads and stores brought by padding.
    for (int i = 0; i < num_spatial_dims; i++) {
      const WindowDimension& window_dim = window.dimensions(i);
      input_indices.push_back(
          builder.getAffineDimExpr(i + 2) * window_dim.stride() +
          builder.getAffineDimExpr(2 + num_spatial_dims + i) -
          window_dim.padding_low());
    }
    std::vector<mlir::Value> input_vars;
    input_vars.push_back(loop_n.getInductionVar());
    input_vars.push_back(loop_c.getInductionVar());
    input_vars.insert(input_vars.end(), output_spatial_indvars.begin(),
                      output_spatial_indvars.end());
    input_vars.insert(input_vars.end(), filter_spatial_indvars.begin(),
                      filter_spatial_indvars.end());

    return builder.create<mlir::arith::ExtFOp>(
        location, builder.getF32Type(),
        builder.createOrFold<mlir::AffineLoadOp>(
            location, input,
            mlir::AffineMap(input_shape_info.affine_map)
                .compose(mlir::AffineMap::get(
                    /*dimCount=*/2 + num_spatial_dims * 2,
                    /*symbolCount=*/0, input_indices, builder.getContext())),
            input_vars));
  }();

  mlir::Value loaded_filter = [&] {
    std::vector<mlir::Value> filter_vars;
    filter_vars.push_back(loop_o.getInductionVar());
    filter_vars.push_back(loop_c.getInductionVar());
    filter_vars.insert(filter_vars.end(), filter_spatial_indvars.begin(),
                       filter_spatial_indvars.end());

    return builder.create<mlir::arith::ExtFOp>(
        location, builder.getF32Type(),
        builder.createOrFold<mlir::AffineLoadOp>(
            location, filter, filter_shape_info.affine_map, filter_vars));
  }();

  auto accum_load_op =
      builder.createOrFold<mlir::AffineLoadOp>(location, output_acc);
  builder.createOrFold<mlir::AffineStoreOp>(
      location,
      builder.create<mlir::arith::AddFOp>(
          location, accum_load_op,
          builder.create<mlir::arith::MulFOp>(location, loaded_input,
                                              loaded_filter)),
      output_acc, llvm::ArrayRef<mlir::Value>());

  builder.setInsertionPointAfter(reduction_loops[0]);
  {
    std::vector<mlir::Value> output_vars;
    output_vars.push_back(loop_n.getInductionVar());
    output_vars.push_back(loop_o.getInductionVar());
    output_vars.insert(output_vars.end(), output_spatial_indvars.begin(),
                       output_spatial_indvars.end());
    builder.createOrFold<mlir::AffineStoreOp>(
        location,
        builder.create<mlir::arith::TruncFOp>(
            location, builder.getF16Type(),
            builder.createOrFold<mlir::AffineLoadOp>(location, output_acc)),
        output, output_shape_info.affine_map, output_vars);
  }

  return InitialMlirConvAnchors{cartesian_product_loops, reduction_loops,
                                output_acc};
}

// Contains the following pattern with anchors:
//   for (cartesian loops...) {
//     %output_acc = alloc() : memref(..., f32)
//     for (reduction loops...) {
//       for (tiled cartesian loops...) {
//         output_acc[...] = 0
//       }
//       for (tiled cartesian loops...) {
//         for (reduction loops...) {
//           output_acc[] += input[...] * filter[...]
//         }
//       }
//       for (tiled cartesian loops...) {
//         output[...] = output_acc[...]
//       }
//     }
//   }
struct TransformedMlirConvAnchors {
  std::vector<mlir::AffineForOp> cartesian_product_loops;
  std::vector<mlir::AffineForOp> reduction_loops;
};

StatusOr<TransformedMlirConvAnchors> TransformMlirConv(
    InitialMlirConvAnchors anchors) {
  std::vector<mlir::AffineForOp> cartesian_product_loops =
      anchors.cartesian_product_loops;
  std::vector<mlir::AffineForOp> reduction_loops = anchors.reduction_loops;
  mlir::memref::AllocOp output_acc = anchors.output_acc;

  // TODO(timshen): consider using pattern matchers for transformations
  //
  // Initial form:
  //   for (cartesian loops...) {
  //     %output_acc = alloc() : memref(f32)
  //     output_acc[] = 0
  //     for (reduction loops...) {
  //       output_acc[] += input[...] * filter[...]
  //     }
  //     output[...] = output_acc[]
  //   }

  // Tile cartesian loops to:
  //   for (cartesian loops...) {
  //     for (tiled cartesian loops...) {
  //       %output_acc = alloc() : memref(f32)
  //       output_acc[] = 0
  //       for (reduction loops...) {
  //         output_acc[] += input[...] * filter[...]
  //       }
  //       output[...] = output_acc[]
  //     }
  //   }
  TileLoop(reduction_loops[0], 4, reduction_loops.back());

  std::vector<mlir::AffineForOp> tiled_cartesian_loops;
  tiled_cartesian_loops.push_back(
      TileLoop(cartesian_product_loops[1], 32, cartesian_product_loops.back()));

  tiled_cartesian_loops.push_back(TileLoop(cartesian_product_loops.back(), 16,
                                           tiled_cartesian_loops.back()));

  // Two hoist operations to interleave the allocation, computation, and
  // writebacks to output_acc:
  // After first hoist:
  //   for (cartesian loops...) {
  //     %output_acc = alloc() : memref(..., f32)
  //     for (tiled cartesian loops...) {
  //       output_acc[...] = 0
  //       for (reduction loops...) {
  //         output_acc[...] += input[...] * filter[...]
  //       }
  //       output[...] = output_acc[...]
  //     }
  //   }
  output_acc = llvm::cast<mlir::memref::AllocOp>(
      HoistAndFix(output_acc, tiled_cartesian_loops.front()));

  // Hoist everything before reduction loops (aka zero initializations of
  // output_acc):
  //   for (cartesian loops...) {
  //     %output_acc = alloc() : memref(..., f32)
  //     for (tiled cartesian loops...) {
  //       output_acc[...] = 0
  //     }
  //     for (tiled cartesian loops...) {
  //       for (reduction loops...) {
  //         output_acc[...] += input[...] * filter[...]
  //       }
  //       output[...] = output_acc[...]
  //     }
  //   }
  HoistAndFix(tiled_cartesian_loops.back().getBody()->begin(),
              reduction_loops.front().getOperation()->getIterator(),
              tiled_cartesian_loops.front());

  // Now hoist all reduction loops outside of tiled cartesian loops.
  // Notice that HoistAndFix automatically add a new set of tiled cartesian
  // loops for hoisted reduction loops to keep the semantics correct.
  //
  // After second hoist:
  //   for (cartesian loops...) {
  //     %output_acc = alloc() : memref(..., f32)
  //     for (tiled cartesian loops...) {
  //       output_acc[...] = 0
  //     }
  //     for (tiled cartesian loops...) {
  //       for (reduction loops...) {
  //         output_acc[] += input[...] * filter[...]
  //       }
  //     }  // compute loop
  //     for (tiled cartesian loops...) {
  //       output[...] = output_acc[...]
  //     }
  //   }
  {
    auto compute_loop = llvm::cast<mlir::AffineForOp>(
        HoistAndFix(reduction_loops.front(), tiled_cartesian_loops[0]));

    // Fix tiled_cartesian_loops to make them point to the tiled compute loops,
    // not the writeback loops to output buffer.
    llvm::SmallVector<mlir::AffineForOp, 4> all_loops;
    getPerfectlyNestedLoops(all_loops, compute_loop);
    absl::c_copy_n(all_loops, tiled_cartesian_loops.size(),
                   tiled_cartesian_loops.data());
  }

  // After exchanging tiled cartesian compute loops with reduction loops:
  //   for (cartesian loops...) {
  //     %output_acc = alloc() : memref(..., f32)
  //     for (tiled cartesian loops...) {
  //       output_acc[...] = 0
  //     }
  //     for (reduction loops...) {
  //       for (tiled cartesian loops...) {
  //         output_acc[] += input[...] * filter[...]
  //       }
  //     }
  //     for (tiled cartesian loops...) {
  //       output[...] = output_acc[...]
  //     }
  //   }
  //
  // ...so that later tiled cartesian loops (with computations in it) can be
  // replaced by CUDA MMA instructions.
  {
    std::vector<mlir::AffineForOp> loops;
    loops.insert(loops.end(), tiled_cartesian_loops.begin(),
                 tiled_cartesian_loops.end());
    loops.insert(loops.end(), reduction_loops.begin(), reduction_loops.end());
    SinkPerfectlyNestedLoops(loops, tiled_cartesian_loops.size());
  }
  return TransformedMlirConvAnchors{cartesian_product_loops, reduction_loops};
}

}  // namespace

StatusOr<mlir::func::FuncOp> EmitConvolutionForwardAsMlir(
    HloInstruction* conv, absl::string_view function_name,
    mlir::MLIRContext* context) {
  OpBuilder builder(context);

  const auto& dim_nums = conv->convolution_dimension_numbers();
  ShapeInfo input_shape_info =
      GetShapeInfo(conv->operand(0)->shape(), dim_nums.input_batch_dimension(),
                   dim_nums.input_feature_dimension(),
                   dim_nums.input_spatial_dimensions(), builder);

  ShapeInfo filter_shape_info = GetShapeInfo(
      conv->operand(1)->shape(), dim_nums.kernel_output_feature_dimension(),
      dim_nums.kernel_input_feature_dimension(),
      dim_nums.kernel_spatial_dimensions(), builder);

  ShapeInfo output_shape_info = GetShapeInfo(
      conv->shape().tuple_shapes(0), dim_nums.output_batch_dimension(),
      dim_nums.output_feature_dimension(), dim_nums.output_spatial_dimensions(),
      builder);

  auto function = mlir::func::FuncOp::create(
      mlir::UnknownLoc::get(builder.getContext()),
      llvm_ir::AsStringRef(function_name),
      builder.getFunctionType(
          {mlir::MemRefType::get(output_shape_info.physical_dimensions,
                                 output_shape_info.element_type,
                                 mlir::AffineMap()),
           mlir::MemRefType::get(input_shape_info.physical_dimensions,
                                 input_shape_info.element_type,
                                 mlir::AffineMap()),
           mlir::MemRefType::get(filter_shape_info.physical_dimensions,
                                 filter_shape_info.element_type,
                                 mlir::AffineMap())},
          {}));

  auto* entry_block = function.addEntryBlock();
  builder.setInsertionPointToStart(entry_block);
  builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());
  builder.setInsertionPointToStart(entry_block);

  mlir::Value input = entry_block->getArgument(1);
  mlir::Value filter = entry_block->getArgument(2);
  mlir::Value output = entry_block->getArgument(0);

  TF_RETURN_IF_ERROR(ConvIsImplemented(conv));

  TF_ASSIGN_OR_RETURN(
      InitialMlirConvAnchors initial_anchors,
      CreateNaiveMlirConv(input, filter, output, input_shape_info,
                          filter_shape_info, output_shape_info, conv->window(),
                          builder));

  TF_ASSIGN_OR_RETURN(TransformedMlirConvAnchors transformed_anchors,
                      TransformMlirConv(initial_anchors));

  // TODO(timshen): Implement a transformation that collects loads to a given
  // buffer, create a local alloc() for the accessed part, redirects all loads
  // and stores to that local alloc(), and create code to initialize /
  // writeback the local alloc() if needed.

  // TODO(timshen): Implement CUDA-specific lowering.

  return function;
}

Status ConvIsImplemented(const HloInstruction* conv) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSexperimentalPSconv_emitterPSconv_emitterDTcc mht_4(mht_4_v, 776, "", "./tensorflow/compiler/mlir/xla/experimental/conv_emitter/conv_emitter.cc", "ConvIsImplemented");

  if (conv->feature_group_count() != 1 || conv->batch_group_count() != 1) {
    return Unimplemented("group count is not implemented.");
  }
  if (window_util::HasWindowReversal(conv->window())) {
    return Unimplemented("Window reversal is not implemented.");
  }
  if (window_util::HasDilation(conv->window())) {
    return Unimplemented("Dilation is not implemented.");
  }
  return Status::OK();
}

}  // namespace experimental
}  // namespace xla
