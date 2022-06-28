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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emission_utilsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emission_utilsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emission_utilsDTcc() {
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

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"

#include <algorithm>
#include <array>
#include <vector>

#include "llvm/IR/IntrinsicsNVPTX.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/mlir/xla/hlo_utils.h"
#include "tensorflow/compiler/mlir/xla/type_to_shape.h"
#include "tensorflow/compiler/xla/service/gpu/target_util.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_type_conversion_util.h"

namespace xla {
namespace gpu {

namespace {

// Return whether the given shape is rank 2 excluding the batch dimensions.
bool IsRank2(const Shape& shape, int64_t batch_dimensions_size) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emission_utilsDTcc mht_0(mht_0_v, 205, "", "./tensorflow/compiler/xla/service/gpu/ir_emission_utils.cc", "IsRank2");

  return shape.rank() == batch_dimensions_size + 2;
}

// Given a shape and a group of contiguous dimensions in the shape, returns
// a tuple of three values (major, middle, minor), where major is the size of
// the dimensions more major then the given dimensions, minor is the size of
// dimensions more minor then the given dimensions, and middle is the size of
// the given dimensions.
std::array<int64_t, 3> PartitionShapeByMiddleDimensions(
    const Shape& shape, absl::Span<const int64_t> dims_middle) {
  CHECK(LayoutUtil::AreDimensionsConsecutive(shape.layout(), dims_middle));
  std::array<int64_t, 3> values = {1, 1, 1};
  enum Segment { kMajor = 0, kMiddle = 1, kMinor = 2 };
  Segment cur_segment = kMinor;

  for (int64_t cur_dim : LayoutUtil::MinorToMajor(shape)) {
    if (cur_segment != kMajor) {
      // Handle change of segments.
      bool cur_dim_in_middle = absl::c_linear_search(dims_middle, cur_dim);
      if (cur_segment == kMinor) {
        if (cur_dim_in_middle) {
          cur_segment = kMiddle;
        }
      } else if (cur_segment == kMiddle) {
        if (!cur_dim_in_middle) {
          cur_segment = kMajor;
        }
      }
    }
    values[cur_segment] *= shape.dimensions(cur_dim);
  }
  return values;
}

Shape GetShapeFromTensorType(mlir::Value value) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emission_utilsDTcc mht_1(mht_1_v, 243, "", "./tensorflow/compiler/xla/service/gpu/ir_emission_utils.cc", "GetShapeFromTensorType");

  constexpr char kDefaultLayoutAttrName[] = "xla_shape";

  mlir::Operation* op = value.getDefiningOp();
  CHECK(op);
  CHECK(value.getType().isa<mlir::TensorType>());
  Shape shape;
  if (auto attr = op->getAttrOfType<mlir::StringAttr>(kDefaultLayoutAttrName)) {
    shape = *xla::ParseShape(
        absl::string_view(attr.getValue().data(), attr.getValue().size()));
  } else {
    shape = TypeToShape(value.getType());
  }
  return shape;
}

}  // namespace

bool IsMatrixMultiplication(const HloInstruction& dot) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emission_utilsDTcc mht_2(mht_2_v, 264, "", "./tensorflow/compiler/xla/service/gpu/ir_emission_utils.cc", "IsMatrixMultiplication");

  if (dot.opcode() != HloOpcode::kDot) {
    return false;
  }
  const Shape& lhs_shape = dot.operand(0)->shape();
  const Shape& rhs_shape = dot.operand(1)->shape();
  const DotDimensionNumbers& dim_numbers = dot.dot_dimension_numbers();

  PrimitiveType output_primitive_type = dot.shape().element_type();
  bool type_is_allowed =
      (output_primitive_type == F16 || output_primitive_type == BF16 ||
       output_primitive_type == F32 || output_primitive_type == F64 ||
       output_primitive_type == C64 || output_primitive_type == C128) ||
      (output_primitive_type == S32 && lhs_shape.element_type() == S8 &&
       lhs_shape.element_type() == S8);
  bool shapes_are_valid =
      type_is_allowed &&
      IsRank2(lhs_shape, dim_numbers.lhs_batch_dimensions_size()) &&
      IsRank2(rhs_shape, dim_numbers.lhs_batch_dimensions_size()) &&
      IsRank2(dot.shape(), dim_numbers.lhs_batch_dimensions_size()) &&
      !ShapeUtil::IsZeroElementArray(lhs_shape) &&
      !ShapeUtil::IsZeroElementArray(rhs_shape);

  if (!shapes_are_valid) {
    return false;
  }

  // The size of the reduction dimension should match. The shape inference
  // guarantees this invariant, so the check here is for programming
  // errors.
  CHECK_EQ(lhs_shape.dimensions(dim_numbers.lhs_contracting_dimensions(0)),
           rhs_shape.dimensions(dim_numbers.rhs_contracting_dimensions(0)));

  return true;
}

std::array<int64_t, 3> GetReductionTiling(
    const ReductionDimensions& reduction_dimensions,
    se::CudaComputeCapability cuda_compute_capability) {
  if (reduction_dimensions.is_row_reduction) {
    int64_t tile_z = std::min(reduction_dimensions.dimensions[0],
                              BatchedReductionRaceFreeBound());
    return {tile_z, 1, 16};
  }

  // Column reduction.
  return {1, 128, 1};
}

const char* const kCusolverCholeskyCallTarget = "__cusolver$cholesky";

bool IsCustomCallToCusolver(const HloInstruction& hlo) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emission_utilsDTcc mht_3(mht_3_v, 318, "", "./tensorflow/compiler/xla/service/gpu/ir_emission_utils.cc", "IsCustomCallToCusolver");

  if (hlo.opcode() != HloOpcode::kCustomCall) {
    return false;
  }
  const auto& target = hlo.custom_call_target();
  return target == kCusolverCholeskyCallTarget;
}

static ReductionDimensions GetReductionKindAndContiguousComponentsImpl(
    const Shape& input_shape, absl::Span<const int64_t> dims_to_reduce) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emission_utilsDTcc mht_4(mht_4_v, 330, "", "./tensorflow/compiler/xla/service/gpu/ir_emission_utils.cc", "GetReductionKindAndContiguousComponentsImpl");

  DimensionVector dims_to_keep;
  for (int64_t dim = 0; dim < input_shape.rank(); ++dim) {
    if (!absl::c_linear_search(dims_to_reduce, dim)) {
      dims_to_keep.push_back(dim);
    }
  }

  if (dims_to_keep.empty()) {
    return {/*is_row_reduction=*/true,
            {1, 1, ShapeUtil::ElementsIn(input_shape)}};
  }

  if (LayoutUtil::AreDimensionsConsecutive(input_shape.layout(),
                                           dims_to_keep)) {
    std::array<int64_t, 3> shape_partition =
        PartitionShapeByMiddleDimensions(input_shape, dims_to_keep);
    if (shape_partition[1] == 1) {
      return {/*is_row_reduction=*/true,
              {1, 1, shape_partition[0] * shape_partition[2]}};
    }
    if (shape_partition[2] == 1) {
      return {/*is_row_reduction=*/false,
              {1, shape_partition[0], shape_partition[1]}};
    }
    return {/*is_row_reduction=*/true, shape_partition};
  }

  std::array<int64_t, 3> shape_partition =
      PartitionShapeByMiddleDimensions(input_shape, dims_to_reduce);

  if (shape_partition[2] == 1) {
    return {/*is_row_reduction=*/true,
            {1, shape_partition[0], shape_partition[1]}};
  }
  return {/*is_row_reduction=*/false, shape_partition};
}

static bool IsUnnestedReductionFasterThanElemental(
    const ReductionDimensions& reduction_dimensions) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emission_utilsDTcc mht_5(mht_5_v, 372, "", "./tensorflow/compiler/xla/service/gpu/ir_emission_utils.cc", "IsUnnestedReductionFasterThanElemental");

  if (reduction_dimensions.is_row_reduction) {
    // For row reduction, the tile block is 1 x tile_size_x, and we are reducing
    // along tile_size_x which needs to be large enough to make the tiling
    // implementation efficient.
    return reduction_dimensions.dimensions[2] >= WarpSize();
  }

  // For column reduction, the tile block is tile_size_y x tile_size_x, and we
  // are reducing along tile_size_y. Only tile_size_y needs to be
  // large enough to make the tiling implementation efficient.
  int64_t major_size = reduction_dimensions.dimensions[1];
  int64_t minor_size = reduction_dimensions.dimensions[2];

  // Rule generated by sweeping the search space of small column reductions.
  bool prefer_elemental_emitter =
      (major_size < WarpSize()) ||
      (major_size < 2 * WarpSize() && minor_size < WarpSize()) ||
      (major_size < 4 * WarpSize() && minor_size < 8) ||
      (major_size < 8 * WarpSize() && minor_size < 3);

  return !prefer_elemental_emitter;
}

// Whether we can/should use the unnested emitter for reduction.
static bool IsReductionFromOrToContiguousDimensionsImpl(
    const Shape& operand_shape, absl::Span<int64_t const> dims_to_reduce) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emission_utilsDTcc mht_6(mht_6_v, 401, "", "./tensorflow/compiler/xla/service/gpu/ir_emission_utils.cc", "IsReductionFromOrToContiguousDimensionsImpl");

  DimensionVector dims_to_keep;
  for (int64_t dim = 0; dim < operand_shape.dimensions().size(); ++dim) {
    if (!absl::c_linear_search(dims_to_reduce, dim)) {
      dims_to_keep.push_back(dim);
    }
  }

  // We support fast codegen for three cases:
  // 1) Row reduction: (K, R)
  // 2) Column reduction: (K, R, K)
  // 3) "Batched" row reduction: (R, K, R)
  return (LayoutUtil::AreDimensionsConsecutive(operand_shape.layout(),
                                               dims_to_keep) ||
          LayoutUtil::AreDimensionsConsecutive(operand_shape.layout(),
                                               dims_to_reduce)) &&
         IsUnnestedReductionFasterThanElemental(
             GetReductionKindAndContiguousComponentsImpl(operand_shape,
                                                         dims_to_reduce));
}

bool IsReductionFromOrToContiguousDimensions(const HloInstruction& reduce) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emission_utilsDTcc mht_7(mht_7_v, 425, "", "./tensorflow/compiler/xla/service/gpu/ir_emission_utils.cc", "IsReductionFromOrToContiguousDimensions");

  return reduce.opcode() == HloOpcode::kReduce &&
         IsReductionFromOrToContiguousDimensionsImpl(reduce.operand(0)->shape(),
                                                     reduce.dimensions());
}

bool IsReductionFromOrToContiguousDimensions(mlir::Operation* op) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emission_utilsDTcc mht_8(mht_8_v, 434, "", "./tensorflow/compiler/xla/service/gpu/ir_emission_utils.cc", "IsReductionFromOrToContiguousDimensions");

  auto reduce = mlir::dyn_cast<mlir::mhlo::ReduceOp>(op);
  if (!reduce) {
    return false;
  }

  mlir::Value first_input = reduce.inputs()[0];
  Shape operand_shape = GetShape(first_input);

  llvm::SmallVector<int64_t> dimensions_to_reduce;
  for (const llvm::APInt& d : reduce.dimensions()) {
    dimensions_to_reduce.push_back(d.getZExtValue());
  }

  return IsReductionFromOrToContiguousDimensionsImpl(operand_shape,
                                                     dimensions_to_reduce);
}

bool IsInputFusibleSlices(mlir::Operation* unnested_hlo,
                          bool verify_no_strides) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emission_utilsDTcc mht_9(mht_9_v, 456, "", "./tensorflow/compiler/xla/service/gpu/ir_emission_utils.cc", "IsInputFusibleSlices");

  auto fusion = mlir::dyn_cast<mlir::lmhlo::FusionOp>(unnested_hlo);
  if (!fusion) {
    return false;
  }

  auto is_non_strided = [](mlir::DenseIntElementsAttr strides) -> bool {
    return absl::c_all_of(
        strides, [](const llvm::APInt& stride) { return stride == 1; });
  };

  for (mlir::Value value : fusion.getFusionResults()) {
    auto slice =
        mlir::dyn_cast_or_null<mlir::mhlo::SliceOp>(value.getDefiningOp());
    if (!slice) {
      return false;
    }
    if (verify_no_strides && !is_non_strided(slice.strides())) {
      return false;
    }
  }
  return true;
}

ReductionDimensions GetReductionKindAndContiguousComponents(
    const HloInstruction& reduce) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emission_utilsDTcc mht_10(mht_10_v, 484, "", "./tensorflow/compiler/xla/service/gpu/ir_emission_utils.cc", "GetReductionKindAndContiguousComponents");

  return GetReductionKindAndContiguousComponentsImpl(reduce.operand(0)->shape(),
                                                     reduce.dimensions());
}

ReductionDimensions GetReductionKindAndContiguousComponents(
    mlir::Operation* reduce) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emission_utilsDTcc mht_11(mht_11_v, 493, "", "./tensorflow/compiler/xla/service/gpu/ir_emission_utils.cc", "GetReductionKindAndContiguousComponents");

  mlir::Value input = reduce->getOperand(0);
  Shape operand_shape = GetShape(input);
  llvm::SmallVector<int64_t> dimensions_to_reduce;
  for (const llvm::APInt& d :
       mlir::cast<mlir::mhlo::ReduceOp>(reduce).dimensions()) {
    dimensions_to_reduce.push_back(d.getZExtValue());
  }
  return GetReductionKindAndContiguousComponentsImpl(operand_shape,
                                                     dimensions_to_reduce);
}

// This emits a device-side call to
// "i32 vprintf(i8* fmt, arguments_type* arguments)" in the driver; see
// http://docs.nvidia.com/cuda/ptx-writers-guide-to-interoperability/index.html#system-calls
llvm::Value* EmitPrintf(absl::string_view fmt,
                        absl::Span<llvm::Value* const> arguments,
                        llvm::IRBuilder<>* builder) {
   std::vector<std::string> mht_12_v;
   mht_12_v.push_back("fmt: \"" + std::string(fmt.data(), fmt.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emission_utilsDTcc mht_12(mht_12_v, 514, "", "./tensorflow/compiler/xla/service/gpu/ir_emission_utils.cc", "EmitPrintf");

  std::vector<llvm::Type*> argument_types;

  // Variadic arguments implicit promotion [1] converts float to double,
  // and bool/char/short are converted to int.
  // [1] https://en.cppreference.com/w/cpp/language/variadic_arguments
  auto requires_int32_promotion = [](llvm::Type* type) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emission_utilsDTcc mht_13(mht_13_v, 523, "", "./tensorflow/compiler/xla/service/gpu/ir_emission_utils.cc", "lambda");

    return type->isIntegerTy(/*BitWidth=*/1) ||
           type->isIntegerTy(/*BitWidth=*/8) ||
           type->isIntegerTy(/*BitWidth=*/16);
  };
  auto requires_double_promotion = [](llvm::Type* type) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emission_utilsDTcc mht_14(mht_14_v, 531, "", "./tensorflow/compiler/xla/service/gpu/ir_emission_utils.cc", "lambda");

    return type->isFloatingPointTy();
  };

  for (auto argument : arguments) {
    llvm::Type* type = argument->getType();
    if (requires_double_promotion(type)) {
      argument_types.push_back(builder->getDoubleTy());
    } else if (requires_int32_promotion(type)) {
      argument_types.push_back(builder->getInt32Ty());
    } else {
      argument_types.push_back(type);
    }
  }
  auto* arguments_type = llvm::StructType::create(argument_types);
  llvm::Value* arguments_ptr = builder->CreateAlloca(arguments_type);
  for (size_t i = 0; i < arguments.size(); ++i) {
    llvm::Value* value = arguments[i];
    llvm::Type* type = value->getType();
    if (requires_double_promotion(type)) {
      value = builder->CreateFPCast(value, builder->getDoubleTy());
    } else if (requires_int32_promotion(type)) {
      value = builder->CreateIntCast(value, builder->getInt32Ty(),
                                     /*isSigned=*/true);
    }
    builder->CreateStore(
        value,
        builder->CreateGEP(arguments_type, arguments_ptr,
                           {builder->getInt64(0), builder->getInt32(i)}));
  }
  llvm::Type* ptr_ty = builder->getInt8Ty()->getPointerTo();
  return builder->CreateCall(
      builder->GetInsertBlock()->getParent()->getParent()->getOrInsertFunction(
          "vprintf",
          llvm::FunctionType::get(builder->getInt32Ty(), {ptr_ty, ptr_ty},
                                  /*isVarArg=*/false)),
      {builder->CreateGlobalStringPtr(llvm_ir::AsStringRef(fmt)),
       builder->CreatePointerCast(arguments_ptr, ptr_ty)});
}

// Helper function to emit call to AMDGPU shfl_down function.
llvm::Value* EmitAMDGPUShflDown(llvm::Value* value, llvm::Value* offset,
                                llvm::IRBuilder<>* b) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emission_utilsDTcc mht_15(mht_15_v, 576, "", "./tensorflow/compiler/xla/service/gpu/ir_emission_utils.cc", "EmitAMDGPUShflDown");

  llvm::Module* module = b->GetInsertBlock()->getModule();
  CHECK_EQ(value->getType()->getPrimitiveSizeInBits(), 32);
  auto* i32_ty = b->getInt32Ty();
  llvm::FunctionCallee shfl_fn = module->getOrInsertFunction(
      llvm_ir::AsStringRef("__ockl_readuplane_i32"),
      llvm::FunctionType::get(/*Result=*/i32_ty, {i32_ty, i32_ty},
                              /*isVarArg=*/false));
  // AMDGPU device function requires first argument as i32.
  llvm::Value* result =
      b->CreateCall(shfl_fn, {b->CreateBitCast(value, i32_ty), offset});
  // AMDGPU device function always returns an i32 type.
  return b->CreateBitCast(result, value->getType());
}

// Helper function to emit call to NVPTX shfl_down intrinsic.
llvm::Value* EmitNVPTXShflDown(llvm::Value* value, llvm::Value* offset,
                               llvm::IRBuilder<>* b) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emission_utilsDTcc mht_16(mht_16_v, 596, "", "./tensorflow/compiler/xla/service/gpu/ir_emission_utils.cc", "EmitNVPTXShflDown");

  llvm::Module* module = b->GetInsertBlock()->getModule();
  llvm::Intrinsic::ID llvm_intrinsic_id;
  CHECK_EQ(value->getType()->getPrimitiveSizeInBits(), 32);
  if (value->getType()->isFloatTy()) {
    llvm_intrinsic_id = llvm::Intrinsic::nvvm_shfl_sync_down_f32;
  } else {
    llvm_intrinsic_id = llvm::Intrinsic::nvvm_shfl_sync_down_i32;
  }
  llvm::Function* intrinsic =
      llvm::Intrinsic::getDeclaration(module, llvm_intrinsic_id, {});
  return b->CreateCall(
      intrinsic, {b->getInt32(-1), value, offset, b->getInt32(WarpSize() - 1)});
}

llvm::Value* EmitFullWarpShuffleDown(llvm::Value* value, llvm::Value* offset,
                                     llvm::IRBuilder<>* builder) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emission_utilsDTcc mht_17(mht_17_v, 615, "", "./tensorflow/compiler/xla/service/gpu/ir_emission_utils.cc", "EmitFullWarpShuffleDown");

  int bit_width = value->getType()->getPrimitiveSizeInBits();
  llvm::Module* module = builder->GetInsertBlock()->getModule();
  llvm::Triple target_triple = llvm::Triple(module->getTargetTriple());

  // Special case for efficiency
  if (value->getType()->isFloatTy() && bit_width == 32) {
    if (target_triple.isNVPTX()) {
      return EmitNVPTXShflDown(value, offset, builder);
    } else if (target_triple.getArch() == llvm::Triple::amdgcn) {
      return EmitAMDGPUShflDown(value, offset, builder);
    } else {
      LOG(FATAL) << "Invalid triple " << target_triple.str();
    }
  }

  // We must split values wider than 32 bits as the "shfl" instruction operates
  // on 32-bit values.
  int num_segments = CeilOfRatio(bit_width, 32);
  llvm::Value* x = builder->CreateBitCast(
      builder->CreateZExt(
          builder->CreateBitCast(value, builder->getIntNTy(bit_width)),
          builder->getIntNTy(32 * num_segments)),
      llvm::VectorType::get(builder->getInt32Ty(), num_segments, false));
  for (int i = 0; i < num_segments; ++i) {
    llvm::Value* insert_val;
    if (target_triple.isNVPTX()) {
      insert_val = EmitNVPTXShflDown(builder->CreateExtractElement(x, i),
                                     offset, builder);
    } else if (target_triple.getArch() == llvm::Triple::amdgcn) {
      insert_val = EmitAMDGPUShflDown(builder->CreateExtractElement(x, i),
                                      offset, builder);
    } else {
      LOG(FATAL) << "Invalid triple " << target_triple.str();
    }
    x = builder->CreateInsertElement(x, insert_val, i);
  }
  return builder->CreateBitCast(
      builder->CreateTrunc(
          builder->CreateBitCast(x, builder->getIntNTy(32 * num_segments)),
          builder->getIntNTy(bit_width)),
      value->getType());
}

llvm::Value* IsBlock0Thread0(llvm::IRBuilder<>* b) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emission_utilsDTcc mht_18(mht_18_v, 662, "", "./tensorflow/compiler/xla/service/gpu/ir_emission_utils.cc", "IsBlock0Thread0");

  llvm::Value* is_thread0 = b->CreateICmpEQ(
      b->getInt32(0),
      EmitCallToTargetIntrinsic(TargetIntrinsicID::kThreadIdx, {}, {}, b));

  llvm::Value* is_block0 = b->CreateICmpEQ(
      b->getInt32(0),
      EmitCallToTargetIntrinsic(TargetIntrinsicID::kBlockIdx, {}, {}, b));
  return b->CreateAnd(is_thread0, is_block0);
}

bool IsFusedReductionOutputConsistent(const HloInstruction* inst,
                                      const HloInstruction* first_reduce) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emission_utilsDTcc mht_19(mht_19_v, 677, "", "./tensorflow/compiler/xla/service/gpu/ir_emission_utils.cc", "IsFusedReductionOutputConsistent");

  if (IsReductionFromOrToContiguousDimensions(*inst)) {
    // Shapes, layouts and dimensions must be the same for all reduces
    // inside of this fusion.
    // TODO(tjoerg): Relax the shape constraint. The datatype does not matter.
    return ShapeUtil::Equal(first_reduce->shape(), inst->shape()) &&
           ShapeUtil::Equal(first_reduce->operand(0)->shape(),
                            inst->operand(0)->shape()) &&
           ShapeUtil::Equal(first_reduce->operand(1)->shape(),
                            inst->operand(1)->shape()) &&
           first_reduce->dimensions() == inst->dimensions();
  }
  return ShapeUtil::CompatibleIgnoringElementType(
             first_reduce->operand(0)->shape(), inst->shape()) &&
         LayoutUtil::Equal(first_reduce->operand(0)->shape().layout(),
                           inst->shape().layout());
}

// Given an LMHLO op, returns the operand index of the first output operand.
//
// Notice that an operand alised to an output isn't an output, even though in
// that case WritesMlirBuffer() returns true on that operand.
//
// An operand is !WritesMlirBuffer() || equals (aliases) to a later operand. An
// output is the opposite, being both WritesMlirBuffer() and does not equal to
// any later operand.
int PartitionLmhloOperandsAndOutputs(mlir::Operation* op) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emission_utilsDTcc mht_20(mht_20_v, 706, "", "./tensorflow/compiler/xla/service/gpu/ir_emission_utils.cc", "PartitionLmhloOperandsAndOutputs");

  CHECK(op->getDialect() == op->getContext()->getLoadedDialect("lmhlo"));

  int i;
  for (i = op->getOperands().size() - 1; i >= 0; i--) {
    const bool aliased =
        std::find(op->getOperands().begin() + i + 1, op->getOperands().end(),
                  op->getOperand(i)) != op->getOperands().end();
    if (!WritesMlirBuffer(op, op->getOperand(i)) || aliased) {
      break;
    }
  }
  return i + 1;
}

std::vector<mlir::Value> GetHloOperands(mlir::Operation* op) {
  if (auto fusion = mlir::dyn_cast<mlir::lmhlo::FusionOp>(op)) {
    return ToStdVector(fusion.getInputBuffers());
  }
  if (op->getDialect() == op->getContext()->getLoadedDialect("lmhlo")) {
    int output_start = PartitionLmhloOperandsAndOutputs(op);
    std::vector<mlir::Value> operands;
    operands.reserve(output_start);
    for (int i = 0; i < output_start; i++) {
      operands.push_back(op->getOperand(i));
    }
    return operands;
  }
  if (op->getDialect() == op->getContext()->getLoadedDialect("mhlo")) {
    return std::vector<mlir::Value>(op->getOperands().begin(),
                                    op->getOperands().end());
  }
  LOG(FATAL) << "Unexpected op: " << MlirToString(op);
}

std::vector<mlir::Value> GetHloOutputs(mlir::Operation* op) {
  if (auto fusion = mlir::dyn_cast<mlir::lmhlo::FusionOp>(op)) {
    return ToStdVector(fusion.getOutputBuffers());
  }
  if (op->getDialect() == op->getContext()->getLoadedDialect("lmhlo")) {
    int output_start = PartitionLmhloOperandsAndOutputs(op);
    std::vector<mlir::Value> outputs;
    for (int i = output_start; i < op->getNumOperands(); i++) {
      outputs.push_back(op->getOperand(i));
    }
    return outputs;
  }
  if (op->getDialect() == op->getContext()->getLoadedDialect("mhlo")) {
    return std::vector<mlir::Value>(op->getResults().begin(),
                                    op->getResults().end());
  }
  LOG(FATAL) << "Unexpected op: " << MlirToString(op);
}

bool WritesMlirBuffer(mlir::Operation* op, mlir::Value operand) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emission_utilsDTcc mht_21(mht_21_v, 763, "", "./tensorflow/compiler/xla/service/gpu/ir_emission_utils.cc", "WritesMlirBuffer");

  llvm::SmallVector<mlir::MemoryEffects::EffectInstance, 2> effects;
  mlir::cast<mlir::MemoryEffectOpInterface>(op).getEffectsOnValue(operand,
                                                                  effects);
  return absl::c_any_of(
      effects, [](const mlir::MemoryEffects::EffectInstance& instance) {
        return mlir::isa<mlir::MemoryEffects::Write>(instance.getEffect());
      });
}

static int64_t GetMemRefSizeInBytes(mlir::MemRefType type) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emission_utilsDTcc mht_22(mht_22_v, 776, "", "./tensorflow/compiler/xla/service/gpu/ir_emission_utils.cc", "GetMemRefSizeInBytes");

  // For i1 memrefs, the underlying allocation is 8 bits.
  if (type.getElementType().isInteger(/*width=*/1)) {
    return type.getNumElements();
  } else {
    return type.cast<mlir::ShapedType>().getSizeInBits() / CHAR_BIT;
  }
}

static int64_t GetAllocationIndex(mlir::BlockArgument func_arg,
                                  std::string* constant_name) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emission_utilsDTcc mht_23(mht_23_v, 789, "", "./tensorflow/compiler/xla/service/gpu/ir_emission_utils.cc", "GetAllocationIndex");

  auto func_op =
      mlir::cast<mlir::func::FuncOp>(func_arg.getParentRegion()->getParentOp());
  if (constant_name) {
    if (auto constant_name_attr = func_op.getArgAttrOfType<mlir::StringAttr>(
            func_arg.getArgNumber(), "lmhlo.constant_name")) {
      *constant_name = constant_name_attr.getValue().str();
    }
  }
  return func_arg.getArgNumber();
}

StatusOr<BufferAllocation::Slice> GetAllocationSlice(
    mlir::Value v, absl::Span<const BufferAllocation> allocations,
    std::string* constant_name) {
  if (constant_name) {
    constant_name->clear();
  }

  int64_t size = GetMemRefSizeInBytes(v.getType().cast<mlir::MemRefType>());

  // We match the following patterns here:
  //  base := ViewOp(arg) | get_global_memref (global_memref) | arg
  //  root := base | MemRefReinterpretCastOp(base)

  if (auto cast = mlir::dyn_cast_or_null<mlir::memref::ReinterpretCastOp>(
          v.getDefiningOp())) {
    v = cast.getViewSource();
  }
  if (auto view =
          mlir::dyn_cast_or_null<mlir::memref::ViewOp>(v.getDefiningOp())) {
    TF_RET_CHECK(view.source().isa<mlir::BlockArgument>());

    return BufferAllocation::Slice(
        &allocations[GetAllocationIndex(
            view.source().cast<mlir::BlockArgument>(), constant_name)],
        mlir::cast<mlir::arith::ConstantOp>(view.byte_shift().getDefiningOp())
            .getValue()
            .cast<mlir::IntegerAttr>()
            .getValue()
            .getSExtValue(),
        size);
  }
  if (auto get_global = mlir::dyn_cast_or_null<mlir::memref::GetGlobalOp>(
          v.getDefiningOp())) {
    auto module = get_global->getParentOfType<mlir::ModuleOp>();
    if (constant_name) {
      *constant_name = get_global.name().str();
    }
    auto global = mlir::cast<mlir::memref::GlobalOp>(
        module.lookupSymbol(get_global.name()));
    int64_t index =
        global->getAttrOfType<mlir::IntegerAttr>("lmhlo.alloc").getInt();
    return BufferAllocation::Slice(&allocations[index], 0,
                                   allocations[index].size());
  }
  if (auto arg = v.dyn_cast<mlir::BlockArgument>()) {
    return BufferAllocation::Slice(
        &allocations[GetAllocationIndex(arg, constant_name)], 0, size);
  }

  return Unimplemented(
      "Operand has to be in the form of ViewOp(arg) or "
      "StaticMemRefCastOp(ViewOp(arg)) or arg");
}

bool CanEmitFusedDynamicUpdateSliceInPlaceForGpu(
    mlir::lmhlo::FusionOp fusion,
    absl::Span<const BufferAllocation> allocations) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emission_utilsDTcc mht_24(mht_24_v, 860, "", "./tensorflow/compiler/xla/service/gpu/ir_emission_utils.cc", "CanEmitFusedDynamicUpdateSliceInPlaceForGpu");

  auto results = fusion.getFusionResults();
  if (results.size() != 1) {
    return false;
  }
  auto dus = mlir::dyn_cast<mlir::mhlo::DynamicUpdateSliceOp>(
      results[0].getDefiningOp());
  if (!dus) {
    return false;
  }

  auto output_buffers = fusion.getOutputBuffers();
  CHECK_EQ(1, output_buffers.size());
  auto parameter = mlir::dyn_cast<mlir::bufferization::ToTensorOp>(
      dus.operand().getDefiningOp());

  if (!parameter) {
    return false;
  }

  auto maybe_lhs = GetAllocationSlice(parameter.memref(), allocations);
  auto maybe_rhs = GetAllocationSlice(output_buffers[0], allocations);
  return maybe_lhs.ok() && maybe_rhs.ok() && *maybe_lhs == *maybe_rhs;
}

Shape GetShape(mlir::Value value) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emission_utilsDTcc mht_25(mht_25_v, 888, "", "./tensorflow/compiler/xla/service/gpu/ir_emission_utils.cc", "GetShape");

  if (value.getType().isa<mlir::MemRefType>()) {
    return TypeToShape(value.getType());
  } else if (value.getType().isa<mlir::TensorType>()) {
    return GetShapeFromTensorType(value);
  } else if (value.getType().isa<mlir::TupleType>()) {
    return TypeToShape(value.getType());
  }
  LOG(FATAL) << "Unexpected value type to get shape for";
  return {};
}

bool ReductionIsRaceFree(const ReductionDimensions& reduction_dimensions,
                         const std::array<int64_t, 3>& reduction_tiling) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emission_utilsDTcc mht_26(mht_26_v, 904, "", "./tensorflow/compiler/xla/service/gpu/ir_emission_utils.cc", "ReductionIsRaceFree");

  return (reduction_dimensions.is_row_reduction &&
          reduction_dimensions.dimensions[2] <=
              MinThreadsXRowReduction() * reduction_tiling[2] &&
          reduction_dimensions.dimensions[0] <=
              BatchedReductionRaceFreeBound()) ||
         (!reduction_dimensions.is_row_reduction &&
          reduction_dimensions.dimensions[1] <=
              WarpSize() * reduction_tiling[1]);
}

}  // namespace gpu
}  // namespace xla
