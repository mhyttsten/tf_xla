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
class MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSlmhlo_to_gpuPSgemm_patternDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSlmhlo_to_gpuPSgemm_patternDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSlmhlo_to_gpuPSgemm_patternDTcc() {
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

// Copyright 2020 The TensorFlow Runtime Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//===- gemm_pattern.cc
//---------------------------------------------------------===//
//
// Pattern to lower lhlogpu_gemm Ops to tfrt cuda dialect.
//
//===----------------------------------------------------------------------===//
#include <assert.h>
#include <stdint.h>

#include <type_traits>
#include <utility>

#include "mlir-hlo/Dialect/lhlo_gpu/IR/lhlo_gpu_ops.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Types.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tfrt/transforms/lmhlo_to_gpu/pattern_utils.h"
#include "tensorflow/compiler/mlir/xla/type_to_shape.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/util.h"
#include "tfrt/gpu/kernels/gpu_ops.h"  // from @tf_runtime
#include "tfrt/gpu/passes/passes.h"  // from @tf_runtime
#include "tfrt/gpu/wrapper/cublas_wrapper.h"  // from @tf_runtime
#include "tfrt/gpu/wrapper/rocblas_wrapper.h"  // from @tf_runtime
#include "tfrt/basic_kernels/opdefs/basic_kernels.h"  // from @tf_runtime
#include "tfrt/basic_kernels/opdefs/types.h"  // from @tf_runtime

namespace tensorflow {
namespace {

// This struct contains the metadata of a matrix, e.g., its base address and
// dimensions.
struct MatrixDescriptor {
  Value data;
  bool transpose;  // Whether this matrix needs to be transposed.
  int64_t num_rows;
  int64_t num_cols;
  int64_t stride;
};

FloatAttr GetBeta(lmhlo_gpu::GEMMOp op) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSlmhlo_to_gpuPSgemm_patternDTcc mht_0(mht_0_v, 233, "", "./tensorflow/compiler/mlir/tfrt/transforms/lmhlo_to_gpu/gemm_pattern.cc", "GetBeta");
 return nullptr; }
Value GetBias(lmhlo_gpu::GEMMOpAdaptor op) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSlmhlo_to_gpuPSgemm_patternDTcc mht_1(mht_1_v, 237, "", "./tensorflow/compiler/mlir/tfrt/transforms/lmhlo_to_gpu/gemm_pattern.cc", "GetBias");
 return nullptr; }

FloatAttr GetBeta(lmhlo_gpu::GEMM_BiasOp op) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSlmhlo_to_gpuPSgemm_patternDTcc mht_2(mht_2_v, 242, "", "./tensorflow/compiler/mlir/tfrt/transforms/lmhlo_to_gpu/gemm_pattern.cc", "GetBeta");
 return op.betaAttr(); }
Value GetBias(lmhlo_gpu::GEMM_BiasOpAdaptor op) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSlmhlo_to_gpuPSgemm_patternDTcc mht_3(mht_3_v, 246, "", "./tensorflow/compiler/mlir/tfrt/transforms/lmhlo_to_gpu/gemm_pattern.cc", "GetBias");
 return op.bias(); }

// Match GEMM auto-tuning, see ComputationTypeFromPrimitive()
Type MlirComputationType(Type element_type,
                         ConversionPatternRewriter& rewriter) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSlmhlo_to_gpuPSgemm_patternDTcc mht_4(mht_4_v, 253, "", "./tensorflow/compiler/mlir/tfrt/transforms/lmhlo_to_gpu/gemm_pattern.cc", "MlirComputationType");

  if (element_type.isF16() || element_type.isBF16())
    return rewriter.getF32Type();

#if !TENSORFLOW_USE_ROCM
  if (auto complex_type = element_type.dyn_cast<mlir::ComplexType>())
    return complex_type.getElementType();
#endif

  return element_type;
}

// Gets the platform specific Gemm algorithm value.
template <class GemmOp>
tfrt::gpu::wrapper::BlasGemmAlgo GetBlasGemmAlgoOrDefault(GemmOp op) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSlmhlo_to_gpuPSgemm_patternDTcc mht_5(mht_5_v, 270, "", "./tensorflow/compiler/mlir/tfrt/transforms/lmhlo_to_gpu/gemm_pattern.cc", "GetBlasGemmAlgoOrDefault");

  if (!op.algorithm().hasValue()) return kBlasGemmDefaultAlgo;
  return {static_cast<int>(op.algorithm().getValue()), kGpuTargetPlatform};
}

// Returns the platform specific matrix transpose operation value.
tfrt::gpu::wrapper::BlasOperation MatrixTransposeToBlasOperation(
    bool transpose) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSlmhlo_to_gpuPSgemm_patternDTcc mht_6(mht_6_v, 280, "", "./tensorflow/compiler/mlir/tfrt/transforms/lmhlo_to_gpu/gemm_pattern.cc", "MatrixTransposeToBlasOperation");

  return transpose ? kBlasOperationTranspose : kBlasOperationNone;
}

// Create all the Ops necessary for the GEMM operation, including the GEMM
// operation itself.
template <class GemmOp>
Value CreateTfrtOps(GemmOp op, typename GemmOp::Adaptor adaptor, Value chain,
                    Value stream, int64_t batch_size, mlir::Type input_type,
                    mlir::Type output_type, MatrixDescriptor lhs_matrix,
                    MatrixDescriptor rhs_matrix, MatrixDescriptor output_matrix,
                    llvm::APFloat alpha_real, llvm::APFloat alpha_imaginary,
                    llvm::APFloat beta_real,
                    ConversionPatternRewriter& rewriter) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSlmhlo_to_gpuPSgemm_patternDTcc mht_7(mht_7_v, 296, "", "./tensorflow/compiler/mlir/tfrt/transforms/lmhlo_to_gpu/gemm_pattern.cc", "CreateTfrtOps");

  auto loc = op.getLoc();
  if (auto bias = GetBias(adaptor)) {
    chain = rewriter.create<tfrt::gpu::MemCopyOp>(loc, adaptor.output(), bias,
                                                  stream, chain);
  }

  auto k_val = lhs_matrix.transpose ? lhs_matrix.num_rows : lhs_matrix.num_cols;

  const Type mlir_compute_type = MlirComputationType(output_type, rewriter);

  auto m = rewriter.create<tfrt::compiler::ConstantI32Op>(
      loc, output_matrix.num_rows);
  auto n = rewriter.create<tfrt::compiler::ConstantI32Op>(
      loc, output_matrix.num_cols);
  auto k = rewriter.create<tfrt::compiler::ConstantI32Op>(loc, k_val);

  // Scale type must match compute type, except for complex types, where
  // it must match the output type
  const Type mlir_scale_type =
      output_type.isa<mlir::ComplexType>() ? output_type : mlir_compute_type;

  auto const_alpha = MakeScalingFactorConstant(rewriter, loc, mlir_scale_type,
                                               alpha_real, alpha_imaginary);

  auto lda =
      rewriter.create<tfrt::compiler::ConstantI32Op>(loc, lhs_matrix.num_rows);
  auto ldb =
      rewriter.create<tfrt::compiler::ConstantI32Op>(loc, rhs_matrix.num_rows);

  llvm::APFloat fp_zero = APFloat::getZero(alpha_imaginary.getSemantics());
  auto const_beta = MakeScalingFactorConstant(rewriter, loc, mlir_scale_type,
                                              beta_real, fp_zero);

  auto ldc = rewriter.create<tfrt::compiler::ConstantI32Op>(
      loc, output_matrix.num_rows);

  tfrt::gpu::wrapper::BlasGemmAlgo algorithm = GetBlasGemmAlgoOrDefault(op);
  auto algo = rewriter.create<tfrt::gpu::BlasGemmAlgoOp>(loc, algorithm);

  Value context = rewriter.create<tfrt::gpu::StreamGetContextOp>(loc, stream);
  auto handle = rewriter.create<tfrt::gpu::BlasCreateOp>(loc, context);

  auto lhs_op = MatrixTransposeToBlasOperation(lhs_matrix.transpose);
  auto rhs_op = MatrixTransposeToBlasOperation(rhs_matrix.transpose);

  const auto input_data_type = MlirTypeToBlasDataType(input_type);
  const auto output_data_type = MlirTypeToBlasDataType(output_type);
  const auto compute_type = MlirTypeToBlasComputeType(mlir_compute_type);
  if (batch_size != 1) {
    auto lhs_stride =
        rewriter.create<tfrt::compiler::ConstantI64Op>(loc, lhs_matrix.stride);
    auto rhs_stride =
        rewriter.create<tfrt::compiler::ConstantI64Op>(loc, rhs_matrix.stride);
    auto output_stride = rewriter.create<tfrt::compiler::ConstantI64Op>(
        loc, output_matrix.stride);
    auto batch =
        rewriter.create<tfrt::compiler::ConstantI32Op>(loc, batch_size);
    return rewriter
        .create<tfrt::gpu::BlasGemmBatchExOp>(
            loc, chain.getType(), handle, stream, lhs_op, rhs_op, m, n, k,
            const_alpha, lhs_matrix.data, input_data_type, lda, lhs_stride,
            rhs_matrix.data, input_data_type, ldb, rhs_stride, const_beta,
            output_matrix.data, output_data_type, ldc, output_stride, batch,
            compute_type, algo, chain)
        .getResult();
  }

  return rewriter
      .create<tfrt::gpu::BlasGemmOp>(
          loc, chain.getType(), handle, stream, lhs_op, rhs_op, m, n, k,
          const_alpha, lhs_matrix.data, input_data_type, lda, rhs_matrix.data,
          input_data_type, ldb, const_beta, output_matrix.data,
          output_data_type, ldc, compute_type, algo, chain)
      .getResult();
}

template <class GemmOp>
FailureOr<Value> GemmOpConversionRewrite(GemmOp op,
                                         typename GemmOp::Adaptor adaptor,
                                         Value chain, Value stream,
                                         ConversionPatternRewriter& rewriter) {
  auto get_element_type = [](Value value) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSlmhlo_to_gpuPSgemm_patternDTcc mht_8(mht_8_v, 381, "", "./tensorflow/compiler/mlir/tfrt/transforms/lmhlo_to_gpu/gemm_pattern.cc", "lambda");

    return value.getType().cast<mlir::MemRefType>().getElementType();
  };
  mlir::Type output_type = get_element_type(op.output());
  mlir::Type input_type = get_element_type(op.lhs());
  if (get_element_type(op.rhs()) != input_type) {
    return rewriter.notifyMatchFailure(op, "Input element type mismatch.");
  }

  const xla::Shape output_shape = xla::gpu::GetShape(op.output());
  const xla::Shape lhs_shape = xla::gpu::GetShape(op.lhs());
  const xla::Shape rhs_shape = xla::gpu::GetShape(op.rhs());
  const mlir::mhlo::DotDimensionNumbersAttr dim_nums =
      op.dot_dimension_numbers();
  absl::Span<const int64_t> output_batch_dims =
      (dim_nums.getLhsBatchingDimensions().size() >
       dim_nums.getRhsBatchingDimensions().size())
          ? dim_nums.getLhsBatchingDimensions()
          : dim_nums.getRhsBatchingDimensions();

  int64_t batch_size = op.batch_size();
  int64_t output_row_dim = output_batch_dims.size();
  int64_t output_col_dim = output_row_dim + 1;

  if (op.rhs_stride() && op.lhs_stride()) {
    if (dim_nums.getLhsBatchingDimensions().size() !=
        dim_nums.getRhsBatchingDimensions().size()) {
      return rewriter.notifyMatchFailure(
          op, "Batching dimension size mismatch for nonzero strides.");
    }
  }

  int64_t output_num_rows = output_shape.dimensions(output_row_dim);
  int64_t output_num_cols = output_shape.dimensions(output_col_dim);

  auto validate_matrix = [&](const xla::Shape& shape,
                             auto batch_dimensions) -> LogicalResult {
    int64_t row_dim = batch_dimensions.size();
    int64_t col_dim = row_dim + 1;
    if (row_dim + 2 != shape.rank()) {
      return rewriter.notifyMatchFailure(op, "Invalid dimensions.");
    }

    for (int64_t batch_dim : batch_dimensions) {
      if (row_dim == batch_dim || col_dim == batch_dim) {
        return rewriter.notifyMatchFailure(
            op, "Batch dimensions overlap the last two dimensions.");
      }
    }

    // Verify that the non-batch dimensions are minor-most. This is required for
    // efficient access.
    if (shape.layout().minor_to_major(row_dim) >= 2 ||
        shape.layout().minor_to_major(col_dim) >= 2) {
      return rewriter.notifyMatchFailure(
          op, "Non-batch dimensions are not minor-most.");
    }
    return success();
  };

  auto valid_lhs =
      validate_matrix(lhs_shape, dim_nums.getLhsBatchingDimensions());
  if (failed(valid_lhs)) return valid_lhs;
  auto valid_rhs =
      validate_matrix(rhs_shape, dim_nums.getRhsBatchingDimensions());
  if (failed(valid_rhs)) return valid_rhs;
  auto valid_output = validate_matrix(output_shape, output_batch_dims);
  if (failed(valid_output)) return valid_output;

  // BLAS gemm expects the inputs and the output are in column-major order.
  // Therefore, we need to convert dot between row-major matrices to that
  // between column-major matrices. The key insight for the conversion is that,
  // in linear storage, matrix M in column-major order is identical to the
  // transpose of M in row-major order. In other words,
  //
  //   column-major(M) = row-major(M^T).
  //
  // Leveraging this insight, we can perform dot between row-major matrices as
  // follows.
  //
  // row-major(C)
  //   = row-major(A x B) = column-major((A x B)^T) = column-major(B^T x A^T)
  //   = gemm(column-major(B^T), column-major(A^T))
  //   = gemm(row-major(B), row-major(A))
  //
  // Although we do not modify the content of A and B in linear memory, we
  // should use the dimensions of B^T and A^T when calling gemm. For example,
  // the leading dimension of the LHS matrix of gemm is the number of rows in
  // B^T and thus the number of columns in B.
  auto make_descriptor = [&](Value data, const xla::Shape& shape,
                             int64_t row_dim, bool transpose,
                             int64_t stride) -> MatrixDescriptor {
    bool is_row_major = xla::LayoutUtil::Minor(shape.layout(), row_dim) != 0;
    bool layout_mismatch =
        xla::LayoutUtil::Minor(shape.layout(), row_dim) !=
        xla::LayoutUtil::Minor(output_shape.layout(), output_row_dim);
    int64_t rows =
        shape.dimensions(row_dim + static_cast<int64_t>(is_row_major));
    int64_t cols =
        shape.dimensions(row_dim + static_cast<int64_t>(!is_row_major));
    return MatrixDescriptor{data, transpose != layout_mismatch, rows, cols,
                            stride};
  };

  bool lhs_transpose = dim_nums.getLhsContractingDimensions()[0] ==
                       dim_nums.getLhsBatchingDimensions().size();
  bool rhs_transpose = dim_nums.getRhsContractingDimensions()[0] ==
                       dim_nums.getRhsBatchingDimensions().size() + 1;

  MatrixDescriptor lhs_matrix = make_descriptor(
      adaptor.lhs(), lhs_shape, dim_nums.getLhsBatchingDimensions().size(),
      lhs_transpose, op.lhs_stride());
  MatrixDescriptor rhs_matrix = make_descriptor(
      adaptor.rhs(), rhs_shape, dim_nums.getRhsBatchingDimensions().size(),
      rhs_transpose, op.rhs_stride());

  if (xla::LayoutUtil::Minor(output_shape.layout(), output_row_dim) != 0) {
    std::swap(lhs_matrix, rhs_matrix);
    std::swap(output_num_cols, output_num_rows);
  }

  const MatrixDescriptor output_matrix{adaptor.output(), /*transpose=*/false,
                                       output_num_rows, output_num_cols,
                                       output_num_rows * output_num_cols};

  auto valid_stride = [](const MatrixDescriptor& matrix) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSlmhlo_to_gpuPSgemm_patternDTcc mht_9(mht_9_v, 509, "", "./tensorflow/compiler/mlir/tfrt/transforms/lmhlo_to_gpu/gemm_pattern.cc", "lambda");

    if (matrix.stride != 0) {
      if (matrix.stride != matrix.num_rows * matrix.num_cols) return false;
    }
    return true;
  };
  if (!valid_stride(lhs_matrix) || !valid_stride(rhs_matrix) ||
      !valid_stride(output_matrix))
    return rewriter.notifyMatchFailure(op, "Invalid nonzero stride.");

  // Use zero with alpha's semantic if no beta_arg is supplied.
  llvm::APFloat beta_real = APFloat::getZero(op.alpha_real().getSemantics());
  if (auto attr = GetBeta(op)) beta_real = attr.getValue();

  return CreateTfrtOps(op, adaptor, chain, stream, batch_size, input_type,
                       output_type, lhs_matrix, rhs_matrix, output_matrix,
                       op.alpha_real(), op.alpha_imag(), beta_real, rewriter);
}

template <class GemmOpType>
struct GemmRewritePattern : tfrt::gpu::GpuAsyncOpConversionPattern<GemmOpType> {
  using typename tfrt::gpu::GpuAsyncOpConversionPattern<GemmOpType>::OpAdaptor;
  using tfrt::gpu::GpuAsyncOpConversionPattern<
      GemmOpType>::GpuAsyncOpConversionPattern;
  FailureOr<Value> matchAndRewriteOp(
      GemmOpType op, OpAdaptor adaptor, Value chain, Value stream,
      ConversionPatternRewriter& rewriter) const override {
    auto result = GemmOpConversionRewrite(op, adaptor, chain, stream, rewriter);
    if (succeeded(result)) rewriter.eraseOp(op);
    return result;
  }
};

}  // namespace

void populateGemmConversionPattern(RewritePatternSet& patterns,
                                   TypeConverter& converter) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSlmhlo_to_gpuPSgemm_patternDTcc mht_10(mht_10_v, 548, "", "./tensorflow/compiler/mlir/tfrt/transforms/lmhlo_to_gpu/gemm_pattern.cc", "populateGemmConversionPattern");

  patterns.add<GemmRewritePattern<lmhlo_gpu::GEMMOp>,
               GemmRewritePattern<lmhlo_gpu::GEMM_BiasOp>>(
      converter, patterns.getContext());
}

}  // namespace tensorflow
