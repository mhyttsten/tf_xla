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
class MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSlmhlo_to_gpuPSpattern_utilsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSlmhlo_to_gpuPSpattern_utilsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSlmhlo_to_gpuPSpattern_utilsDTcc() {
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

#include "tensorflow/compiler/mlir/tfrt/transforms/lmhlo_to_gpu/pattern_utils.h"

#include "tfrt/gpu/wrapper/cublas_wrapper.h"  // from @tf_runtime
#include "tfrt/gpu/wrapper/cudnn_wrapper.h"  // from @tf_runtime
#include "tfrt/gpu/wrapper/miopen_wrapper.h"  // from @tf_runtime
#include "tfrt/gpu/wrapper/rocblas_wrapper.h"  // from @tf_runtime
#include "tfrt/basic_kernels/opdefs/basic_kernels.h"  // from @tf_runtime

namespace tensorflow {

#if TENSORFLOW_USE_ROCM
const tfrt::gpu::wrapper::Platform kGpuTargetPlatform =
    tfrt::gpu::wrapper::Platform::ROCm;
const tfrt::gpu::wrapper::BlasGemmAlgo kBlasGemmDefaultAlgo =
    rocblas_gemm_algo_standard;
const tfrt::gpu::wrapper::BlasOperation kBlasOperationTranspose =
    rocblas_operation_transpose;
const tfrt::gpu::wrapper::BlasOperation kBlasOperationConjTranspose =
    rocblas_operation_conjugate_transpose;
const tfrt::gpu::wrapper::BlasOperation kBlasOperationNone =
    rocblas_operation_none;
const tfrt::gpu::wrapper::BlasFillMode kBlasFillModeLower = rocblas_fill_lower;
const tfrt::gpu::wrapper::BlasFillMode kBlasFillModeUpper = rocblas_fill_upper;
const tfrt::gpu::wrapper::BlasSideMode kBlasSideLeft = rocblas_side_left;
const tfrt::gpu::wrapper::BlasSideMode kBlasSideRight = rocblas_side_right;
const tfrt::gpu::wrapper::BlasDiagType kBlasDiagUnit = rocblas_diagonal_unit;
const tfrt::gpu::wrapper::BlasDiagType kBlasDiagNonUnit =
    rocblas_diagonal_non_unit;
#else
const tfrt::gpu::wrapper::Platform kGpuTargetPlatform =
    tfrt::gpu::wrapper::Platform::CUDA;
const tfrt::gpu::wrapper::BlasGemmAlgo kBlasGemmDefaultAlgo =
    CUBLAS_GEMM_DEFAULT;
const tfrt::gpu::wrapper::BlasOperation kBlasOperationTranspose = CUBLAS_OP_T;
const tfrt::gpu::wrapper::BlasOperation kBlasOperationConjTranspose =
    CUBLAS_OP_C;
const tfrt::gpu::wrapper::BlasOperation kBlasOperationNone = CUBLAS_OP_N;
const tfrt::gpu::wrapper::BlasFillMode kBlasFillModeLower =
    CUBLAS_FILL_MODE_LOWER;
const tfrt::gpu::wrapper::BlasFillMode kBlasFillModeUpper =
    CUBLAS_FILL_MODE_UPPER;
const tfrt::gpu::wrapper::BlasSideMode kBlasSideLeft = CUBLAS_SIDE_LEFT;
const tfrt::gpu::wrapper::BlasSideMode kBlasSideRight = CUBLAS_SIDE_RIGHT;
const tfrt::gpu::wrapper::BlasDiagType kBlasDiagUnit = CUBLAS_DIAG_UNIT;
const tfrt::gpu::wrapper::BlasDiagType kBlasDiagNonUnit = CUBLAS_DIAG_NON_UNIT;
#endif

tfrt::gpu::wrapper::BlasDataType MlirTypeToBlasDataType(mlir::Type type) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSlmhlo_to_gpuPSpattern_utilsDTcc mht_0(mht_0_v, 232, "", "./tensorflow/compiler/mlir/tfrt/transforms/lmhlo_to_gpu/pattern_utils.cc", "MlirTypeToBlasDataType");

#if TENSORFLOW_USE_ROCM
  if (type.isF16()) return rocblas_datatype_f16_r;
  if (type.isF32()) return rocblas_datatype_f32_r;
  if (type.isF64()) return rocblas_datatype_f64_r;
  if (auto complex_type = type.dyn_cast<mlir::ComplexType>()) {
    auto element_type = complex_type.getElementType();
    if (element_type.isF32()) return rocblas_datatype_f32_c;
    if (element_type.isF64()) return rocblas_datatype_f64_c;
  }
  if (type.isSignlessInteger(/*width=*/8)) return rocblas_datatype_i8_r;
  if (type.isSignlessInteger(/*width=*/32)) return rocblas_datatype_i32_r;
#else
  if (type.isF16()) return CUDA_R_16F;
  // Introduced in CUDA 11.
  if (type.isBF16()) return /*CUDA_R_16BF=*/cudaDataType(14);
  if (type.isF32()) return CUDA_R_32F;
  if (type.isF64()) return CUDA_R_64F;
  if (auto complex_type = type.dyn_cast<mlir::ComplexType>()) {
    auto element_type = complex_type.getElementType();
    if (element_type.isF32()) return CUDA_C_32F;
    if (element_type.isF64()) return CUDA_C_64F;
  }
  if (type.isSignlessInteger(/*width=*/8)) return CUDA_R_8I;
  if (type.isSignlessInteger(/*width=*/32)) return CUDA_R_32I;
#endif
  llvm_unreachable("unsupported type");
}

tfrt::gpu::wrapper::BlasComputeType MlirTypeToBlasComputeType(mlir::Type type) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSlmhlo_to_gpuPSpattern_utilsDTcc mht_1(mht_1_v, 264, "", "./tensorflow/compiler/mlir/tfrt/transforms/lmhlo_to_gpu/pattern_utils.cc", "MlirTypeToBlasComputeType");

#if TENSORFLOW_USE_ROCM
  if (type.isF16()) return rocblas_datatype_f16_r;
  if (type.isF32()) return rocblas_datatype_f32_r;
  if (type.isF64()) return rocblas_datatype_f64_r;
  if (auto complex_type = type.dyn_cast<mlir::ComplexType>()) {
    auto element_type = complex_type.getElementType();
    if (element_type.isF32()) return rocblas_datatype_f32_c;
    if (element_type.isF64()) return rocblas_datatype_f64_c;
  }
  if (type.isSignlessInteger(/*width=*/32)) return rocblas_datatype_i32_r;
#else
  if (type.isF16()) return CUBLAS_COMPUTE_16F;
  if (type.isF32()) return CUBLAS_COMPUTE_32F;
  if (type.isF64()) return CUBLAS_COMPUTE_64F;
  if (type.isSignlessInteger(/*width=*/32)) return CUBLAS_COMPUTE_32I;
#endif
  llvm_unreachable("unsupported type");
}

tfrt::gpu::wrapper::DnnDataType MlirTypeToDnnDataType(mlir::Type type) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSlmhlo_to_gpuPSpattern_utilsDTcc mht_2(mht_2_v, 287, "", "./tensorflow/compiler/mlir/tfrt/transforms/lmhlo_to_gpu/pattern_utils.cc", "MlirTypeToDnnDataType");

#if TENSORFLOW_USE_ROCM
  if (type.isF16()) return miopenHalf;
  if (type.isBF16()) return miopenBFloat16;
  if (type.isF32()) return miopenFloat;
  if (type.isF64()) return miopenDouble;
  if (type.isSignlessInteger(/*width=*/8)) return miopenInt8;
  if (type.isSignlessInteger(/*width=*/32)) return miopenInt32;
  // if (type.isSignlessInteger(/*width=*/64)) NOT SUPPORTED ON ROCM
  if (type.isUnsignedInteger(/*width=*/8)) return miopenInt8;
#else
  if (type.isF16()) return CUDNN_DATA_HALF;
  if (type.isBF16()) return CUDNN_DATA_BFLOAT16;
  if (type.isF32()) return CUDNN_DATA_FLOAT;
  if (type.isF64()) return CUDNN_DATA_DOUBLE;
  if (type.isSignlessInteger(/*width=*/8)) return CUDNN_DATA_INT8;
  if (type.isSignlessInteger(/*width=*/32)) return CUDNN_DATA_INT32;
  if (type.isSignlessInteger(/*width=*/64)) return CUDNN_DATA_INT64;
  if (type.isUnsignedInteger(/*width=*/8)) return CUDNN_DATA_UINT8;
#endif
  llvm_unreachable("unsupported type");
}

tfrt::gpu::wrapper::DnnDataType MlirTypeToDnnDataType(
    mlir::Type type, se::dnn::DataLayout data_layout) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSlmhlo_to_gpuPSpattern_utilsDTcc mht_3(mht_3_v, 314, "", "./tensorflow/compiler/mlir/tfrt/transforms/lmhlo_to_gpu/pattern_utils.cc", "MlirTypeToDnnDataType");

  switch (data_layout) {
#if TENSORFLOW_USE_ROCM
    case se::dnn::DataLayout::kBatchDepthYX4:
      if (type.isSignlessInteger(/*width=*/8)) {
        return miopenInt8x4;
      }
      if (type.isUnsignedInteger(/*width=*/8)) {
        return miopenInt8x4;
      }
      break;
    case se::dnn::DataLayout::kBatchDepthYX32:
      llvm_unreachable("unsupported type");
#else
    case se::dnn::DataLayout::kBatchDepthYX4:
      if (type.isSignlessInteger(/*width=*/8)) {
        return CUDNN_DATA_INT8x4;
      }
      if (type.isUnsignedInteger(/*width=*/8)) {
        return CUDNN_DATA_UINT8x4;
      }
      break;
    case se::dnn::DataLayout::kBatchDepthYX32:
      if (type.isSignlessInteger(/*width=*/8)) {
        return CUDNN_DATA_INT8x32;
      }
      break;
#endif
    default:
      break;
  }
  return MlirTypeToDnnDataType(type);
}

tfrt::gpu::wrapper::DnnDataType MlirTypeToDnnDataType(
    mlir::Type type, se::dnn::FilterLayout filter_layout) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSlmhlo_to_gpuPSpattern_utilsDTcc mht_4(mht_4_v, 352, "", "./tensorflow/compiler/mlir/tfrt/transforms/lmhlo_to_gpu/pattern_utils.cc", "MlirTypeToDnnDataType");

  switch (filter_layout) {
#if TENSORFLOW_USE_ROCM
    case se::dnn::FilterLayout::kOutputInputYX4:
      if (type.isSignlessInteger(/*width=*/8)) {
        return miopenInt8x4;
      }
      if (type.isUnsignedInteger(/*width=*/8)) {
        return miopenInt8x4;
      }
      break;
    case se::dnn::FilterLayout::kOutputInputYX32:
      llvm_unreachable("unsupported type");
#else
    case se::dnn::FilterLayout::kOutputInputYX4:
      if (type.isSignlessInteger(/*width=*/8)) {
        return CUDNN_DATA_INT8x4;
      }
      if (type.isUnsignedInteger(/*width=*/8)) {
        return CUDNN_DATA_UINT8x4;
      }
      break;
    case se::dnn::FilterLayout::kOutputInputYX32:
      if (type.isSignlessInteger(/*width=*/8)) {
        return CUDNN_DATA_INT8x32;
      }
      break;
#endif
    default:
      break;
  }
  return MlirTypeToDnnDataType(type);
}

mlir::Value MakeScalingFactorConstant(mlir::OpBuilder& builder,
                                      mlir::Location loc, mlir::Type type,
                                      llvm::APFloat value_real,
                                      llvm::APFloat value_imaginary) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSlmhlo_to_gpuPSpattern_utilsDTcc mht_5(mht_5_v, 392, "", "./tensorflow/compiler/mlir/tfrt/transforms/lmhlo_to_gpu/pattern_utils.cc", "MakeScalingFactorConstant");

  bool losesInfo = false;
  if (type.isBF16()) {
    value_real.convert(llvm::APFloat::BFloat(),
                       llvm::RoundingMode::NearestTiesToEven, &losesInfo);
    return builder.create<tfrt::compiler::ConstantBF16Op>(loc, value_real);
  }
  if (type.isF32()) {
    value_real.convert(llvm::APFloat::IEEEsingle(),
                       llvm::RoundingMode::NearestTiesToEven, &losesInfo);
    return builder.create<tfrt::compiler::ConstantF32Op>(loc, value_real);
  }
  if (type.isF64()) {
    value_real.convert(llvm::APFloat::IEEEdouble(),
                       llvm::RoundingMode::NearestTiesToEven, &losesInfo);
    return builder.create<tfrt::compiler::ConstantF64Op>(loc, value_real);
  }
  if (type.isa<mlir::ComplexType>()) {
    auto element_type = type.cast<ComplexType>().getElementType();
    if (element_type.isF32()) {
      value_real.convert(llvm::APFloat::IEEEsingle(),
                         llvm::RoundingMode::NearestTiesToEven, &losesInfo);
      value_imaginary.convert(llvm::APFloat::IEEEsingle(),
                              llvm::RoundingMode::NearestTiesToEven,
                              &losesInfo);
      return builder.create<tfrt::compiler::ConstantComplexF32Op>(
          loc, value_real, value_imaginary);
    }
    if (element_type.isF64()) {
      value_real.convert(llvm::APFloat::IEEEdouble(),
                         llvm::RoundingMode::NearestTiesToEven, &losesInfo);
      value_imaginary.convert(llvm::APFloat::IEEEdouble(),
                              llvm::RoundingMode::NearestTiesToEven,
                              &losesInfo);
      return builder.create<tfrt::compiler::ConstantComplexF64Op>(
          loc, value_real, value_imaginary);
    }
  }
  if (type.isSignlessInteger(/*width=*/32)) {
    llvm::APSInt value_int;
    bool is_exact = false;
    value_real.convertToInteger(
        value_int, llvm::RoundingMode::NearestTiesToEven, &is_exact);
    return builder.create<tfrt::compiler::ConstantI32Op>(
        loc, value_int.getExtValue());
  }

  llvm_unreachable("unsupported type");
}

mlir::Value MakeBitPatternConstant(mlir::OpBuilder& builder, mlir::Location loc,
                                   mlir::Type type, uint32_t bit_pattern) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSlmhlo_to_gpuPSpattern_utilsDTcc mht_6(mht_6_v, 446, "", "./tensorflow/compiler/mlir/tfrt/transforms/lmhlo_to_gpu/pattern_utils.cc", "MakeBitPatternConstant");

  llvm::APInt value(32, bit_pattern);
  if (type.isSignlessInteger(/*width=*/32)) {
    return builder.create<tfrt::compiler::ConstantI32Op>(loc,
                                                         value.getZExtValue());
  }
  if (type.isUnsignedInteger(/*width=*/32)) {
    return builder.create<tfrt::compiler::ConstantUI32Op>(loc,
                                                          value.getZExtValue());
  }
  if (type.isF32()) {
    llvm::APFloat value_float(value.bitsToFloat());  // Like reinterpret_cast.
    return builder.create<tfrt::compiler::ConstantF32Op>(loc, value_float);
  }

  llvm_unreachable("unsupported type");
}

}  // namespace tensorflow
