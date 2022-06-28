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

#ifndef TENSORFLOW_COMPILER_MLIR_TFRT_BENCHMARKS_BENCHMARK_H_
#define TENSORFLOW_COMPILER_MLIR_TFRT_BENCHMARKS_BENCHMARK_H_
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
class MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSbenchmarksPSbenchmarkDTh {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSbenchmarksPSbenchmarkDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSbenchmarksPSbenchmarkDTh() {
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


#define EIGEN_USE_THREADS

#include <memory>
#include <utility>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "tensorflow/compiler/mlir/tfrt/jit/tf_jitrt_pipeline.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tfrt/jitrt/jitrt.h"  // from @tf_runtime
#include "tfrt/dtype/dtype.h"  // from @tf_runtime
#include "tfrt/host_context/host_context.h"  // from @tf_runtime
#include "tfrt/tensor/dense_host_tensor.h"  // from @tf_runtime
#include "tfrt/tensor/tensor_shape.h"  // from @tf_runtime

namespace tensorflow {

// This header is a part of the library with private visibility and will be
// used only to build benchmarks for different functions in this folder, so
// it is ok to put convenience using-declarations here.

using ::tfrt::HostContext;
using ::tfrt::RemainingResults;
using ::tfrt::jitrt::JitExecutable;
using ::tfrt::jitrt::MemrefDesc;
using ::tfrt::jitrt::Type;

// Constants to make shape specification more readable.
// kStaticDim refers to the static shape in IR taken from ARGS of the benchmark.
ABSL_CONST_INIT extern const bool kStaticDim;
// kDynamicDim refers to the dynamic shape `?` in IR.
ABSL_CONST_INIT extern const bool kDynamicDim;

// Generate random Eigen Tensor of the given dimensions:
//   (rand<T>() + offset) * scale
template <typename T, int rank>
Eigen::Tensor<T, rank, Eigen::RowMajor> GenRandomTensor(
    std::array<ssize_t, rank> dims, T scale = static_cast<T>(1.0),
    T offset = static_cast<T>(0.0)) {
  Eigen::DSizes<Eigen::Index, rank> dsizes;
  for (int d = 0; d < rank; ++d) dsizes[d] = dims[d];
  Eigen::Tensor<T, rank, Eigen::RowMajor> tensor(dsizes);
  tensor.setRandom();
  tensor = scale * (tensor + offset);  // shift random numbers
  return tensor;
}

// -------------------------------------------------------------------------- //
// Run benchmark by compiling MLIR function using TFRT JitRt API.
// -------------------------------------------------------------------------- //

// Record data ptrs of inputs to free the returned memrefs only if necessary.
struct ResultConversionCtx {
  explicit ResultConversionCtx(llvm::SmallVector<void*>&& ptrs)
      : input_ptrs(std::move(ptrs)) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSbenchmarksPSbenchmarkDTh mht_0(mht_0_v, 245, "", "./tensorflow/compiler/mlir/tfrt/benchmarks/benchmark.h", "ResultConversionCtx");
}
  llvm::SmallVector<void*> input_ptrs;
};

// Result converter that simply frees the memrefs returned from the compiled
// functions. We are not interested in the computed results, and constructing
// async values and tensors adds noticeable overhead.
mlir::LogicalResult FreeReturnedMemref(const ResultConversionCtx&,
                                       RemainingResults results,
                                       unsigned result_index, const Type* type,
                                       const Type* runtime_type,
                                       void* result_ptr);

// Compile serialized mlir module and convert entrypoint function into TFRT JIT
// executable.
JitExecutable& CreateJitExecutable(const HostContext& host,
                                   llvm::StringRef mlir_input,
                                   llvm::StringRef function_name,
                                   bool lower_from_tensorflow,
                                   const TfJitRtPipelineOptions& tf_jitrt_opts);

// Converts Eigen Tensor to Memref descriptor.
template <typename T, int rank>
MemrefDesc TensorToMemrefDesc(Eigen::Tensor<T, rank, Eigen::RowMajor>& tensor) {
  tfrt::TensorShape shape(tensor.dimensions().values);
  MemrefDesc desc;
  desc.dtype = tfrt::GetDType<T>();
  desc.data = tensor.data();
  desc.offset = 0;
  shape.GetDimensions(&desc.sizes);
  shape.GetStrides(&desc.strides);
  return desc;
}

// Converts Tensorflow Tensor to Memref descriptor.
MemrefDesc TensorToMemrefDesc(const Tensor& tensor);

// -------------------------------------------------------------------------- //
// Initialize Eigen tensor.
// -------------------------------------------------------------------------- //

template <typename T, int RANK>
struct InitEigenTensor {
  static Eigen::Tensor<T, RANK, Eigen::RowMajor> Get(
      const std::array<ssize_t, RANK>&);
};

#define INIT_TENSOR(RANK, UNROLL)                         \
  template <typename T>                                   \
  struct InitEigenTensor<T, RANK> {                       \
    static Eigen::Tensor<T, RANK, Eigen::RowMajor> Get(   \
        const std::array<ssize_t, RANK>& shape) {         \
      Eigen::Tensor<T, RANK, Eigen::RowMajor> dst UNROLL; \
      return dst;                                         \
    }                                                     \
  };

template <typename T>
struct InitEigenTensor<T, 0> {
  static Eigen::Tensor<T, 0, Eigen::RowMajor> Get(
      const std::array<ssize_t, 0>&) {
    return Eigen::Tensor<T, 0, Eigen::RowMajor>();
  }
};

INIT_TENSOR(1, (shape[0]));
INIT_TENSOR(2, (shape[0], shape[1]));
INIT_TENSOR(3, (shape[0], shape[1], shape[2]));

// -------------------------------------------------------------------------- //
// Run benchmark using Eigen expression evaluation.
// -------------------------------------------------------------------------- //

// Explicitly control if Eigen assignment should use SIMD instructions or not.
template <bool vectorize, typename Device, typename Dst, typename Expr>
struct ExecuteAssignOp {
  static void run(Device& d, Dst& dst, const Expr& expr) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSbenchmarksPSbenchmarkDTh mht_1(mht_1_v, 324, "", "./tensorflow/compiler/mlir/tfrt/benchmarks/benchmark.h", "run");

    using Assign = Eigen::TensorAssignOp<Dst, const Expr>;
    using Executor =
        Eigen::internal::TensorExecutor<const Assign, Device,
                                        /*Vectorizable=*/vectorize>;
    Executor::run(Assign(dst, expr), d);
  }
};

// -------------------------------------------------------------------------- //
// Common utilities.
// -------------------------------------------------------------------------- //

static constexpr int64_t kDynSize = mlir::ShapedType::kDynamicSize;

// Prints an MLIR tensor type, i.e. for `shape` {1, kDynSize} and `element_type`
// "f32" the output is "tensor<1x?xf32>".
std::string PrintTensorType(llvm::ArrayRef<int64_t> shape,
                            llvm::StringRef element_type);

// Prints an MLIR dense array attribute, i.e. for `array` {1, 2} the output is
// "dense<[1, 2]>".
std::string PrintDenseArray(llvm::ArrayRef<int32_t> array);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TFRT_BENCHMARKS_BENCHMARK_H_
