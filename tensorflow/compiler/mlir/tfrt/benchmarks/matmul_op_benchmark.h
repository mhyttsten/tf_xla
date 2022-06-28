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

#ifndef TENSORFLOW_COMPILER_MLIR_TFRT_BENCHMARKS_MATMUL_BENCHMARK_H_
#define TENSORFLOW_COMPILER_MLIR_TFRT_BENCHMARKS_MATMUL_BENCHMARK_H_
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
class MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSbenchmarksPSmatmul_op_benchmarkDTh {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSbenchmarksPSmatmul_op_benchmarkDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSbenchmarksPSmatmul_op_benchmarkDTh() {
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


#include <utility>

#include "tensorflow/compiler/mlir/tfrt/benchmarks/benchmark.h"
#include "tensorflow/compiler/mlir/tfrt/utils/host_context.h"

namespace tensorflow {

// This header is a part of the library with private visibility and will be
// used only to build benchmarks for different functions in this folder, so
// it is ok to put convenience using-declarations here.

using ::tfrt::AsyncValue;
using ::tfrt::AsyncValuePtr;
using ::tfrt::HostContext;
using ::tfrt::RCReference;
using ::tfrt::RemainingResults;
using ::tfrt::RequestContext;
using ::tfrt::RequestContextBuilder;
using ::tfrt::jitrt::Executable;
using ::tfrt::jitrt::HostContextAsyncTaskRunner;
using ::tfrt::jitrt::JitExecutable;
using ::tfrt::jitrt::MemrefDesc;
using ::tfrt::jitrt::ReturnValueConverter;

// -------------------------------------------------------------------------- //
// Run benchmark by compiling MLIR function using TFRT JitRt API.
// -------------------------------------------------------------------------- //

template <typename T>
void RunMatMulMlirBenchmark(::testing::benchmark::State& state,
                            llvm::StringRef mlir_input,
                            llvm::StringRef function_name) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSbenchmarksPSmatmul_op_benchmarkDTh mht_0(mht_0_v, 219, "", "./tensorflow/compiler/mlir/tfrt/benchmarks/matmul_op_benchmark.h", "RunMatMulMlirBenchmark");

  // MatMul: [m, k] x [k, n]
  ssize_t m = state.range(0);
  ssize_t k = state.range(1);
  ssize_t n = state.range(2);

  std::unique_ptr<HostContext> host = CreateSingleThreadedHostContext();

  TfJitRtPipelineOptions tf_jitrt_opts;
  JitExecutable& jit_executable =
      CreateJitExecutable(*host, mlir_input, function_name,
                          /*lower_from_tensorflow=*/true, tf_jitrt_opts);

  // Build an ExecutionContext from the HostContext.
  llvm::Expected<RCReference<RequestContext>> req_ctx =
      RequestContextBuilder(host.get(), /*resource_context=*/nullptr).build();
  tfrt::ExecutionContext exec_ctx(std::move(*req_ctx));

  // Generate random input data.
  std::array<ssize_t, 2> lhs_dims = {m, k};
  std::array<ssize_t, 2> rhs_dims = {k, n};

  Eigen::Tensor<T, 2, Eigen::RowMajor> lhs = GenRandomTensor<T, 2>(lhs_dims);
  Eigen::Tensor<T, 2, Eigen::RowMajor> rhs = GenRandomTensor<T, 2>(rhs_dims);

  std::array<MemrefDesc, 2> operands = {TensorToMemrefDesc(lhs),
                                        TensorToMemrefDesc(rhs)};

  auto result_values = std::array<RCReference<AsyncValue>, 2>{{}};
  RemainingResults results(result_values);

  // Record data ptrs of inputs.
  llvm::SmallVector<void*> input_ptrs;
  for (auto& operand : operands) {
    input_ptrs.push_back(operand.data);
  }

  // Free memory owned by the returned memrefs.
  ResultConversionCtx result_ctx(std::move(input_ptrs));
  ReturnValueConverter<ResultConversionCtx> converter(results, result_ctx);
  converter.AddConversion(FreeReturnedMemref);

  // Execute async tasks in the HostContext work queue.
  Executable::ExecuteOpts opts;
  HostContextAsyncTaskRunner async_task_runner(host.get());
  opts.async_task_runner = &async_task_runner;

  // Get an executable that might be specialized to the operands.
  llvm::Expected<AsyncValuePtr<Executable>> executable =
      jit_executable.GetExecutable(operands);
  if (auto err = executable.takeError())
    LOG(FATAL) << "Failed to specialize executable";

  // Wait for the compilation completion.
  host->Await({executable->CopyRef()});

  CHECK(!executable->IsError())
      << "Failed to get executable: " << StrCat(executable->GetError());
  CHECK(!(*executable)->IsAsync()) << "async results are not supported";

  // Initialize call frame with MemrefDesc operands.
  Executable::CallFrame call_frame;
  if (auto err = (*executable)->InitializeCallFrame(operands, &call_frame))
    LOG(FATAL) << "Failed to initialize call frame";

  for (auto _ : state) {
    (*executable)->Execute(call_frame, opts);
    if (auto err = (*executable)->ReturnResults(converter, &call_frame))
      LOG(FATAL) << "Failed to return compiled kernel results";
  }

  state.SetItemsProcessed(state.iterations() * m * k * n);
}

// -------------------------------------------------------------------------- //
// Run benchmark using Eigen expression evaluation.
// -------------------------------------------------------------------------- //

template <typename T>
void RunMatMulEigenBenchmark(::testing::benchmark::State& state) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSbenchmarksPSmatmul_op_benchmarkDTh mht_1(mht_1_v, 301, "", "./tensorflow/compiler/mlir/tfrt/benchmarks/matmul_op_benchmark.h", "RunMatMulEigenBenchmark");

  // MatMul: [m, k] x [k, n]
  ssize_t m = state.range(0);
  ssize_t k = state.range(1);
  ssize_t n = state.range(2);

  // Generate random input data.
  std::array<ssize_t, 2> lhs_dims = {m, k};
  std::array<ssize_t, 2> rhs_dims = {k, n};

  Eigen::Tensor<T, 2, Eigen::RowMajor> lhs = GenRandomTensor<T, 2>(lhs_dims);
  Eigen::Tensor<T, 2, Eigen::RowMajor> rhs = GenRandomTensor<T, 2>(rhs_dims);

  using Device = Eigen::DefaultDevice;
  Device d;

  Eigen::Tensor<T, 2, Eigen::RowMajor> dst(m, n);
  dst.setZero();

  Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> contract_pairs;
  contract_pairs[0] = Eigen::IndexPair<Eigen::DenseIndex>(1, 0);

  for (auto _ : state) {
    auto expr = lhs.contract(rhs, contract_pairs);

    using Dst = decltype(dst);
    using Expr = decltype(expr);
    ExecuteAssignOp</*vectorize=*/true, Device, Dst, Expr>::run(d, dst, expr);
  }

  state.SetItemsProcessed(state.iterations() * m * k * n);
}

}  // namespace tensorflow

// -------------------------------------------------------------------------- //
// Macros to dispatch to different MatMul shapes.
// -------------------------------------------------------------------------- //

#define BM_TFMlir(NAME, MLIR_INPUT, FN, TYPE)                               \
  static void BM_mlir_##NAME##_##TYPE(::testing::benchmark::State& state) { \
    RunMatMulMlirBenchmark<TYPE>(state, MLIR_INPUT, FN);                    \
  }                                                                         \
  BENCHMARK(BM_mlir_##NAME##_##TYPE)

#define BM_Eigen(NAME, TYPE)                                                 \
  static void BM_eigen_##NAME##_##TYPE(::testing::benchmark::State& state) { \
    RunMatMulEigenBenchmark<TYPE>(state);                                    \
  }                                                                          \
  BENCHMARK(BM_eigen_##NAME##_##TYPE)

#endif  // TENSORFLOW_COMPILER_MLIR_TFRT_BENCHMARKS_MATMUL_BENCHMARK_H_
