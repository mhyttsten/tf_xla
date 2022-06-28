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
class MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSbenchmarksPSbenchmark_mlir_functionDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSbenchmarksPSbenchmark_mlir_functionDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSbenchmarksPSbenchmark_mlir_functionDTcc() {
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

#include "tensorflow/compiler/mlir/tfrt/benchmarks/benchmark_mlir_function.h"

#include <functional>
#include <memory>
#include <utility>

#include "llvm/Support/SourceMgr.h"
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tfrt/runtime_fallback/runtime_fallback_executor.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/passes.h"
#include "tensorflow/compiler/mlir/tfrt/utils/host_context.h"
#include "tensorflow/core/framework/tensor.h"
#include "tfrt/host_context/execution_context.h"  // from @tf_runtime
#include "tfrt/host_context/resource_context.h"  // from @tf_runtime

namespace tensorflow {

using ::tfrt::ArrayRef;
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

// Returns random tensors generated based on the input specs.
static llvm::SmallVector<Tensor> GetInputTensors(
    llvm::ArrayRef<InputTensorSpec> input_specs) {
  llvm::SmallVector<Tensor> input_tensors;

  for (const InputTensorSpec& spec : input_specs) {
    TensorShape shape;
    CHECK(TensorShapeUtils::MakeShape(spec.dims, &shape).ok());
    input_tensors.emplace_back(spec.dtype, shape);

    // Initialize tensors with random data.
    switch (spec.dtype) {
      case DT_FLOAT:
        input_tensors.back().flat<float>().setRandom();
        break;
      case DT_INT64:
        input_tensors.back().flat<int64_t>().setRandom();
        break;
      default:
        CHECK(false) << "Unsupported dtype: " << spec.dtype;
    }
  }

  return input_tensors;
}

// -------------------------------------------------------------------------- //
// Run function benchmark via the TF JitRt compilation.
// -------------------------------------------------------------------------- //

void RunJitRtBenchmark(::testing::benchmark::State& state,
                       llvm::StringRef mlir_input,
                       llvm::StringRef function_name,
                       llvm::ArrayRef<InputTensorSpec> input_specs,
                       bool vectorize, bool codegen_transpose) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSbenchmarksPSbenchmark_mlir_functionDTcc mht_0(mht_0_v, 251, "", "./tensorflow/compiler/mlir/tfrt/benchmarks/benchmark_mlir_function.cc", "RunJitRtBenchmark");

  // Number of worker threads.
  int64_t num_threads = state.range(0);

  // Host context for running compute tasks.
  std::unique_ptr<HostContext> host =
      num_threads > 0 ? CreateMultiThreadedHostContext(num_threads)
                      : CreateSingleThreadedHostContext();

  TfJitRtPipelineOptions tf_jitrt_opts;
  tf_jitrt_opts.vectorize = vectorize;
  tf_jitrt_opts.codegen_transpose = codegen_transpose;
  JitExecutable& jit_executable =
      CreateJitExecutable(*host, mlir_input, function_name,
                          /*lower_from_tensorflow=*/true, tf_jitrt_opts);

  // Build an ExecutionContext from the HostContext.
  llvm::Expected<RCReference<RequestContext>> req_ctx =
      RequestContextBuilder(host.get(), /*resource_context=*/nullptr).build();
  tfrt::ExecutionContext exec_ctx(std::move(*req_ctx));

  // Generate random inputs based on the tensor specs.
  llvm::SmallVector<Tensor> input_tensors = GetInputTensors(input_specs);

  // Record data ptrs of inputs.
  llvm::SmallVector<void*> input_ptrs;
  // Convert input tensors to memref descriptors.
  llvm::SmallVector<MemrefDesc> operands;
  for (const Tensor& tensor : input_tensors) {
    input_ptrs.push_back(tensor.data());
    operands.emplace_back(TensorToMemrefDesc(tensor));
  }

  // Get an executable that might be specialized to the operands.
  llvm::Expected<AsyncValuePtr<Executable>> executable =
      jit_executable.GetExecutable(operands);
  if (auto err = executable.takeError())
    LOG(FATAL) << "Failed to specialize executable: " << tfrt::StrCat(err);

  // Wait for the compilation completion.
  host->Await({executable->CopyRef()});

  CHECK(!executable->IsError())
      << "Failed to get executable: " << tfrt::StrCat(executable->GetError());
  CHECK(!(*executable)->IsAsync()) << "async results are not supported";

  // Placeholders for returned values.
  unsigned num_results = (*executable)->num_results();
  llvm::SmallVector<RCReference<AsyncValue>> result_values(num_results);
  RemainingResults results(result_values);

  // Free memory owned by the returned memrefs.
  ResultConversionCtx result_ctx(std::move(input_ptrs));
  ReturnValueConverter<ResultConversionCtx> converter(results, result_ctx);
  converter.AddConversion(FreeReturnedMemref);

  // Initialize call frame with MemrefDesc operands.
  Executable::CallFrame call_frame;
  if (auto err = (*executable)->InitializeCallFrame(operands, &call_frame))
    LOG(FATAL) << "Failed to initialize call frame";

  // Execute async tasks in the HostContext work queue.
  Executable::ExecuteOpts opts;
  HostContextAsyncTaskRunner async_task_runner(host.get());
  opts.async_task_runner = &async_task_runner;

  // Execute compiled kernel and return results.
  auto execute = [&]() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSbenchmarksPSbenchmark_mlir_functionDTcc mht_1(mht_1_v, 321, "", "./tensorflow/compiler/mlir/tfrt/benchmarks/benchmark_mlir_function.cc", "lambda");

    call_frame.args[0] = nullptr;  // reset kernel context argument
    (*executable)->Execute(call_frame, opts);
    if (auto err = (*executable)->ReturnResults(converter, &call_frame))
      LOG(FATAL) << "Failed to return compiled kernel results";
  };

  // Warm up to compile the kernel outside of the benchmark loop.
  execute();

  for (auto _ : state) {
    execute();
  }
}

// -------------------------------------------------------------------------- //
// Run function benchmark via the TF->TFRT fallback lowering.
// -------------------------------------------------------------------------- //

void RunTfrtBenchmark(::testing::benchmark::State& state,
                      llvm::StringRef mlir_input, llvm::StringRef function_name,
                      ArrayRef<InputTensorSpec> input_specs) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSbenchmarksPSbenchmark_mlir_functionDTcc mht_2(mht_2_v, 345, "", "./tensorflow/compiler/mlir/tfrt/benchmarks/benchmark_mlir_function.cc", "RunTfrtBenchmark");

  // Number of worker threads (intra-op concurrency for the fallback ops).
  int64_t num_threads = state.range(0);
  RuntimeFallbackExecutor executor(num_threads);

  executor.Prepare(mlir_input);

  // Generate random inputs based on the tensor specs.
  llvm::SmallVector<Tensor> input_tensors = GetInputTensors(input_specs);

  for (auto _ : state) {
    executor.Execute(function_name, input_tensors);
  }
}

// -------------------------------------------------------------------------- //
// Run arbitrary benchark written as a function.
// -------------------------------------------------------------------------- //

void RunEigenBenchmark(
    ::testing::benchmark::State& state,
    std::function<void(llvm::ArrayRef<Tensor>,
                       llvm::Optional<Eigen::ThreadPoolDevice>)>
        compute,
    llvm::ArrayRef<InputTensorSpec> input_specs) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSbenchmarksPSbenchmark_mlir_functionDTcc mht_3(mht_3_v, 372, "", "./tensorflow/compiler/mlir/tfrt/benchmarks/benchmark_mlir_function.cc", "RunEigenBenchmark");

  // Number of worker threads.
  int64_t num_threads = state.range(0);

  // Maybe construct an Eigen thread pool device for evaluating expressions.
  Eigen::ThreadPool thread_pool(num_threads);
  llvm::Optional<Eigen::ThreadPoolDevice> device;
  if (num_threads > 0) device.emplace(&thread_pool, num_threads);

  // Generate random inputs based on the tensor specs.
  llvm::SmallVector<Tensor> input_tensors = GetInputTensors(input_specs);

  // Call the user defined compute function.
  for (auto _ : state) {
    compute(input_tensors, device);
  }
}

}  // namespace tensorflow
