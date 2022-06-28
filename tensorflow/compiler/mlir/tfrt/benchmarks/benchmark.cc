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
class MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSbenchmarksPSbenchmarkDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSbenchmarksPSbenchmarkDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSbenchmarksPSbenchmarkDTcc() {
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

#include "tensorflow/compiler/mlir/tfrt/benchmarks/benchmark.h"

#include <string>

#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "tensorflow/core/platform/logging.h"
#include "tfrt/jitrt/jitrt_compiler.h"  // from @tf_runtime
#include "tfrt/host_context/concurrent_work_queue.h"  // from @tf_runtime
#include "tfrt/host_context/host_allocator.h"  // from @tf_runtime

namespace tensorflow {

using ::tfrt::HostContext;
using ::tfrt::jitrt::CompilationOptions;
using ::tfrt::jitrt::CompilationPipelineOptions;
using ::tfrt::jitrt::MemrefType;

const bool kStaticDim = false;
const bool kDynamicDim = true;

mlir::LogicalResult FreeReturnedMemref(const ResultConversionCtx& ctx,
                                       RemainingResults results,
                                       unsigned result_index, const Type* type,
                                       const Type* runtime_type,
                                       void* result_ptr) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSbenchmarksPSbenchmarkDTcc mht_0(mht_0_v, 214, "", "./tensorflow/compiler/mlir/tfrt/benchmarks/benchmark.cc", "FreeReturnedMemref");

  DCHECK(llvm::isa<MemrefType>(runtime_type)) << "expected memref result";
  // Cast result to the arbitrary chosen memref type and rank because we only
  // need to know the base pointer value.
  auto* memref = static_cast<StridedMemRefType<float, 0>*>(result_ptr);
  if (llvm::find(ctx.input_ptrs, memref->data) == ctx.input_ptrs.end()) {
    free(memref->basePtr);
  }
  return mlir::success();
}

JitExecutable& CreateJitExecutable(
    const HostContext& host, llvm::StringRef mlir_input,
    llvm::StringRef function_name, bool lower_from_tensorflow,
    const TfJitRtPipelineOptions& tf_jitrt_opts) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSbenchmarksPSbenchmarkDTcc mht_1(mht_1_v, 231, "", "./tensorflow/compiler/mlir/tfrt/benchmarks/benchmark.cc", "CreateJitExecutable");

  // Options for the default JitRt compilation pipeline (lowering to LLVM).
  CompilationPipelineOptions copts;
  copts.alignment = EIGEN_MAX_ALIGN_BYTES;
  copts.num_worker_threads = host.GetNumWorkerThreads();

  CompilationOptions opts;
  opts.register_dialects = [](mlir::DialectRegistry& registry) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSbenchmarksPSbenchmarkDTcc mht_2(mht_2_v, 241, "", "./tensorflow/compiler/mlir/tfrt/benchmarks/benchmark.cc", "lambda");

    mlir::RegisterAllTensorFlowDialects(registry);
    tfrt::jitrt::RegisterDefaultJitRtDialects(registry);
  };
  opts.create_compilation_pipeline =
      [&, copts, lower_from_tensorflow](mlir::PassManager& pm) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSbenchmarksPSbenchmarkDTcc mht_3(mht_3_v, 249, "", "./tensorflow/compiler/mlir/tfrt/benchmarks/benchmark.cc", "lambda");

        if (lower_from_tensorflow)
          tensorflow::CreateTfJitRtPipeline(pm, tf_jitrt_opts);
        tfrt::jitrt::CreateDefaultJitRtCompilationPipeline(pm, copts);
      };
  opts.create_specialization_pipeline = CreateJitRtSpecializationPipeline;
  opts.calling_convention = CompilationOptions::DefaultCallingConvention(
      mlir::bufferization::BufferizeTypeConverter());

  // Cache all jit executables, otherwise different benchmark runs will produce
  // different .so files and the same compiled function will have different
  // records in the perf profile.
  static auto* cache = new llvm::StringMap<std::unique_ptr<JitExecutable>>();

  std::string key =
      llvm::formatv("{0}/{1}/{2}", mlir_input.data(), copts.num_worker_threads,
                    hash_value(tf_jitrt_opts));

  // Compile and cache MLIR function.
  auto it = cache->find(key);
  if (it == cache->end()) {
    llvm::Expected<JitExecutable> jit_executable =
        JitExecutable::Instantiate(mlir_input, function_name, opts);
    if (auto err = jit_executable.takeError())
      LOG(FATAL) << "Failed to instantiate JitExecutable from the function: "
                 << function_name.str() << "; error: " << tfrt::StrCat(err);

    auto storage = std::make_unique<JitExecutable>(std::move(*jit_executable));
    it = cache->insert_or_assign(key, std::move(storage)).first;
  }

  return *(it->getValue());
}

MemrefDesc TensorToMemrefDesc(const Tensor& tensor) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSbenchmarksPSbenchmarkDTcc mht_4(mht_4_v, 286, "", "./tensorflow/compiler/mlir/tfrt/benchmarks/benchmark.cc", "TensorToMemrefDesc");

  llvm::SmallVector<ssize_t> dims(tensor.shape().dims());
  for (int d = 0; d < tensor.shape().dims(); ++d)
    dims[d] = tensor.shape().dim_size(d);

  tfrt::DType dtype;
  if (tensor.dtype() == DT_FLOAT)
    dtype = tfrt::GetDType<float>();
  else if (tensor.dtype() == DT_INT64)
    dtype = tfrt::GetDType<int64_t>();
  else
    LOG(FATAL) << "Unsupported tensor dtype: " << tensor.dtype();

  tfrt::TensorShape shape(dims);
  MemrefDesc desc;
  desc.dtype = dtype;
  desc.data = tensor.data();
  desc.offset = 0;
  shape.GetDimensions(&desc.sizes);
  shape.GetStrides(&desc.strides);
  return desc;
}

std::string PrintTensorType(llvm::ArrayRef<int64_t> shape,
                            llvm::StringRef element_type) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSbenchmarksPSbenchmarkDTcc mht_5(mht_5_v, 313, "", "./tensorflow/compiler/mlir/tfrt/benchmarks/benchmark.cc", "PrintTensorType");

  std::string result{"tensor<"};
  llvm::raw_string_ostream ss(result);
  for (int64_t dim : shape) {
    if (mlir::ShapedType::isDynamic(dim)) {
      ss << '?';
    } else {
      ss << dim;
    }
    ss << 'x';
  }
  ss << element_type << '>';
  return result;
}

std::string PrintDenseArray(llvm::ArrayRef<int32_t> array) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSbenchmarksPSbenchmarkDTcc mht_6(mht_6_v, 331, "", "./tensorflow/compiler/mlir/tfrt/benchmarks/benchmark.cc", "PrintDenseArray");

  std::string result{"dense<["};
  llvm::raw_string_ostream ss(result);
  for (auto elem : llvm::enumerate(array)) {
    if (elem.index() > 0) ss << ',';
    ss << elem.value();
  }
  ss << "]>";
  return result;
}

}  // namespace tensorflow
