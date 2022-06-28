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
class MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSruntime_fallbackPSruntime_fallback_executorDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSruntime_fallbackPSruntime_fallback_executorDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSruntime_fallbackPSruntime_fallback_executorDTcc() {
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

/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tfrt/runtime_fallback/runtime_fallback_executor.h"

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/passes.h"
#include "tensorflow/compiler/mlir/tfrt/utils/host_context.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/threadpool.h"
#include "tensorflow/core/platform/threadpool_interface.h"
#include "tensorflow/core/runtime_fallback/kernel/kernel_fallback_execute_compat.h"
#include "tensorflow/core/runtime_fallback/runtime/kernel_utils.h"
#include "tensorflow/core/tfrt/utils/fallback_tensor.h"
#include "tfrt/bef/bef_buffer.h"  // from @tf_runtime
#include "tfrt/bef_converter/mlir_to_bef.h"  // from @tf_runtime
#include "tfrt/bef_executor/bef_file.h"  // from @tf_runtime
#include "tfrt/host_context/async_value.h"  // from @tf_runtime
#include "tfrt/host_context/execution_context.h"  // from @tf_runtime
#include "tfrt/host_context/function.h"  // from @tf_runtime
#include "tfrt/host_context/host_context.h"  // from @tf_runtime
#include "tfrt/host_context/resource_context.h"  // from @tf_runtime
#include "tfrt/support/ref_count.h"  // from @tf_runtime

namespace tensorflow {

using ::tfrt::AsyncValue;
using ::tfrt::BEFFile;
using ::tfrt::ExecutionContext;
using ::tfrt::Function;
using ::tfrt::HostContext;
using ::tfrt::MakeAvailableAsyncValueRef;
using ::tfrt::RCReference;
using ::tfrt::RequestContext;
using ::tfrt::RequestContextBuilder;
using ::tfrt::ResourceContext;

using ::tensorflow::Env;
using ::tensorflow::thread::ThreadPool;
using ::tensorflow::thread::ThreadPoolInterface;

using ::tensorflow::tfrt_stub::FallbackTensor;

// -------------------------------------------------------------------------- //
// Run function via the TF->TFRT fallback lowering.
// -------------------------------------------------------------------------- //

namespace {
// Thread pool for running `intra-op` tasks scheduled by the fallback kernels.
class IntraOpThreadPool : public ThreadPoolInterface {
 public:
  explicit IntraOpThreadPool(int64_t num_threads)
      : tpool_(Env::Default(), "intra-op",
               std::max(1, static_cast<int32_t>(num_threads))) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSruntime_fallbackPSruntime_fallback_executorDTcc mht_0(mht_0_v, 247, "", "./tensorflow/compiler/mlir/tfrt/runtime_fallback/runtime_fallback_executor.cc", "IntraOpThreadPool");
}

  void Schedule(std::function<void()> fn) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSruntime_fallbackPSruntime_fallback_executorDTcc mht_1(mht_1_v, 252, "", "./tensorflow/compiler/mlir/tfrt/runtime_fallback/runtime_fallback_executor.cc", "Schedule");

    tpool_.Schedule(std::move(fn));
  }

  int NumThreads() const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSruntime_fallbackPSruntime_fallback_executorDTcc mht_2(mht_2_v, 259, "", "./tensorflow/compiler/mlir/tfrt/runtime_fallback/runtime_fallback_executor.cc", "NumThreads");
 return tpool_.NumThreads(); }
  int CurrentThreadId() const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSruntime_fallbackPSruntime_fallback_executorDTcc mht_3(mht_3_v, 263, "", "./tensorflow/compiler/mlir/tfrt/runtime_fallback/runtime_fallback_executor.cc", "CurrentThreadId");
 return tpool_.CurrentThreadId(); }
  void Cancel() override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSruntime_fallbackPSruntime_fallback_executorDTcc mht_4(mht_4_v, 267, "", "./tensorflow/compiler/mlir/tfrt/runtime_fallback/runtime_fallback_executor.cc", "Cancel");
}

 private:
  ThreadPool tpool_;
};
}  // namespace

RuntimeFallbackExecutor::RuntimeFallbackExecutor(int64_t num_threads)
    : intra_op_(std::make_unique<IntraOpThreadPool>(num_threads)) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSruntime_fallbackPSruntime_fallback_executorDTcc mht_5(mht_5_v, 278, "", "./tensorflow/compiler/mlir/tfrt/runtime_fallback/runtime_fallback_executor.cc", "RuntimeFallbackExecutor::RuntimeFallbackExecutor");

  // Create a HostContext for running TFRT functions. Concurrent work queue acts
  // similar to the Tensorflow `inter-op` thread pool, so we'll match the size.
  host_context_ = num_threads ? CreateMultiThreadedHostContext(num_threads)
                              : CreateSingleThreadedHostContext();
  tfrt::RegisterStaticKernels(host_context_->GetMutableRegistry());

  // Build an ExecutionContext from the HostContext.
  auto builder = RequestContextBuilder(host_context_.get(), &resource_context_);

  // Get tensorflow::EagerContext for the kernel fallback.
  auto* eager_context_resource =
      resource_context_
          .GetOrCreateResource<tensorflow::tfd::EagerContextResource>(
              tensorflow::tfd::kEagerContextResourceName);
  auto expected_eager_context = eager_context_resource->GetTFEagerContext();
  auto* eager_context = expected_eager_context.get();

  // Initialize fallback kernels state with a custom intra-op thread pool.
  auto status = tensorflow::tfd::SetUpKernelFallbackCompatRequestContext(
      &builder, /*runner_table=*/nullptr, eager_context, intra_op_.get());
  CHECK(status.ok()) << "Failed to setup request context: "
                     << status.error_message();

  auto req_ctx = std::move(builder).build();
  if (auto err = req_ctx.takeError())
    LOG(FATAL) << "Failed to build a request context";

  exec_ctx_ = std::make_unique<tfrt::ExecutionContext>(std::move(*req_ctx));
}

void RuntimeFallbackExecutor::Prepare(llvm::StringRef mlir_input) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSruntime_fallbackPSruntime_fallback_executorDTcc mht_6(mht_6_v, 312, "", "./tensorflow/compiler/mlir/tfrt/runtime_fallback/runtime_fallback_executor.cc", "RuntimeFallbackExecutor::Prepare");

  // We only support IR written in the Tensorflow dialect.
  mlir::DialectRegistry registry;
  mlir::RegisterAllTensorFlowDialects(registry);
  mlir::MLIRContext context(registry);

  llvm::SourceMgr source_mgr;
  source_mgr.AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBuffer(mlir_input, "test_ir"), llvm::SMLoc());

  // Parse a kernel source code into the MLIR Module.
  mlir::OwningOpRef<mlir::ModuleOp> module(
      mlir::parseSourceFile<mlir::ModuleOp>(source_mgr, &context));
  CHECK(module) << "failed to parse mlir module";

  // Collect all diagnostics emitted while lowering parsed kernel module.
  std::string diagnostic_str;
  llvm::raw_string_ostream os(diagnostic_str);
  mlir::SourceMgrDiagnosticHandler handler(source_mgr, module->getContext(),
                                           os);

  // Convert TF to TFRT fallback dialect.
  TfrtPipelineOptions pipeline_opts;
  pipeline_opts.default_device = kDefaultHostDeviceName;
  pipeline_opts.hoist_invariant_ops = true;
  pipeline_opts.enable_native_ops = false;
  pipeline_opts.cost_threshold = 1024;
  pipeline_opts.upper_cost_threshold = 100000;
  pipeline_opts.merge_inter_dependent_streams = true;
  pipeline_opts.func_use_fallback_tensor = true;

  mlir::PassManager pm(module->getContext());
  pm.addPass(CreateTfToTfrtConversionPass(pipeline_opts));

  CHECK(mlir::succeeded(pm.run(*module)))
      << "Failed to lower module to TFRT: " << os.str();

  // Convert module to BEF.
  bef_buffer_ =
      tfrt::ConvertMLIRToBEF(*module, /*disable_optional_sections=*/false);
  CHECK(!bef_buffer_.empty()) << "Failed to convert module to BEF";

  bef_file_ =
      BEFFile::Open(bef_buffer_, host_context_->GetKernelRegistry(),
                    host_context_->diag_handler(), host_context_->allocator());
  CHECK(bef_file_) << "Failed to open BEF";

  // Run TFRT initialization function to pre-instantiate fallback kernels.
  RunTfrtInitializer();
}

llvm::SmallVector<Tensor> RuntimeFallbackExecutor::Execute(
    llvm::StringRef function_name, llvm::ArrayRef<Tensor> arguments) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSruntime_fallbackPSruntime_fallback_executorDTcc mht_7(mht_7_v, 367, "", "./tensorflow/compiler/mlir/tfrt/runtime_fallback/runtime_fallback_executor.cc", "RuntimeFallbackExecutor::Execute");

  // Get the kernel entrypoint function.
  const Function* compute = bef_file_->GetFunction(function_name);
  CHECK(compute) << "Entrypoint function not found";
  CHECK_EQ(arguments.size() + 1, compute->num_arguments())
      << "Wrong number of arguments for function " << function_name.str();

  // Prepare function arguments from ready Chain and input Tensors.
  llvm::SmallVector<tfrt::AsyncValue*> exec_arguments;
  exec_arguments.reserve(compute->num_arguments());
  exec_arguments.push_back(tfrt::GetReadyChain().release());
  for (const Tensor& input_tensor : arguments) {
    auto av = MakeAvailableAsyncValueRef<FallbackTensor>(input_tensor);
    exec_arguments.push_back(av.release());
  }

  // Space for returned values.
  llvm::SmallVector<RCReference<AsyncValue>> results(compute->num_results());

  compute->Execute(*exec_ctx_, exec_arguments, results);

  // Wait for the function execution to finish, as well as the side-effects.
  host_context_->Await(results);

  // Check that all results are available.
  llvm::SmallVector<Tensor> ret_values;
  for (unsigned i = 1; i < results.size(); ++i) {
    if (auto* error = results[i]->GetErrorIfPresent())
      LOG(FATAL) << "Failed to execute a function: " << StrCat(*error);
    ret_values.push_back(results[i]->get<tfrt_stub::FallbackTensor>().tensor());
  }

  // Deallocate arguments.
  for (auto* argument : exec_arguments) argument->DropRef();
  return ret_values;
}

// Run TFRT fallback initialization function to instantiate all fallback
// kernels ahead of executing the compute function.
void RuntimeFallbackExecutor::RunTfrtInitializer() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSruntime_fallbackPSruntime_fallback_executorDTcc mht_8(mht_8_v, 409, "", "./tensorflow/compiler/mlir/tfrt/runtime_fallback/runtime_fallback_executor.cc", "RuntimeFallbackExecutor::RunTfrtInitializer");

  const Function* func = bef_file_->GetFunction("_tfrt_fallback_init");
  CHECK(func) << "TFRT initialization function was not found";
  CHECK_EQ(func->argument_types().size(), 1);

  llvm::SmallVector<RCReference<AsyncValue>, 1> results;
  results.resize(func->result_types().size());
  CHECK_EQ(results.size(), 1);

  func->Execute(*exec_ctx_, tfrt::GetReadyChain().GetAsyncValue(), results);

  host_context_->Await(results);

  CHECK(!results[0]->IsError()) << "Failed to run TFRT initialization function";
}

}  // namespace tensorflow
