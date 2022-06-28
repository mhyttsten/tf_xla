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
class MHTracer_DTPStensorflowPScompilerPSjitPSkernelsPSxla_opsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSjitPSkernelsPSxla_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSjitPSkernelsPSxla_opsDTcc() {
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

#include "tensorflow/compiler/jit/kernels/xla_ops.h"

#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/synchronization/notification.h"
#include "absl/types/optional.h"
#include "tensorflow/compiler/jit/defs.h"
#include "tensorflow/compiler/jit/encapsulate_subgraphs_pass.h"
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/jit/xla_activity_listener.h"
#include "tensorflow/compiler/jit/xla_cluster_util.h"
#include "tensorflow/compiler/jit/xla_platform_info.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/tf2xla_util.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/executable_run_options.h"
#include "tensorflow/compiler/xla/service/compiler.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_executable_run_options.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/monitoring/counter.h"
#include "tensorflow/core/platform/casts.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/refcount.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/util/stream_executor_util.h"

// OP_REQUIRES_OK_RETURN is the same as OP_REQUIRES_OK except that
// in error case, it returns RET instead of void.
#define OP_REQUIRES_OK_RETURN(CTX, RET, ...)                \
  do {                                                      \
    ::tensorflow::Status _s(__VA_ARGS__);                   \
    if (!TF_PREDICT_TRUE(_s.ok())) {                        \
      (CTX)->CtxFailureWithWarning(__FILE__, __LINE__, _s); \
      return RET;                                           \
    }                                                       \
  } while (0)

namespace tensorflow {

namespace {

auto* xla_launch_counter = monitoring::Counter<1>::New(
    "/tensorflow/core/xla_launch_counter",
    "The number of times a XlaLaunch is called.", "device");

// A closure describing how to run a compiled version of a TensorFlow function.
//
// It may seem unusual to stick the resource variable snapshots in this class.
// This is necessary: we need to use the snapshots observed by the compiler as
// the initial values for the resource variables (and cannot snapshot them again
// during execution) because otherwise we risk observing a different snapshot
// with shapes different from what we compiled for.
class XlaExecutableClosure {
 public:
  explicit XlaExecutableClosure(
      xla::LocalClient* client, xla::LocalExecutable* executable,
      const XlaCompiler::CompilationResult* compilation_result,
      ResourceVarsSnapshot resource_var_snapshots, int num_constant_args)
      : client_(client),
        executable_(executable),
        compilation_result_(compilation_result),
        resource_var_snapshots_(std::move(resource_var_snapshots)),
        num_constant_args_(num_constant_args) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSkernelsPSxla_opsDTcc mht_0(mht_0_v, 263, "", "./tensorflow/compiler/jit/kernels/xla_ops.cc", "XlaExecutableClosure");
}

  XlaExecutableClosure(XlaExecutableClosure&&) = default;
  XlaExecutableClosure& operator=(XlaExecutableClosure&&) = default;

  xla::LocalClient* client() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSkernelsPSxla_opsDTcc mht_1(mht_1_v, 271, "", "./tensorflow/compiler/jit/kernels/xla_ops.cc", "client");
 return client_; }
  xla::LocalExecutable* executable() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSkernelsPSxla_opsDTcc mht_2(mht_2_v, 275, "", "./tensorflow/compiler/jit/kernels/xla_ops.cc", "executable");
 return executable_; }
  const XlaCompiler::CompilationResult* compilation_result() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSkernelsPSxla_opsDTcc mht_3(mht_3_v, 279, "", "./tensorflow/compiler/jit/kernels/xla_ops.cc", "compilation_result");

    return compilation_result_;
  }
  const ResourceVarsSnapshot& resource_var_snapshots() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSkernelsPSxla_opsDTcc mht_4(mht_4_v, 285, "", "./tensorflow/compiler/jit/kernels/xla_ops.cc", "resource_var_snapshots");

    return resource_var_snapshots_;
  }
  int num_constant_args() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSkernelsPSxla_opsDTcc mht_5(mht_5_v, 291, "", "./tensorflow/compiler/jit/kernels/xla_ops.cc", "num_constant_args");
 return num_constant_args_; }

 private:
  xla::LocalClient* client_;
  xla::LocalExecutable* executable_;
  const XlaCompiler::CompilationResult* compilation_result_;
  ResourceVarsSnapshot resource_var_snapshots_;
  int num_constant_args_;

  TF_DISALLOW_COPY_AND_ASSIGN(XlaExecutableClosure);
};

// This maintains a mapping from a globally unique ID to XlaExecutableClosure
// instances.
class XlaExecutableClosureStore {
 public:
  XlaExecutableClosureStore() : key_counter_(0) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSkernelsPSxla_opsDTcc mht_6(mht_6_v, 310, "", "./tensorflow/compiler/jit/kernels/xla_ops.cc", "XlaExecutableClosureStore");
}

  using KeyT = string;

  KeyT Produce(XlaExecutableClosure result) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSkernelsPSxla_opsDTcc mht_7(mht_7_v, 317, "", "./tensorflow/compiler/jit/kernels/xla_ops.cc", "Produce");

    mutex_lock l(mutex_);
    KeyT key = absl::StrCat(key_counter_++);
    bool insert_successful = closures_.emplace(key, std::move(result)).second;
    DCHECK(insert_successful);
    (void)insert_successful;
    return key;
  }

  XlaExecutableClosure Consume(const KeyT& key) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSkernelsPSxla_opsDTcc mht_8(mht_8_v, 329, "", "./tensorflow/compiler/jit/kernels/xla_ops.cc", "Consume");

    mutex_lock l(mutex_);
    auto it = closures_.find(key);
    DCHECK(it != closures_.end());
    XlaExecutableClosure value = std::move(it->second);
    closures_.erase(it);
    return value;
  }

  static XlaExecutableClosureStore* Global() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSkernelsPSxla_opsDTcc mht_9(mht_9_v, 341, "", "./tensorflow/compiler/jit/kernels/xla_ops.cc", "Global");

    static XlaExecutableClosureStore* instance = new XlaExecutableClosureStore;
    return instance;
  }

 private:
  mutex mutex_;
  int64_t key_counter_ TF_GUARDED_BY(mutex_);
  absl::flat_hash_map<KeyT, XlaExecutableClosure> closures_
      TF_GUARDED_BY(mutex_);

  TF_DISALLOW_COPY_AND_ASSIGN(XlaExecutableClosureStore);
};

se::Stream* GetStream(OpKernelContext* ctx) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSkernelsPSxla_opsDTcc mht_10(mht_10_v, 358, "", "./tensorflow/compiler/jit/kernels/xla_ops.cc", "GetStream");

  return ctx->op_device_context() ? ctx->op_device_context()->stream()
                                  : nullptr;
}

XlaComputationLaunchContext GetLaunchContext(
    const XlaPlatformInfo& platform_info, OpKernelContext* ctx,
    xla::LocalClient* client, se::DeviceMemoryAllocator* allocator) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSkernelsPSxla_opsDTcc mht_11(mht_11_v, 368, "", "./tensorflow/compiler/jit/kernels/xla_ops.cc", "GetLaunchContext");

  se::Stream* stream = GetStream(ctx);
  int device_ordinal = stream ? stream->parent()->device_ordinal()
                              : client->default_device_ordinal();
  XlaComputationLaunchContext launch_context(
      client, allocator, device_ordinal,
      /*allocate_xla_tensors=*/platform_info.is_on_xla_device(),
      /*use_multiple_streams=*/platform_info.UseMultipleStreams());
  return launch_context;
}

StatusOr<xla::ExecutionOutput> RunExecutable(
    const XlaPlatformInfo& platform_info,
    const XlaComputationLaunchContext& launch_context,
    std::vector<xla::ExecutionInput> execution_inputs,
    xla::ExecutableRunOptions run_options, xla::LocalExecutable* executable,
    OpKernelContext* ctx, se::DeviceMemoryAllocator* allocator) {
  VLOG(2) << "Executing Xla Computation.";
  Env* env = Env::Default();
  auto start_time = env->NowMicros();

  se::Stream* stream = GetStream(ctx);
  run_options.set_stream(GetStream(ctx));
  run_options.set_allocator(allocator);
  run_options.set_intra_op_thread_pool(&ctx->eigen_cpu_device());
  run_options.set_rng_seed(GetXLARandomSeed());
  StatusOr<xla::ExecutionOutput> execution_output;
  bool run_synchronous =
      !stream || platform_info.platform_id() == se::host::kHostPlatformId;
  if (run_synchronous) {
    execution_output =
        executable->Run(std::move(execution_inputs), run_options);
  } else {
    execution_output =
        executable->RunAsync(std::move(execution_inputs), run_options);
  }

  auto elapsed = env->NowMicros() - start_time;
  VLOG(2) << "Elapsed time for Xla Executable Run: " << elapsed << "us";
  return execution_output;
}

}  // namespace

XlaLocalLaunchBase::XlaLocalLaunchBase(OpKernelConstruction* ctx,
                                       const std::vector<int>& constants,
                                       const std::vector<int>& resources,
                                       const NameAttrList& function,
                                       bool has_ref_vars)
    : OpKernel(ctx),
      constants_(constants),
      resources_(resources),
      function_(function),
      platform_info_(XlaPlatformInfoFromDevice(ctx->device())),
      has_ref_vars_(has_ref_vars) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSkernelsPSxla_opsDTcc mht_12(mht_12_v, 425, "", "./tensorflow/compiler/jit/kernels/xla_ops.cc", "XlaLocalLaunchBase::XlaLocalLaunchBase");
}

static Status CompileToLocalExecutable(
    OpKernelContext* ctx, const NameAttrList& function, bool has_ref_vars,
    const XlaPlatformInfo& platform_info,
    absl::Span<const Tensor* const> inputs,
    absl::Span<VariableInfo const> variable_infos,
    absl::Span<const int> constants,
    XlaCompilationCache::CompileMode compile_mode,
    bool may_alias_resource_update, xla::LocalClient** client,
    const XlaCompiler::CompilationResult** compilation_result,
    xla::LocalExecutable** executable) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSkernelsPSxla_opsDTcc mht_13(mht_13_v, 439, "", "./tensorflow/compiler/jit/kernels/xla_ops.cc", "CompileToLocalExecutable");

  // We store information about the JIT-compiled XLA computation
  // in the ResourceMgr.
  ResourceMgr* rm = ctx->resource_manager();
  if (!rm) {
    return errors::Internal("No resource manager.");
  }

  XlaCompilationCache* cache;
  TF_RETURN_IF_ERROR(rm->LookupOrCreate<XlaCompilationCache>(
      rm->default_container(), "xla_cache", &cache,
      [&](XlaCompilationCache** cache) {
        return BuildXlaCompilationCache(ctx->device(), ctx->function_library(),
                                        platform_info, cache);
      }));
  // Hold the reference to the JIT during evaluation. (We could probably
  // free it sooner because the ResourceMgr will retain a reference, but
  // this is more obviously correct.)
  core::ScopedUnref cache_ref(cache);

  *client = static_cast<xla::LocalClient*>(cache->client());

  XlaCompiler::Options options =
      GenerateCompilerOptions(*cache, *ctx->function_library(), ctx->device(),
                              GetStream(ctx), platform_info, has_ref_vars);

  XlaCompiler::CompileOptions compile_options;
  compile_options.is_entry_computation = true;
  // Optimization: where possible, have the computation return a naked array
  // rather than a one-element tuple.
  compile_options.always_return_tuple = false;
  compile_options.alias_resource_update = !has_ref_vars &&
                                          may_alias_resource_update;

  StatusOr<std::vector<XlaCompiler::Argument>> args =
      XlaComputationLaunchContext::BuildXlaCompilerArguments(
          constants, inputs, variable_infos,
          static_cast<Device*>(ctx->device()));
  TF_RETURN_IF_ERROR(args.status());
  return cache->Compile(options, function, *args, compile_options, compile_mode,
                        compilation_result, executable);
}

void XlaLocalLaunchBase::Compute(OpKernelContext* ctx) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSkernelsPSxla_opsDTcc mht_14(mht_14_v, 485, "", "./tensorflow/compiler/jit/kernels/xla_ops.cc", "XlaLocalLaunchBase::Compute");

  VLOG(1) << "XlaLocalLaunchOpBase::Compute "
          << Canonicalize(function_.name(), AttrSlice(&function_.attr()));
  xla_launch_counter->GetCell(platform_info_.device_type().type_string())
      ->IncrementBy(1);

  std::vector<const Tensor*> inputs = InputsFromContext(ctx);
  xla::LocalClient* client;
  const XlaCompiler::CompilationResult* compilation_result;
  xla::LocalExecutable* executable;

  std::vector<VariableInfo> variable_infos;
  {
    OP_REQUIRES_OK(
        ctx, GetVariableInfosFromInputs(ctx->resource_manager(), ctx->device(),
                                        inputs, resources_, &variable_infos));
    OP_REQUIRES_OK(ctx, LockVariables(absl::MakeSpan(variable_infos)));
    Status s = CompileToLocalExecutable(
        ctx, function_, /*has_ref_vars=*/has_ref_vars_, platform_info_, inputs,
        variable_infos, constants_, XlaCompilationCache::CompileMode::kStrict,
        /*may_alias_resource_update=*/true, &client, &compilation_result,
        &executable);
    OP_REQUIRES_OK(ctx, s);
  }

  std::map<int, const Tensor*> resource_var_ptrs;
  for (int i = 0; i < resources_.size(); i++) {
    resource_var_ptrs[resources_[i]] = variable_infos[i].var()->tensor();
  }
  std::shared_ptr<se::DeviceMemoryAllocator> allocator =
      GetAllocator(ctx->device(), GetStream(ctx), platform_info_);
  XlaComputationLaunchContext launch_context =
      GetLaunchContext(platform_info_, ctx, client, allocator.get());

  const xla::HloInputOutputAliasConfig& input_output_alias =
      executable->executable()->module().input_output_alias_config();
  StatusOr<std::vector<xla::ExecutionInput>> execution_inputs =
      launch_context.PopulateInputs(ctx, compilation_result, resource_var_ptrs,
                                    /*missing_ctx_input_prefix=*/0,
                                    input_output_alias);
  OP_REQUIRES_OK(ctx, execution_inputs.status());

  // Execute the computation.
  xla::gpu::GpuExecutableRunOptions gpu_options;
  xla::DeviceAssignment device_assignment;
  xla::ExecutableRunOptions run_options;
  if (compilation_result->collective_info.has_value()) {
    OP_REQUIRES_OK(ctx, ResolveDeviceAssignment(
                            ctx, *compilation_result->collective_info,
                            run_options, device_assignment, gpu_options));
  } else {
    VLOG(2) << "No collective info provided: skipping device assignment";
  }

  // Hardcode run id to always be zero: TF distributed strategy differentiates
  // between subsequent runs using dependency edges.
  // This is safe, as only TF dist-strat can produce distributed ops, and we can
  // rely on TF dist-strat invariants.
  xla::RunId run_id(0);
  run_options.set_run_id(run_id);

  StatusOr<xla::ExecutionOutput> execution_output = RunExecutable(
      platform_info_, launch_context, std::move(*execution_inputs), run_options,
      executable, ctx, allocator.get());
  OP_REQUIRES(ctx, execution_output.ok(), execution_output.status());

  OP_REQUIRES_OK(
      ctx, launch_context.PopulateOutputs(
               ctx, compilation_result, execution_output->ConsumeResult(),
               /*missing_ctx_input_prefix=*/0, absl::MakeSpan(variable_infos),
               input_output_alias, resource_var_ptrs));

  VLOG(1) << "Done";
}

namespace {
// Helper static functions to construct parameters for
// XlaLocalLaunchBase constructor from OpKernelConstruction.
std::vector<int> ConstantsVector(OpKernelConstruction* ctx) {
  DataTypeVector constant_types;
  OP_REQUIRES_OK_RETURN(ctx, std::vector<int>(),
                        ctx->GetAttr("Tconstants", &constant_types));
  std::vector<int> constants(constant_types.size());
  std::iota(constants.begin(), constants.end(), 0);
  return constants;
}

std::vector<int> ResourcesVector(OpKernelConstruction* ctx) {
  DataTypeVector constant_types;
  OP_REQUIRES_OK_RETURN(ctx, std::vector<int>(),
                        ctx->GetAttr("Tconstants", &constant_types));

  DataTypeVector arg_types;
  OP_REQUIRES_OK_RETURN(ctx, std::vector<int>(),
                        ctx->GetAttr("Targs", &arg_types));

  int num_resources;
  OP_REQUIRES_OK_RETURN(ctx, std::vector<int>(),
                        ctx->GetAttr("Nresources", &num_resources));

  std::vector<int> resources(num_resources);
  std::iota(resources.begin(), resources.end(),
            constant_types.size() + arg_types.size());
  return resources;
}

NameAttrList FunctionAttr(OpKernelConstruction* ctx) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSkernelsPSxla_opsDTcc mht_15(mht_15_v, 594, "", "./tensorflow/compiler/jit/kernels/xla_ops.cc", "FunctionAttr");

  const NameAttrList* func;
  OP_REQUIRES_OK_RETURN(ctx, NameAttrList(), ctx->GetAttr("function", &func));
  return *func;
}

bool MustCompileAttr(OpKernelConstruction* ctx) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSkernelsPSxla_opsDTcc mht_16(mht_16_v, 603, "", "./tensorflow/compiler/jit/kernels/xla_ops.cc", "MustCompileAttr");

  bool must_compile;
  OP_REQUIRES_OK_RETURN(ctx, false,
                        ctx->GetAttr("must_compile", &must_compile));
  return must_compile;
}

bool HasRefVars(OpKernelConstruction* ctx) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSkernelsPSxla_opsDTcc mht_17(mht_17_v, 613, "", "./tensorflow/compiler/jit/kernels/xla_ops.cc", "HasRefVars");

  bool has_ref_vars;
  OP_REQUIRES_OK_RETURN(ctx, false,
                        ctx->GetAttr(kXlaHasReferenceVarsAttr, &has_ref_vars));
  return has_ref_vars;
}

}  // namespace

XlaLocalLaunchOp::XlaLocalLaunchOp(OpKernelConstruction* ctx)
    : XlaLocalLaunchBase(ctx, ConstantsVector(ctx), ResourcesVector(ctx),
                         FunctionAttr(ctx), /*has_ref_vars=*/true) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSkernelsPSxla_opsDTcc mht_18(mht_18_v, 627, "", "./tensorflow/compiler/jit/kernels/xla_ops.cc", "XlaLocalLaunchOp::XlaLocalLaunchOp");
}

XlaLocalLaunchOp::~XlaLocalLaunchOp() {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSkernelsPSxla_opsDTcc mht_19(mht_19_v, 632, "", "./tensorflow/compiler/jit/kernels/xla_ops.cc", "XlaLocalLaunchOp::~XlaLocalLaunchOp");

  VLOG(1) << "XlaLocalLaunchOp destroyed";
}

XlaCompileOp::XlaCompileOp(OpKernelConstruction* ctx)
    : OpKernel(ctx),
      constants_(ConstantsVector(ctx)),
      resources_(ResourcesVector(ctx)),
      function_(FunctionAttr(ctx)),
      platform_info_(XlaPlatformInfoFromDevice(ctx->device())),
      must_compile_(MustCompileAttr(ctx)),
      has_ref_vars_(HasRefVars(ctx)) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSkernelsPSxla_opsDTcc mht_20(mht_20_v, 646, "", "./tensorflow/compiler/jit/kernels/xla_ops.cc", "XlaCompileOp::XlaCompileOp");
}

void XlaCompileOp::Compute(OpKernelContext* ctx) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSkernelsPSxla_opsDTcc mht_21(mht_21_v, 651, "", "./tensorflow/compiler/jit/kernels/xla_ops.cc", "XlaCompileOp::Compute");

  VLOG(3) << "XlaCompileOp " << def().name()
          << (must_compile_ ? "(must-compile)" : "");
  xla::LocalClient* client;
  const XlaCompiler::CompilationResult* kernel;
  xla::LocalExecutable* executable;
  ResourceVarsSnapshot variables;

  std::vector<const Tensor*> inputs = InputsFromContext(ctx);
  bool cannot_compile_cluster;
  {
    mutex_lock guard(cannot_compile_cluster_mu_);
    cannot_compile_cluster = cannot_compile_cluster_;
  }
  XlaCompilationCache::CompileMode compile_mode = [&] {
    if (must_compile_) {
      return XlaCompilationCache::CompileMode::kStrict;
    }
    return GetXlaOpsCommonFlags().tf_xla_async_compilation
               ? XlaCompilationCache::CompileMode::kAsync
               : XlaCompilationCache::CompileMode::kLazy;
  }();

  if (GetXlaOpsCommonFlags().tf_xla_always_defer_compilation ||
      cannot_compile_cluster) {
    executable = nullptr;
  } else {
    std::vector<VariableInfo> variable_infos;
    OP_REQUIRES_OK(
        ctx, GetVariableInfosFromInputs(ctx->resource_manager(), ctx->device(),
                                        inputs, resources_, &variable_infos));
    OP_REQUIRES_OK(ctx, LockVariables(absl::MakeSpan(variable_infos)));

    // Do not alias resource updates as locking variables in XlaCompile and
    // unlocking them in XlaRun may lead to deadlocks.
    Status status = CompileToLocalExecutable(
        ctx, function_, has_ref_vars_, platform_info_, inputs, variable_infos,
        constants_, compile_mode, /*may_alias_resource_update=*/false, &client,
        &kernel, &executable);
    OP_REQUIRES_OK(ctx, SnapshotResourceVariables(ctx, resources_,
                                                  variable_infos, &variables));
    if (compile_mode != XlaCompilationCache::CompileMode::kLazy ||
        status.code() != error::UNIMPLEMENTED) {
      OP_REQUIRES_OK(ctx, status);
    }

    if (status.code() == error::UNIMPLEMENTED) {
      LOG(WARNING) << "Compilation failed:" << status.ToString()
                   << ".  Falling back to TF function call.";

      BroadcastOptimizationRemark(
          XlaOptimizationRemark::UNIMPLEMENTED_OPERATION, status.ToString())
          .IgnoreError();
      executable = nullptr;
      mutex_lock guard(cannot_compile_cluster_mu_);
      cannot_compile_cluster_ = true;
    }
  }

  AllocatorAttributes host_alloc_attrs;
  host_alloc_attrs.set_gpu_compatible(true);
  host_alloc_attrs.set_on_host(true);
  Allocator* cpu_allocator = ctx->device()->GetAllocator(host_alloc_attrs);

  // Async compilation returns nullptr executable without an error.
  if (!executable) {
    DCHECK(!must_compile_);
    Tensor compilation_key(cpu_allocator, DT_STRING, TensorShape({}));

    Tensor compilation_successful(cpu_allocator, DT_BOOL, TensorShape({}));
    compilation_successful.scalar<bool>()() = false;
    ctx->set_output(0, Tensor(cpu_allocator, DT_STRING, TensorShape({})));
    ctx->set_output(1, compilation_successful);
    return;
  }

  // Each execution of an XlaCompile op creates a new XlaExecutableClosure, even
  // if it didn't have to compile the cluster because of a compilation-cache
  // hit.  This is because we at least need new snapshots of the resource
  // variables.
  XlaExecutableClosureStore::KeyT key =
      XlaExecutableClosureStore::Global()->Produce(XlaExecutableClosure(
          client, executable, kernel, std::move(variables), constants_.size()));

  Tensor compilation_key(cpu_allocator, DT_STRING, TensorShape({}));
  compilation_key.flat<tstring>()(0) = key;

  Tensor compilation_successful(cpu_allocator, DT_BOOL, TensorShape({}));
  compilation_successful.flat<bool>()(0) = true;

  ctx->set_output(0, compilation_key);
  ctx->set_output(1, compilation_successful);
}

XlaRunOp::XlaRunOp(OpKernelConstruction* ctx)
    : OpKernel(ctx), platform_info_(XlaPlatformInfoFromDevice(ctx->device())) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSkernelsPSxla_opsDTcc mht_22(mht_22_v, 749, "", "./tensorflow/compiler/jit/kernels/xla_ops.cc", "XlaRunOp::XlaRunOp");
}

void XlaRunOp::Compute(OpKernelContext* ctx) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSkernelsPSxla_opsDTcc mht_23(mht_23_v, 754, "", "./tensorflow/compiler/jit/kernels/xla_ops.cc", "XlaRunOp::Compute");

  VLOG(3) << "XlaRunOp " << def().name();
  Tensor key_tensor = ctx->input(ctx->num_inputs() - 1);
  const XlaExecutableClosureStore::KeyT& key = key_tensor.flat<tstring>()(0);

  XlaExecutableClosure closure =
      XlaExecutableClosureStore::Global()->Consume(key);
  std::shared_ptr<se::DeviceMemoryAllocator> allocator =
      GetAllocator(ctx->device(), GetStream(ctx), platform_info_);
  XlaComputationLaunchContext launch_context =
      GetLaunchContext(platform_info_, ctx, closure.client(), allocator.get());

  // We're missing the must-be-constant inputs, tell `PopulateInputs`
  // about this.  We don't actually need these inputs because they've
  // already been baked into the compiled kernel.
  const xla::HloInputOutputAliasConfig& input_output_alias =
      closure.executable()->executable()->module().input_output_alias_config();
  StatusOr<std::vector<xla::ExecutionInput>> execution_inputs;
  std::map<int, const Tensor*> snapshot_ptrs;
  {
    tensorflow::profiler::TraceMe hlo_module_activity(
        [&] {
          return absl::StrCat(
              "Populate Inputs (",
              closure.compilation_result()->xla_input_shapes.size(), ")");
        },
        tensorflow::profiler::TraceMeLevel::kInfo);

    for (auto& p : closure.resource_var_snapshots()) {
      snapshot_ptrs.emplace(p.first,
                            p.second.has_value() ? &p.second.value() : nullptr);
    }
    execution_inputs = launch_context.PopulateInputs(
        ctx, closure.compilation_result(), snapshot_ptrs,
        /*missing_ctx_input_prefix=*/closure.num_constant_args(),
        input_output_alias);
    OP_REQUIRES_OK(ctx, execution_inputs.status());
  }

  xla::ExecutableRunOptions run_options;
  StatusOr<xla::ExecutionOutput> execution_output = RunExecutable(
      platform_info_, launch_context, std::move(*execution_inputs), run_options,
      closure.executable(), ctx, allocator.get());
  OP_REQUIRES(ctx, execution_output.ok(), execution_output.status());

  tensorflow::profiler::TraceMe hlo_module_activity(
      [&] {
        return absl::StrCat("Populate Outputs (", ctx->num_outputs(), ")");
      },
      tensorflow::profiler::TraceMeLevel::kInfo);

  StatusOr<std::vector<VariableInfo>> variable_infos = GatherVariableInfo(
      ctx, *closure.compilation_result(), closure.num_constant_args());
  OP_REQUIRES_OK(ctx, variable_infos.status());
  OP_REQUIRES_OK(ctx, LockVariables(absl::MakeSpan(*variable_infos)));
  OP_REQUIRES_OK(
      ctx,
      launch_context.PopulateOutputs(
          ctx, closure.compilation_result(), execution_output->ConsumeResult(),
          /*missing_ctx_input_prefix=*/closure.num_constant_args(),
          absl::MakeSpan(*variable_infos), input_output_alias, snapshot_ptrs));
}

XlaMergeOp::XlaMergeOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSkernelsPSxla_opsDTcc mht_24(mht_24_v, 820, "", "./tensorflow/compiler/jit/kernels/xla_ops.cc", "XlaMergeOp::XlaMergeOp");
}

void XlaMergeOp::Compute(OpKernelContext* ctx) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSkernelsPSxla_opsDTcc mht_25(mht_25_v, 825, "", "./tensorflow/compiler/jit/kernels/xla_ops.cc", "XlaMergeOp::Compute");

  VLOG(3) << "XlaMergeOp " << def().name();
  int i = 0;
  if (ctx->has_input(i) || ctx->has_input(++i)) {
    ctx->set_output(0, ctx->input(i));
  }
}

REGISTER_KERNEL_BUILDER(Name("XlaLaunch").Device(DEVICE_CPU), XlaLocalLaunchOp);

REGISTER_KERNEL_BUILDER(Name("XlaLaunch")
                            .Device(DEVICE_GPU)
                            .HostMemory("constants")
                            .HostMemory("resources"),
                        XlaLocalLaunchOp);

REGISTER_KERNEL_BUILDER(Name("_XlaCompile").Device(DEVICE_CPU), XlaCompileOp);
REGISTER_KERNEL_BUILDER(Name("_XlaCompile")
                            .Device(DEVICE_GPU)
                            .HostMemory("constants")
                            .HostMemory("key")
                            .HostMemory("compilation_successful")
                            .HostMemory("resources"),
                        XlaCompileOp);

REGISTER_KERNEL_BUILDER(Name("_XlaRun").Device(DEVICE_CPU), XlaRunOp);
REGISTER_KERNEL_BUILDER(Name("_XlaRun").Device(DEVICE_GPU).HostMemory("key"),
                        XlaRunOp);

REGISTER_KERNEL_BUILDER(Name("_XlaMerge").Device(DEVICE_CPU), XlaMergeOp);
REGISTER_KERNEL_BUILDER(Name("_XlaMerge").Device(DEVICE_GPU), XlaMergeOp);

}  // namespace tensorflow
