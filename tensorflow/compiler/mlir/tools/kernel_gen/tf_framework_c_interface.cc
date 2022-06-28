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
class MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStf_framework_c_interfaceDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStf_framework_c_interfaceDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStf_framework_c_interfaceDTcc() {
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

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tools/kernel_gen/tf_framework_c_interface.h"

#include <cstddef>
#include <string>
#include <utility>

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/TargetSelect.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"  // from @llvm-project
#include "mlir/ExecutionEngine/OptUtils.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tools/kernel_gen/compile_cache_item.pb.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/ir/tf_framework_ops.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/kernel_creator.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/tf_jit_cache.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/stream_executor/stream.h"

#if defined(GOOGLE_CUDA) || defined(TENSORFLOW_USE_ROCM)
#include "tensorflow/compiler/mlir/tools/kernel_gen/tf_gpu_runtime_wrappers.h"
#endif

static constexpr absl::string_view kTFJitCacheDirEnvVar = "TF_JIT_CACHE_DIR";

namespace mlir {
namespace kernel_gen {
namespace tf_framework {
namespace {

using tensorflow::Allocator;
using tensorflow::AllocatorAttributes;

Allocator* GetAllocator(void* op_kernel_ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStf_framework_c_interfaceDTcc mht_0(mht_0_v, 221, "", "./tensorflow/compiler/mlir/tools/kernel_gen/tf_framework_c_interface.cc", "GetAllocator");

  auto* ctx = static_cast<tensorflow::OpKernelContext*>(op_kernel_ctx);
  // TODO(pifon): Figure out how to set AllocatorAttributes correctly.
  AllocatorAttributes attrs;
  return ctx->get_allocator(attrs);
}

}  // namespace

extern "C" void* _mlir_ciface_tf_alloc(void* op_kernel_ctx, size_t num_elements,
                                       size_t element_size,
                                       int32_t output_index,
                                       int32_t num_candidates,
                                       int32_t* candidate_input_indices) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStf_framework_c_interfaceDTcc mht_1(mht_1_v, 237, "", "./tensorflow/compiler/mlir/tools/kernel_gen/tf_framework_c_interface.cc", "_mlir_ciface_tf_alloc");

  static constexpr int kAmbiguousOutputIndex = -1;
  auto* ctx = static_cast<tensorflow::OpKernelContext*>(op_kernel_ctx);
  if (output_index != kAmbiguousOutputIndex) {
    // Create a 1D shape, because the shapes don't have to match exactly for
    // input forwarding. Only the number of elements must be the same.
    tensorflow::TensorShape output_shape;
    output_shape.AddDim(num_elements);

    // Iterate over indices of all inputs that can potentially be used for
    // forwarding.
    for (int i = 0; i < num_candidates; ++i) {
      auto tensor = ctx->forward_input(candidate_input_indices[i], output_index,
                                       ctx->expected_output_dtype(output_index),
                                       output_shape,
                                       ctx->output_memory_type(output_index),
                                       ctx->output_alloc_attr(output_index));
      if (tensor != nullptr) {
        return tensor->data();
      }
    }

    CHECK(!ctx->output_expects_forwarding(output_index));
  }

  // If no forwarding happened, allocate a chunk of memory.
  return GetAllocator(op_kernel_ctx)
      ->AllocateRaw(Allocator::kAllocatorAlignment,
                    num_elements * element_size);
}

extern "C" void _mlir_ciface_tf_dealloc(void* op_kernel_ctx, void* ptr) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStf_framework_c_interfaceDTcc mht_2(mht_2_v, 271, "", "./tensorflow/compiler/mlir/tools/kernel_gen/tf_framework_c_interface.cc", "_mlir_ciface_tf_dealloc");

  GetAllocator(op_kernel_ctx)->DeallocateRaw(ptr);
}

extern "C" void _mlir_ciface_tf_report_error(void* op_kernel_ctx,
                                             int32_t error_code, char* msg) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("msg: \"" + (msg == nullptr ? std::string("nullptr") : std::string((char*)msg)) + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStf_framework_c_interfaceDTcc mht_3(mht_3_v, 280, "", "./tensorflow/compiler/mlir/tools/kernel_gen/tf_framework_c_interface.cc", "_mlir_ciface_tf_report_error");

  Optional<ErrorCode> symbol = symbolizeErrorCode(error_code);
  if (!symbol.hasValue()) {
    LOG(ERROR) << "No valid conversion from integer value = " << error_code
               << "to ErrorCode attribute";
    return;
  }
  auto* ctx = static_cast<tensorflow::OpKernelContext*>(op_kernel_ctx);
  ctx->CtxFailureWithWarning(
      tensorflow::Status{ConvertAttrToEnumValue(symbol.getValue()), msg});
}

static void ReportError(void* op_kernel_ctx, ErrorCode error_code,
                        const char* msg) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("msg: \"" + (msg == nullptr ? std::string("nullptr") : std::string((char*)msg)) + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStf_framework_c_interfaceDTcc mht_4(mht_4_v, 297, "", "./tensorflow/compiler/mlir/tools/kernel_gen/tf_framework_c_interface.cc", "ReportError");

  _mlir_ciface_tf_report_error(op_kernel_ctx, static_cast<uint32_t>(error_code),
                               const_cast<char*>(msg));
}

namespace {

std::string GetFileCachePath(const std::string cache_dir,
                             const std::string& code) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("cache_dir: \"" + cache_dir + "\"");
   mht_5_v.push_back("code: \"" + code + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStf_framework_c_interfaceDTcc mht_5(mht_5_v, 310, "", "./tensorflow/compiler/mlir/tools/kernel_gen/tf_framework_c_interface.cc", "GetFileCachePath");

  size_t hash = llvm::hash_value(code);
  return tensorflow::io::JoinPath(cache_dir, std::to_string(hash));
}

// A callback to register all externally defined symbols needed by the kernel.
llvm::orc::SymbolMap TFFrameworkSymbolMap(llvm::orc::MangleAndInterner mangle) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStf_framework_c_interfaceDTcc mht_6(mht_6_v, 319, "", "./tensorflow/compiler/mlir/tools/kernel_gen/tf_framework_c_interface.cc", "TFFrameworkSymbolMap");

  llvm::orc::SymbolMap symbol_map;
  auto bind = [&](llvm::StringRef name, auto symbol_ptr) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStf_framework_c_interfaceDTcc mht_7(mht_7_v, 324, "", "./tensorflow/compiler/mlir/tools/kernel_gen/tf_framework_c_interface.cc", "lambda");

    symbol_map[mangle(name)] = llvm::JITEvaluatedSymbol(
        llvm::pointerToJITTargetAddress(symbol_ptr), llvm::JITSymbolFlags());
  };

  // Register TF framework symbols.
  bind("_mlir_ciface_tf_alloc", &_mlir_ciface_tf_alloc);
  bind("_mlir_ciface_tf_dealloc", &_mlir_ciface_tf_dealloc);
  bind("_mlir_ciface_tf_report_error", &_mlir_ciface_tf_report_error);
#if defined(GOOGLE_CUDA) || defined(TENSORFLOW_USE_ROCM)
  bind("_mlir_ciface_tf_launch_kernel", &_mlir_ciface_tf_launch_kernel);
#endif

  // Register malloc/free to avoid unexpected implementations from shared libs.
  bind("malloc", &malloc);
  bind("free", &free);

  return symbol_map;
}

llvm::Expected<std::unique_ptr<ExecutionEngine>> Compile(
    const std::string code, llvm::SmallVectorImpl<std::string>& architectures,
    llvm::SmallVectorImpl<int64_t>& tile_sizes,
    llvm::SmallVectorImpl<int64_t>& unroll_factors, int64_t max_supported_rank,
    bool enable_ftz, bool index_64bit, bool cpu_codegen) {
  std::string cache_dir;
  if (const char* dir = getenv(kTFJitCacheDirEnvVar.data())) {
    cache_dir = dir;
  }

  // Check if we already have a partially compiled module in the filesystem
  // based cache.
  CompilationCacheItem item;
  auto tenv = tensorflow::Env::Default();
  if (!cache_dir.empty() && tenv->RecursivelyCreateDir(cache_dir).ok()) {
    std::string data;
    if (tensorflow::ReadFileToString(tenv, GetFileCachePath(cache_dir, code),
                                     &data)
            .ok()) {
      item.ParseFromString(data);
      if (item.original_module() != code) {
        item.Clear();
      }
    }
  }

  // Create the kernel.
  mlir::OwningOpRef<mlir::ModuleOp> module;
  mlir::MLIRContext context;

  if (item.result_module().empty()) {
    // Otherwise, compile the module now.
    tensorflow::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> status_or_module =
        tensorflow::kernel_gen::GenerateKernelForTfCode(
            context, code, architectures, tile_sizes, unroll_factors,
            max_supported_rank, /*embed_memref_prints=*/false,
            /*print_ptx=*/false, /*print_llvmir=*/false, enable_ftz,
            index_64bit, cpu_codegen,
            /*jit_compile=*/false,
            /*jit_i64_indexed_for_large_tensors=*/false,
            /*apply_cl_options=*/false);
    if (!status_or_module.ok()) return nullptr;
    module = std::move(status_or_module.ValueOrDie());

    if (!cache_dir.empty() && tenv->RecursivelyCreateDir(cache_dir).ok()) {
      // Save the compilation result here for future processes to use.
      item.set_original_module(code);
      llvm::raw_string_ostream stream(*item.mutable_result_module());
      module.get().print(stream);
      stream.flush();

      tensorflow::WriteStringToFile(tenv, GetFileCachePath(cache_dir, code),
                                    item.SerializeAsString())
          .IgnoreError();
    }
  } else {
    module = tensorflow::kernel_gen::SetupContextAndParseModule(
                 context, item.result_module())
                 .ValueOrDie();
  }

  // Initialize LLVM targets.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  // Create execution engine with an inner optimization pipeline.
  auto opt_pipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/2, /*sizeLevel=*/0, /*targetMachine=*/nullptr);
  mlir::ExecutionEngineOptions engine_options;
  engine_options.transformer = opt_pipeline;
  llvm::Expected<std::unique_ptr<ExecutionEngine>> engine =
      mlir::ExecutionEngine::create(module.get(), engine_options);
  if (!engine) return nullptr;

  // Finally, register the missing symbols.
  engine.get()->registerSymbols(TFFrameworkSymbolMap);
  return engine;
}

template <typename T, typename U = T>
llvm::SmallVector<T, 8> SmallVectorFromCArray(int64_t num_elements,
                                              U* elements_ptr) {
  llvm::SmallVector<T, 8> result;
  result.reserve(num_elements);
  for (int i = 0; i < num_elements; ++i) result.push_back(elements_ptr[i]);
  return result;
}

}  // namespace

extern "C" void* _mlir_ciface_tf_jit_compile(
    void* op_kernel_ctx, char* code, int64_t num_tile_sizes,
    int64_t* tile_sizes_ptr, int64_t num_unroll_factors,
    int64_t* unroll_factors_ptr, int64_t max_supported_rank, bool enable_ftz,
    bool index_64bit, bool cpu_codegen) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("code: \"" + (code == nullptr ? std::string("nullptr") : std::string((char*)code)) + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStf_framework_c_interfaceDTcc mht_8(mht_8_v, 442, "", "./tensorflow/compiler/mlir/tools/kernel_gen/tf_framework_c_interface.cc", "_mlir_ciface_tf_jit_compile");

  // Get the resource manager.
  auto* ctx = static_cast<tensorflow::OpKernelContext*>(op_kernel_ctx);
  tensorflow::ResourceMgr* rm = ctx->resource_manager();
  if (!rm) {
    ReportError(op_kernel_ctx, ErrorCode::UNKNOWN, "No resource manager.");
    return nullptr;
  }

  // Get the JIT cache.
  JITCache* jit_cache = nullptr;
  auto status = rm->LookupOrCreate<JITCache>(rm->default_container(),
                                             JITCache::kDefaultResourceName,
                                             &jit_cache, JITCache::Create);
  tensorflow::core::ScopedUnref jit_cache_ref(jit_cache);
  if (!status.ok()) {
    ReportError(op_kernel_ctx, ErrorCode::UNKNOWN,
                "Failed to find or create JIT cache.");
    return nullptr;
  }

  // Determine the unique architecture for the current GPU, if any.
  SmallVector<std::string, 1> architectures;
#if defined(GOOGLE_CUDA)
  stream_executor::CudaComputeCapability cc =
      ctx->op_device_context()->stream()->GetCudaComputeCapability();
  architectures.push_back(absl::StrCat("sm_", cc.major, cc.minor));
#elif defined(TENSORFLOW_USE_ROCM)
  stream_executor::RocmComputeCapability cc =
      ctx->op_device_context()->stream()->GetRocmComputeCapability();
  architectures.push_back(cc.gcn_arch_name());
#endif

  // Construct `SmallVector`s from arguments.
  llvm::SmallVector<int64_t, 8> tile_sizes =
      SmallVectorFromCArray<int64_t>(num_tile_sizes, tile_sizes_ptr);
  llvm::SmallVector<int64_t, 8> unroll_factors =
      SmallVectorFromCArray<int64_t>(num_unroll_factors, unroll_factors_ptr);

  // Lookup or compile the execution module.
  ExecutionEngine* engine = jit_cache->LookupOrCompile(code, [&]() {
    return Compile(code, architectures, tile_sizes, unroll_factors,
                   max_supported_rank, enable_ftz, index_64bit, cpu_codegen);
  });
  if (engine == nullptr) {
    ReportError(op_kernel_ctx, ErrorCode::UNKNOWN, "JIT compilation failed.");
    return nullptr;
  }
  return engine;
}

extern "C" void _mlir_ciface_tf_jit_execute(void* op_kernel_ctx, void* callable,
                                            void* result, int64_t num_args,
                                            void* args_ptr) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStf_framework_c_interfaceDTcc mht_9(mht_9_v, 498, "", "./tensorflow/compiler/mlir/tools/kernel_gen/tf_framework_c_interface.cc", "_mlir_ciface_tf_jit_execute");

  // JIT compilation must have failed earlier if there is no callable ptr.
  // Return some empty memory descriptor to prevent a crash.
  if (callable == nullptr) {
    auto* desc = static_cast<::UnrankedMemRefType<void>*>(result);
    desc->rank = 0;
    auto* inner_desc = static_cast<StridedMemRefType<int8_t, 0>*>(
        malloc(sizeof(StridedMemRefType<int8_t, 0>)));
    inner_desc->basePtr = nullptr;
    inner_desc->data = nullptr;
    inner_desc->offset = 0;
    desc->descriptor = inner_desc;
    return;
  }

  // Build the argument array according to `ExecutionEngine`'s calling
  // convention.
  auto* typed_args_ptr = static_cast<::UnrankedMemRefType<void>*>(args_ptr);
  llvm::SmallVector<void*, 8> args_array = {&op_kernel_ctx};
  for (int i = 0; i < num_args; i++) {
    auto& desc = typed_args_ptr[i];
    args_array.push_back(&desc.rank);
    args_array.push_back(&desc.descriptor);
  }
  args_array.push_back(result);

  llvm::Error invocation_result =
      static_cast<ExecutionEngine*>(callable)->invokePacked("main", args_array);
  if (invocation_result)
    ReportError(op_kernel_ctx, ErrorCode::UNKNOWN, "JIT invocation failed.");
}

}  // namespace tf_framework
}  // namespace kernel_gen
}  // namespace mlir
