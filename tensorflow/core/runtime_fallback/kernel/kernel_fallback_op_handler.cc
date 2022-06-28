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
class MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPSkernel_fallback_op_handlerDTcc {
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
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPSkernel_fallback_op_handlerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPSkernel_fallback_op_handlerDTcc() {
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
#include "tensorflow/core/runtime_fallback/kernel/kernel_fallback_op_handler.h"

#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/runtime_fallback/kernel/kernel_fallback_compat_request_state.h"
#include "tensorflow/core/runtime_fallback/kernel/kernel_fallback_execute_compat.h"
#include "tensorflow/core/runtime_fallback/kernel/kernel_fallback_tensor.h"
#include "tensorflow/core/runtime_fallback/runtime/kernel_utils.h"
#include "tensorflow/core/runtime_fallback/util/attr_util.h"
#include "tensorflow/core/tfrt/fallback/op_kernel_runner.h"
#include "tensorflow/core/tfrt/utils/error_util.h"
#include "tfrt/core_runtime/dispatch_utils.h"  // from @tf_runtime
#include "tfrt/core_runtime/op_invocation.h"  // from @tf_runtime
#include "tfrt/core_runtime/op_metadata_function.h"  // from @tf_runtime
#include "tfrt/core_runtime/tensor_handle.h"  // from @tf_runtime
#include "tfrt/host_context/async_value_ref.h"  // from @tf_runtime
#include "tfrt/host_context/execution_context.h"  // from @tf_runtime
#include "tfrt/support/error_util.h"  // from @tf_runtime
#include "tfrt/support/string_util.h"  // from @tf_runtime
#include "tfrt/tensor/string_host_tensor.h"  // from @tf_runtime

namespace tensorflow {
namespace tfd {

class KernelFallbackOpHandler : public tfrt::OpHandler {
 public:
  ~KernelFallbackOpHandler() override;

  llvm::Expected<tfrt::CoreRuntimeOp> MakeOp(
      tfrt::string_view op_name) override;

  // TODO(b/166199701) obtain result device from the result tensor, similar to
  // what runtime fallback op handler does.
  tfrt::RCReference<tfrt::Device> GetDeviceRef() { return device_; }

  tfrt::Device* device() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPSkernel_fallback_op_handlerDTcc mht_0(mht_0_v, 221, "", "./tensorflow/core/runtime_fallback/kernel/kernel_fallback_op_handler.cc", "device");
 return device_.get(); }

 private:
  explicit KernelFallbackOpHandler(tfrt::CoreRuntime* runtime,
                                   tfrt::RCReference<tfrt::Device> device);
  friend llvm::Expected<tfrt::OpHandler*> CreateKernelFallbackOpHandler(
      tfrt::CoreRuntime* runtime, tfrt::RCReference<tfrt::Device> device);

  llvm::Error Initialize();
  tfrt::RCReference<tfrt::Device> device_;
};

namespace {

using ::tensorflow::tfrt_stub::OpKernelRunner;
using tfrt::AsyncValue;
using tfrt::AsyncValueRef;
using tfrt::Chain;
using tfrt::CoreRuntime;
using tfrt::CoreRuntimeOp;
using tfrt::ExecutionContext;
using tfrt::Expected;
using tfrt::OpAttrsRef;
using tfrt::OpHandler;
using tfrt::OpInvocation;
using tfrt::OpMetadataFn;
using tfrt::raw_ostream;
using tfrt::RCReference;
using tfrt::string_view;
using tfrt::TensorMetadata;

using CompatDispatchFn = AsyncValueRef<Chain> (*)(
    const ExecutionContext& exec_ctx, tfrt::string_view op_name,
    tfrt::string_view device_name, tfrt::ArrayRef<tfrt::Tensor*> arguments,
    tfrt::MutableArrayRef<RCReference<AsyncValue>> results,
    const KernelFallbackCompatRequestState& fallback_request_state,
    const OpKernelRunner& op_kernel_runner);

struct CompatOpEntry {
  // TODO(tfrt-devs): Avoid having string here, which can be expensive to copy.
  std::string op_name;
  OpMetadataFn metadata_fn = nullptr;
  // All ops use the same dispatch function.
  CompatDispatchFn dispatch_fn =
      &KernelFallbackExecuteCompatCoreRuntimeDispatch;
  KernelFallbackCompatRequestState* fallback_request_state = nullptr;
  OpKernelRunner* op_kernel_runner = nullptr;
};

struct KernelFallbackOpHandlerCompatTraits {
  using InputTensorTy = tfrt::Tensor;
  using OpEntryTy = CompatOpEntry;
  using OpHandlerInfoTy = KernelFallbackOpHandler*;

  static void Dispatch(const CompatOpEntry& op_entry,
                       KernelFallbackOpHandler* tf_op_handler,
                       llvm::ArrayRef<tfrt::Tensor*> inputs,
                       const OpAttrsRef& attrs,
                       llvm::ArrayRef<TensorMetadata> result_mds,
                       llvm::MutableArrayRef<RCReference<AsyncValue>> results,
                       AsyncValueRef<Chain>* chain,
                       const ExecutionContext& exec_ctx) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPSkernel_fallback_op_handlerDTcc mht_1(mht_1_v, 285, "", "./tensorflow/core/runtime_fallback/kernel/kernel_fallback_op_handler.cc", "Dispatch");

    auto ch = op_entry.dispatch_fn(
        exec_ctx, op_entry.op_name, tf_op_handler->device()->name(), inputs,
        results, *op_entry.fallback_request_state, *op_entry.op_kernel_runner);

    if (chain) *chain = std::move(ch);
  }

  // TODO(fishx): Remove this method.
  static tfrt::Variant<tfrt::RCReference<tfrt::Device>,
                       tfrt::AsyncValueRef<tfrt::RCReference<tfrt::Device>>>
  GetResultDevice(KernelFallbackOpHandler* kernel_fallback_op_handler,
                  const tfrt::AsyncValueRef<tfrt::Tensor>& result_tensor_av,
                  const ExecutionContext& exec_ctx) {
    return kernel_fallback_op_handler->GetDeviceRef();
  }

  static tfrt::Variant<tfrt::RCReference<tfrt::Device>,
                       tfrt::AsyncValueRef<tfrt::RCReference<tfrt::Device>>>
  GetResultDevice(const CompatOpEntry& op_entry,
                  KernelFallbackOpHandler* kernel_fallback_op_handler,
                  const tfrt::AsyncValueRef<tfrt::Tensor>& result_tensor_av,
                  int index, const ExecutionContext& exec_ctx) {
    auto* op_kernel = op_entry.op_kernel_runner->op_kernel();
    DCHECK(index < op_kernel->num_outputs());
    // NOTE: For DT_RESOURCE, we use the resource device as the device of the
    // resource handle.
    if (op_kernel->output_memory_types()[index] == MemoryType::HOST_MEMORY &&
        op_kernel->output_type(index) != DT_RESOURCE) {
      return exec_ctx.host()->GetHostDeviceRef();
    } else {
      return kernel_fallback_op_handler->GetDeviceRef();
    }
  }
};

class OpLocationKey {
 public:
  explicit OpLocationKey(tfrt::Location loc) : loc_(loc) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPSkernel_fallback_op_handlerDTcc mht_2(mht_2_v, 326, "", "./tensorflow/core/runtime_fallback/kernel/kernel_fallback_op_handler.cc", "OpLocationKey");
}

  template <typename H>
  friend H AbslHashValue(H h, const OpLocationKey& key) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPSkernel_fallback_op_handlerDTcc mht_3(mht_3_v, 332, "", "./tensorflow/core/runtime_fallback/kernel/kernel_fallback_op_handler.cc", "AbslHashValue");

    // NOTE: Each BEF file has its own LocationHandler. Using LocationHandler
    // as part of cache key here can avoid cache collision between different
    // BEF file.
    return H::combine(std::move(h), key.loc_.data, key.loc_.GetHandler());
  }

  friend bool operator==(const OpLocationKey& x, const OpLocationKey& y) {
    return x.loc_.data == y.loc_.data &&
           x.loc_.GetHandler() == y.loc_.GetHandler();
  }

 private:
  tfrt::Location loc_;
};

// OpKernelRunnerCache is similar to OpKernelRunnerTable but thread-safe.
class OpKernelRunnerCache {
 public:
  OpKernelRunnerCache() = default;

  StatusOr<OpKernelRunner*> GetOrCreate(
      tfrt::Location loc, absl::string_view op_name,
      absl::string_view device_name, int num_args,
      const std::function<Status(tensorflow::AttrValueMap*)>& attr_builder,
      const tensorflow::DeviceMgr& device_manager,
      const tensorflow::ProcessFunctionLibraryRuntime&
          process_function_library_runtime);

 private:
  mutable mutex mu_;
  absl::flat_hash_map<OpLocationKey, std::unique_ptr<OpKernelRunner>> map_
      TF_GUARDED_BY(mu_);
};

StatusOr<OpKernelRunner*> OpKernelRunnerCache::GetOrCreate(
    tfrt::Location loc, absl::string_view op_name,
    absl::string_view device_name, int num_args,
    const std::function<Status(tensorflow::AttrValueMap*)>& attr_builder,
    const tensorflow::DeviceMgr& device_manager,
    const tensorflow::ProcessFunctionLibraryRuntime&
        process_function_library_runtime) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("op_name: \"" + std::string(op_name.data(), op_name.size()) + "\"");
   mht_4_v.push_back("device_name: \"" + std::string(device_name.data(), device_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPSkernel_fallback_op_handlerDTcc mht_4(mht_4_v, 378, "", "./tensorflow/core/runtime_fallback/kernel/kernel_fallback_op_handler.cc", "OpKernelRunnerCache::GetOrCreate");

  OpLocationKey key(loc);
  {
    tf_shared_lock lock(mu_);
    auto it = map_.find(key);
    if (it != map_.end()) {
      DCHECK_EQ(it->second->op_kernel()->name(), op_name);
      return it->second.get();
    }
  }

  mutex_lock lock(mu_);

  auto it = map_.find(key);
  if (it != map_.end()) {
    DCHECK_EQ(it->second->op_kernel()->name(), op_name);
    return it->second.get();
  }

  VLOG(1) << "KernelFallbackExecuteCompat creating op " << op_name
          << " at location " << loc.data << " on device " << device_name;

  TF_ASSIGN_OR_RETURN(
      auto runner,
      OpKernelRunner::Create(op_name, device_name, num_args, attr_builder,
                             device_manager, process_function_library_runtime));

  auto runner_uptr = std::make_unique<OpKernelRunner>(std::move(runner));

  auto* runner_ptr = runner_uptr.get();
  auto r = map_.emplace(key, std::move(runner_uptr)).second;
  DCHECK(r);

  return runner_ptr;
}

}  // namespace

Expected<CoreRuntimeOp> KernelFallbackOpHandler::MakeOp(string_view op_name) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("op_name: \"" + std::string(op_name.data(), op_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPSkernel_fallback_op_handlerDTcc mht_5(mht_5_v, 420, "", "./tensorflow/core/runtime_fallback/kernel/kernel_fallback_op_handler.cc", "KernelFallbackOpHandler::MakeOp");

  // NOTE(fishx): Copying string here will cost extra overhead in graph
  // execution. Because in current implementation, we needs to prepare the op
  // before each executions.
  // TODO(fishx): Avoid this heap allocation by getting op registration
  // information from current TF.
  op_name.consume_front("tf.");
  return CoreRuntimeOp(
      [op_name = op_name.str(), this](const OpInvocation& invocation) {
        auto propagate_error = [&invocation](Status s) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPSkernel_fallback_op_handlerDTcc mht_6(mht_6_v, 432, "", "./tensorflow/core/runtime_fallback/kernel/kernel_fallback_op_handler.cc", "lambda");

          auto error = tfrt::EmitErrorAsync(
              invocation.exec_ctx,
              tfrt::StrCat("Error running kernel fallback OpHandler ",
                           invocation.op_name, ":", s.error_message()),
              tfrt::ConvertTfErrorCodeToTfrtErrorCode(s));
          for (auto& result : invocation.results) {
            result = tfrt::TensorHandle::CreateError(error.CopyRef());
          }
          if (invocation.chain) {
            *invocation.chain = error.CopyRef();
          }
        };
        // If the op does not have outputs, then it is expected to output an
        // out chain.
        bool update_chain = invocation.results.empty();
        CompatOpEntry fallback_op_entry;
        fallback_op_entry.op_name = std::move(op_name);

        // Convert the argument tensors to RuntimeFallbackTensors.
        for (auto& argument : invocation.arguments) {
          argument = argument.TransferToSameDevice(
              invocation.exec_ctx, KernelFallbackTensor::kTensorType);
        }

        fallback_op_entry.fallback_request_state =
            invocation.exec_ctx.request_ctx()
                ->GetDataIfExists<KernelFallbackCompatRequestState>();

        if (!fallback_op_entry.fallback_request_state) {
          propagate_error(tensorflow::errors::NotFound(
              "KernelFallbackCompatRequestState not found in RequestContext."));
          return;
        }

        DCHECK(invocation.exec_ctx.location());

        DCHECK(invocation.exec_ctx.request_ctx()->resource_context());
        auto* runner_cache = invocation.exec_ctx.request_ctx()
                                 ->resource_context()
                                 ->GetOrCreateResource<OpKernelRunnerCache>(
                                     kOpKernelRunnerCacheResourceName);

        auto kernel_runner_or_status = runner_cache->GetOrCreate(
            invocation.exec_ctx.location(),
            ToAbslStringView(fallback_op_entry.op_name),
            ToAbslStringView(device()->name()), invocation.arguments.size(),
            [&attrs = invocation.attrs, host = invocation.exec_ctx.host()](
                tensorflow::AttrValueMap* attr_value_map) {
              if (auto error =
                      tfd::FillAttrValueMap(attrs, host, attr_value_map))
                return tensorflow::errors::InvalidArgument(tfrt::StrCat(error));
              return Status::OK();
            },
            fallback_op_entry.fallback_request_state->device_manager(),
            fallback_op_entry.fallback_request_state
                ->process_function_library_runtime());

        if (!kernel_runner_or_status.ok()) {
          propagate_error(kernel_runner_or_status.status());
          return;
        }
        fallback_op_entry.op_kernel_runner =
            kernel_runner_or_status.ValueOrDie();

        tfrt::ExecuteOnOpHandler<KernelFallbackOpHandlerCompatTraits>(
            update_chain, invocation, fallback_op_entry, this);
      },
      // device and arg_tensor_type are currently not used in kernel fallback
      // ops.
      /*is_fallback=*/true, /*device=*/device_);
}

llvm::Expected<tfrt::OpHandler*> CreateKernelFallbackOpHandler(
    tfrt::CoreRuntime* runtime, tfrt::RCReference<tfrt::Device> device) {
  std::unique_ptr<KernelFallbackOpHandler> op_handler(
      new KernelFallbackOpHandler(runtime, std::move(device)));
  if (auto error = op_handler->Initialize()) {
    return std::move(error);
  }
  auto op_handler_ptr = op_handler.get();
  runtime->TakeOpHandler(std::move(op_handler));
  return op_handler_ptr;
}

KernelFallbackOpHandler::KernelFallbackOpHandler(
    CoreRuntime* runtime, RCReference<tfrt::Device> device)
    : OpHandler("tfkernel", runtime, nullptr), device_(std::move(device)) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPSkernel_fallback_op_handlerDTcc mht_7(mht_7_v, 522, "", "./tensorflow/core/runtime_fallback/kernel/kernel_fallback_op_handler.cc", "KernelFallbackOpHandler::KernelFallbackOpHandler");
}

KernelFallbackOpHandler::~KernelFallbackOpHandler() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPSkernel_fallback_op_handlerDTcc mht_8(mht_8_v, 527, "", "./tensorflow/core/runtime_fallback/kernel/kernel_fallback_op_handler.cc", "KernelFallbackOpHandler::~KernelFallbackOpHandler");
}

llvm::Error KernelFallbackOpHandler::Initialize() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPSkernel_fallback_op_handlerDTcc mht_9(mht_9_v, 532, "", "./tensorflow/core/runtime_fallback/kernel/kernel_fallback_op_handler.cc", "KernelFallbackOpHandler::Initialize");

  return llvm::Error::success();
}

}  // namespace tfd
}  // namespace tensorflow
