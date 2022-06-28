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
class MHTracer_DTPStensorflowPScorePStfrtPSeagerPStfrt_contextDTcc {
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
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPStfrt_contextDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePStfrtPSeagerPStfrt_contextDTcc() {
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

#include "tensorflow/core/tfrt/eager/tfrt_context.h"

#include <string>
#include <utility>

#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/runtime_fallback/runtime/kernel_utils.h"
#include "tensorflow/core/runtime_fallback/runtime/runtime_fallback_op_handler.h"
#include "tensorflow/core/tfrt/common/global_state.h"
#include "tensorflow/core/tfrt/eager/core_runtime/op_handler_registry.h"
#include "tensorflow/core/tpu/virtual_device.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tfrt/core_runtime/core_runtime.h"  // from @tf_runtime
#include "tfrt/host_context/concurrent_work_queue.h"  // from @tf_runtime
#include "tfrt/host_context/host_allocator.h"  // from @tf_runtime

namespace tfrt {
namespace tf {

TfrtContext::TfrtContext(
    const tensorflow::SessionOptions& opts,
    tensorflow::ContextDevicePlacementPolicy default_device_placement_policy,
    bool is_async) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPStfrt_contextDTcc mht_0(mht_0_v, 208, "", "./tensorflow/core/tfrt/eager/tfrt_context.cc", "TfrtContext::TfrtContext");

  tensorflow::tfd::EagerContextResource* eager_context_resource =
      resource_context_
          .GetOrCreateResource<tensorflow::tfd::EagerContextResource>(
              tensorflow::tfd::kEagerContextResourceName, opts,
              default_device_placement_policy, is_async);
  auto eager_context_expected = eager_context_resource->GetTFEagerContext();
  DCHECK(eager_context_expected) << StrCat(eager_context_expected.takeError());
  eager_context_ = eager_context_expected.get();

  eager_ctx_thread_pool_ = std::make_unique<ThreadPoolInterfaceWrapper>(
      eager_context_->GetThreadPool()->AsEigenThreadPool());

  local_thread_pool_.reset(tensorflow::NewThreadPoolFromSessionOptions(opts));

  local_thread_pool_wrapper_ = std::make_unique<ThreadPoolInterfaceWrapper>(
      local_thread_pool_->AsEigenThreadPool());

  tf_thread_pool_work_queue_ =
      std::make_unique<tensorflow::tfrt_stub::TfThreadPoolWorkQueue>(
          /*intra_op_threadpool=*/local_thread_pool_wrapper_.get(),
          /*inter_op_threadpool=*/eager_ctx_thread_pool_.get());
  LOG(INFO) << "Created work queue from TF thread pool. inter op thread pool "
            << "# threads: " << eager_ctx_thread_pool_->NumThreads()
            << " intra op thread pool # threads: "
            << local_thread_pool_wrapper_->NumThreads();

  // Default cpu device name is "/job:localhost/replica:0/task:0/device:CPU:0".
  const std::string& host_cpu_name = eager_context_->HostCPU()->name();

  auto diag_handler = [](const DecodedDiagnostic& diag) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPStfrt_contextDTcc mht_1(mht_1_v, 241, "", "./tensorflow/core/tfrt/eager/tfrt_context.cc", "lambda");

    LOG(ERROR) << diag.message;
  };

  auto rt = CoreRuntime::Create(diag_handler, CreateMallocAllocator(),
                                CreateMultiThreadedWorkQueue(
                                    /*num_threads=*/4,
                                    /*num_blocking_threads=*/64),
                                host_cpu_name);
  DCHECK(rt) << StrCat(rt.takeError());
  corert_ = std::move(rt.get());
  host_context_ = corert_->GetHostContext();

  // Create multiple (currently virtual) CPU devices according to options.
  // TODO(b/174877837): Support multiple physical cpu devices.
  int requested_num_cpus = 1;
  auto iter = opts.config.device_count().find("CPU");
  if (iter != opts.config.device_count().end()) {
    requested_num_cpus = iter->second;
  }

  std::string cpu_name_prefix{host_cpu_name};
  cpu_name_prefix.pop_back();  // remove the `id` from host cpu device name.
  for (int i = 1; i < requested_num_cpus; ++i) {
    host_context_->GetDeviceManager()->MaybeAddDevice(TakeRef(
        new CpuDevice(absl::StrCat(cpu_name_prefix, std::to_string(i)))));
  }

  // Specifically register RuntimeFallbackOpHandler.
  auto runtime_fallback_op_handler =
      tensorflow::tfd::CreateRuntimeFallbackOpHandler(corert_.get(), "");
  DCHECK(runtime_fallback_op_handler)
      << StrCat(runtime_fallback_op_handler.takeError());
  fallback_op_handler_ = runtime_fallback_op_handler.get();
  corert_->RegisterOpHandler("tf", fallback_op_handler_);

  RegisterOpHandlers(corert_.get(), &resource_context_,
                     eager_context_->local_device_mgr());

  // Set the global host context singleton.
  tensorflow::tfrt_global::GlobalHostContext::Set(corert_->GetHostContext());
}

const tensorflow::DeviceNameUtils::ParsedName& TfrtContext::HostCPUParsedName()
    const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPStfrt_contextDTcc mht_2(mht_2_v, 288, "", "./tensorflow/core/tfrt/eager/tfrt_context.cc", "TfrtContext::HostCPUParsedName");

  return eager_context_->HostCPU()->parsed_name();
}

bool TfrtContext::IsAsync() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPStfrt_contextDTcc mht_3(mht_3_v, 295, "", "./tensorflow/core/tfrt/eager/tfrt_context.cc", "TfrtContext::IsAsync");
 return eager_context_->Executor().Async(); }

TfrtContext::~TfrtContext() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPStfrt_contextDTcc mht_4(mht_4_v, 300, "", "./tensorflow/core/tfrt/eager/tfrt_context.cc", "TfrtContext::~TfrtContext");
}

}  // namespace tf
}  // namespace tfrt
