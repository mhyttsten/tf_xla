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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSutilsPSsimple_delegateDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSutilsPSsimple_delegateDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSutilsPSsimple_delegateDTcc() {
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
#include "tensorflow/lite/delegates/utils/simple_delegate.h"

#include <limits>
#include <memory>
#include <vector>

#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/delegates/utils.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/minimal_logging.h"

namespace tflite {
namespace {
TfLiteRegistration GetDelegateKernelRegistration(
    SimpleDelegateInterface* delegate) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSutilsPSsimple_delegateDTcc mht_0(mht_0_v, 200, "", "./tensorflow/lite/delegates/utils/simple_delegate.cc", "GetDelegateKernelRegistration");

  TfLiteRegistration kernel_registration;
  kernel_registration.profiling_string = nullptr;
  kernel_registration.builtin_code = kTfLiteBuiltinDelegate;
  kernel_registration.custom_name = delegate->Name();
  kernel_registration.version = 1;
  kernel_registration.free = [](TfLiteContext* context, void* buffer) -> void {
    delete reinterpret_cast<SimpleDelegateKernelInterface*>(buffer);
  };
  kernel_registration.init = [](TfLiteContext* context, const char* buffer,
                                size_t length) -> void* {
    const TfLiteDelegateParams* params =
        reinterpret_cast<const TfLiteDelegateParams*>(buffer);
    if (params == nullptr) {
      TF_LITE_KERNEL_LOG(context, "NULL TfLiteDelegateParams passed.");
      return nullptr;
    }
    auto* delegate =
        reinterpret_cast<SimpleDelegateInterface*>(params->delegate->data_);
    std::unique_ptr<SimpleDelegateKernelInterface> delegate_kernel(
        delegate->CreateDelegateKernelInterface());
    if (delegate_kernel->Init(context, params) != kTfLiteOk) {
      return nullptr;
    }
    return delegate_kernel.release();
  };
  kernel_registration.prepare = [](TfLiteContext* context,
                                   TfLiteNode* node) -> TfLiteStatus {
    if (node->user_data == nullptr) {
      TF_LITE_KERNEL_LOG(context, "Delegate kernel was not initialized");
      return kTfLiteError;
    }
    SimpleDelegateKernelInterface* delegate_kernel =
        reinterpret_cast<SimpleDelegateKernelInterface*>(node->user_data);
    return delegate_kernel->Prepare(context, node);
  };
  kernel_registration.invoke = [](TfLiteContext* context,
                                  TfLiteNode* node) -> TfLiteStatus {
    SimpleDelegateKernelInterface* delegate_kernel =
        reinterpret_cast<SimpleDelegateKernelInterface*>(node->user_data);
    TFLITE_DCHECK(delegate_kernel != nullptr);
    return delegate_kernel->Eval(context, node);
  };

  return kernel_registration;
}

TfLiteStatus DelegatePrepare(TfLiteContext* context,
                             TfLiteDelegate* base_delegate) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSutilsPSsimple_delegateDTcc mht_1(mht_1_v, 251, "", "./tensorflow/lite/delegates/utils/simple_delegate.cc", "DelegatePrepare");

  auto* delegate =
      reinterpret_cast<SimpleDelegateInterface*>(base_delegate->data_);
  auto delegate_options = delegate->DelegateOptions();
  if (delegate_options.max_delegated_partitions <= 0)
    delegate_options.max_delegated_partitions = std::numeric_limits<int>::max();

  TF_LITE_ENSURE_STATUS(delegate->Initialize(context));
  delegates::IsNodeSupportedFn node_supported_fn =
      [=](TfLiteContext* context, TfLiteNode* node,
          TfLiteRegistration* registration,
          std::string* unsupported_details) -> bool {
    return delegate->IsNodeSupportedByDelegate(registration, node, context);
  };
  // TODO(b/149484598): Update to have method that gets all supported nodes.
  delegates::GraphPartitionHelper helper(context, node_supported_fn);
  TF_LITE_ENSURE_STATUS(helper.Partition(nullptr));

  std::vector<int> supported_nodes = helper.GetNodesOfFirstNLargestPartitions(
      delegate_options.max_delegated_partitions,
      delegate_options.min_nodes_per_partition);

  TFLITE_LOG_PROD_ONCE(tflite::TFLITE_LOG_INFO,
                       "%s delegate: %d nodes delegated out of %d nodes with "
                       "%d partitions.\n",
                       delegate->Name(), supported_nodes.size(),
                       helper.num_total_nodes(), helper.num_partitions());
  TfLiteRegistration delegate_kernel_registration =
      GetDelegateKernelRegistration(delegate);

  return context->ReplaceNodeSubsetsWithDelegateKernels(
      context, delegate_kernel_registration,
      BuildTfLiteIntArray(supported_nodes).get(), base_delegate);
}
}  // namespace

TfLiteDelegate* TfLiteDelegateFactory::CreateSimpleDelegate(
    std::unique_ptr<SimpleDelegateInterface> simple_delegate, int64_t flag) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSutilsPSsimple_delegateDTcc mht_2(mht_2_v, 291, "", "./tensorflow/lite/delegates/utils/simple_delegate.cc", "TfLiteDelegateFactory::CreateSimpleDelegate");

  if (simple_delegate == nullptr) {
    return nullptr;
  }
  auto delegate = new TfLiteDelegate();
  delegate->Prepare = &DelegatePrepare;
  delegate->flags = flag;
  delegate->CopyFromBufferHandle = nullptr;
  delegate->CopyToBufferHandle = nullptr;
  delegate->FreeBufferHandle = nullptr;
  delegate->data_ = simple_delegate.release();
  return delegate;
}

void TfLiteDelegateFactory::DeleteSimpleDelegate(TfLiteDelegate* delegate) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSutilsPSsimple_delegateDTcc mht_3(mht_3_v, 308, "", "./tensorflow/lite/delegates/utils/simple_delegate.cc", "TfLiteDelegateFactory::DeleteSimpleDelegate");

  if (!delegate) return;
  SimpleDelegateInterface* simple_delegate =
      reinterpret_cast<SimpleDelegateInterface*>(delegate->data_);
  delete simple_delegate;
  delete delegate;
}
}  // namespace tflite
